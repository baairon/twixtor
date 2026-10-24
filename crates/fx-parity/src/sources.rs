//! Test patterns, written as ffmpeg `geq` expressions so every depth gets exact
//! values instead of a lossy YUV round trip. The stills feed frame effects; the
//! clip sources (a few frames with real motion) feed temporal effects.

pub struct Source {
    pub name: &'static str,
    /// Depths the pattern exists at.
    pub depths: &'static [u32],
}

pub const SOURCES: &[Source] = &[
    // red, green and blue ramps across, down and along the diagonal, opaque
    Source { name: "gradient", depths: &[8, 16, 32] },
    // SMPTE style bars: 75% bars, reverse bars, -I / white / +Q / PLUGE row
    Source { name: "bars", depths: &[8, 16, 32] },
    // fixed hash noise in every colour channel, opaque
    Source { name: "noise", depths: &[8, 16, 32] },
    // alpha ramps 0..1 across while the colour changes down, so colour under zero alpha is non-trivial
    Source { name: "alpha", depths: &[8, 16, 32] },
    // gradient stretched to -0.5..1.5, the unclamped values only a float project carries
    Source { name: "range", depths: &[32] },
];

pub fn find(name: &str) -> Option<&'static Source> {
    SOURCES.iter().find(|s| s.name == name)
}

/// ffmpeg raw pixel format for a depth, as `vfx` reads and writes it.
pub fn pix_fmt(depth: u32) -> &'static str {
    match depth {
        8 => "rgba",
        16 => "rgba64le",
        _ => "gbrapf32le",
    }
}

/// Position along each axis, 0..1 with both end points hit. A one pixel axis sits at the middle.
pub const FX: &str = "if(eq(W,1),0.5,X/(W-1))";
pub const FY: &str = "if(eq(H,1),0.5,Y/(H-1))";
pub const FD: &str = "if(eq(W+H,2),0.5,(X+Y)/(W+H-2))";

/// Modulus of the noise hash. Squares of residues stay below 2^53, so doubles are exact.
pub const NOISE_MOD: u64 = 67_108_859;

fn noise_channel(c: u32) -> String {
    let m = NOISE_MOD;
    format!(
        "st(1,mod(((Y*W+X)*4+{c}+1)*40503+12345,{m}));\
         st(2,mod(ld(1)*ld(1)+7919,{m}));\
         st(3,mod(ld(2)*48271+ld(1),{m}));\
         st(4,mod(ld(3)*ld(3)+ld(2),{m}));\
         st(5,mod(ld(4)*69621+ld(3),{m}));\
         ld(5)/{m}"
    )
}

fn bars_channel(top: &str, mid: &str, bottom: [f64; 6]) -> String {
    // bar index across 7 bars, and position across 21 slots for the bottom row
    let k = "min(floor(X*7/W),6)";
    let u = "floor(X*21/W)";
    let k_top = top.replace('K', k);
    let k_mid = mid.replace('K', k);
    // -I, white, +Q, black, PLUGE (-4% clamps to 0, 0, +4%), black
    let b = bottom;
    let row = format!(
        "if(lt({u},3),{},if(lt({u},6),{},if(lt({u},9),{},if(lt({u},17),{},if(lt({u},18),{},{})))))",
        b[0], b[1], b[2], b[3], b[4], b[5]
    );
    format!("if(lt(Y*12,H*8),{k_top},if(lt(Y*12,H*9),{k_mid},{row}))")
}

/// Red, green, blue and alpha expressions for a pattern. 0..1 spans the depth's range.
pub fn channels(name: &str) -> [String; 4] {
    let s = |v: &str| v.to_string();
    match name {
        "gradient" => [s(FX), s(FY), s(FD), s("1")],
        "alpha" => [s(FY), format!("1-{FY}"), s("0.5"), s(FX)],
        "range" => [format!("{FX}*2-0.5"), format!("{FY}*2-0.5"), format!("1.5-{FD}*2"), s("1")],
        "noise" => [noise_channel(0), noise_channel(1), noise_channel(2), s("1")],
        "bars" => [
            bars_channel("0.75*lt(mod(K,4),2)", "0.75*eq(mod(K,2),0)*eq(mod(K,4),2)", [0.0, 1.0, 0.196, 0.0, 0.04, 0.0]),
            bars_channel("0.75*lt(K,4)", "0.75*eq(mod(K,2),0)*gte(K,4)", [0.129, 1.0, 0.0, 0.0, 0.04, 0.0]),
            bars_channel("0.75*eq(mod(K,2),0)", "0.75*eq(mod(K,2),0)", [0.298, 1.0, 0.416, 0.0, 0.04, 0.0]),
            s("1"),
        ],
        other => panic!("unknown source `{other}`"),
    }
}

/// The lavfi graph that renders one frame of `name` at `depth` and `w`x`h`.
pub fn graph(name: &str, depth: u32, w: usize, h: usize) -> String {
    let (planar, max) = match depth {
        8 => ("gbrap", Some(255)),
        16 => ("gbrap16le", Some(65535)),
        _ => ("gbrapf32le", None),
    };
    let quantise = |e: &str| match max {
        Some(m) => format!("clip(round(({e})*{m}),0,{m})"),
        None => format!("({e})"),
    };
    let [r, g, b, a] = channels(name).map(|e| quantise(&e));
    format!("color=c=black:s={w}x{h}:r=1,format={planar},geq=r='{r}':g='{g}':b='{b}':a='{a}'")
}

// ---- clips for temporal effects -------------------------------------------------------------

/// A moving test pattern, rendered as a clip of several frames.
pub struct ClipSource {
    pub name: &'static str,
}

/// The names stay apart from the stills', so a source name says which generator makes it.
pub const CLIP_SOURCES: &[ClipSource] = &[
    // ffmpeg's own testsrc2: coloured shapes, a running counter, gradients, with alpha 255
    ClipSource { name: "testsrc2" },
    // opaque gradients whose phase moves a little every frame
    ClipSource { name: "movegrad" },
    // the `noise` hash with the frame number mixed in: new noise every frame, fixed seed
    ClipSource { name: "noisefx" },
    // colour ramps with an alpha ramp that slides across the frame
    ClipSource { name: "alphaslide" },
];

pub fn find_clip(name: &str) -> Option<&'static ClipSource> {
    CLIP_SOURCES.iter().find(|s| s.name == name)
}

/// The shorter side testsrc2 is drawn with before it is averaged down to the clip size.
const TESTSRC_MIN_SIDE: usize = 192;

/// Frame rate every clip is generated at. A case runs it at the rate its `fps` says;
/// the raw frames carry no rate, so the files do not depend on that choice.
pub const CLIP_RENDER_RATE: &str = "24";

/// `noise_channel` with the frame number `N` mixed into the hash input.
/// Frame 0 therefore equals the `noise` still.
fn clip_noise_channel(c: u32) -> String {
    let m = NOISE_MOD;
    [
        format!("st(1,mod(((Y*W+X)*4+{c}+1+N*1000003)*40503+12345,{m}));"),
        format!("st(2,mod(ld(1)*ld(1)+7919,{m}));"),
        format!("st(3,mod(ld(2)*48271+ld(1),{m}));"),
        format!("st(4,mod(ld(3)*ld(3)+ld(2),{m}));"),
        format!("st(5,mod(ld(4)*69621+ld(3),{m}));"),
        format!("ld(5)/{m}"),
    ]
    .concat()
}

/// Red, green, blue and alpha expressions of a `geq` based clip source; `N` is the frame
/// number. `None` for sources ffmpeg draws itself.
pub fn clip_channels(name: &str, frames: usize) -> Option<[String; 4]> {
    let s = |v: &str| v.to_string();
    let span = frames.max(1);
    Some(match name {
        "movegrad" => [
            format!("mod({FX}+N*0.07,1)"),
            format!("mod({FY}+N*0.045,1)"),
            format!("mod({FD}+N*0.11,1)"),
            s("1"),
        ],
        "noisefx" => [clip_noise_channel(0), clip_noise_channel(1), clip_noise_channel(2), s("1")],
        "alphaslide" => [s(FY), format!("1-{FY}"), s("0.5"), format!("clip({FX}*2-N*2/{span}+0.5,0,1)")],
        _ => return None,
    })
}

/// The lavfi graph that renders a clip of `name` at `depth` and `w`x`h`, `frames` long
/// (the caller stops the stream with `-frames:v`).
pub fn clip_graph(name: &str, depth: u32, w: usize, h: usize, frames: usize) -> String {
    let Some(channels) = clip_channels(name, frames) else {
        // testsrc2 draws 8 bit RGB; the output conversion widens it for the deeper formats.
        // Its motion is slower than a pixel at tiny sizes (a 16x12 clip stands still), so it is
        // drawn with at least 192 lines and averaged down.
        let k = TESTSRC_MIN_SIDE.div_ceil(w.min(h)).max(1);
        let down = if k > 1 { format!(",scale={w}:{h}:flags=area") } else { String::new() };
        return format!("{name}=s={}x{}:r={CLIP_RENDER_RATE}:alpha=255{down},format=rgba", w * k, h * k);
    };
    let (planar, max) = match depth {
        8 => ("gbrap", Some(255)),
        16 => ("gbrap16le", Some(65535)),
        _ => ("gbrapf32le", None),
    };
    let quantise = |e: &str| match max {
        Some(m) => format!("clip(round(({e})*{m}),0,{m})"),
        None => format!("({e})"),
    };
    let [r, g, b, a] = channels.map(|e| quantise(&e));
    format!("color=c=black:s={w}x{h}:r={CLIP_RENDER_RATE},format={planar},geq=r='{r}':g='{g}':b='{b}':a='{a}'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphs_are_well_formed_for_every_source_and_depth() {
        for src in SOURCES {
            for &depth in src.depths {
                let g = graph(src.name, depth, 37, 23);
                let depth_of = |c: char| g.chars().filter(|x| *x == c).count();
                assert_eq!(depth_of('('), depth_of(')'), "{} @{depth}", src.name);
                assert_eq!(depth_of('\''), 8, "four quoted expressions: {} @{depth}", src.name);
                assert!(!g.contains(" "), "no spaces to escape");
            }
        }
    }

    #[test]
    fn every_source_has_a_pix_fmt_and_is_findable() {
        for src in SOURCES {
            assert!(find(src.name).is_some());
            for &d in src.depths {
                assert!(["rgba", "rgba64le", "gbrapf32le"].contains(&pix_fmt(d)));
            }
        }
        assert!(find("nope").is_none());
    }

    #[test]
    fn clip_graphs_are_well_formed_for_every_source_and_depth() {
        for src in CLIP_SOURCES {
            for depth in [8u32, 16, 32] {
                let g = clip_graph(src.name, depth, 37, 23, 12);
                let count = |c: char| g.chars().filter(|x| *x == c).count();
                assert_eq!(count('('), count(')'), "{} @{depth}", src.name);
                assert!(!g.contains(' '), "no spaces to escape: {g}");
                if clip_channels(src.name, 12).is_some() {
                    assert_eq!(count('\''), 8, "four quoted expressions: {} @{depth}", src.name);
                    assert!(g.contains('N'), "a clip source moves with the frame number: {}", src.name);
                } else {
                    // 23 lines are drawn with 9 times as many, then averaged down to 37x23
                    assert!(g.starts_with("testsrc2=s=333x207:") && g.contains(",scale=37:23:flags=area,"), "{g}");
                }
            }
        }
    }

    #[test]
    fn testsrc2_is_drawn_large_enough_to_move_and_left_alone_when_it_already_is() {
        assert!(clip_graph("testsrc2", 8, 16, 12, 12).starts_with("testsrc2=s=256x192:"));
        assert!(clip_graph("testsrc2", 8, 5, 4, 12).starts_with("testsrc2=s=240x192:"));
        let big = clip_graph("testsrc2", 8, 320, 180, 12);
        assert!(big.starts_with("testsrc2=s=640x360:") && big.contains("scale=320:180:flags=area"), "{big}");
        let native = clip_graph("testsrc2", 8, 640, 480, 12);
        assert!(native.starts_with("testsrc2=s=640x480:") && !native.contains("scale="), "{native}");
    }

    #[test]
    fn still_and_clip_names_never_collide() {
        for c in CLIP_SOURCES {
            assert!(find(c.name).is_none(), "{} is a still too", c.name);
            assert!(find_clip(c.name).is_some());
        }
        assert!(find_clip("gradient").is_none());
    }

    #[test]
    fn the_first_noisefx_frame_is_the_noise_still() {
        let [clip_r, ..] = clip_channels("noisefx", 12).unwrap();
        let [still_r, ..] = channels("noise");
        assert_eq!(clip_r.replace("+N*1000003", ""), still_r);
    }

    #[test]
    fn noise_hash_stays_exact_in_doubles() {
        let m = NOISE_MOD as f64;
        assert!(m * m < 9.007e15);
        assert!(m * 69621.0 + m < 9.007e15);
        // the clip variant adds N * 1000003 to the first state before multiplying by 40503
        let first = ((1000.0 * 1000.0 * 4.0 + 4.0) + 99_999.0 * 1_000_003.0) * 40503.0 + 12345.0;
        assert!(first < 9.007e15, "clips of 100k frames at 1000x1000 stay exact");
    }
}

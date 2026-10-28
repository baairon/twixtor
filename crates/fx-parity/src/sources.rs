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

// ---- clips for the reference case families ---------------------------------------------------

/// A clip made for the reference case families of the open parity gaps (`fx-parity gen
/// --families`). These stay out of [`CLIP_SOURCES`], whose order hands out the rates and the
/// layer clips of the default matrix, so no default case changes when one is added here.
pub struct FamilySource {
    pub name: &'static str,
    /// What the clip shows, for the checklist.
    pub about: &'static str,
}

/// The family clips. Motion is in whole pixels per frame and every expression uses only `+ - * /`,
/// `floor`, `mod`, `abs`, `sqrt`, `min`, `max`, `clip` and comparisons, which IEEE 754 rounds the
/// same on every machine, so a family input renders to the same bytes wherever it is generated.
pub const FAMILY_SOURCES: &[FamilySource] = &[
    FamilySource { name: "fgdisc", about: "a textured disc moving 2 px right and 1 px down per frame over a still textured background, hard edged" },
    FamilySource { name: "fgdisc_matte", about: "the hard edged matte of the fgdisc disc: white inside, black outside, opaque" },
    FamilySource { name: "fgdisc_soft", about: "the fgdisc disc as a soft edged matte, a linear ramp 4 px wide across its edge" },
    FamilySource {
        name: "fg2disc",
        about: "two textured discs crossing over a still background: disc A as in fgdisc and in front, disc B moving 2 px left and 1 px down behind it",
    },
    FamilySource { name: "fg2disc_matte", about: "the two overlapping hard mattes of fg2disc: disc A in red, disc B in green, opaque" },
    FamilySource { name: "fgvec_prev", about: "the Vectors: Prev image of the fgdisc disc at Max Displace 8: (-2, -1) inside the disc, zero outside" },
    FamilySource { name: "fgvec_next", about: "the Vectors: Next image of the fgdisc disc at Max Displace 8: (2, 1) inside the disc, zero outside" },
    FamilySource { name: "wraptex", about: "a texture periodic across the width, scrolling 3 px right per frame through the left and right edges" },
    FamilySource { name: "wrappole", about: "the wraptex texture scrolling 3 px right and 2 px up per frame, so the motion also crosses the top edge" },
    FamilySource { name: "alphaonly", about: "flat grey colour, with the wraptex pattern scrolling in the alpha channel only" },
    FamilySource { name: "stilltex", about: "a still texture, the same in every frame" },
    FamilySource { name: "flatgrey", about: "flat 50% grey, opaque, the same in every frame" },
];

pub fn find_family(name: &str) -> Option<&'static FamilySource> {
    FAMILY_SOURCES.iter().find(|s| s.name == name)
}

/// The Max Displace the `fgvec_*` images are coded for; a case that reads them sets
/// `vectors_max_displace` to it.
pub const FAMILY_VECTOR_SCALE: u32 = 8;

/// A triangle wave of period 1 at `t`, from 0 to 1.
fn tri(t: &str) -> String {
    format!("abs(mod({t},1)-0.5)*2")
}

/// A plaid of four triangle waves at `(u, v)`, 0..1; `k` picks one of two sets of periods.
fn plaid(u: &str, v: &str, k: u32) -> String {
    let p: [f64; 4] = if k == 0 { [7.3, 5.1, 9.7, 11.3] } else { [6.1, 8.9, 4.7, 13.7] };
    format!(
        "({}+{}+{}+{})/4",
        tri(&format!("({u})/{}", p[0])),
        tri(&format!("({v})/{}", p[1])),
        tri(&format!("(({u})+({v}))/{}", p[2])),
        tri(&format!("(({u})-2*({v}))/{}", p[3]))
    )
}

/// A texture periodic across the width: each term moves by whole periods when `u` grows by W.
fn wrap_plaid(u: &str, v: &str, k: u32) -> String {
    let (a, b, c) = if k == 0 { (2, 3, 5) } else { (1, 4, 7) };
    format!(
        "({}+{}+{})/3",
        tri(&format!("{a}*({u})/W+({v})/9.1")),
        tri(&format!("{b}*({u})/W-({v})/6.7")),
        tri(&format!("{c}*({u})/W+({v})/13.3"))
    )
}

/// Disc A of the family clips, centred at `(floor(0.3 W) + 2N, floor(0.35 H) + N)`.
const DISC_A: (&str, &str) = ("floor(0.3*W)+2*N", "floor(0.35*H)+N");
/// Disc B of `fg2disc`, centred at `(floor(0.65 W) - 2N, floor(0.3 H) + N)`.
const DISC_B: (&str, &str) = ("floor(0.65*W)-2*N", "floor(0.3*H)+N");
/// The radius of both discs.
const RADIUS: &str = "max(1,floor(0.22*min(W,H)))";

/// The squared distance of the pixel from a disc centre.
fn dist2((cx, cy): (&str, &str)) -> String {
    format!("(X-({cx}))*(X-({cx}))+(Y-({cy}))*(Y-({cy}))")
}

/// 1 inside the disc (a pixel centre at most the radius away), else 0.
fn inside(c: (&str, &str)) -> String {
    format!("lte({},{RADIUS}*{RADIUS})", dist2(c))
}

/// The still background of the disc clips, per channel.
fn background(c: usize) -> String {
    match c {
        0 => format!("0.8*{}", plaid("X", "Y", 1)),
        1 => format!("0.8*{}", plaid("X", "Y", 0)),
        _ => "0.4".to_string(),
    }
}

/// A disc's texture, carried along with the disc: sampled at the pixel minus the disc's offset
/// `(dx, dy)` since frame 0. `swap` gives disc B other colours than disc A.
fn disc_colour(c: usize, (dx, dy): (&str, &str), swap: bool) -> String {
    let (u, v) = (format!("X-({dx})"), format!("Y-({dy})"));
    let k = u32::from(swap);
    match c {
        0 => plaid(&u, &v, k),
        1 => plaid(&u, &v, 1 - k),
        _ => format!("0.25+0.5*{}", plaid(&u, &v, k)),
    }
}

/// Red, green, blue and alpha expressions of a family clip; `N` is the frame number. None for a
/// name that is not a family clip.
pub fn family_channels(name: &str) -> Option<[String; 4]> {
    let s = |v: &str| v.to_string();
    let rgb = |f: &dyn Fn(usize) -> String| [f(0), f(1), f(2), s("1")];
    let vector = |sign: &str| {
        let (a, scale) = (inside(DISC_A), FAMILY_VECTOR_SCALE);
        [format!("0.5+0.5*({sign}2)*{a}/{scale}"), format!("0.5+0.5*({sign}1)*{a}/{scale}"), s("0"), s("1")]
    };
    let wrap = |v: &str| {
        let u = "X-3*N";
        [wrap_plaid(u, v, 0), wrap_plaid(u, v, 1), format!("0.2+0.6*{}", wrap_plaid(u, v, 0)), s("1")]
    };
    Some(match name {
        "fgdisc" => rgb(&|c| format!("if({},{},{})", inside(DISC_A), disc_colour(c, ("2*N", "N"), false), background(c))),
        "fgdisc_matte" => {
            let m = inside(DISC_A);
            [m.clone(), m.clone(), m, s("1")]
        }
        "fgdisc_soft" => {
            let m = format!("clip(({RADIUS}+2-sqrt({}))/4,0,1)", dist2(DISC_A));
            [m.clone(), m.clone(), m, s("1")]
        }
        "fg2disc" => rgb(&|c| {
            let a = disc_colour(c, ("2*N", "N"), false);
            let b = disc_colour(c, ("-2*N", "N"), true);
            format!("if({},{a},if({},{b},{}))", inside(DISC_A), inside(DISC_B), background(c))
        }),
        "fg2disc_matte" => [inside(DISC_A), inside(DISC_B), s("0"), s("1")],
        "fgvec_prev" => vector("-"),
        "fgvec_next" => vector(""),
        "wraptex" => wrap("Y"),
        "wrappole" => wrap("Y+2*N"),
        "alphaonly" => [s("0.5"), s("0.5"), s("0.5"), wrap_plaid("X-3*N", "Y", 0)],
        "stilltex" => rgb(&|c| match c {
            0 => plaid("X", "Y", 0),
            1 => plaid("X", "Y", 1),
            _ => format!("0.3+0.4*{}", plaid("Y", "X", 0)),
        }),
        "flatgrey" => [s("0.5"), s("0.5"), s("0.5"), s("1")],
        _ => return None,
    })
}

/// The lavfi graph that renders a family clip at `depth` and `w`x`h` (the caller stops the
/// stream with `-frames:v`), quantised like the `geq` clips of [`clip_graph`].
pub fn family_graph(name: &str, depth: u32, w: usize, h: usize) -> Option<String> {
    let channels = family_channels(name)?;
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
    Some(format!("color=c=black:s={w}x{h}:r={CLIP_RENDER_RATE},format={planar},geq=r='{r}':g='{g}':b='{b}':a='{a}'"))
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

    #[test]
    fn family_clips_stay_out_of_the_matrix_rotation_and_never_collide() {
        let matrix: Vec<&str> = CLIP_SOURCES.iter().map(|s| s.name).collect();
        assert_eq!(matrix, ["testsrc2", "movegrad", "noisefx", "alphaslide"], "the rotation of the default matrix is untouched");
        for (i, f) in FAMILY_SOURCES.iter().enumerate() {
            assert!(find(f.name).is_none() && find_clip(f.name).is_none(), "{} is also a still or a matrix clip", f.name);
            assert!(FAMILY_SOURCES[..i].iter().all(|o| o.name != f.name), "{} twice", f.name);
            assert!(find_family(f.name).is_some() && family_channels(f.name).is_some(), "{}", f.name);
            assert!(f.about.is_ascii() && !f.about.is_empty(), "{}", f.name);
            assert!(f.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'), "{} is file safe", f.name);
        }
        for name in ["fgdisc", "fgdisc_matte", "fgdisc_soft", "fg2disc", "wraptex", "alphaonly", "stilltex"] {
            assert!(find_family(name).is_some(), "{name}");
        }
        assert!(find_family("testsrc2").is_none() && family_channels("noise").is_none() && family_graph("movegrad", 8, 4, 4).is_none());
    }

    #[test]
    fn family_graphs_are_well_formed_and_use_only_exactly_rounded_operations() {
        for f in FAMILY_SOURCES {
            for depth in [8u32, 16, 32] {
                let g = family_graph(f.name, depth, 64, 48).unwrap();
                let count = |c: char| g.chars().filter(|x| *x == c).count();
                assert_eq!(count('('), count(')'), "{} @{depth}", f.name);
                assert_eq!(count('\''), 8, "four quoted expressions: {} @{depth}", f.name);
                assert!(!g.contains(' '), "no spaces to escape: {g}");
                assert!(g.starts_with("color=c=black:s=64x48:r=24,format="), "{g}");
                // a word that is not one of these would be a function whose last bits a C library chooses
                let allowed = ["abs", "mod", "floor", "sqrt", "min", "max", "clip", "round", "lte", "if", "color", "black", "format", "gbrap", "gbrap16le", "gbrapf32le", "geq", "c", "s", "r", "g", "b", "a", "x", "le", "f32le", "W", "H", "X", "Y", "N"];
                let words = g.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| w.chars().next().is_some_and(|c| c.is_ascii_alphabetic()));
                for w in words {
                    assert!(allowed.contains(&w), "{}: `{w}` in {g}", f.name);
                }
            }
        }
        // the clips that move use the frame number, the still ones do not
        for (name, moves) in [("fgdisc", true), ("fgdisc_matte", true), ("fg2disc_matte", true), ("wraptex", true), ("alphaonly", true), ("fgvec_next", true), ("stilltex", false), ("flatgrey", false)] {
            let [r, g, b, a] = family_channels(name).unwrap();
            assert_eq!([r, g, b, a].iter().any(|e| e.contains('N')), moves, "{name}");
        }
        // only alphaonly moves in alpha
        assert!(family_channels("alphaonly").unwrap()[3].contains('N'));
        assert!(FAMILY_SOURCES.iter().filter(|f| f.name != "alphaonly").all(|f| family_channels(f.name).unwrap()[3] == "1"));
    }

    #[test]
    fn the_disc_mattes_match_the_disc_clip() {
        // the matte and the colour clip pick their pixels with the same test
        let [mr, mg, mb, _] = family_channels("fgdisc_matte").unwrap();
        assert!(mr == mg && mg == mb);
        assert!(family_channels("fgdisc").unwrap()[0].starts_with(&format!("if({mr},")));
        let [a, b, _, _] = family_channels("fg2disc_matte").unwrap();
        assert_eq!(a, mr, "disc A of fg2disc is the fgdisc disc");
        assert!(family_channels("fg2disc").unwrap()[1].contains(&format!("if({b},")), "disc B sits behind disc A");
        // the vector images are coded for the scale the cases set and are opposite to each other
        let next = family_channels("fgvec_next").unwrap();
        let prev = family_channels("fgvec_prev").unwrap();
        assert!(next[0].ends_with(&format!("/{FAMILY_VECTOR_SCALE}")) && next[0].contains("(2)*") && prev[0].contains("(-2)*"));
        assert!(next[0].contains(&mr) && next[2] == "0" && next[3] == "1");
    }
}

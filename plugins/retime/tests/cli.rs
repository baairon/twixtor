//! End to end through the real `vfx` program: an ffmpeg `testsrc2` clip as raw frames,
//! `vfx apply retime` with a frame blend retime, and the output compared with values computed
//! independently here.
//!
//! The test builds `vfx` with the cargo that runs it (`cargo build -p fx-cli`, in the same target
//! directory and profile) and needs `ffmpeg` on the PATH; without ffmpeg it prints a note and
//! passes, so a machine without it stays green. The pixel formats are the ones `vfx` reads and
//! writes: `rgba`, `rgba64le`, `gbrapf32le` (planar G B R A).

mod common;

use fx_core::{Depth, Frame, Pixels};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const W: usize = 64;
const H: usize = 48;
const SOURCE_FRAMES: usize = 12;

fn workspace() -> PathBuf {
    // plugins/retime -> the workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).expect("workspace root").to_path_buf()
}

/// Builds `vfx` once and returns its path, or None when it cannot be built.
fn vfx() -> Option<&'static Path> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let profile_dir = exe.parent()?.parent()?.to_path_buf(); // target/<profile>/deps/cli-hash -> target/<profile>
        let release = profile_dir.file_name().is_some_and(|n| n == "release");
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let mut build = Command::new(cargo);
        build.args(["build", "-p", "fx-cli", "--quiet"]).current_dir(workspace());
        if release {
            build.arg("--release");
        }
        if !build.status().ok()?.success() {
            return None;
        }
        let bin = profile_dir.join(if cfg!(windows) { "vfx.exe" } else { "vfx" });
        bin.exists().then_some(bin)
    })
    .as_deref()
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("retime-cli-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pix_fmt(depth: Depth) -> &'static str {
    match depth {
        Depth::U8 => "rgba",
        Depth::U16 => "rgba64le",
        Depth::F32 => "gbrapf32le",
    }
}

/// A `testsrc2` clip of `SOURCE_FRAMES` frames as raw frames in the wire format of `depth`.
fn make_clip(dir: &Scratch, depth: Depth) -> PathBuf {
    let out = dir.path("clip.raw");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={W}x{H}:rate=24:duration=2"))
        .args(["-frames:v", &SOURCE_FRAMES.to_string(), "-pix_fmt", pix_fmt(depth), "-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg could not make the clip");
    out
}

fn apply(bin: &Path, depth: Depth, input: &Path, output: &Path, frames: usize, params: &[&str]) -> Output {
    apply_sized(bin, depth, (W, H), input, output, frames, params)
}

fn apply_sized(bin: &Path, depth: Depth, (w, h): (usize, usize), input: &Path, output: &Path, frames: usize, params: &[&str]) -> Output {
    let mut cmd = Command::new(bin);
    cmd.args(["apply", "retime", "--width", &w.to_string(), "--height", &h.to_string(), "--depth", &depth.bits().to_string()])
        .args(["--fps", "24", "--in"])
        .arg(input)
        .arg("--out")
        .arg(output)
        .args(["--frames", &frames.to_string()]);
    for p in params {
        cmd.args(["--param", p]);
    }
    cmd.output().expect("run vfx")
}

fn read_frames(path: &Path, depth: Depth) -> Vec<Frame> {
    read_frames_sized(path, depth, (W, H))
}

fn read_frames_sized(path: &Path, depth: Depth, (w, h): (usize, usize)) -> Vec<Frame> {
    let bytes = std::fs::read(path).unwrap();
    let len = w * h * depth.bytes_per_pixel();
    assert_eq!(bytes.len() % len, 0, "a trailing partial frame");
    bytes.chunks(len).map(|c| Frame::from_bytes(w, h, depth, c)).collect()
}

/// The blend of two OPAQUE 8 bpc pixels at weight one half, per channel (premultiply is the identity at alpha 255).
fn mix8(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    std::array::from_fn(|i| ((((b[i] as f32 - a[i] as f32) * 0.5f32) + a[i] as f32) + 0.5f32) as i64 as u8)
}

/// Half of an opaque 8 bpc pixel against the transparent frame, then the output finish: premultiplied
/// `trunc((0 - c) * 0.5 + c + 0.5)`, alpha 128, and unpremultiply with `255 / 128`.
fn fade8(a: [u8; 4]) -> [u8; 4] {
    let alpha = ((((0.0f32 - a[3] as f32) * 0.5f32) + a[3] as f32) + 0.5f32) as i64 as u8;
    let s = 255.0f32 / alpha as f32;
    let mut out = [0u8; 4];
    for i in 0..3 {
        let c = ((((0.0f32 - a[i] as f32) * 0.5f32) + a[i] as f32) + 0.5f32) as i64 as u8;
        let mut v = c as f32 * s;
        if 255.0f32 < v {
            v = 255.0;
        }
        out[i] = ((v as f64) + 0.5) as i64 as u8;
    }
    out[3] = alpha;
    out
}

/// What a frame becomes when it passes the engine on an exact position. 8 and 32 bpc opaque pixels
/// come back bit for bit. At 16 bpc the engine works in the range 0..=32768, which has half as many
/// levels as the wire (`Frame::u16_to_q15`, `Frame::q15_to_u16`): an opaque pixel comes back as the
/// nearest level of that coarser range.
fn exact_pass(f: &Frame) -> Frame {
    let mut out = f.clone();
    if let Pixels::U16(px) = &mut out.pixels {
        for p in px.iter_mut() {
            *p = p.map(|v| Frame::q15_to_u16(Frame::u16_to_q15(v)));
        }
    }
    out
}

fn px8(f: &Frame) -> &[[u8; 4]] {
    match &f.pixels {
        Pixels::U8(p) => p,
        _ => panic!("not an 8 bpc frame"),
    }
}

#[test]
fn half_speed_frame_blend_of_a_testsrc2_clip_at_8_bpc() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("half");
    let clip = make_clip(&dir, Depth::U8);
    let out = dir.path("out.raw");
    let r = apply(bin, Depth::U8, &clip, &out, 2 * SOURCE_FRAMES, &["vectors=1", "speed_percent=50", "frame_interp=2"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let src = read_frames(&clip, Depth::U8);
    let got = read_frames(&out, Depth::U8);
    assert_eq!(src.len(), SOURCE_FRAMES);
    assert_eq!(got.len(), 2 * SOURCE_FRAMES, "the frame count is --frames");
    let distinct = src.iter().enumerate().filter(|(i, f)| src[..*i].iter().all(|g| g.pixels != f.pixels)).count();
    assert!(distinct >= 4, "the test clip moves: {distinct} distinct frames");
    let mut real_blends = 0;
    for (n, g) in got.iter().enumerate() {
        let k = n / 2;
        let want: Vec<[u8; 4]> = if n % 2 == 0 {
            // output frame 2k is source frame k, bit for bit (testsrc2 is opaque)
            px8(&src[k]).to_vec()
        } else if k + 1 < SOURCE_FRAMES {
            px8(&src[k]).iter().zip(px8(&src[k + 1])).map(|(&a, &b)| mix8(a, b)).collect()
        } else {
            // past the last frame the neighbour is transparent
            px8(&src[k]).iter().map(|&a| fade8(a)).collect()
        };
        assert_eq!(px8(g), &want[..], "output frame {n}");
        if n % 2 == 1 && k + 1 < SOURCE_FRAMES && g.pixels != src[k].pixels && g.pixels != src[k + 1].pixels {
            real_blends += 1;
        }
    }
    assert!(real_blends >= 2, "{real_blends} odd frames differ from both neighbours");
}

#[test]
fn the_other_blend_modes_and_depths_run_end_to_end() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("modes{}", depth.bits()));
        let clip = make_clip(&dir, depth);
        let src = read_frames(&clip, depth);
        // exact frames: the default parameters (Vectors Best, Weighted Blend) copy the clip through the round trip
        let out = dir.path("exact.raw");
        let r = apply(bin, depth, &clip, &out, SOURCE_FRAMES, &[]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let expected: Vec<Frame> = src.iter().map(exact_pass).collect();
        assert_eq!(read_frames(&out, depth), expected, "{depth:?} opaque frames survive the engine on exact positions");
        // Nearest at half speed repeats every frame twice
        let out = dir.path("nearest.raw");
        let r = apply(bin, depth, &clip, &out, 2 * SOURCE_FRAMES, &["vectors=1", "speed_percent=50", "frame_interp=1"]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let got = read_frames(&out, depth);
        assert_eq!(got.len(), 2 * SOURCE_FRAMES);
        for (n, g) in got.iter().enumerate() {
            assert_eq!(g, &expected[n / 2], "{depth:?} nearest frame {n}");
        }
        // Weighted Blend without vectors is the plain Blend
        let (wb, bl) = (dir.path("wb.raw"), dir.path("blend.raw"));
        for (file, interp) in [(&wb, "frame_interp=3"), (&bl, "frame_interp=2")] {
            let r = apply(bin, depth, &clip, file, 2 * SOURCE_FRAMES, &["vectors=1", "speed_percent=50", interp]);
            assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        }
        assert_eq!(std::fs::read(&wb).unwrap(), std::fs::read(&bl).unwrap(), "{depth:?}");
        // the blended odd frames lie between their neighbours (opaque data: per channel min and max)
        let blended = read_frames(&bl, depth);
        for n in (1..2 * SOURCE_FRAMES - 1).step_by(2) {
            let (a, b) = (&src[n / 2], &src[n / 2 + 1]);
            match (&blended[n].pixels, &a.pixels, &b.pixels) {
                (Pixels::U8(g), Pixels::U8(a), Pixels::U8(b)) => {
                    for ((g, a), b) in g.iter().zip(a).zip(b) {
                        for i in 0..3 {
                            assert!(g[i] >= a[i].min(b[i]) && g[i] <= a[i].max(b[i]), "8 bpc frame {n}");
                        }
                    }
                }
                (Pixels::U16(g), Pixels::U16(a), Pixels::U16(b)) => {
                    for ((g, a), b) in g.iter().zip(a).zip(b) {
                        for i in 0..3 {
                            // the 15 bit range of the engine and the rounding of the chain move a value by a level or two
                            assert!(g[i] as i32 >= a[i].min(b[i]) as i32 - 3 && g[i] as i32 <= a[i].max(b[i]) as i32 + 3, "16 bpc frame {n}");
                        }
                    }
                }
                (Pixels::F32(g), Pixels::F32(a), Pixels::F32(b)) => {
                    for ((g, a), b) in g.iter().zip(a).zip(b) {
                        for i in 0..3 {
                            let (lo, hi) = (a[i].min(b[i]), a[i].max(b[i]));
                            let want = ((b[i] - a[i]) * 0.5f32 + a[i]) + 0.0f32;
                            assert_eq!(g[i], want, "32 bpc frame {n}");
                            assert!(g[i] >= lo - 1e-6 && g[i] <= hi + 1e-6);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

/// The synthetic clip with a known motion (`common/moving.rs`) as raw frames in the wire format of `depth`.
fn make_moving_clip(dir: &Scratch, depth: Depth) -> PathBuf {
    let out = dir.path("moving.raw");
    let bytes: Vec<u8> = common::moving::DEFAULT.clip(depth, W, H, SOURCE_FRAMES).iter().flat_map(|f| f.to_bytes()).collect();
    std::fs::write(&out, bytes).unwrap();
    out
}

/// The default parameters (Vectors Best, Frame Interp Blend, Smart Blend off) with only the speed changed:
/// the optical flow path through the real binary. The run succeeds, the frame count is `--frames`, the even
/// output frames (exact positions) are the source frames, and the odd ones are much closer to the analytic
/// in-between frame of the moving pattern than the plain frame blend of the same run with No Vectors.
#[test]
fn half_speed_with_the_default_vectors_runs_the_optical_flow_path_end_to_end() {
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("flow{}", depth.bits()));
        let clip = make_moving_clip(&dir, depth);
        let src = read_frames(&clip, depth);
        assert_eq!(src.len(), SOURCE_FRAMES);
        let frames = 2 * SOURCE_FRAMES - 2; // positions 0 to 10.5: every odd frame has both neighbours
        let out = dir.path("flow.raw");
        let r = apply(bin, depth, &clip, &out, frames, &["speed_percent=50"]);
        assert!(r.status.success(), "{depth:?}: {}", String::from_utf8_lossy(&r.stderr));
        let got = read_frames(&out, depth);
        assert_eq!(got.len(), frames, "the frame count is --frames");
        let plain_out = dir.path("plain.raw");
        let r = apply(bin, depth, &clip, &plain_out, frames, &["speed_percent=50", "vectors=1"]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let plain = read_frames(&plain_out, depth);
        let (mut worst, mut closest) = (0.0f64, f64::MAX);
        for (n, g) in got.iter().enumerate() {
            if n % 2 == 0 {
                assert_eq!(g, &exact_pass(&src[n / 2]), "{depth:?} output frame {n} is source frame {}", n / 2);
            } else {
                let truth = common::moving::DEFAULT.frame_at(depth, W, H, n as f64 / 2.0);
                let (flow_err, _) = common::moving::error(g, &truth, 9);
                let (plain_err, _) = common::moving::error(&plain[n], &truth, 9);
                worst = worst.max(flow_err);
                closest = closest.min(plain_err / flow_err);
                // measured at 64 x 48: flow error 0.0013 to 0.0017, plain blend 0.019 to 0.023, 11.8 to 17 times closer
                assert!(flow_err < 0.0025, "{depth:?} frame {n}: flow error {flow_err}");
                assert!(flow_err * 8.0 < plain_err, "{depth:?} frame {n}: flow error {flow_err} against plain blend error {plain_err}");
            }
        }
        eprintln!("{depth:?}: worst flow error {worst:.5}, at least {closest:.1}x closer than the plain blend");
    }
}

/// Output frame `n` (odd, half speed) of the vector path computed in process from explicit inputs: the two source
/// frames of `clip`, the selection of the half speed position with the vectors quality of Best (4), `solve_flows`
/// and `compose_vectors`. Only the warping code, interpolation, sensitivity and render quality come from the settings
/// of `params`; the front-end's vectors quality and the routing of the core are not used.
fn vector_frame(clip: &[Frame], params: &[(&str, &str)], n: usize) -> Frame {
    use retime_engine::compose::{compose_vectors, solve_flows, Rendered};
    use retime_engine::image::Image;
    use retime_engine::timeline::Selection;
    let rig = common::Rig::new(clip.to_vec(), params).with_frames(clip.len() as u64);
    let settings = retime_engine::frontend::read_settings(&retime::WIRING, &rig.cx(n as u64), &rig.env()).unwrap();
    let k = (n - 1) / 2;
    let (a, b) = (Image::from_frame(&clip[k]), Image::from_frame(&clip[k + 1]));
    let pos = n as f32 / 2.0;
    let sel = Selection { s0: pos, s1: pos + 0.5, a: k as i32, b: k as i32 + 1, wa: 0.5, exact: false, vectors_quality: 4, interp: settings.interp };
    let flows = solve_flows(&settings, &sel, &a, &b).unwrap();
    Rendered::Image(compose_vectors(&settings, &sel, &a, &b, &flows).unwrap()).into_frame()
}

/// A 3 x 2 or 2 x 2 clip was below the 2 x 2 working size of Best; the motion search never halves below 2 x 2 (the
/// tiny frames rule, parity gap G7), so through the real binary the in-between frames of a clip whose texture moves by
/// a sub-pixel amount run the vector path at the clip's own size and the run exits 0. Every in-between frame is the
/// vector path computed in process from explicit inputs and differs from the binary's own `vectors=1` run, so a fall
/// back to the frame blend (in the core or in the front-end) fails here; the frames on source frames are the same in
/// both runs.
#[test]
fn tiny_clips_run_the_vector_path_through_the_binary() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("tiny");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for (w, h) in [(3usize, 2usize), (2, 2)] {
            let clip = common::moving::tiny_texture_clip(depth, w, h, 4);
            let input = dir.path(&format!("tiny{}_{w}x{h}.raw", depth.bits()));
            std::fs::write(&input, clip.iter().flat_map(|f| f.to_bytes()).collect::<Vec<u8>>()).unwrap();
            for mode in [&[("frame_interp", "2")][..], &[("frame_interp", "3"), ("smart_blend", "1")]] {
                let mut params = vec![("speed_percent", "50"), ("sensitivity", "100")];
                params.extend_from_slice(mode);
                let run = |extra: &[&str], name: &str| -> Vec<Frame> {
                    let mut args: Vec<String> = params.iter().map(|(id, v)| format!("{id}={v}")).collect();
                    args.extend(extra.iter().map(|e| e.to_string()));
                    let args: Vec<&str> = args.iter().map(String::as_str).collect();
                    let out = dir.path(name);
                    let r = apply_sized(bin, depth, (w, h), &input, &out, 6, &args);
                    assert!(r.status.success(), "{depth:?} {w}x{h} {args:?}: {}", String::from_utf8_lossy(&r.stderr));
                    read_frames_sized(&out, depth, (w, h))
                };
                let got = run(&[], "out.raw");
                let plain = run(&["vectors=1"], "plain.raw");
                assert_eq!((got.len(), plain.len()), (6, 6));
                for n in 0..6 {
                    let case = format!("{depth:?} {w}x{h} {mode:?} frame {n}");
                    if n % 2 == 1 {
                        assert_eq!(got[n], vector_frame(&clip, &params, n), "{case}");
                        assert_ne!(got[n], plain[n], "{case}: the frame blend");
                    } else {
                        assert_eq!(got[n], plain[n], "{case}: on a source frame");
                    }
                }
            }
        }
    }
}

/// Clips with a side of 1 pixel have no motion search: every mode renders through the real binary, exits 0, and gives
/// the bytes of the same run with `vectors=1`.
#[test]
fn one_pixel_clips_render() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("onepx");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for (w, h) in [(1usize, 1usize), (1, 3), (5, 1)] {
            let clip: Vec<Frame> = (0..4).map(|k| common::moving::DEFAULT.frame_at(depth, w, h, k as f64)).collect();
            let input = dir.path(&format!("in{}_{w}x{h}.raw", depth.bits()));
            std::fs::write(&input, clip.iter().flat_map(|f| f.to_bytes()).collect::<Vec<u8>>()).unwrap();
            for mode in [&["smart_blend=0", "frame_interp=2"][..], &["smart_blend=1", "frame_interp=3", "blur_compensation=2"], &["frame_interp=1", "vectors=2"]] {
                let run = |extra: &[&str], name: &str| {
                    let out = dir.path(name);
                    let mut params = vec!["speed_percent=37.5"];
                    // the No Vectors run replaces the mode's own Vectors value (vfx refuses a repeated id)
                    params.extend(mode.iter().filter(|p| extra.is_empty() || !p.starts_with("vectors=")));
                    params.extend_from_slice(extra);
                    let r = apply_sized(bin, depth, (w, h), &input, &out, 6, &params);
                    assert!(r.status.success(), "{depth:?} {w}x{h} {params:?}: {}", String::from_utf8_lossy(&r.stderr));
                    std::fs::read(&out).unwrap()
                };
                let searched = run(&[], "searched.raw");
                let plain = run(&["vectors=1"], "plain.raw");
                assert_eq!(searched.len(), 6 * w * h * depth.bytes_per_pixel());
                assert!(searched == plain, "{depth:?} {w}x{h} {mode:?}: differs from vectors=1");
            }
        }
    }
}

/// Smart Blend (`smart_blend=1`) through the real binary at the three depths, on the moving texture and on the scene with
/// an occlusion: success, the exact frames are the source frames, every pixel is opaque, and the in-between frames are close
/// to the analytic frame (the texture: within 0.0025, 8 times closer than the frame blend; the occlusion scene: the whole
/// frame error at most 0.6 of the frame blend's, measured about 0.45).
#[test]
fn smart_blend_runs_end_to_end() {
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("smart{}", depth.bits()));
        let frames = 2 * SOURCE_FRAMES - 2;
        let (w, h) = (96usize, 64usize);
        for (name, size, clip_frames) in [
            ("texture", (W, H), common::moving::DEFAULT.clip(depth, W, H, SOURCE_FRAMES)),
            ("occlusion", (w, h), common::moving::OCCLUSION.clip(depth, w, h, SOURCE_FRAMES)),
        ] {
            let clip = dir.path(&format!("{name}.raw"));
            std::fs::write(&clip, clip_frames.iter().flat_map(|f| f.to_bytes()).collect::<Vec<u8>>()).unwrap();
            let (out, plain_out) = (dir.path("smart.raw"), dir.path("plain.raw"));
            let r = apply_sized(bin, depth, size, &clip, &out, frames, &["speed_percent=50", "smart_blend=1"]);
            assert!(r.status.success(), "{depth:?} {name}: {}", String::from_utf8_lossy(&r.stderr));
            let r = apply_sized(bin, depth, size, &clip, &plain_out, frames, &["speed_percent=50", "vectors=1"]);
            assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
            let (got, plain) = (read_frames_sized(&out, depth, size), read_frames_sized(&plain_out, depth, size));
            assert_eq!(got.len(), frames);
            for (n, g) in got.iter().enumerate() {
                if n % 2 == 0 {
                    assert_eq!(g, &exact_pass(&clip_frames[n / 2]), "{depth:?} {name} frame {n} is source frame {}", n / 2);
                    continue;
                }
                assert!(common::moving::all_opaque(g), "{depth:?} {name} frame {n} has a pixel that is not opaque");
                let s = n as f64 / 2.0;
                let truth = if name == "texture" {
                    common::moving::DEFAULT.frame_at(depth, W, H, s)
                } else {
                    common::moving::OCCLUSION.frame_at(depth, w, h, s)
                };
                if name == "texture" {
                    let (flow_err, _) = common::moving::error(g, &truth, 9);
                    let (plain_err, _) = common::moving::error(&plain[n], &truth, 9);
                    assert!(flow_err < 0.0025 && flow_err * 8.0 < plain_err, "{depth:?} frame {n}: smart {flow_err}, plain blend {plain_err}");
                } else {
                    let all = |_: usize, _: usize| true;
                    let (smart_err, plain_err) = (common::moving::error_in(g, &truth, all), common::moving::error_in(&plain[n], &truth, all));
                    assert!(smart_err < plain_err * 0.6, "{depth:?} frame {n}: smart {smart_err}, plain blend {plain_err}");
                }
            }
        }
    }
}

/// A real video test pattern (ffmpeg `testsrc2`: sharp edges, text, a moving gradient and counters) with the default
/// parameters except the speed: the flow path runs to the end on content that is not a smooth texture, every depth
/// of the pipeline keeps the exact frames, and the in-between frames are no frame blend.
#[test]
fn the_default_parameters_run_on_a_testsrc2_clip() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("testsrc");
    let clip = make_clip(&dir, Depth::U8);
    let src = read_frames(&clip, Depth::U8);
    let frames = 2 * SOURCE_FRAMES - 2;
    let out = dir.path("flow.raw");
    let r = apply(bin, Depth::U8, &clip, &out, frames, &["speed_percent=50"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let got = read_frames(&out, Depth::U8);
    assert_eq!(got.len(), frames);
    let plain_out = dir.path("plain.raw");
    let r = apply(bin, Depth::U8, &clip, &plain_out, frames, &["speed_percent=50", "vectors=1"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let plain = read_frames(&plain_out, Depth::U8);
    let (mut different, mut moving) = (0, 0);
    for (n, g) in got.iter().enumerate() {
        if n % 2 == 0 {
            assert_eq!(px8(g), px8(&src[n / 2]), "output frame {n} is source frame {}", n / 2);
        } else {
            assert!(px8(g).iter().all(|p| p[3] == 255), "frame {n} is opaque");
            // testsrc2 repeats frames at this size: between two equal frames there is nothing to interpolate
            if src[n / 2].pixels != src[n / 2 + 1].pixels {
                moving += 1;
                if g.pixels != plain[n].pixels {
                    different += 1;
                }
            } else {
                assert_eq!(g.pixels, plain[n].pixels, "frame {n} lies between equal frames and is that frame");
            }
        }
    }
    assert!(moving >= 3 && different == moving, "{different} of {moving} in-between frames between different frames differ from the frame blend");
}

#[test]
fn the_effect_is_listed_as_temporal_with_its_two_layers() {
    let bin = vfx().expect("fx-cli builds");
    let out = Command::new(bin).args(["list", "--json"]).output().unwrap();
    assert!(out.status.success());
    let list: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entry = list
        .as_array()
        .expect("a list of effects")
        .iter()
        .find(|e| e["id"] == "retime")
        .expect("retime is registered");
    assert_eq!(entry["kind"], "temporal");
    let layers: Vec<(&str, bool)> =
        entry["layers"].as_array().unwrap().iter().map(|l| (l["id"].as_str().unwrap(), l["optional"].as_bool().unwrap())).collect();
    assert_eq!(layers, vec![("alt_source_a", true), ("alt_source_b", true)]);
}

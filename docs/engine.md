# Engine

How `plugins/retime-engine` computes one output frame, which module does each stage, and what it costs. Each module's `//!` header holds its exact arithmetic (operation order, precision, rounding); this file is the map, not a second copy of those contracts.

## One output frame

| Step | What happens | Module |
|---|---|---|
| 1. parameters | params, time context and layers become `EngineSettings`; popups read as integers, 16.16 fixed sliders converted with `((v as f64) * (1.0 / 65536.0)) as f32 * scale` | `frontend.rs`, `host.rs` |
| 2. position | the time map gives the source positions `s0` of this output frame and `s1` of the next ([timing](timing.md)) | `timeline.rs` |
| 3. selection | frames `a = trunc(s0)` and `b = a + 1`, the weight of `a` `1 - (s0 - a)`, the cut-label override of `src_mark_segments`, and the exactness test (0.005, or any motion with `blur_compensation` above 0.005) | `timeline.rs` (`select`, `marker_override`, `exact_position`) |
| 4. plan | the source frames the render fetches, announced before rendering (the pre-render) | `planner.rs`, the effects' `prerender.rs` |
| 5. fetch | each frame turned into layer time, fetched and premultiplied, for the retimed layer and the tracking layer (`alt_source_b` when set); the FG matte layers are fetched the same way, at the layer times of the main frames | `host.rs` |
| 6. compose | `display=2` copies the source; missing data gives the error fill; an exact position, `vectors=1` or a frame with a side of 1 takes the fast path (a copy or a cross-fade of `a` and `b`; with FG matte layers, of the displayed group); anything else takes the vector path | `compose.rs` |
| 7. vector path | flow solves on the tracking layer's frames `a` and `b`, warps and composition of the retimed layer's frames, then the blur compensation tail; with FG matte layers one solve and one warp per layer ([FG matte layers](#fg-matte-layers)) | `compose/vector.rs`, `layers.rs` |
| 8. output | the engine image converted back to the wire format | `frontend.rs`, `image.rs` |

What the engine cannot serve is an `unsupported` error, never a silent fallback ([parameters: refusals](parameters.md#refusals)). Where a stage runs a stated rule because the original's behaviour there is not established, it records that gap (`fx_core::gaps::record` with an entry of `gaps.rs`) and `vfx` prints its note once per run ([README: limits](../README.md#limits)); recording changes no pixel.

## Working format

- Engine images are interleaved A, R, G, B with colour **premultiplied** by alpha, rows tightly packed, top row first (`image.rs`).
- 8 bpc is 0 to 255 and float is 1.0 for white. 16 bpc uses 0 to 65535 inside the engine; the range 0 to 32768 exists only at the boundary conversions, which is why opaque 16 bpc samples can come back 1 off ([reference: raw frame format](reference.md#raw-frame-format)).
- Arithmetic is single precision in the order written, with no fused multiply add and no reciprocal estimates, plus the double precision steps each function notes. Integer stores truncate or round as each module states. Parity at the bit level depends on these rules ([parity: the bar](parity.md#the-bar)).
- The motion path works on frames bottom row first, as the plugins do: the search, the warps and the fills run on the reversed rows and the result is turned back (`motion_rows_reversed`).
- At the edge rows of a frame, the vertical pass of every separable filter (the pyramids, the blurs, the hole fill) adds the tap above before the tap below, as the plugins do; that order decides the last bits of the sums.
- Where an operation meets two NaNs, the result follows the emulator's NaN rule by default (`nanrule.rs`, cargo feature `emulator-nan`; [parity: the NaN rule](parity.md#the-nan-rule)). Numbers are not affected.
- The Delinearize step uses a port of the C runtime's `powf` (`crt_powf.rs`), whose last bit differs from a correctly rounded power for rare arguments.

## Motion search

| Stage | What happens | Module |
|---|---|---|
| luma | colour to one 0 to 255 channel: `0.3 R + 0.59 G + 0.11 B` (plus `0.3 A` with `track_rgb_a`); with `video_360` in `retime` only R, stretched from video range 16 to 235 | `prep.rs` (`luma`) |
| resolution | `vectors` 5, 4, 3/2 halve the frame 1, 2, 3 times (`vector_resolution` 1 once, 2 not at all); draft quality halves once more, at most 3 times; never below 2x2, and a frame with a side of 1 is not searched (the fast path; `create-vectors` writes the zero vector) | `prep.rs` (`flow_resolution`, `searchable`) |
| prep | Delinearize (1/2.2 gamma), then Contrast/Edge Enhance, on the search copy only | `prep.rs` |
| pyramid | a level is added while both halves stay above 2 pixels (a side of 5 or less gives one level); each level a band-pass image `orig - blur5(orig)` with the binomial `[1 4 6 4 1] / 16` filter | `flow.rs` (`build_pyramid`), `prep.rs` (`level_count`), `flow/conv.rs`, `flow/sample.rs` |
| solve | coarse to fine: per level a gradient step on the structure tensor of the band-pass images, then a block-matching refinement sweep; the flow is upsampled to the next level | `flow.rs` (`solve`, `solve_pair`) |
| sensitivity | sets the start level and a damping factor: `x = log2(s * s) + L` for `s` = sensitivity / 100 and L levels; below 0 no motion is followed, at L or more the search starts at the coarsest level undamped | `flow.rs` |
| layer solves | with FG matte layers each layer is solved on the luma of its own layer images (the frames times the layer's composition matte), masked where the layers above cover it; FG1's solve has no mask. Block matching is never gated by a mask | `layers/plates.rs`, `flow/extras.rs` (`solve_pyramids_with`), `flow.rs` |
| tracking points | the points in use of a layer move its motion at the finest level only: each point adds its displacement at the pixel it falls on, a pixel with several takes their mean, and every coarser level takes the field from the one above with the image pyramid's blur and resample, scaled by the size ratio. The displacement is set again right after the finest block matching | `points.rs` (`user_displacement`), `flow/extras.rs` |
| 360 | with `video_360` in `retime-fgbg` the motion search, the fill and the blur use their wrap variants around the sphere: horizontal taps wrap by whole widths, vertical taps reflect over the poles with a half turn. In a layered render `track_rgb_a` switches the luma pyramids' filters to those variants | `flow.rs`, `flow/extras.rs`, `prep.rs`, `flow/conv.rs`, `flow/sample.rs` |

The returned flow lives at the working resolution, `(W >> R, H >> R)`, and every warp resamples it bilinearly. Exact frames never run the search.

## Warps

| Code | Choice | Warp | Module |
|---|---|---|---|
| 0 | `retime` default, `retime-fgbg` `warping=1`, `vectors-in` `warping=1` | inverse: `W(p) = img(p + F(p) * k)`, bilinear, with an edge fade; outside the frame the border repeats | `warp.rs` |
| 1 | `smart_blend=1`, `retime-fgbg` `warping=2` | inverse with Smart Blend: each warp returns a coverage image, the composite takes each pixel from the frame that covers it, then the hole fill | `smartblend.rs`, `holefill.rs` |
| 2 | `retime-fgbg` `warping=3`, `vectors-in` `warping=2` | forward: each source frame pushed along its own flow as a mesh of triangles shaded from their corner colours (the Gouraud variant, which both plugins use), accumulated as a running mean per pixel, with Smart Blend and the hole fill | `forward.rs`, `forward/mesh.rs`, `forward/raster.rs` |
| 3 | `retime-fgbg` `warping=4` | forward without the Smart Blend step or the hole fill: pixels no triangle covers stay empty | `forward.rs` |

The forward mesh is rendered cell by cell in a fixed order, so its accumulation, and with it the last bits, does not depend on the thread count. A flow that is all zero, or a warp amount below 1e-5, gives a plain copy. With 360 Video in `retime-fgbg` the inverse warps sample across the left and right edges (the 360 mode of `warp.rs`); the forward warp itself never wraps, but the hole fill after it does.

Under `--device gpu` the forward warp runs on the device except for its mesh: the vertex positions and the per vertex mip levels are computed on the host by the CPU's own code (the mip level takes a `logf`, whose bits come from the host C runtime) and uploaded. The device scan converts every triangle with the CPU's arithmetic, stores each covered pixel as a record keyed by the triangle's index, sorts each pixel's records by that index and folds them in that order, which is the CPU's fold; the pyramid is built on the device with the CPU's blur and resample. The warp is handed back to the CPU, which then gives its own bytes or error, for a triangle covering more than 16384 pixels, a pixel covered by more than 512 triangles, more than 4 covered pixels per pixel on average, a texture read below the first column or row (only NaN or infinite vectors, or vertices about a million pixels outside the frame, lead there), frames above 16384 pixels on a side, and the degenerate last row write outside the frame (which the CPU refuses too). A float result holding a NaN is always recomputed on the CPU.

## Composition

- **`frame_interp`**: Nearest takes the nearer warped frame; Blend mixes the two warped frames by their distance to the position (`blend`, `lerp` in `blend.rs`); Motion Weighted Blend adds the sharpening step of `filters.rs` (a difference mask, its clean-up, a masked composite, `sharpen_weight` and `sharpen_step`) to each half.
- **Smart Blend** (`smartblend.rs`): per pixel, `over` composites of the two warped frames in both orders, mixed by the position's weight, keeping the larger alpha. With `video_360` and Blend it uses the binary-alpha variant, which treats alpha as either opaque or absent.
- **Hole fill** (`holefill.rs`): a pull-push pyramid fill, channel by channel, alpha first. Known pixels come back unchanged; a hole receives a smooth blend of the known pixels around it. Used by the Smart Blend modes, the layer plates and `vectors-in`. Two NaNs that meet in it follow the NaN rule. With 360 Video the horizontal passes wrap, so a hole at the left or right edge is filled from both sides of the seam.

## FG matte layers

`retime-fgbg`, `vectors-in` and `create-vectors` take up to three FG matte layers (`layers.rs`, `matte.rs`, `points.rs`). A layer counts only after the one before it; FG1 is on top and the background (Main_BG) is what the mattes leave free.

- **Mattes** (`matte.rs`): the matte is the `fgN_matte_chan` channel of the layer as stored (not premultiplied by the matte layer's own alpha), at the output size. The composition matte of FG k is its raw matte cut by every FG layer above it, each cut quantised to the units of the depth; `fgN_inv_matte_shrink` chokes the cutting matte with the plugin's own choke kernel (a fraction mixes the two nearest whole shrinks, a negative value grows it). A shrink on any layer makes exact frames take the motion path.
- **Motion**: every layer of the displayed group gets its own flow, solved on the luma of its own layer images ([Motion search](#motion-search)) with its own sensitivity and tracking points. `vectors-in` instead reads its vector layers once per layer and caps each layer's vector alpha by the layer's composition matte. A layer whose Track is off in `create-vectors` takes the second tracker: its points' displacement spread over the frame by the pull-push hole fill (gap G5, this project's rule).
- **Composition** (`retime-fgbg`, `vectors-in`; `layers/plates.rs`): every layer of the group is interpolated on its own, bottom layer first. Its layer images are the two frames times its composition matte. With two sides (Blend, Weighted Blend) each side gets a plate, the layer image OVER the other side's layer image warped to its time by the layer's own flow. Each plate is unpremultiplied, its colour filled where it is not opaque, its alpha replaced by the layer's own raw matte, and premultiplied again. The two plates then go through the single layer composition of the warping code, including the blur compensation with that layer's flow. The background's interpolated layer is the base, and every FG layer is composed OVER it with the plugin's OVER. A single group display (`display` 3 to 6) shows one interpolated layer; its fast path is the layer image composed over itself and refilled.
- Without matte layers and points only the single layer path runs. Each FG layer adds solves and warps of its own.

## Blur compensation

`warp/blurcomp.rs`. After the warps, `blur_amount` decides whether to run and computes the amount from the speed; a positive amount blurs along the flow with the motion blur primitive, a negative one builds a flow-directed unsharp mask; colour is then cleared where alpha is 0. With 360 Video the taps go around the sphere: x wraps by whole widths, and the bilinear taps that cross a pole reflect and move x by half a turn.

## Vector images

`vectors.rs` and `create_vectors.rs`.

- **Encoding** (`create-vectors`): the flow between source frame n and its neighbour (`track_frame`) is sampled bilinearly at every output pixel, scaled to output pixels, divided by `max_displace`, and stored with R = dx and G = dy centred on 127, 32767 or 0.5; B is 0 and alpha opaque. Motion beyond `max_displace` is clipped. With FG matte layers every layer of `layers_output` has its own flow, solved on its own layer images (a layer with Track off takes the second tracker, gap G5). The vectors of the group are composited from the bottom up with the raw mattes, then take the same clip and store. A one-layer output is that layer's own flow, unweighted, with its visible matte in blue. A frame with a side of 1 is the zero vector image.
- **Debug views** (`debug.rs`, `create-vectors` `display=3`): frame X of the pair is pushed along the flow (the merged flow with FG layers) by the forward warp at amount 1, with no fill, and compared with frame Y. Warp shows the push. The other views use W': without FG layers, the push laid over Y over a transparent image with the plugin's OVER, which also covers the uncovered last row and column. Blend mixes W' and Y half and half; Diff is the grey absolute difference of their lumas, times 2.5; the overlays are red and cyan (R from W', G and B from Y); Detail and Edge compare a detail and an edge measure of the two lumas. The lumas are summed in double precision. With an FG stack, the Debug display lays each layer's Y and then the warp of its X over the output.
- **Draw Geom** (`geom.rs`): the points in use of the `draw_geom` group drawn after the source copy, later points on top. `create-vectors` draws the original's glyph, an antialiased cross: a horizontal and then a vertical segment of half length 3 and width 1.5. `retime-fgbg` still draws an opaque disc of radius 1.5 on its Source display, where the original draws the cross (gap G3).
- **Decoding** (`vectors-in`): the R and G channels back into a flow, a hole mask from the alpha channel, the hole fill over the flow, then `vec_scale_x` and `vec_scale_y`. With one vector layer the other direction is derived from it. With FG matte layers each layer's vector alpha is first capped by its composition matte.

## Device

`device.rs` and `device/`: an opt-in OpenCL backend. The CPU code stays the reference and everything on the device is additive; a device is used only where it gives the CPU's bytes.

**Not ported yet.** The kernels follow the CPU rules as they were before the parity fixes. Since then the CPU path changed in several stages: the vertical tap order, the Debug views, the tracking point pyramid, the 360 wrap, the per-layer plates and the NaN rule. Their simulator tests fail until the kernels are ported. Until then a `--device gpu` render can differ from the CPU render, and the parity results cover the CPU path only.

- **Loading.** `device/cl.rs` loads the system OpenCL runtime at run time (Windows `LoadLibraryExW("OpenCL.dll", LOAD_LIBRARY_SEARCH_SYSTEM32)`, Linux `dlopen("libOpenCL.so.1")`, macOS the OpenCL framework path) and declares the OpenCL 1.2 entry points it needs by hand. It is the only file of the workspace with FFI: the crate root denies `unsafe_code` and `tests/no_unsafe.rs` scans the workspace. A missing runtime or entry point is an error, never a panic.
- **Probe.** Info queries only (`device/probe.rs`, `vfx devices`). A device must be `FULL_PROFILE`, little endian, OpenCL C 1.2 or later, available with a compiler, have `cl_khr_fp64`, single precision DENORM, INF_NAN, ROUND_TO_NEAREST and CORRECTLY_ROUNDED_DIVIDE_SQRT, and double precision DENORM, INF_NAN and ROUND_TO_NEAREST; each unmet requirement is one `reason`. A run may place three quarters of a device's global memory on it, and `gpu` prefers the first usable device with memory of its own over one that shares the host's.
- **Kernels.** OpenCL C 1.2 sources as Rust string constants (`device/kernels/`), each program starting with the prelude: `#pragma OPENCL FP_CONTRACT OFF`, range tested truncating conversions (out of range and NaN give the minimum of the type, as x86 `cvtt` and the CPU helpers do), the stores of the CPU code, hexadecimal float literals only. Build options exactly `-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt`. A lint keeps `mad`, `fma`, `native_`, `half_`, `pow`, `log`, `exp`, `fmin`, `fmax`, `mix`, decimal float literals and unbounded loops out of every source.
- **Three copies.** Each kernel has the CPU function it ports, its OpenCL C source and a line for line Rust transcription. `device/sim.rs` runs the transcription work item by work item in shuffled seeded orders, and the tests compare it with the CPU function bit for bit, without a device.
- **Gate.** `device/selfcheck.rs` runs starter kernels (bit copies, Blend at 8, 16 and 32 bits, every conversion helper in single and double precision, a 2D layout check, a dependent loop) on a device over an adversarial corpus (every 8 bit pair, subnormals, -0.0, NaN, infinities, .5 ties, out of range casts, products where a fused multiply add rounds differently) and compares with the CPU bit for bit. Any difference refuses the device with the first failing input. NaN payloads are reported apart: IEEE 754 leaves them open and a GPU may write another payload than x86, so a stage never takes a float output that holds a NaN from a device; it reruns that stage on the CPU.
- **Stages.** Device versions of the CPU stages, each with a form that keeps the images on the device (`*_enqueue`) and one with the CPU function's arguments (`*_gpu`): the per pixel composition (`device/warpstage/`: blend, lerp, copy, the inverse warp in hold, fade, coverage and 360 mode, motion blur, Blur Compensation, Smart Blend and its binary variant); the Weighted Blend chain (`device/wbstage/`); vector coding, the matte operations, the Debug view compositions, Draw Geom, the point splat and the per pixel prep (`device/vecstage/`); convolution, resampling, the pyramids, the hole fill and the matte choke (`device/convstage/`); the motion solver (`device/solver/`); and the forward warp (`device/fwdstage/`). Where the CPU loops a data dependent number of times, a kernel loops at most a small constant and otherwise raises a flag; that stage then runs on the CPU, as does any stage a kernel flags for a case the CPU refuses, and any OpenCL error.
- **Host steps.** A step whose bits come from the host C runtime stays on the CPU under any device: the `pow` of Delinearize, the `logf` of the forward warp's mip level, the Gaussian taps (an f64 `powf`) and the solver's start level and damping, computed on the host and passed in.

### Executor

`--device gpu` and `gpu=N` ([reference: renderer](reference.md#renderer)) run the engine through the device executor (`plugins/retime-engine/src/device/exec/`). The CPU code is the reference and is unchanged; the device takes over single stages behind one seam (`stages::Stages`), and a render through the CPU stage set is the CPU render.

**Gates.** A device is used only when it meets the requirements of the probe, passes the self-check at open and builds the whole program (`-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt`, `FP_CONTRACT OFF`). The first frame of every run that reaches the device is rendered on the CPU as well and the two are compared byte for byte (the frame and the outcome); any difference refuses the device for the rest of the run, with one note.

**Per stage fallback.** Any device error of a stage (a driver error, a float output that holds a NaN, a case its kernels do not cover, the memory budget, a device busy on the calling thread) reruns that stage with its CPU function. The NaN rule: both GPUs of the development machine write other NaN payloads than the CPU, so a float output holding a NaN is never taken from the device.

**Stage placement:**

| stage | runs on | why |
|---|---|---|
| `flow::solve`, `flow::solve_pair` (pyramids built on the device and kept resident in the frame cache) | device | the solve is the largest part of a vector frame |
| `flow::extras::solve_pyramids_with` (masks, user displacement, wrap; host pyramids uploaded) | device | the solve dominates its upload |
| `compose::compose_vectors`: the warps, Blend, Weighted Blend (the sharpening), Smart Blend (binary alpha too), the hole fill and the blur compensation of one vector frame, warping codes 0 to 3, as one resident chain | device, except with the wrap-around (360) flag | the two frames and the flows go up once, every intermediate stays on the device, only the finished frame comes back |
| `forward::forward_warp_with`, textured variant (the forward warps of the chain's fallback, the Warp debug view; the mesh and its mip levels on the host, [Warps](#warps)) | device | 4 to 7 times faster than the single threaded CPU rasteriser |
| `forward::forward_warp_with`, Gouraud variant (the flow inversion of Vectors In and the `create-vectors` Debug views) | CPU | no device kernel; the executor keeps `pyramid=false` on the CPU |
| the layered compositions of `retime-fgbg` with matte layers (premultiply, per layer warps, over, fill), vector coding, mattes, overlays | CPU | bit exact on the device as single stages, but not chained yet and alone each costs more in transfers than it saves |
| the wrap-around (360) composition | CPU | its fill and blur record the G6 note; kept on the CPU |
| `delinearize`, the f64 `powf` / `ln` scalars, `matte::effective` with a shrink | CPU | not on the device |

A chain that refuses at any step (a driver error, a raised refusal flag, the memory budget, a forward warp beyond its device bounds, a finished float frame holding a NaN) is composed again by `compose::compose_vectors` on the CPU from the same frames and flows, so its bytes are the CPU's. `VFX_DEVICE_REPORT=1` names each stage's device calls, fallbacks by kind and CPU calls, the first refusal text of a stage that fell back, and the open times.

**Memory.** A run may place 75% of the device's global memory on it: a quarter of that is the frame cache (resident luma pyramids keyed by the exact frame, compared sample bit for sample bit, least recently used out), the rest is the working set of one stage; a stage that would not fit runs on the CPU.

**Cold start.** Opening a device runs the self-check and the build of the program at the same time (two contexts). Built programs are kept as device binaries in the user's temporary directory (`vfx-opencl-programs`, one file per program, keyed by the platform, device and driver names and versions, the build options and the whole program text, compared exactly, with a digest of the binary); a matching entry is loaded instead of compiling, anything else builds from source and rewrites the entry, and `VFX_DEVICE_PROGRAM_CACHE=0` turns it off. Nothing prunes the folder. The gate itself is unchanged. Measured on the development machine with warm entries: the gate takes about 0.15 s (gpu=0) and 0.4 s (gpu=1), the whole `vfx apply --check` with a device about 0.4 s (gpu=0) and 1.3 to 2.0 s (gpu=1), against 0.5 to 1.1 s and 1.9 to 2.6 s before the cache and the concurrent build. The first run after a driver update or a change of the kernels compiles from source (6 to 11 s on that machine). Loading the OpenCL runtime and listing the platforms (both vendors' drivers load) is about 0.2 s of that (`vfx devices`); the rest of the gpu=1 figure is outside the gate (the AMD runtime's own setup and teardown).

**Determinism.** The device bytes equal the CPU bytes by construction (every kernel follows its CPU function's operation order and precision, shown per stage on both GPUs) and by the two gates. `fx-parity devcheck` is the end to end gate.

### Measured

On 2026-10-28, on the development machine: RTX 4060 Laptop = gpu=0, Radeon 780M = gpu=1, 16 thread CPU. 1080p 8 bpc, a rotating and panning `testsrc2` (24 fps), Speed 40%, wall time of the whole `vfx apply` run including the device open, median of 3 runs. The machine was shared with other jobs (no run of this study overlapped another of its own runs). "Best" is the default Vectors setting, "High" is `vectors=4`; Forward is `retime-fgbg` with `warping=3`; the matte run is `retime-fgbg` with one FG matte (a moving box). Every device run gave the CPU's bytes.

30 output frames:

| run | cpu s (fps) | gpu=0 s (fps) | gpu=1 s (fps) |
|---|---|---|---|
| retime Blend, Best | 4.01 (7.5) | 4.24 (7.1) | 5.91 (5.1) |
| retime Blend, High | 2.56 (11.7) | 2.42 (12.4) | 3.96 (7.6) |
| retime Weighted Blend, Best | 7.76 (3.9) | 3.98 (7.5) | 6.08 (4.9) |
| retime Smart Blend (Blend), Best | 7.74 (3.9) | 4.62 (6.5) | 7.10 (4.2) |
| retime-fgbg Forward, Best | 34.12 (0.9) | 10.14 (3.0) | 15.15 (2.0) |
| retime-fgbg Forward, High | 31.54 (1.0) | 8.51 (3.5) | 12.45 (2.4) |
| create-vectors, half resolution | 3.69 (8.1) | 3.73 (8.0) | 4.64 (6.5) |
| create-vectors, full resolution | 8.09 (3.7) | 4.40 (6.8) | 6.66 (4.5) |
| retime-fgbg, one FG matte | 9.97 (3.0) | 10.29 (2.9) | 11.34 (2.6) |

100 output frames:

| run | cpu s (fps) | gpu=0 s (fps) | gpu=1 s (fps) |
|---|---|---|---|
| retime Blend, Best | 14.33 (7.0) | 10.84 (9.2) | 15.07 (6.6) |
| retime Weighted Blend, Best | 27.99 (3.6) | 11.96 (8.4) | 17.24 (5.8) |
| retime Smart Blend (Blend), Best | 27.04 (3.7) | 14.85 (6.7) | 20.52 (4.9) |
| retime-fgbg Forward, Best | 112.92 (0.9) | 28.42 (3.5) | 39.30 (2.5) |
| create-vectors, half resolution | 12.29 (8.1) | 8.84 (11.3) | 12.15 (8.2) |
| retime-fgbg, one FG matte | 30.02 (3.3) | 28.00 (3.6) | 32.49 (3.1) |

Where the device time goes (100 frames, `VFX_DEVICE_REPORT=1`, milliseconds per call, gpu=0 / gpu=1): `solve_pair` at Best 75 to 86 / 101 to 120, at High 30 / 39; the compose chain 12.5 / 13 (Blend), 26 / 35 (Weighted Blend), 52 / 73 (Smart Blend), 199 / 310 (Forward: two forward warps with coverage, the Smart Blend and the hole fill); `solve` of create-vectors 47 / 67; `solve_pyramids_with` of the matte run 36 / 43 (four a frame). The gate takes 0.15 to 0.19 s / 0.41 to 0.55 s. A run on gpu=0 with no device stage at all (`vectors=1`, 100 frames) took 3.2 s against 2.2 s on the CPU: about 0.4 s is the open, the rest was not attributed.

What wins and what loses: Forward wins on both GPUs at every length (3.4 to 4.0 times on gpu=0, 2.2 to 2.9 times on gpu=1), and so do Weighted Blend and Smart Blend (1.7 to 2.3 times on gpu=0, 1.1 to 1.6 times on gpu=1). Blend at Best loses at 30 frames on both GPUs (the open, the cross check frame and the runtime setup outweigh the about 45 ms gpu=0 saves per vector frame) and wins at 100 frames on gpu=0 (1.3 times); on gpu=1 it does not win (its solve is about as fast as the CPU's). create-vectors at half resolution breaks even at 30 frames on gpu=0 and wins at 100 (1.4 times); at full resolution it wins at 30 frames on both. The matte run of `retime-fgbg` stays within 12% of the CPU either way (gpu=0 3% slower at 30 frames and 7% faster at 100, gpu=1 12% and 8% slower): its layered composition still runs on the CPU and its four solves a frame run one after another on the device. With `--device gpu` the RTX 4060 is chosen first (its own memory), where only short Blend clips and the matte run do not gain; a static placement that keeps a stage family on the CPU would not change that, because the loss is the per run cost, not a slower stage.

Byte gate (`fx-parity devcheck`, both GPUs, every case identical to the CPU run: clip, exit code and stderr): the reference families set 117 of 117 (no fallback), 96 cases of the default and sweep sets (fallbacks only in Vectors In runs, the forward warp's device bound on wild decoded vectors), and a set of 160 cases for the chains (every warping code and interpolation mode, Smart Blend with and without the binary alpha, blur compensation, 8, 16 and 32 bpc, frames from 1x1 to 64x48, the wrap-around flag, one FG matte, Vectors In with one and two vector layers): 160 of 160, fallbacks again only from that device bound.

### Open costs

- The layered `retime-fgbg` compositions with FG mattes (premultiply, the per layer warps, OVER, the alpha hole fill, the blend and the tail) run on the CPU stage by stage. Every step has a device stage, so a second resident chain is the next step.
- The solver at the Best working size is the largest part of every vector frame on the device and not much below the CPU's; Blend and create-vectors gain only on long clips because of it.
- The forward warp chain includes the host mesh (with the host `logf`) and two blocking read backs inside each warp.
- Per run: the gate, the runtime and platform setup, the one cross check frame (one CPU frame) and a share on gpu=0 that the study did not attribute.

## Determinism

Rows are independent and run through rayon row chunks, every output pixel depends only on input data, and the forward warp accumulates in a fixed order, so output bytes do not depend on the thread count. `fx-parity selfcheck` and every effect's `tests/threads.rs` check this. The same command on the same file and machine gave a byte-identical output file (measured).

A device never changes bytes: a device that differs from the CPU on any input of the gate is refused, and the CPU bytes are the reference. The CPU bytes themselves depend on the platform C runtime in two places, the `pow` of Delinearize and the `logf` of the forward warp's mip level, so they are fixed per machine and operating system, not across operating systems.

## Cost

Measured on one desktop; the order of magnitude, not a guarantee.

| Work | Cost |
|---|---|
| an exact frame (a copy) at 1080p | a few hundredths of a second |
| an in-between frame at 1080p, 8 bpc, by `vectors` | `5` about 0.2 s, `4` 0.13 s, `3` or `2` 0.1 s, `1` 0.04 s |
| 18 output frames at 1080p, by `vectors` | `1` 1.1 s, `2` or `3` 1.7 s, `4` 1.9 s, `5` 2.6 s |
| `smart_blend=1` | about 1.7 times the render time |
| forward warping (`retime-fgbg` `warping=3`) | about 1.5 times on a small clip, about 6 times at 1080p |
| one forward warp at 1080p (measured, laptop: RTX 4060 Laptop GPU, Radeon 780M) | CPU 0.47 to 0.57 s at every depth (the rasteriser is single threaded); `--device gpu` with the frame on the device 0.07 to 0.11 s on the RTX 4060 and 0.12 to 0.13 s on the 780M, of which about 0.05 s is the mesh on the host; device kernels alone 0.02 s and 0.035 s; at 3840x2160 all device launches of one warp together 0.04 s (RTX 4060) and 0.07 to 0.09 s (780M) |
| `blur_compensation` above 0 at exactly 100% | about 3 times, with every frame still a copy |
| 120 frames at 40%, 8 bpc, by size | 480x270 5 s, 960x540 9 s, 1920x1080 30 s, 3840x2164 94 s |

`tools/speed_curve.py` estimates a curve's render time at 0.25 s per in-between frame at 1080p. These figures are of the single layer path on the CPU; each FG matte layer adds solves and warps of its own. Device timings: [Device: measured](#measured).

## Module map

| Module | Stage |
|---|---|
| `timeline.rs` | rate model, Speed and Frame Number time maps, frame selection, exactness |
| `planner.rs` | the frames a render fetches, and the pre-render order |
| `frontend.rs`, `host.rs` | parameters to settings, frame fetch and premultiplication |
| `compose.rs`, `compose/vector.rs` | the render of one frame: fast path, vector path |
| `prep.rs` | luma, search resolution, Delinearize, Contrast/Edge Enhance |
| `flow.rs`, `flow/conv.rs`, `flow/sample.rs`, `flow/extras.rs` | pyramid, optical flow solver with its layer mask, tracking point and wrap-around variants, convolution, bilinear sampling |
| `warp.rs`, `warp/blurcomp.rs` | inverse warp, blur compensation |
| `forward.rs`, `forward/mesh.rs`, `forward/pyramid.rs`, `forward/raster.rs` | forward warp |
| `smartblend.rs`, `holefill.rs` | Smart Blend composites, hole fill |
| `blend.rs`, `filters.rs` | Blend, cross-fade, Weighted Blend |
| `vectors.rs`, `create_vectors.rs` | vector image encoding and decoding, the matte weighted merge, the `create-vectors` job |
| `image.rs` | engine images |
| `matte.rs` | FG matte layers: matte channel, shrink and grow, tracking mattes, OVER weights, layer premultiply and OVER, composite holes |
| `points.rs`, `geom.rs` | tracking points: the layer of a point, the user displacement, the second tracker; the Draw Geom discs |
| `layers.rs` | the n-layer render of `retime-fgbg` and `vectors-in` |
| `debug.rs` | the Debug views of `create-vectors` |
| `gaps.rs` | the parity gaps register: each open point's id, title and controls, and the gaps each effect can note |
| `device.rs`, `device/cl.rs`, `device/probe.rs`, `device/image.rs` | OpenCL runtime loading and bindings, device probe and requirements, device images |
| `device/kernels/`, `device/sim.rs`, `device/selfcheck.rs` | kernel sources and prelude, work item simulator, the device gate |
| `device/warpstage/`, `device/wbstage/`, `device/vecstage/`, `device/convstage/`, `device/solver/`, `device/fwdstage/` | the device stages: composition, Weighted Blend, vectors and mattes, convolution and pyramids, the motion solver, the forward warp |
| `device/exec/`, `device/progcache.rs` | the device executor: open and gate (`open.rs`), the OpenCL backend with the frame cache (`backend.rs`, `cache.rs`), the resident compose chain (`chain.rs`), per stage fallback, the cross check (`mod.rs`); the cache of program binaries (`progcache.rs`) |

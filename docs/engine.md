# Engine

How `plugins/retime-engine` computes one output frame, which module does each stage, and what it costs. Each module's `//!` header holds its exact arithmetic (operation order, precision, rounding); this file is the map, not a second copy of those contracts.

## One output frame

| Step | What happens | Module |
|---|---|---|
| 1. parameters | params, time context and layers become `EngineSettings`; popups read as integers, 16.16 fixed sliders converted with `((v as f64) * (1.0 / 65536.0)) as f32 * scale` | `frontend.rs`, `host.rs` |
| 2. position | the time map gives the source positions `s0` of this output frame and `s1` of the next ([timing](timing.md)) | `timeline.rs` |
| 3. selection | frames `a = trunc(s0)` and `b = a + 1`, the weight of `a` `1 - (s0 - a)`, the cut-label override of `src_mark_segments`, and the exactness test (0.005, or any motion with `blur_compensation` above 0.005) | `timeline.rs` (`select`, `marker_override`, `exact_position`) |
| 4. plan | the source frames the render fetches, announced before rendering (the pre-render) | `planner.rs`, the effects' `prerender.rs` |
| 5. fetch | each frame turned into layer time, fetched and premultiplied, for the retimed layer and the tracking layer (`alt_source_b` when set) | `host.rs` |
| 6. compose | `display=2` copies the source; missing data gives the error fill; an exact position or `vectors=1` takes the fast path (a copy or a cross-fade of `a` and `b`); anything else takes the vector path | `compose.rs` |
| 7. vector path | flow solves on the tracking layer's frames `a` and `b`, warps and composition of the retimed layer's frames, then the blur compensation tail | `compose/vector.rs` |
| 8. output | the engine image converted back to the wire format | `frontend.rs`, `image.rs` |

What the engine cannot serve is an `unsupported` error, never a silent fallback ([parameters: refusals](parameters.md#refusals)).

## Working format

- Engine images are interleaved A, R, G, B with colour **premultiplied** by alpha, rows tightly packed, top row first (`image.rs`).
- 8 bpc is 0 to 255 and float is 1.0 for white. 16 bpc uses 0 to 65535 inside the engine; the range 0 to 32768 exists only at the boundary conversions, which is why opaque 16 bpc samples can come back 1 off ([README: raw frame format](../README.md#raw-frame-format)).
- Arithmetic is single precision in the order written, with no fused multiply add and no reciprocal estimates, plus the double precision steps each function notes. Integer stores truncate or round as each module states. Parity at the bit level depends on these rules ([README: parity](../README.md#parity)).

## Motion search

| Stage | What happens | Module |
|---|---|---|
| luma | colour to one 0 to 255 channel: `0.3 R + 0.59 G + 0.11 B` (plus `0.3 A` with `track_rgb_a`); with `video_360` only R, stretched from video range 16 to 235 | `prep.rs` (`luma`) |
| resolution | `vectors` 5, 4, 3/2 halve the frame 1, 2, 3 times (`vector_resolution` 1 once, 2 not at all); draft quality halves once more, at most 3 times | `prep.rs` (`flow_resolution`) |
| prep | Delinearize (1/2.2 gamma), then Contrast/Edge Enhance, on the search copy only | `prep.rs` |
| pyramid | levels halving down to 2x2; each level a band-pass image `orig - blur5(orig)` with the binomial `[1 4 6 4 1] / 16` filter | `flow.rs` (`build_pyramid`), `flow/conv.rs`, `flow/sample.rs` |
| solve | coarse to fine: per level a gradient step on the structure tensor of the band-pass images, then a block-matching refinement sweep; the flow is upsampled to the next level | `flow.rs` (`solve`, `solve_pair`) |
| sensitivity | sets the start level and a damping factor: `x = log2(s * s) + L` for `s` = sensitivity / 100 and L levels; below 0 no motion is followed, at L or more the search starts at the coarsest level undamped | `flow.rs` |

The returned flow lives at the working resolution, `(W >> R, H >> R)`, and every warp resamples it bilinearly. Exact frames never run the search.

## Warps

| Code | Choice | Warp | Module |
|---|---|---|---|
| 0 | `retime` default, `retime-fgbg` `warping=1`, `vectors-in` `warping=1` | inverse: `W(p) = img(p + F(p) * k)`, bilinear, with an edge fade; outside the frame the border repeats | `warp.rs` |
| 1 | `smart_blend=1`, `retime-fgbg` `warping=2` | inverse with Smart Blend: each warp returns a coverage image, the composite takes each pixel from the frame that covers it, then the hole fill | `smartblend.rs`, `holefill.rs` |
| 2 | `retime-fgbg` `warping=3`, `vectors-in` `warping=2` | forward: each source frame pushed along its own flow as a textured triangle mesh over a mip pyramid, accumulated as a running mean per pixel, with Smart Blend and the hole fill | `forward.rs`, `forward/mesh.rs`, `forward/pyramid.rs`, `forward/raster.rs` |
| 3 | `retime-fgbg` `warping=4` | forward without the Smart Blend step or the hole fill: pixels no triangle covers stay empty | `forward.rs` |

The forward mesh is rendered cell by cell in a fixed order, so its accumulation, and with it the last bits, does not depend on the thread count. A flow that is all zero, or a warp amount below 1e-5, gives a plain copy.

## Composition

- **`frame_interp`**: Nearest takes the nearer warped frame; Blend mixes the two warped frames by their distance to the position (`blend`, `lerp` in `blend.rs`); Weighted Blend adds the sharpening step of `filters.rs` (a difference mask, its clean-up, a masked composite, `sharpen_weight` and `sharpen_step`) to each half.
- **Smart Blend** (`smartblend.rs`): per pixel, `over` composites of the two warped frames in both orders, mixed by the position's weight, keeping the larger alpha. With `video_360` and Blend it uses the binary-alpha variant, which treats alpha as either opaque or absent.
- **Hole fill** (`holefill.rs`): a pull-push pyramid fill, channel by channel, alpha first. Known pixels come back unchanged; a hole receives a smooth blend of the known pixels around it. Used by the Smart Blend modes and by `vectors-in`.

## Blur compensation

`warp/blurcomp.rs`. After the warps, `blur_amount` decides whether to run and computes the amount from the speed; a positive amount blurs along the flow with the motion blur primitive, a negative one builds a flow-directed unsharp mask; colour is then cleared where alpha is 0.

## Vector images

`vectors.rs` and `create_vectors.rs`.

- **Encoding** (`create-vectors`): the flow between source frame n and its neighbour (`track_frame`) is sampled bilinearly at every output pixel, scaled to output pixels, divided by `max_displace`, and stored with R = dx and G = dy centred on 127, 32767 or 0.5; B is 0 and alpha opaque. Motion beyond `max_displace` is clipped.
- **Decoding** (`vectors-in`): the R and G channels back into a flow, a hole mask from the alpha channel, the hole fill over the flow, then `vec_scale_x` and `vec_scale_y`. With one vector layer the other direction is derived from it.

## Determinism

Rows are independent and run through rayon row chunks, every output pixel depends only on input data, and the forward warp accumulates in a fixed order, so output bytes do not depend on the thread count. `fx-parity selfcheck` and every effect's `tests/threads.rs` check this. The same command on the same file and machine gave a byte-identical output file (measured).

## Cost

Measured on one desktop; the order of magnitude, not a guarantee.

| Work | Cost |
|---|---|
| an exact frame (a copy) at 1080p | a few hundredths of a second |
| an in-between frame at 1080p, 8 bpc, by `vectors` | `5` about 0.2 s, `4` 0.13 s, `3` or `2` 0.1 s, `1` 0.04 s |
| 18 output frames at 1080p, by `vectors` | `1` 1.1 s, `2` or `3` 1.7 s, `4` 1.9 s, `5` 2.6 s |
| `smart_blend=1` | about 1.7 times the render time |
| forward warping (`retime-fgbg` `warping=3`) | about 1.5 times on a small clip, about 6 times at 1080p |
| `blur_compensation` above 0 at exactly 100% | about 3 times, with every frame still a copy |
| 120 frames at 40%, 8 bpc, by size | 480x270 5 s, 960x540 9 s, 1920x1080 30 s, 3840x2164 94 s |

`tools/speed_curve.py` estimates a curve's render time at 0.25 s per in-between frame at 1080p.

## Module map

| Module | Stage |
|---|---|
| `timeline.rs` | rate model, Speed and Frame Number time maps, frame selection, exactness |
| `planner.rs` | the frames a render fetches, and the pre-render order |
| `frontend.rs`, `host.rs` | parameters to settings, frame fetch and premultiplication |
| `compose.rs`, `compose/vector.rs` | the render of one frame: fast path, vector path |
| `prep.rs` | luma, search resolution, Delinearize, Contrast/Edge Enhance |
| `flow.rs`, `flow/conv.rs`, `flow/sample.rs` | pyramid, optical flow solver, convolution, bilinear sampling |
| `warp.rs`, `warp/blurcomp.rs` | inverse warp, blur compensation |
| `forward.rs`, `forward/mesh.rs`, `forward/pyramid.rs`, `forward/raster.rs` | forward warp |
| `smartblend.rs`, `holefill.rs` | Smart Blend composites, hole fill |
| `blend.rs`, `filters.rs` | Blend, cross-fade, Weighted Blend |
| `vectors.rs`, `create_vectors.rs` | vector image encoding and decoding |
| `image.rs` | engine images |
| `matte.rs` | reserved, empty |

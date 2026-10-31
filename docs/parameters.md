# Parameters

Every control of the four effects: values, defaults, ranges and what each one changes. `target/release/vfx list --json` is the source of truth for ids, kinds, ranges and defaults; where this file disagrees with it, the listing is right. Value syntax and error messages: [reference: parameter values](reference.md#parameter-values). How keyed speeds and frame numbers become source positions: [timing](timing.md).

An *exact* output frame is one whose source position lies within 0.005 of a whole source frame: that frame is copied. Every other output frame is an *in-between* frame, built from the two source frames around the position ([timing: terms](timing.md#terms)).

## retime

Listed as **Retime**. 17 params, layers `alt_source_a` and `alt_source_b`.

| Id | Name | Values | Default | Range |
|---|---|---|---|---|
| `time_remap_mode` | Time Remap Mode | `1` Speed, `2` Frame Number | `1` | |
| `speed_percent` | Speed % | number, percent of the clip's own pace | `100` | -10000 to 10000 |
| `frame_num` | Frame Num | number, a source frame counted from 0 | `0` | -200000 to 200000 |
| `src_time_offset` | Src Time Offset | number of source frames | `0` | 0 to 30000 |
| `in_fps_is_out_fps` | In FPS is Out FPS | `1` on, `0` off | `1` | |
| `input_frame_rate` | Input: Frame Rate | frames per second | `29.969999313354492` | 0.10000000149011612 to 1000 |
| `input_fields` | Input: Fields | `1` None, `2` Upper Field First, `3` Lower Field First | `1` | |
| `frame_interp` | Frame Interp | `1` Nearest, `2` Blend, `3` Motion Weighted Blend | `2` | |
| `vectors` | Motion Vectors | `1` No Motion Vectors, `2` Sloppy, `3` Medium, `4` High, `5` Best | `5` | |
| `prep` | Image Prep | `1` None, `2` Contrast/Edge Enhance, `3` Delinearize | `1` | |
| `sensitivity` | Motion Sensitivity | number | `70` | 0 to 100 |
| `smart_blend` | Smart Blend | `1` on, `0` off | `0` | |
| `blur_compensation` | Motion Blur Compensation | number | `0` | 0 to 10 |
| `src_mark_segments` | Src: Mark Segments | `1` Cut A, `2` Cut B, `3` Cut C, `4` No MV, Use Nearest, `5` No MV, Frame Blend | `1` | |
| `display` | Display | `1` Twixtored Output, `2` Source | `1` | |
| `video_360` | 360 Video | `1` on, `0` off | `0` | |
| `plenty_memory` | Plenty Memory? | `1` on, `0` off | `1` | |

`input_frame_rate`'s default and minimum are the 32-bit floats 29.97 and 0.1 widened to doubles; the double `0.1` itself is below the minimum and refused.

### time_remap_mode

Selects what is keyed. `1` Speed: the source position is the running sum of `speed_percent`. `2` Frame Number: the source position is `frame_num` itself. The other mode's control is not read. Details: [timing: Speed mode](timing.md#speed-mode), [Frame Number mode](timing.md#frame-number-mode).

### speed_percent

Speed mode only. 100 plays the clip at its own pace, 50 at half (output twice as long), 200 at double, 0 holds the position reached, a negative value plays backwards. The position starts at 0 at output frame 0, so a negative first key reads before the clip ([timing: clip edges](timing.md#clip-edges)).

### frame_num

Frame Number mode only. The source frame shown at each output frame, counted from 0; a fraction shows an in-between frame. With `in_fps_is_out_fps=0` the value is scaled by the ratio of the two rates, so it still counts frames of the clip at its own rate. After its last key the last value holds.

### src_time_offset

`retime` only; `retime-fgbg` and `vectors-in` have no such parameter and refuse it as `unknown param`. In Speed mode output frame 0 shows source frame N instead of 0: source position p shows clip frame p + N. Only the integer part is used (`5.7` acts as `5`). Keys still count output frames from 0. With `in_fps_is_out_fps=0` N counts source frames at `input_frame_rate`: `src_time_offset=3` with `input_frame_rate=12` in a 24 fps clip starts 6 clip frames in. Frame Number mode ignores it for the retime. `display=2` adds it in either mode, counted in clip frames. The whole clip is still decoded and stored whatever the offset.

### in_fps_is_out_fps

On: every frame of the clip is a source frame, and `input_frame_rate` is not read. Off: the clip is read as holding `input_frame_rate` real frames per second inside a file that plays at its own rate; source frames sit at that spacing and the clip frames between them are not used. Neither setting changes the pace: at 100% output frame k lines up in time with clip frame k, and the output keeps the clip's rate (`vfx --clip-fps` reads footage of another rate by time, [timing: rate model](timing.md#rate-model)). `vectors-in` treats the on state as a 30 fps clip ([vectors-in](#vectors-in)). Rate model: [timing: rate model](timing.md#rate-model).

### input_frame_rate

Read only when `in_fps_is_out_fps=0`. A value within 0.00002 of 29.97, 59.94 or 23.976 snaps to exactly that decimal, so 30000/1001 footage needs 29.97003 (and 24000/1001 needs 23.976024, 60000/1001 needs 59.94006) to be read at its true rate. Measured on a 30000/1001 clip at 100%: with 29.97 every output frame from 5127 on is an in-between frame instead of a copy; with 29.97003 none of 5,400 frames is.

### input_fields

No frame is ever split into fields. A field order (`2` or `3`) only doubles the source frame rate: each clip frame counts as two source frames. At 100% the output is unchanged; slowed down, pictures repeat instead of being interpolated (measured: at 50% every frame shows twice, at 25% three of every four output frames repeat a clip frame).

### frame_interp

How an in-between frame combines its two source frames.

| Value | With `vectors` above 1 | With `vectors=1` |
|---|---|---|
| `1` Nearest | the nearer source frame alone, moved along the motion (the earlier one at exactly half way) | the nearer frame copied: frames repeat (0, 0, 1, 1 at 50%) |
| `2` Blend | both neighbours moved to the in-between moment and mixed by their distance to it | a cross-fade of the two frames |
| `3` Motion Weighted Blend | Blend with a sharpening step on each moved frame | identical to Blend |

None of them changes an exact frame.

### vectors

How finely motion is searched. `1` No Motion Vectors skips the search: in-between frames are copies or cross-fades. `5` searches at half the frame's width and height, `4` at a quarter, `3` and `2` at an eighth (`2` and `3` give identical output). `--quality draft` halves once more, never below an eighth. The search never halves below 2x2: a frame too small for its halvings is searched at the last size of at least 2x2 (6x6 at `4`: 3x3), and a frame with a side of 1 pixel has no search and renders as `vectors=1` (`create-vectors`: the zero vector). No result of the original is established at those sizes, so the rule is this project's own ([parity: rules without a note](parity.md#rules-without-a-note)). Exact frames never run the search. Cost per setting: [engine: cost](engine.md#cost).

### prep

Changes only the copy of the picture the motion search reads; the output is always built from the untouched source frames. `2` mixes an edge map into that copy. `3` brightens it with a 1/2.2 gamma, for footage stored in linear light.

### sensitivity

Read only when `vectors` is above 1. Limits how far a pixel may travel between two source frames and still be followed: lower values start the search on a finer scale and damp the motion found. `0` follows no motion; the output then equals `vectors=1` byte for byte (measured in all three `frame_interp` modes). `100` keeps whatever the search finds. Named `main_bg_sensitivity` in `retime-fgbg` and `create-vectors`.

### smart_blend

`retime` only (`retime-fgbg` has it as `warping=2`). Each pixel of an in-between frame comes from the source frame that actually covers it, and pixels neither covers are filled from their surroundings, instead of the default warp repeating the border pixels where content enters or leaves the frame ([engine: composition](engine.md#composition)). Measured cost: about 1.7 times the render time.

### blur_compensation

Blurs along the motion where the clip plays faster than 100% and sharpens along it where it plays slower. Strength is (source frames advanced per output frame - 1) x the value, which at matching rates is (|speed| / 100 - 1) x the value: zero at exactly 100%. It needs the motion search: with `vectors=1`, or where `src_mark_segments` turns the search off, it does nothing. Any value above 0 sends every output frame whose position moves through the motion path, exact frames included, so away from 100% frames that land on source frames change too; at exactly 100% they stay copies, but a 1080p test still took about three times as long.

### src_mark_segments

Labels source frames. For each in-between frame the labels of its two source frames decide how it is built; later rows win:

| Earlier frame | Later frame | In-between frame |
|---|---|---|
| `1`, `2` or `3` | the same label | built normally |
| `1`, `2` or `3` | another label | the nearer frame copied, no motion search |
| `4` | any | the nearer frame copied, no motion search |
| `5` | any | a cross-fade of the two, no motion search, whatever `frame_interp` says |

Its keys count source frames (clip frames), not output frames: the label is sampled at each source frame's own time. Labels step; they never interpolate. Because the `5` row wins, the pair that leaves a `5` span is cross-faded even across a label change. Example: a cut between source frames 23 and 24 is `1@0;2@24`; at 50% output frame 47 (position 23.5) is then a copy of frame 23.

### display

`2` shows the footage untouched at the output frame's own time (plus `src_time_offset` clip frames, in either mode), ignoring speed and every other setting.

### alt_source_a, alt_source_b

Optional layers, listed as **Alt Color Source** (`alt_source_a`) and **Alt Motion Source** (`alt_source_b`). `tools/apply.py` scales each to the output's size (stretched, not cropped) and pixel format, and resamples it to the input's rate; with `--size` or `--fps` a layer keeps its own rate and `vfx` reads it by time (`--layer-fps`).

| Layer | Role | Output pixels come from |
|---|---|---|
| `alt_source_a` | replaces the main clip as the footage that is retimed, searched and shown by `display=2` | A |
| `alt_source_b` | the motion search only; the warps follow B's motion | the main clip, or A |

Past a layer's last frame, A gives an empty picture and B gives the search an empty picture.

### video_360

In `retime` it is not a spherical-video switch: nothing wraps at the frame edges. On, the motion search reads only the red channel (as video range, 16 to 235) instead of a mix of all three, and with `smart_blend=1` and `frame_interp=2` the composite uses a binary-alpha variant. Both change in-between frames. In `retime-fgbg` it does wrap (below).

### plenty_memory

Caching only; output was byte for byte the same with `0`.

## retime-fgbg

Listed as **Retime (FG/BG)**. 63 params, layers `alt_source_a`, `alt_source_b`, `fg1_matte`, `fg2_matte`, `fg3_matte`. The FG matte layers (FG1 first; FG2 counts only after FG1, FG3 after FG2) split the frame into layers with one motion each, composed bottom up with FG1 on top ([engine: FG matte layers](engine.md#fg-matte-layers)); without them only the single layer path runs.

Every `retime` control except `src_time_offset` and `smart_blend`, with `sensitivity` named `main_bg_sensitivity` (listed as Main_BG Sensitivity, 0 to 100, default 70), plus:

| Id | Values | Default | Effect |
|---|---|---|---|
| `warping` | `1` Inverse, `2` Inverse w/ Smart Blend, `3` Forward, `4` Forward, no Smart Blend | `1` | the warp of in-between frames, below |
| `display` | `1` Twixtored Output, `2` Source, `3` FG1 Twixtored, `4` FG2 Twixtored, `5` FG3 Twixtored, `6` Background Twixtored | `1` | the layers composed: `1` all, `3` to `5` FG1 to FG3 alone (a missing one is the background), `6` the background (Main_BG, the whole frame warped with the background motion); without matte layers `3` to `6` render as `1` |
| `draw_geom` | `1` None, `2` All, `3` FG1, `4` FG2, `5` FG3, `6` Main_BG | `2` | on the Source display, the points in use of that group (`2` all) drawn as opaque discs of radius 1.5 pixels in their `ptN_color`, later points on top; the original draws an antialiased cross there, which this rewrite does not do yet (gap G3) |
| `track_rgb_a` | checkbox | `0` | adds the alpha channel to what the motion search reads; in a layered render it also switches the luma pyramids' filters to their 360 variants |
| `video_360` | checkbox | `0` | the wrap-around flag: the motion search, the inverse warps, the hole fill and the blur go around the sphere (x wraps, the poles reflect with a half turn); the forward warp itself does not wrap; the wrap of the layer masks in a layered render and of the draft quality blur is this project's reading (gap G6) |

| `warping` | In-between frame | Measured |
|---|---|---|
| `1` Inverse | each pixel pulled from both source frames along the motion; border pixels repeat outside the frame | byte for byte `retime` at its defaults (lossless test clip) |
| `2` Inverse w/ Smart Blend | the same pull, each pixel from the frame that covers it, gaps filled | byte for byte `retime` with `smart_blend=1` |
| `3` Forward | each source frame pushed along its own motion, with Smart Blend and gap filling | about 1.5 times the time on a small clip, about 6 times at 1080p |
| `4` Forward, no Smart Blend | the push of `3` without the Smart Blend step or gap filling: pixels nothing lands on stay empty | |

With matte layers, `fg1_matte_chan` to `fg3_matte_chan` (`1` Red to `4` Alpha) pick each layer's matte channel, `fg1_sensitivity` to `fg3_sensitivity` (FG1 Motion Sensitivity to FG3 Motion Sensitivity) its search sensitivity, and `fg1_inv_matte_shrink` to `fg3_inv_matte_shrink` choke, in pixels, the matte with which that layer cuts the layers below, with the plugin's own choke kernel; a shrink also makes exact frames take the motion path. `pt1_use` to `pt12_use` (`1` Don't Use, `2` FG 1, `3` FG 2, `4` FG3, `5` Main_BG Layer; without that layer the point goes to Main_BG) put the point `ptN_pos` on a layer when its value is the same at both source frames of the pair: on frames that search motion, the point's displacement enters the search at the finest level and the coarser levels take it from the image pyramid. `ptN_color` (Pt N: Color) is the colour Draw Geom draws it in. Without matte layers or points in use none of these changes anything.

## create-vectors

Listed as **Create Vectors**. 59 params, layers `alt_source_a` (Alt Motion Src), `fg1_matte`, `fg2_matte`, `fg3_matte`. Does not retime: output frame n is an encoded vector image of the motion between source frame n and a neighbour ([engine: vector images](engine.md#vector-images)).

| Id | Values | Default | Effect |
|---|---|---|---|
| `display` | `1` Source, `2` Motion Vectors, `3` Debug | `2` | `1` copies the source and draws the tracking points `draw_geom` names; `3` shows the Debug view of `debug_view_method` (the frame pair swaps: Previous compares frame n + 1 pushed onto frame n, Next frame n - 1) with the same overlay |
| `vector_resolution` | `1` Normal (half res), `2` Full res | `1` | the search resolution; `1` matches `vectors=5` in `retime` |
| `track_frame` | `1` Previous, `2` Next | `1` | the neighbour of frame n: `1` makes `vectors_prev`, `2` makes `vectors_next` |
| `max_displace` | 1 to 8192 | `2048` | the largest motion in pixels the image holds; larger motion is clipped. Stored in steps of `max_displace` / 127 pixels at 8 bits and / 32767 at 16 bits (16-pixel steps at 8 bits with the default) |
| `prep` | as in `retime` | `1` | as in `retime` |
| `track_rgb_a` | checkbox | `0` | adds the alpha channel to what the search reads |
| `main_bg_sensitivity` | 0 to 100 | `70` | as `sensitivity` in `retime` |
| `main_bg_track` | checkbox, listed as Main_BG: Image Track | `1` | `0` replaces the search of the background layer by its tracking points (the second tracker, gap G5): the motion is the points' displacement spread over the frame, none at all without a point |

With `fg1_matte` (then `fg2_matte`, `fg3_matte`; a layer counts only after the one before it) every layer gets its own motion: FG k with `fgk_sensitivity` and `fgk_track` (FGk: Image Track), its matte the `fgk_matte_chan` channel (`1` Red to `4` Alpha) of the layer, FG1 on top; the background (Main_BG) is the part the FG mattes leave free, and `fgk_inv_matte_shrink` (pixels) shrinks that free part around FG k. `layers_output` picks what the image holds: `1` All composites the layers' motion from the bottom up with their mattes, `2` to `4` one FG layer (its own motion, unweighted, with its visible matte in the blue channel), `5` MAIN_BG the background motion everywhere; a layer that does not exist gives MAIN_BG. `pt1_use` to `pt12_use` (`1` Don't Use, `2` FG1, `3` FG2, `4` FG3, `5` Main_BG Layer; without that layer the point goes to Main_BG) put the point `ptN_pos` on a layer when the value is the same at both frames of the pair: the point's displacement enters the search at the finest level and the coarser levels take it from the image pyramid. `debug_view_method` (`1` Warp to `5` Edge Compare, default `2`) is read with `display=3`; with matte layers and a one-layer `layers_output` its Blend view is a plain blend where the original runs a step that was not read (gap G4); `draw_geom` (`1` None, `2` All, `3` FG1 to `6` Main_BG, default `2`) draws the points in use at the frame time as the original's antialiased cross in `ptN_color` (Pt N: Color) on the Source and Debug displays. A frame 1 pixel wide or high has no motion search: Vectors and Debug show the zero vector whatever these controls. Without matte layers or points in use, these controls change nothing.

## vectors-in

Listed as **VectorsIn**. 19 params, layers `source` (listed as **Color Source**), `vectors_prev`, `vectors_next`, `fg1_matte`, `fg2_matte`, `fg3_matte`. Retimes with the vector images of `create-vectors` instead of searching for motion. The footage is the `source` layer when given, else the input clip. With only one vector layer the other direction is derived from it; with neither, every frame is the error fill ([reference: exit codes](reference.md#exit-codes)).

| Id | Values | Default | Effect |
|---|---|---|---|
| `display` | `1` Twixtored Output, `2` Source, `3` FG1 Twixtored to `5` FG3 Twixtored, `6` Background Twixtored | `1` | the layer groups as in `retime-fgbg`; without matte layers `3` to `6` render as `1` |
| `src_mark_segments` | as in `retime` | `1` | as in `retime` |
| `vectors_max_displace` | 0 to 10000 | `2048` | the `max_displace` the vector images were written with |
| `vec_scale_x`, `vec_scale_y` | -10 to 10 | `1` | multiply the stored motion; negative flips it |
| `in_fps_is_out_fps`, `input_frame_rate` | as in `retime` | `1`, 29.97 | see the rules below |
| `time_remap_mode`, `speed_percent`, `frame_num` | as in `retime` | | as in `retime` |
| `frame_interp` | as in `retime` | `2` | as in `retime` |
| `warping` | `1` Inverse, `2` Forward | `2` | as `warping` `1` and `3` in `retime-fgbg` |
| `blur_compensation` | as in `retime` | `0` | as in `retime` |

With matte layers the vector layers are read once per layer of the `display` group: each read caps the vector alpha by the layer's composition matte, then decodes and fills as without, and each layer is interpolated with its own vectors, as in `retime-fgbg`; `fgN_matte_chan` and `fgN_inv_matte_shrink` act as in `retime-fgbg`. Without matte layers they change nothing. Not offered: `vectors`, `prep`, `sensitivity`, `smart_blend`, `src_time_offset`, `input_fields`, `video_360`, `plenty_memory`. There is no output frame rate control: the plugin registers none, the output keeps the clip's rate, and `output_frame_rate` is refused as `unknown param`.

| Rule | Otherwise |
|---|---|
| `vectors_max_displace` equals the `max_displace` the images were made with | every motion is scaled by the ratio |
| the vector clips are at least as long as the output (`--frames`) | output frames past their end are the error fill |
| the last source position stays at or before the last source frame | the vector image there compares the last frame with nothing, and that output frame warps |
| at a clip rate other than 30 fps, `in_fps_is_out_fps=0` and the clip's rate in `input_frame_rate` | with `in_fps_is_out_fps=1` the clip is read as 30 fps and frame timing turns irregular |
| the vector clips are lossless (`tools/apply.py --lossless`, depth 16 for fine steps) | yuv420p and tv range change the stored motion |

Verified pair on the [smoke test](reference.md#smoke-test) clip: the vector file has 96 frames (ffv1, `gbrap16le`), and the mean colour of the output's last frame is `104 96 110`, not the `255 0 0` of the error fill.

```sh
python tools/apply.py target/demo/clip.mp4 create-vectors --param track_frame=2 --param max_displace=32 --frames 96 --depth 16 --lossless -o target/demo/vectors_next.mkv
python tools/apply.py target/demo/clip.mp4 vectors-in --layer vectors_next=target/demo/vectors_next.mkv --param in_fps_is_out_fps=0 --param input_frame_rate=24 --param vectors_max_displace=32 --param speed_percent=50 --frames 96 --depth 16 -o target/demo/slow_vectors.mp4
```

## How the effects differ

| Control | `retime` | `create-vectors` | `retime-fgbg` | `vectors-in` |
|---|---|---|---|---|
| `time_remap_mode`, `speed_percent`, `frame_num` | yes | no | yes | yes |
| `src_time_offset` | yes | no | no | no |
| `in_fps_is_out_fps`, `input_frame_rate` | yes | no | yes | yes (30 fps when on) |
| `input_fields` | yes | no | yes | no |
| `frame_interp` | yes | no | yes | yes |
| `vectors` | yes | `vector_resolution` | yes | no: motion comes from the vector clips |
| `prep` | yes | yes | yes | no |
| sensitivity | `sensitivity` | `main_bg_sensitivity` | `main_bg_sensitivity` | none |
| smart blend | `smart_blend` | no | `warping=2` | no |
| `warping` | no | no | 4 choices | 2 choices |
| `blur_compensation` | yes | no | yes | yes |
| `src_mark_segments` | yes | no | yes | yes |
| `video_360`, `plenty_memory` | yes | no | yes | no |
| `track_rgb_a` | no | adds alpha to the search | adds alpha to the search (and the 360 filter variants in a layered render) | no |
| `display` | 2 choices | 3 choices | 6 choices | 6 choices |
| layers | `alt_source_a`, `alt_source_b` | `alt_source_a`, `fg1_matte` to `fg3_matte` | `alt_source_a`, `alt_source_b`, `fg1_matte` to `fg3_matte` | `source`, `vectors_prev`, `vectors_next`, `fg1_matte` to `fg3_matte` |

## Refusals

A refusal stops the run with exit 3 and `vfx: error: frame N: unsupported: <message>`; `tools/apply.py` reports it under `[error] the pipeline failed:` and exits 1. `--check` does not catch these. Every control renders; what is left are inputs that no setting can serve. A control whose rule is still open prints a note instead ([README: limits](../README.md#limits)).

| Effect | Trigger | From frame | Message starts |
|---|---|---|---|
| all | an output position beyond 2^23 frames in Speed mode | that frame | `output position beyond 2^23 frames is not supported` |
| `vectors-in` | NaN samples in a 32 bpc vector layer, on a frame that warps with them (a frame with a side of 1 never reads the vector layers) | that frame | `inverse warp: a sample coordinate is NaN or never converges` or `motion blur: a tap coordinate is NaN ...` |

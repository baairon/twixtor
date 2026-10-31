# Timing

How the retiming effects map each output frame to a source position, and what that position produces. The arithmetic is `plugins/retime-engine/src/timeline.rs`. Key syntax: [reference: parameter values](reference.md#parameter-values).

## Terms

| Term | Meaning |
|---|---|
| source frame | a frame of the clip passed in, counted from 0 |
| source position | the source frame an output frame shows, possibly fractional: 17.9 is 90% of the way from frame 17 to 18 |
| exact frame | a position within 0.005 of a whole frame: that frame is copied, not built |
| in-between frame | any other position: built from source frames `trunc(p)` and `trunc(p) + 1` |
| tick | the unit keys are stored in: 1/100 of a frame at integer rates, 1/D of a frame at a rate N/D ([reference: time base](reference.md#parameter-values)) |

## Speed mode

`time_remap_mode=1`. The source position of output frame n is the running sum of the speed in half-frame steps, in 32-bit floats:

```
P(n) = sum for j = 1 to n of (S(j - 1/2) + S(j)) / 2
```

S is the speed as a fraction (`speed_percent` / 100), each sample read at the tick `trunc(t + 0.05)` of its half frame t. A fractional output position adds a partial step, the result is multiplied by the ratio of the source and output rates ([rate model](#rate-model)), and a negative output position gives 0.

- The sum exceeds the exact area under the curve by (S(n) - S(0)) / 4 frames: a quarter frame for every 100 points by which the current speed differs from the speed at frame 0. The shift vanishes when the speed returns to its starting value.
- A key does not mark a source frame: in `100@0;50@24;100@48` the key at output frame 24 lands on source position 17.9 and the key at 48 on 36.0.
- A hold key's jump takes effect half a frame before the key: in `200@0h;30@60h;100@78` the position at frame 60 is 119.2, not 120.
- `tools/speed_curve.py` runs this same sum ([what speed_curve.py models](#what-speed_curvepy-models)).
- The sum reads the speed again from frame 0 for every output frame. A segment with a handle costs a 64 step solve per read, so the curves keep every solved tick (at most 2^20 of them) and hand back that same value; the memo changes no output.

## Frame Number mode

`time_remap_mode=2`. The source position is `frame_num` sampled at the output frame's time, times the rate ratio. Fractions give in-between frames. Two equal eased keys, or two equal keys whose handles there have a speed of 0, stay exactly constant between them, and after the last key the last value holds. A falling key pair (`23@0;0@23`) plays backwards from the first output frame.

## Key shapes

Between two keys a float control moves by its progress `u` from 0 to 1 through the segment:

| Keys | Progress | Value halfway between 100 and 200 |
|---|---|---|
| plain, plain | `u` | 150 |
| eased, eased | `u*u*(3-2*u)` | 150 |
| eased start, plain end | `u*u*(2-u)` | 137.5 |
| plain start, eased end | `u*(1+u*(1-u))` | 162.5 |
| hold | the key's value until the next key | |
| a handle at either end | the cubic Bezier below | `100@0o0/33.3;200@10i0/33.3`: 150 |

A plain key keeps each neighbouring segment's straight-line slope, so the rate of change jumps at the key; an eased key has zero slope there. None of the first four shapes overshoots its two values. `tools/speed_curve.py` warns about every plain key where the speed's rate of change jumps by 1 point a frame or more; a key with handles was shaped on purpose and gets no warning.

A segment with a handle at either end is a cubic Bezier (After Effects' Keyframe Velocity model). With k ticks a second, n ticks between the keys, u = (t - ta) / n and d3 = vb - va, the end leaving key a is the control point (x1, d1): a handle `oS/I` gives (I/100, ((S/k) * (I/100)) * n), an eased key (1/3, 0), a plain key (1/3, d3/3). The end arriving at key b is (x2, d2): a handle `iS/I` gives (1 - I/100, d3 - ((S/k) * (I/100)) * n), an eased key (1 - 1/3, d3), a plain key (1 - 1/3, d3 - d3/3). In exact arithmetic those plain and eased ends give the four shapes above, which segments without handles keep bit for bit. The curve parameter comes from 64 bisection steps in f64: lo = 0, hi = 1, then 64 times m = (lo + hi) * 0.5, r = 1 - m, x = ((3*r*r*m)*x1 + (3*r*m*m)*x2) + m*m*m, lo = m if x < u, else hi = m; then s = (lo + hi) * 0.5, r = 1 - s and the value is va + (((3*r*r*s)*d1 + (3*r*s*s)*d2) + (s*s*s)*d3), the key's own value at u = 0. Only `+ - * /` in that order, so `tools/speed_curve.py` gives the same bits (`crates/fx-core/tests/fixtures/bezier_cases.hex`). With influences of 0.1 to 100 the time curve never turns back; at influence 100 on both ends it is flat halfway, where the solve is good to a few millionths of the segment (`100@0o0/100;200@10i0/100` gives 149.99968569895339 at frame 5). A handle may overshoot its two values: a float is held inside its valid range (`speed_percent` -10000 to 10000), an angle is not; a point moves on the straight line between its keys with va = 0 and vb = sqrt(dx*dx + dy*dy), and stays put when that length is 0. The plain end next to a handle, the influence of an eased end, the clamp, the solve and the point path are stated rules of this project ([parity: rules without a note](parity.md#rules-without-a-note)).

## Output length

`--frames N` sets how many output frames render. Without it the output has as many frames as the input clip (read at another rate with `vfx --clip-fps`, the output frames that start inside it), whatever the curve; nothing stretches or trims the clip to fit the curve.

| Quantity | Formula (speeds as fractions) |
|---|---|
| output frames for a constant speed S over C source frames | C / S |
| source frames a segment covers, both ends plain or both eased | length x (left speed + right speed) / 2 |
| the same, one end eased | the average sits 5/12 of the way from the left speed to the right (start eased) or 7/12 (end eased) |
| the same, a handle at either end | no closed form: `tools/speed_curve.py` runs the sum |
| delay of a stretch of L frames at average speed a | L x (1 - a) frames, relative to 100% |
| longest render at constant speed S that stays inside a C-frame clip | (C - 1) / S + 1, rounded down |

At 50% a 48-frame clip renders at most 95 frames inside the clip (positions 0 to 47.0).

## Clip edges

| Source position, clip of N frames | Output frame |
|---|---|
| 0 to N - 1 | the clip |
| between N - 1 and N | the last frame with its alpha reduced by the distance past it (half at N - 0.5; 80% at N - 0.8, 40% at N - 0.4); a lossless output keeps the alpha, a lossy one shows the last frame |
| N or more | empty (transparent, black once encoded) |
| between -1 and 0 | a picture extrapolated backwards from frames 0 and 1 |
| -1 or below | empty |

The last frame does not hold. At -100% from the start, the output is empty from output frame 1 on.

## Exact frames

An output frame is copied when its position is within 0.005 of a whole frame, except that with `blur_compensation` above 0.005 every frame whose position moves between output frames goes through the motion path. Copies cost a few hundredths of a second at 1080p; in-between frames run the full pipeline ([engine](engine.md)).

A constant run at a whole multiple of 100% shows exact frames only if it starts on the grid of whole source frames, and its offset depends on every ramp and plateau before it. `100@0;100@24;200@36;200@72` puts the 200% run 0.25 of a frame off the grid, so all 36 of its frames are in-between frames; moving the keys from frame 24 on by +0.25 frames (`100@0;100@24.25;200@36.25;200@72.25`) puts it on the grid. Lengthening a stretch at speed s by one frame lowers every later position by 1 - s, so an offset d is removed by lengthening it by d / (1 - s) frames. `tools/speed_curve.py` reports the final run's offset on its `tail` line and prints the corrected keys.

## In-between frame counts

Below 100%, each source frame covers 1 / S output frames. On a steady run that starts on the grid:

| Speed | Output frames per source frame | Exact copies | Built |
|---|---|---|---|
| 100% | 1 | all | none |
| 50% | 2 | 1 in 2 | 1 in 2 |
| 40% | 2.5 | 1 in 5 | 4 in 5 |
| 30% | 3.3 | 1 in 10 | 9 in 10 |
| 25% | 4 | 1 in 4 | 3 in 4 |
| 22% | 4.5 | 1 in 50 | 49 in 50 |
| 20% | 5 | 1 in 5 | 4 in 5 |

A run that starts off the grid has no exact frames. Above 100%, speeds that are not whole multiples of 100% also build frames: at 150% every second frame is built, at 200% and 300% none. The least number of new pictures per second of output is fps x (1 - S).

## Rate model

| Setting | Source frames | Rate ratio applied to positions |
|---|---|---|
| `in_fps_is_out_fps=1` (default) | every clip frame | 1 (both rates are taken as 30) |
| `in_fps_is_out_fps=0` | clip frames at `input_frame_rate` spacing | `input_frame_rate` / clip rate |
| `input_fields` `2` or `3` | each clip frame counted twice | source rate doubled |

- `vfx` reads a clip at the output rate unless `--clip-fps R` (`--layer-fps ID=R` for a layer) gives its own: then clip frame i covers i/R to (i+1)/R seconds and the effect's checkout at tick t gets clip frame floor(t x R / time_scale), in exact integers. At the output rate (the default) output frame k lines up in time with clip frame k at 100%. `tools/apply.py --fps` passes the footage's own rate this way.
- With `in_fps_is_out_fps=0` and `input_frame_rate=30` on a 60 fps clip, output frame 2k shows clip frame 2k and 2k + 1 is built between clip frames 2k and 2k + 2: the stored repeat is never read (measured on a file storing every frame twice: 119 of 120 output frames distinct, against 60).
- Footage whose real rate R is higher than the output rate F supplies real frames down to a speed of 100 x F / R percent: 50% for 120 fps footage in a 60 fps output, 25% for 240 in 60, 40% for 60 in 24. Read at its own rate (`--clip-fps R` with `in_fps_is_out_fps=0` and `input_frame_rate=R`), 60 fps footage at 40% in a 24 fps output shows footage frame n at output frame n.
- With another `--clip-fps` the default `--frames` is the output frames that start inside the clip, ceil(C x F / R) (25 frames at 24 give 63 at 60), and keys still count output frames. That a checkout between two frames gets the footage frame at its exact time, and this default length, are stated rules ([parity: rules without a note](parity.md#rules-without-a-note)).
- `input_frame_rate` snapping: [parameters: input_frame_rate](parameters.md#input_frame_rate). `src_time_offset`: [parameters: src_time_offset](parameters.md#src_time_offset).

## Audio

`tools/apply.py --audio retime` plays the input's audio along the same time map as the picture, varispeed: the pitch follows the speed, and nothing preserves it. `tools/audio_map.py` does it; ffmpeg only decodes and encodes.

**The map.** Output position t, in output frames and continuous, shows clip position p(t):

| Setting | p(t) |
|---|---|
| Speed mode (`time_remap_mode=1`) in `retime`, `retime-fgbg` and `vectors-in` | linear between the knots of the half frame sum of [Speed mode](#speed-mode) (32-bit floats, the speed read at tick trunc(L x k / 2 + 0.05)), plus trunc(`src_time_offset`) x out_fps / in_fps in `retime` |
| Frame Number mode (`time_remap_mode=2`) | `frame_num` at the continuous tick t x L, with its keys, eases and handles; a hold stays put until the next key and jumps there |
| `display=2` (Source) | t + trunc(`src_time_offset`) in `retime`, t in `retime-fgbg` and `vectors-in` |
| `create-vectors`, a frame effect, any other effect | t |

L is the ticks a frame of the time base. The mode, the display, the offset and the rate settings are read at the start of each output frame, as the effect reads them for that frame, and hold over it. The rate settings (`in_fps_is_out_fps`, `input_frame_rate`, `input_fields`) act only through the offset: the rate ratio and the duration of a source frame cancel everywhere else. `src_time_offset=3` with `in_fps_is_out_fps=0` and `input_frame_rate=12` in a 24 fps clip is 6 clip frames; `input_fields=2` halves that to 3. At every output frame p is the position the picture shows; `python tools/audio_map.py positions` prints them.

**Sound and silence.** Output sample i plays the input at clip time p(i x fps / SR) / R, with R the rate of the decoded clip (the output rate, or the input's own when `tools/apply.py --fps` passes `--clip-fps`): input sample (p / R - offset) x SR, linearly interpolated, on every channel. The offset is a_start - v_start + lead / R, a_start and v_start being the audio and video streams' ffprobe `start_time`, taken to the nearest whole sample. `lead` is the number of copies of its first frame the decode writes in front of the clip: ffmpeg's raw decode is constant rate and starts at the file's start time, the earliest `start_time` of any stream, so a video stream that starts after another stream gets about (v_start - file start) x R copies; mpegts, mpeg program streams and hls take their start from the streams the decode reads and get none. `tools/apply.py` works the lead and the frame count out from the video packets, with one ffprobe pass that decodes nothing and the arithmetic of ffmpeg's constant rate video sync.

The audio is silent where p < 0, where p reaches the clip's frame count (the picture is empty from there; between N - 1 and N it still shows the last frame), where the sample lies outside the decoded audio, and where p stands still (a hold, a run at 0%, a flat `frame_num`). Every edge between sound and silence fades linearly over 2 ms; a jump of p between two places that both sound cuts straight across; a falling p plays backwards. 100% with the audio and the video starting together copies every sample bit for bit.

**Length.** Each stream gets floor(N x SR / fps + 1/2) samples, as long as the video, with no `-shortest`. N is `--frames`, else the frames `vfx` renders: the container's count, else the frames the decode writes, counted from the video packets, so matroska has an exact length too. Only when the packets cannot give the count (a variable rate input, packets without timestamps) does the map render ceil(duration x fps) + 1 frames, duration being the video's length from the file's start time, and `-shortest` cuts it.

**Sync.** The audio plays the clip time the picture shows within 1 ms at every frame while the effect reads its source at or below the output rate and without fields (`input_fields` off, and `in_fps_is_out_fps=1` or an `input_frame_rate` no higher than the output rate). What is left there is the copy rule of [Exact frames](#exact-frames): a picture within 0.005 of a frame (0.21 ms at 24 fps) shows that frame, while the audio keeps the exact position. With `input_fields` set, or an `input_frame_rate` above the output rate, the picture steps in whole clip frames while the audio slides along the continuous map, so the gap stays under one clip frame; the audio is not stepped to match, because a stepped map would click at every step. `tools/test_apply.py` holds both bounds on a clip whose frame n is gray 512 x n and whose audio at time t is t / 8, and the clicks of an aac track to 2 ms.

## What speed_curve.py models

`tools/speed_curve.py` runs the Speed mode sum above with the effect's tick rules and the default time base, so its positions agree with a render to a few thousandths of a frame. It models a clip that plays at the output rate (`in_fps_is_out_fps=1`) and starts at clip frame 0. It does not model:

- `time_remap_mode=2`;
- `in_fps_is_out_fps=0` with a typed `input_frame_rate`, or `input_fields`;
- clip rates other than the output rate (`vfx --clip-fps`);
- `src_time_offset`: with an offset k its position p is clip frame p + k, so it exits 3 for positions the offset brings back inside the clip. `--keys="-100@0" --fps 24 --clip-frames 630 --frames 41` warns `output frames 1 to 40 read source frames outside the clip (0 to 629): positions -1.0 to -40.0`, which with `src_time_offset=40` are clip frames 39 to 0.

Its exit 3 covers every output frame more than 0.005 outside 0 to N - 1, which includes the reduced-alpha band of [clip edges](#clip-edges). It does model Bezier handles, with the effect's arithmetic. `tools/audio_map.py positions` prints the clip position of every output frame for Frame Number mode, `src_time_offset`, the rate settings and another clip rate as well ([Audio](#audio)), with no table of speeds or warnings.

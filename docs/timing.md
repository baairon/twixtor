# Timing

How the retiming effects map each output frame to a source position, and what that position produces. The arithmetic is `plugins/retime-engine/src/timeline.rs`. Key syntax: [README: parameter values](../README.md#parameter-values).

## Terms

| Term | Meaning |
|---|---|
| source frame | a frame of the clip passed in, counted from 0 |
| source position | the source frame an output frame shows, possibly fractional: 17.9 is 90% of the way from frame 17 to 18 |
| exact frame | a position within 0.005 of a whole frame: that frame is copied, not built |
| in-between frame | any other position: built from source frames `trunc(p)` and `trunc(p) + 1` |
| tick | the unit keys are stored in: 1/100 of a frame at integer rates, 1/D of a frame at a rate N/D ([README: time base](../README.md#parameter-values)) |

## Speed mode

`time_remap_mode=1`. The source position of output frame n is the running sum of the speed in half-frame steps, in 32-bit floats:

```
P(n) = sum for j = 1 to n of (S(j - 1/2) + S(j)) / 2
```

S is the speed as a fraction (`speed_percent` / 100), each sample read at the tick `trunc(t + 0.1)` of its half frame t. A fractional output position adds a partial step, the result is multiplied by the ratio of the source and output rates ([rate model](#rate-model)), and a negative output position gives 0.

- The sum exceeds the exact area under the curve by (S(n) - S(0)) / 4 frames: a quarter frame for every 100 points by which the current speed differs from the speed at frame 0. The shift vanishes when the speed returns to its starting value.
- A key does not mark a source frame: in `100@0;50@24;100@48` the key at output frame 24 lands on source position 17.9 and the key at 48 on 36.0.
- A hold key's jump takes effect half a frame before the key: in `200@0h;30@60h;100@78` the position at frame 60 is 119.2, not 120.
- `tools/speed_curve.py` runs this same sum ([what speed_curve.py models](#what-speed_curvepy-models)).

## Frame Number mode

`time_remap_mode=2`. The source position is `frame_num` sampled at the output frame's time, times the rate ratio. Fractions give in-between frames. Two equal eased keys stay exactly constant between them, and after the last key the last value holds. A falling key pair (`23@0;0@23`) plays backwards from the first output frame.

## Key shapes

Between two keys a float control moves by its progress `u` from 0 to 1 through the segment:

| Keys | Progress | Value halfway between 100 and 200 |
|---|---|---|
| plain, plain | `u` | 150 |
| eased, eased | `u*u*(3-2*u)` | 150 |
| eased start, plain end | `u*u*(2-u)` | 137.5 |
| plain start, eased end | `u*(1+u*(1-u))` | 162.5 |
| hold | the key's value until the next key | |

A plain key keeps each neighbouring segment's straight-line slope, so the rate of change jumps at the key; an eased key has zero slope there. No shape overshoots its two values. `tools/speed_curve.py` warns about every plain key where the speed's rate of change jumps by 1 point a frame or more.

## Output length

`--frames N` sets how many output frames render. Without it the output has as many frames as the input clip, whatever the curve; nothing stretches or trims the clip to fit the curve.

| Quantity | Formula (speeds as fractions) |
|---|---|
| output frames for a constant speed S over C source frames | C / S |
| source frames a segment covers, both ends plain or both eased | length x (left speed + right speed) / 2 |
| the same, one end eased | the average sits 5/12 of the way from the left speed to the right (start eased) or 7/12 (end eased) |
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

- The output always has the clip's frame rate, and at 100% output frame k lines up in time with clip frame k.
- With `in_fps_is_out_fps=0` and `input_frame_rate=30` on a 60 fps clip, output frame 2k shows clip frame 2k and 2k + 1 is built between clip frames 2k and 2k + 2: the stored repeat is never read (measured on a file storing every frame twice: 119 of 120 output frames distinct, against 60).
- Footage whose real rate R is higher than the output rate F supplies real frames down to a speed of 100 x F / R percent: 50% for 120 fps footage in a 60 fps output, 25% for 240 in 60, 40% for 60 in 24.
- `input_frame_rate` snapping: [parameters: input_frame_rate](parameters.md#input_frame_rate). `src_time_offset`: [parameters: src_time_offset](parameters.md#src_time_offset).

## What speed_curve.py models

`tools/speed_curve.py` runs the Speed mode sum above with the effect's tick rules and the default time base, so its positions agree with a render to a few thousandths of a frame. It models a clip that plays at the output rate (`in_fps_is_out_fps=1`) and starts at clip frame 0. It does not model:

- `time_remap_mode=2`;
- `in_fps_is_out_fps=0` with a typed `input_frame_rate`, or `input_fields`;
- `src_time_offset`: with an offset k its position p is clip frame p + k, so it exits 3 for positions the offset brings back inside the clip. `--keys="-100@0" --fps 24 --clip-frames 630 --frames 41` warns `output frames 1 to 40 read source frames outside the clip (0 to 629): positions -1.0 to -40.0`, which with `src_time_offset=40` are clip frames 39 to 0.

Its exit 3 covers every output frame more than 0.005 outside 0 to N - 1, which includes the reduced-alpha band of [clip edges](#clip-edges).

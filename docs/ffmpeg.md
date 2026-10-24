# ffmpeg pairing

How `vfx` exchanges frames with ffmpeg, and the exact commands `tools/apply.py` builds around it. The wrapper's flags, lookup order, refusals, verification and output lines are in [README: wrapper](../README.md#wrapper); this file covers what happens to the pixels, the rate, the colour and the sound.

## Raw frames

`vfx` reads and writes ffmpeg's own raw formats, whole frames back to back with no header ([README: raw frame format](../README.md#raw-frame-format)).

| `--depth` | Decode to `vfx` | Read `vfx` output back |
|---|---|---|
| 8 | `-f rawvideo -pix_fmt rgba -` | `-f rawvideo -pix_fmt rgba -video_size WxH -framerate R -i -` |
| 16 | `-f rawvideo -pix_fmt rgba64le -` | `-f rawvideo -pix_fmt rgba64le -video_size WxH -framerate R -i -` |
| 32 | `-f rawvideo -pix_fmt gbrapf32le -` | `-f rawvideo -pix_fmt gbrapf32le -video_size WxH -framerate R -i -` |

A frame is W x H x 4, 8 or 16 bytes. Alpha is straight. ffmpeg's YUV to RGB step needs the source's colour matrix (`-vf scale=in_color_matrix=bt709` for HD), or it assumes BT.601 ([colour](#colour)).

## Piping without the wrapper

Each line below ran on the smoke test clip and wrote 12 frames:

```sh
ffmpeg -v error -i target/demo/clip.mp4 -f rawvideo -pix_fmt rgba64le - | target/release/vfx apply retime --width 160 --height 96 --depth 16 --fps 24 --param speed_percent=50 --frames 12 | ffmpeg -v error -f rawvideo -pix_fmt rgba64le -video_size 160x96 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p target/demo/pipe16.mp4
ffmpeg -v error -i target/demo/clip.mp4 -f rawvideo -pix_fmt gbrapf32le - | target/release/vfx apply retime --width 160 --height 96 --depth 32 --fps 24 --param speed_percent=50 --frames 12 | ffmpeg -v error -f rawvideo -pix_fmt gbrapf32le -video_size 160x96 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p target/demo/pipe32.mp4
```

A temporal effect needs random access, so `vfx` first copies piped stdin whole to a spool file in `--tmp-dir` (removed on exit), then renders. `--fps` must be the exact rate (`30000/1001`, not `29.97`, for NTSC footage), because the time base follows the fraction.

## The apply.py pipeline

`tools/apply.py` runs four kinds of ffmpeg and `vfx` command, printed as `[cmd]` lines by `--dry-run`. Placeholders: `IDX` the input's first video stream that is not cover art, `M` the colour matrix, `R` the rate as `N/D`, `FMT` the raw format of `--depth`, `S` the scratch folder.

| Stage | Command |
|---|---|
| layer decode, per `--layer` | `ffmpeg -v error -nostdin -y -i LAYER -map 0:IDX -an -sn -dn -vf [fps=R,]scale=W:H:in_color_matrix=M -pix_fmt FMT -f rawvideo S/layer_ID.FMT` (`fps=R` when the layer's rate differs or varies, the size only when it differs) |
| decode | `ffmpeg -v error -nostdin -i INPUT -map 0:IDX -an -sn -dn -vf [fps=R,]scale=in_color_matrix=M -pix_fmt FMT -f rawvideo -` |
| effect | `vfx apply EFFECT --width W --height H --depth D --fps R [--frames N] --tmp-dir S [--layer ID=S/layer_ID.FMT] [--param K=V ...]` |
| encode | `ffmpeg -v error -nostdin -n -f rawvideo -pix_fmt FMT -video_size WxH -framerate R -i - -i INPUT -map 0:v:0 -map 1:a? -vf scale=out_color_matrix=M:out_range=tv,setparams=colorspace=...:range=tv -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy [-shortest] [-movflags +faststart] OUTPUT.partial-PID.EXT` |
| encode, `--lossless` | the same inputs and maps, then `-c:v ffv1 -level 3 -pix_fmt bgra` (8 bpc) or `rgba64le` (16 bpc), `-c:a copy`, no `-vf` |

Decode, effect and encode run as one pipe; the layer decodes run first, one after another. The partial file is probed and verified before it is moved onto the output path.

### Frame rate

- The input's rate is `r_frame_rate` from ffprobe. When `r_frame_rate` and `avg_frame_rate` differ by more than 1/1000 of `r_frame_rate`, the input counts as variable rate: its rate becomes `avg_frame_rate` and the decode adds `fps=R`, which repeats or drops frames to fit. The `[plan]` line shows the rate picked (`52.555 fps` for `avg_frame_rate=7200/137`).
- `vfx` always gets the rate as an exact fraction (`--fps 24/1`, `--fps 30000/1001`), and the encode writes the same rate.
- The output keeps the input's rate; only `--frames` changes its length. Without `--frames`, a variable-rate input gives as many frames as its resampling makes.

### Size and orientation

The size is the displayed size: a 90 or 270 degree rotation (side data or the `rotate` tag) swaps width and height. A layer of another size is scaled to the input's size, stretched. The lossy path needs even sides (yuv420p); `--lossless` takes any size.

### Colour

| Source `color_space` | Matrix used to decode and encode | Tags written |
|---|---|---|
| `bt709`, `bt470bg`, `smpte170m`, `bt2020nc`, `smpte240m` or `fcc` | that matrix (`bt470bg` as swscale's `bt601`, `bt2020nc` as `bt2020`) | the source's own `colorspace`, `color_primaries` and `color_trc`; a tag it lacks stays unset |
| anything else, or untagged | `bt709` when the longer side is at least 1280 or the height at least 720, else `smpte170m` | all three set to that name |

The encode always writes `range=tv`. Decoding and encoding with the same matrix returns the RGB the effect wrote as the same YUV. HDR inputs (`color_transfer` `smpte2084` or `arib-std-b67`) are refused; an HDR file without those tags is not detected.

A `--lossless` output is RGB with no matrix. Re-encoding it to YUV with ffmpeg alone uses BT.601 coefficients and writes no tags (`color_range`, `color_space`, `color_transfer` and `color_primaries` all read `unknown`). Naming the matrix in the filter chain converts and tags it:

```sh
ffmpeg -i lossless.mkv -vf "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv" -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
```

That output reads back `tv,bt709,bt709,bt709`.

### Streams and audio

- Video: only the first stream that is not cover art (`attached_pic`) is decoded.
- Audio: `-map 1:a?` copies every audio stream of the input unchanged (`-c:a copy`): no retiming, no re-encode, no level change. When `--frames` makes the video shorter than the audio by more than a frame, `-shortest` cuts it at the end of the video; when the video is longer, the audio ends early.
- Subtitle and data streams are not copied.
- There is no option to drop, replace or retime the audio.

### Encoding

| Path | Settings | Readback (the smoke test's `slow.mp4`) |
|---|---|---|
| default | libx264, CRF 18, x264's default `medium` preset, yuv420p, tv range, 8-bit 4:2:0 at any `--depth`; `-movflags +faststart` for `.mp4`, `.m4v` and `.mov` | `profile=High`, `bframes=3`, `open_gop=0`, `keyint=250`, `crf=18.0` |
| `--lossless` | ffv1 level 3 in RGB, alpha kept, no matrix or range conversion; `.mkv` by default; refused with `--depth 32` | 8 bpc stored as `bgra`; 16 bpc requested as `rgba64le` and stored as `gbrap16le`, the nearest format ffv1 has |

No bitrate, GOP or B-frame setting is passed. The x264 settings are readable from the file with `grep -a -o -E "(bframes|keyint|open_gop|crf)=[0-9.]+" out.mp4`.

## Scratch and disk

- Scratch lives in an `apply_XXXXXXXX` folder in the system temp folder, removed when the run ends; `--dry-run` prints it on the `[temp]` line. Python picks the temp folder from `TMPDIR`, then `TEMP`, then `TMP`, on every platform; a variable naming a folder that does not exist is skipped without a message. Git Bash sets `TMPDIR=/tmp`, so setting `TEMP` alone changes nothing there.
- The whole input is decoded into one raw spool file before the first frame renders, whatever `--frames` or `src_time_offset` say: frames x width x height x bytes per pixel (4, 8 or 16). Each `--layer` adds a raw file of its own frames at the same size and depth.

| 1080p, 60 fps | Frames | `--depth 8` | `--depth 16` | `--depth 32` |
|---|---|---|---|---|
| one frame | 1 | 8.3 MB | 16.6 MB | 33.2 MB |
| 10 s | 600 | 5.0 GB | 10.0 GB | 19.9 GB |
| 30 s | 1,800 | 14.9 GB | 29.9 GB | 59.7 GB |
| 60 s | 3,600 | 29.9 GB | 59.7 GB | 119.4 GB |
| 5 min | 18,000 | 149 GB | 299 GB | 597 GB |

GB is 10^9 bytes.

## Input requirements

What the effects assume about the clip, and what a clip that breaks the assumption produces.

| Assumption | Otherwise |
|---|---|
| a constant frame rate: frame k happened at k / rate | `tools/apply.py` resamples to the average rate, repeating and dropping frames; gaps become repeats |
| every frame a new picture | a repeated frame is a hold followed by a jump once slowed; the effect cannot tell a repeat from a still moment (`tools/repeat_keys.py` finds and replaces isolated repeats; a regular pattern such as every frame stored twice is declared with `in_fps_is_out_fps=0` and `input_frame_rate`) |
| one continuous shot | in-between frames across a hard cut morph the two shots (about 100 / speed - 1 blended frames per cut) unless the cut is labelled with `src_mark_segments` |
| the retime starts at the clip's first frame | keys and `--frames` count from clip frame 0; a later start needs a trimmed clip or `src_time_offset` (`retime` only) |
| progressive frames | fields are never separated ([parameters: input_fields](parameters.md#input_fields)) |
| SDR | HDR is refused |
| even sides | refused on the lossy path; `--lossless` accepts odd sides |
| motion of less than about 5% of the image per frame | estimates degrade above it; very fast or crossing motion warps or ghosts at any setting |

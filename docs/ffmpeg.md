# ffmpeg pairing

How `vfx` exchanges frames with ffmpeg, and the exact commands `tools/apply.py` builds around it. The wrapper's flags, lookup order, refusals, verification and output lines are in [README: wrapper](../README.md#wrapper); this file covers what happens to the pixels, the rate, the size, the colour and the sound.

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

`tools/apply.py` runs ffmpeg, `vfx` and, with `--audio retime`, `tools/audio_map.py` commands, printed as `[cmd]` lines by `--dry-run`. Placeholders: `IDX` the input's first video stream that is not cover art, `M` the colour matrix, `R` the output rate as `N/D`, `RI` and `RL` the rates of the input and of a layer, `FMT` the raw format of `--depth`, `S` the scratch folder, `W:H` the output size.

| Stage | Command |
|---|---|
| layer decode, per `--layer` | `ffmpeg -v error -nostdin -y -i LAYER -map 0:IDX -an -sn -dn -vf [fps=R,]scale=W:H:in_color_matrix=M -pix_fmt FMT -f rawvideo S/layer_ID.FMT` (`fps=R` when the layer's rate differs or varies, the size only when it differs; with `--size` or `--fps` the layer keeps its own rate, `fps=` only making a variable one constant at its average, and `vfx` reads it with `--layer-fps`) |
| audio decode, `--audio retime` | `ffmpeg -v error -nostdin -y -i INPUT -map 0:A -c:a pcm_f32le -f f32le S/audio_1.f32 [-map 0:A2 -c:a pcm_f32le -f f32le S/audio_2.f32 ...]`, one output a stream, A the audio stream's index |
| audio retime, per stream | `python -I tools/audio_map.py render --effect EFFECT --fps R --frames N --clip-frames C [--clip-fps RI] [--param K=V ...] --rate SR --channels CH --offset O --in S/audio_K.f32 --out S/audio_K_retimed.f32`, with N the output's frames, C the frames of the decoded clip and O the offset of [timing: audio](timing.md#audio) |
| decode | `ffmpeg -v error -nostdin -i INPUT -map 0:IDX -an -sn -dn -vf [fps=RI,][settb=D/N,setpts=N,fps=R:round=up,]scale=[W:H:]in_color_matrix=M -pix_fmt FMT -f rawvideo -` (`fps=RI` for a variable rate; the `settb` group for a frame effect under `--fps`, N/D the input's rate; `W:H` under `--size`) |
| effect | `vfx apply EFFECT --width W --height H --depth D --fps R [--frames N] --tmp-dir S [--layer ID=S/layer_ID.FMT] [--clip-fps RI] [--layer-fps ID=RL] [--device D] [--param K=V ...] [--param in_fps_is_out_fps=0 --param input_frame_rate=RI]` (`--clip-fps` and `--layer-fps` for a temporal effect under `--size` or `--fps`; `--device` only when given to `apply.py`; the two params under `--fps` as in [Frame rate](#frame-rate)) |
| encode | `ffmpeg -v error -nostdin -n -f rawvideo -pix_fmt FMT -video_size WxH -framerate R -i - -i INPUT -map 0:v:0 -map 1:a? -vf scale=out_color_matrix=M:out_range=tv,setparams=colorspace=...:range=tv -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy [-shortest] [-movflags +faststart] OUTPUT.partial-PID.EXT` |
| encode, a size with an odd side | the same, with `-pix_fmt yuv444p` at that exact size: the output's size, or a declared stage's |
| encode, HDR input | the same inputs and maps, then `-vf scale=out_color_matrix=M:out_range=tv,setparams=colorspace=S:color_primaries=P:color_trc=T:range=tv -c:v libx265 -crf 18 -pix_fmt yuv420p10le` (`yuv444p10le` for an odd side) `[-tag:v hvc1] -x265-params log-level=error[:master-display=G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)][:max-cll=C,F] -c:a copy [-shortest] [-movflags +faststart]`, `hvc1` for `.mp4`, `.m4v` and `.mov` ([HDR](#hdr)) |
| encode, `--lossless` | the same inputs and maps, then `-c:v ffv1 -level 3 -pix_fmt bgra` (8 bpc) or `rgba64le` (16 bpc), `-c:a copy` (`-c:a flac` with `--audio retime`), no `-vf` (an HDR input: `-vf setparams=color_primaries=P:color_trc=T`, which changes no pixel) |
| encode, `--audio retime` or `drop` | the audio input is `-f f32le -ar SR -ch_layout L -i S/audio_K_retimed.f32` a stream (`-ac CH` without a known layout), mapped `-map K:a:0` in place of `-map 1:a?`, encoded `-c:a aac -b:a:K BR [-ar:a:K 96000]` (BR the source's bit rate, else 96000 a channel) with `-metadata:s:a:K language=... title=...`, and `-shortest` only when the input does not say its frame count; under `drop` there is no audio input and no audio map |
| stage `ffmpeg-filter`, as `--pre-stage` or `--post-stage` | `python tools/ffstage.py apply ffmpeg-filter --width {width} --height {height} --depth {depth} --fps {fps} --param graph=G [--param expect=declared] [--param unsafe=1]`, which runs `ffmpeg -v error -nostdin -f rawvideo -pix_fmt FMT -video_size WxH -framerate R -i - -vf setparams=colorspace=M:color_primaries=M:color_trc=M,G -fps_mode passthrough -pix_fmt FMT -f rawvideo -` (M by the size rule of [Colour](#colour)). Under `expect=preserve` (the default) bytes out equal bytes in; a graph that changes the size, rate or count is refused with exit 3. Under `expect=declared` the stage goes with `--geometry declared` (`--contract frames/2`). An input that ends inside a frame exits 4, although ffmpeg exits 0 on it. |
| `ffmpeg-filter` dry run, for `--describe` and before every run | `ffmpeg -v error -nostdin -f lavfi -i color=c=gray:s=WxH:r=R,format=FMT,trim=end_frame=N -vf setparams=...,G -fps_mode passthrough -pix_fmt FMT -f framemd5 -` with N at most 60 frames and 2 seconds of video: the size from `#dimensions`, the rate from `#tb` and the timestamps (it must stay R, an exact fraction), the count from the frame lines |
| `ffmpeg-filter` probes | `ffmpeg -version` (9.0.2 is the measured version; another is reported as unverified), `ffmpeg -hide_banner -pix_fmts` (FMT readable and writable), `ffmpeg -hide_banner -filters` (every filter of G present). Without `--param unsafe=1` a graph may only use the frame filters `boxblur`, `colorbalance`, `colorchannelmixer`, `colorlevels`, `copy`, `crop`, `deflicker`, `drawbox`, `drawgrid`, `edgedetect`, `eq`, `fade`, `format`, `fps`, `framerate`, `gblur`, `geq`, `hflip`, `hue`, `lut`, `lutrgb`, `lutyuv`, `minterpolate`, `negate`, `noise`, `null`, `pad`, `reverse`, `rotate`, `scale`, `select`, `setdar`, `setparams`, `setpts`, `setsar`, `smartblur`, `sobel`, `tblend`, `tmix`, `transpose`, `trim`, `unsharp`, `vflip`, `vignette`, which read and write nothing but the frames; any other filter is refused with exit 3 (`unsupported`), and a filter this ffmpeg lacks is refused first as `has no filter`. A graph ffmpeg refuses in the dry run also exits 3 |

Decode, effect and encode run as one pipe; the layer decodes, then the audio steps of `--audio retime`, run first, one after another. Under `--fps` a pre stage of a temporal effect reads the input at its own rate: its `{fps}` and `{frames}` are the input's, and only the post stages see the output's. The partial file is probed and verified before it is moved onto the output path.

### Frame rate

- The input's rate is `r_frame_rate` from ffprobe. When `r_frame_rate` and `avg_frame_rate` differ by more than 1/1000 of `r_frame_rate`, the input counts as variable rate: its rate becomes `avg_frame_rate` and the decode adds `fps=R`, which repeats or drops frames to fit. The `[plan]` line shows the rate picked (`52.555 fps` for `avg_frame_rate=7200/137`).
- `vfx` always gets the rate as an exact fraction (`--fps 24/1`, `--fps 30000/1001`), and the encode writes the same rate.
- The output keeps the input's rate unless `--fps R` sets another. A temporal effect then reads the input at its own rate RI (`vfx --clip-fps RI`, a variable rate input first made constant at its average as above) and each layer at its own (`--layer-fps`): clip frame i covers i/RI to (i+1)/RI seconds, so no `fps` filter drops or repeats a real frame on the way in. A frame effect cannot read by time, so its decode converts: `settb=D/N,setpts=N` renumbers the input's frames at its exact rate and `fps=R:round=up` gives each output frame the input frame that covers its start.
- Without `--frames` the output has the output frames that start inside the input, ceil(C x R / RI): 25 frames at 24 give 63 at 60, 60 frames at 60 give 24 at 24. A variable-rate input's count is not known before the run (`frames` is null); it gives as many frames as its resampling makes, converted the same way.
- `retime`, `retime-fgbg` and `vectors-in` get `in_fps_is_out_fps=0` and `input_frame_rate=RI` (`%.10g`, so 30000/1001 is `29.97002997`, read at its true rate) unless the user gives either; so at 100% the footage keeps its pace and slowed down it shows its real frames (60 fps footage at 40% in a 24 fps output: output frame n is footage frame n). Keys, `frame_num` and `--frames` count output frames.
- A value equal to the input's rate (the average of a variable rate input) changes nothing.

### Size and orientation

The size is the displayed size: a 90 or 270 degree rotation (side data or the `rotate` tag) swaps width and height. `--size WxH` scales the input to W x H in the decode (`scale=W:H:in_color_matrix=M`, stretched, before the effect); a layer of another size is scaled to the output's size, stretched. A size equal to the input's changes nothing.

yuv420p needs even sides, and of the chroma layouts only 4:4:4 has a crop unit of one pixel, so on the lossy path a size with an odd side (the output's, or the one a declared stage gives) encodes yuv444p at that exact size (yuv444p10le for HDR); the plan says `encoded yuv444p (the size has an odd side)`. Not every player or browser decodes 4:4:4 (H.264 High 4:4:4 Predictive, HEVC Main 4:4:4 10). `--lossless` takes any size as it is.

### Colour

| Source `color_space` | Matrix used to decode and encode | Tags written |
|---|---|---|
| `bt709`, `bt470bg`, `smpte170m`, `bt2020nc`, `smpte240m` or `fcc` | that matrix (`bt470bg` as swscale's `bt601`, `bt2020nc` as `bt2020`) | the source's own `colorspace`, `color_primaries` and `color_trc`; a tag it lacks stays unset |
| anything else, or untagged | `bt709` when the longer side is at least 1280 or the height at least 720, else `smpte170m` | all three set to that name |

The encode always writes `range=tv`. Decoding and encoding with the same matrix returns the RGB the effect wrote as the same YUV. An HDR input has rules of its own ([HDR](#hdr)).

A `--lossless` output is RGB with no matrix. Re-encoding it to YUV with ffmpeg alone uses BT.601 coefficients and writes no tags (`color_range`, `color_space`, `color_transfer` and `color_primaries` all read `unknown`). Naming the matrix in the filter chain converts and tags it:

```sh
ffmpeg -i lossless.mkv -vf "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv" -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
```

That output reads back `tv,bt709,bt709,bt709`.

### HDR

An input whose `color_transfer` is `smpte2084` (PQ) or `arib-std-b67` (HLG) is HDR. Its signal goes through as stored, with no tone mapping: the decode turns the YUV into R'G'B' with the input's matrix and nothing else, so the effect warps and blends PQ or HLG code values, not light, and an identity gives the input's code values back. A layer, HDR or not, is read the same way, with its own matrix.

| | HDR input |
|---|---|
| matrix (decode and encode) and `colorspace` tag | as tagged when swscale has that matrix (`bt2020nc` as `bt2020`, `bt709`, ...), else `bt2020nc`; never the size convention |
| `color_primaries` | as tagged, else `bt2020` |
| `color_trc` | the input's |
| range | tv |
| encode | libx265 CRF 18, `yuv420p10le` (`yuv444p10le` for an odd side), `-tag:v hvc1` for `.mp4`, `.m4v` and `.mov`, at any `--depth` (8 rounds the 10 bit code values to 8 on the way to the effect) |
| HDR metadata | the master display and the content light level of the video stream's side data (what the container holds), else of its first frame's (the SEI of HEVC): written back as `-x265-params log-level=error:master-display=G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min):max-cll=C,F` (chromaticity in 0.00002, luminance in 0.0001 cd/m2, rounded half up), each only when the input has it whole and in range |
| `--lossless` | ffv1 in RGB, tagged with the primaries and the transfer (`setparams`); no master display or light level is written |
| libx265 missing, or without 10 bit 4:2:0 and 4:4:4 | refused before any work (`tool_missing`); `--lossless` needs no libx265 |

An HDR file without those transfer tags is not detected: it renders as SDR, as it always has (the matrix of the size convention, libx264 yuv420p). Retag it first, the stream with the metadata bitstream filter of its codec and the container with ffmpeg's three colour options (the filter alone leaves a container's old tags standing), without re-encoding:

```sh
ffmpeg -i in.mp4 -map 0 -c copy -bsf:v hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9 -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc pq.mp4
ffmpeg -i in.mp4 -map 0 -c copy -bsf:v h264_metadata=colour_primaries=9:transfer_characteristics=18:matrix_coefficients=9 -color_primaries bt2020 -color_trc arib-std-b67 -colorspace bt2020nc hlg.mp4
```

`transfer_characteristics` 16 is PQ and 18 HLG, `colour_primaries` and `matrix_coefficients` 9 are BT.2020 (non-constant luminance). With ffmpeg 9.0.2 both lines turn an untagged or a BT.709-tagged 10 bit file, in `.mp4` or `.mkv`, into one ffprobe reads as `bt2020nc`, `bt2020` and the transfer, in the container and in the stream (`tools/test_apply_hdr.py`, `RetagRecipeTests`).

Passing the signal as stored is a stated rule; a 16 bpc render of PQ code values in an unmanaged project of the host would test it ([parity: rules without a note](parity.md#rules-without-a-note)).

### Streams and audio

- Video: only the first stream that is not cover art (`attached_pic`) is decoded.
- Audio, `--audio copy` (the default): `-map 1:a?` copies every audio stream of the input unchanged (`-c:a copy`): no retiming, no re-encode, no level change. When `--frames` makes the video shorter than the audio by more than a frame, `-shortest` cuts it at the end of the video; when the video is longer, the audio ends early.
- Audio, `--audio retime`: every stream is decoded to raw `pcm_f32le` at its own rate, played along the effect's time map by `tools/audio_map.py` ([timing: audio](timing.md#audio)) and encoded again, aac or flac (`--lossless`), at its own rate (aac at most 96 kHz), channel count, language and title, in the input's order; the channel layout is kept too, except where the encoder relabels an equivalent one (aac writes `5.1(side)` as `5.1`). Each stream is as long as the video, with no `-shortest`, except for an input whose frame count the video packets cannot give, where `-shortest` cuts the audio on the encoder's frames (aac 1024 samples, flac up to 4608).
- Audio, `--audio drop`: no audio input, no audio map, no audio.
- Subtitle and data streams are not copied.

### Encoding

| Path | Settings | Readback (the smoke test's `slow.mp4`) |
|---|---|---|
| default | libx264, CRF 18, x264's default `medium` preset, yuv420p, tv range, 8-bit 4:2:0 at any `--depth`; `-movflags +faststart` for `.mp4`, `.m4v` and `.mov` | `profile=High`, `bframes=3`, `open_gop=0`, `keyint=250`, `crf=18.0` |
| a size with an odd side | as default, `yuv444p` at the exact size | `profile=High 4:4:4 Predictive` |
| HDR input | libx265, CRF 18, x265's default `medium` preset, `yuv420p10le` (`yuv444p10le` for an odd side), tv range, the input's tags and HDR metadata ([HDR](#hdr)); `-tag:v hvc1` and `-movflags +faststart` for `.mp4`, `.m4v` and `.mov` | `profile=Main 10` (`Rext` for 4:4:4), `bframes=4`, `keyint=250`, `crf=18.0`, `hdr10` with the master display |
| `--lossless` | ffv1 level 3 in RGB, alpha kept, no matrix or range conversion; `.mkv` by default; refused with `--depth 32`; an HDR input keeps its primaries and transfer as tags | 8 bpc stored as `bgra`; 16 bpc requested as `rgba64le` and stored as `gbrap16le`, the nearest format ffv1 has |

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
| SDR, or HDR tagged `smpte2084` or `arib-std-b67` | an HDR file without those tags renders as SDR; retag it first ([HDR](#hdr)) |
| even sides for yuv420p | an odd side encodes yuv444p at the exact size on the lossy path; `--lossless` takes any size |
| motion of less than about 5% of the image per frame | estimates degrade above it; very fast or crossing motion warps or ghosts at any setting |

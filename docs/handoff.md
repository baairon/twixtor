# Handoff convention

How ffmpeg, twixtor and sapphire (and any later plugin) pass raw frames to each other, report failures, discover each other's stages and hand tasks between tools as packets. The contract files in `contracts/` are the specification; `python -I tools/handoff.py contracts` checks that every contract file parses and that every schema validates its own examples.

## Files

| File | Role |
|---|---|
| `contracts/tool.json` | registry of the tools that take part: id, markers, built binaries, stage providers, and the list of shared files |
| `contracts/tool.schema.json` | schema of the registry |
| `contracts/error.schema.json` | the error document and the one error table (`x-codes`), the stage exits and the frames/1 exits |
| `contracts/stage-manifest.schema.json` | what a stage provider offers: provider id, version, command, and its stages |
| `contracts/describe.schema.json` | the answer of a declared geometry stage to `apply STAGE --describe` |
| `contracts/run-report.schema.json` | the keys every successful wrapper `--json` result shares, with `role` and `index` per step |
| `contracts/handoff-packet.schema.json` | packet `schema_version` 2 |
| `contracts/handoff-packet-v1.schema.json` | packet `schema_version` 1, kept so old packets stay readable |
| `tools/handoff.py` | the control plane: packets, discovery, `stages`, `sync`, `contracts`, `doctor` (Python 3.12 standard library, run with `python -I`) |
| `tools/test_handoff.py` | its test suite; nothing is skipped |

`handoff.py sync` copies the files named by the `shared` list of `contracts/tool.json` byte for byte to every peer and confirms equality by SHA-256; `handoff.py doctor` fails when a contract file differs between peers.

## Contract versions

| Contract | Spoken by default by | Adds |
|---|---|---|
| `frames/1` | twixtor `tools/apply.py` (its default) | the raw frame contract below; with stages, a `contract` key, a `stages` array and the error object in `--json` |
| `frames/2` | `sapphire`, `tools/handoff.py`; twixtor `tools/apply.py` with `--contract frames/2` | the error object on every failure, the normative exit codes of the table below, the stage exits 2, 3 and 4 mapped for every stage, `role` and `index` in reports, declared geometry and `--describe` |

A wrapper takes `--contract frames/1` or `--contract frames/2`. twixtor `tools/apply.py` defaults to frames/1 and without the flag behaves byte for byte as before; `sapphire` defaults to frames/2. A consumer refuses a manifest or a result whose contract major it does not know (`frames/1` and `frames/2` are known).

## Frame contract

| Depth | ffmpeg pixel format | Bytes per pixel | Layout | Sample |
|---|---|---|---|---|
| 8 | `rgba` | 4 | packed R, G, B, A per pixel | unsigned 8 bit, 0 to 255 |
| 16 | `rgba64le` | 8 | packed R, G, B, A per pixel | unsigned 16 bit little endian, 0 to 65535 |
| 32 | `gbrapf32le` | 16 | four whole planes in the order G, B, R, A, rows top to bottom | 32 bit float little endian, 1.0 is full scale |

Rules for every stage, in both contracts:

- Frames are whole frames back to back, with no header and no padding; a frame is `width * height * bytes per pixel` bytes.
- Alpha is straight (not premultiplied).
- A frame is at most 4 GiB.
- A stream that ends inside a frame is an error (`partial_frame`); a stage that reads a partial last frame exits 4.
- A stage writes as many frames as it reads, unless its manifest says otherwise through declared geometry.
- Frame rates are exact fractions such as `30000/1001`, never `29.97`.

## Errors and exit codes

Every `--json` failure is one error document:

```json
{"ok": false, "contract": "frames/2",
 "error": {"code": "stage_timeout", "stage": "pre1", "exit_code": null,
           "message": "pre-stage 1 moved no byte for 120 seconds and was killed", "stderr_tail": []}}
```

`code` is a row of the table below, `stage` the label of the failing step (or null), `exit_code` the status of the failing child process (or null when no child failed), `message` one line, `stderr_tail` the last lines of the child's standard error. A tool may add keys next to `error` (`verify` and `doctor` add their checks). Text mode prints one line `error: <code>: <message>` on standard error.

| Code | frames/2 exit | frames/1 exit | When |
|---|---|---|---|
| `stage_failed` | 1 | 1 | a process exited non zero outside the stage exits 2, 3 and 4, a layer decode failed, or the output did not verify or could not be moved into place |
| `usage` | 2 | 1 | a bad flag or flag combination, an unknown placeholder value, or a stage that exited 2 |
| `unsupported` | 3 | 1 | a valid request that is refused: stage exit 3, unknown effect, HDR input, odd side without `--lossless`, unknown geometry value, an fps change by a declared stage, a replay of an unregistered program |
| `bad_input` | 4 | 1 | an input that is missing, unreadable or malformed, stage exit 4, an invalid packet, a hash or expectation mismatch, a revision conflict |
| `broken_pipe` | 4 | 1 | a stage stopped reading, or exited 0 after writing fewer whole frames than it was sent |
| `partial_frame` | 4 | 1 | a producer's output ended inside a frame, or a file is not a whole number of frames |
| `stage_timeout` | 5 | 1 | no byte moved on a judged process for `--stage-timeout` seconds, a replayed command ran past its limit, or a packet lock stayed held |
| `tool_missing` | 6 | 1 | a program (ffmpeg, ffprobe, vfx, sapphire, a stage program) or a peer repository cannot be found or started |
| `interrupted` | 130 | 130 | interrupted (Ctrl+C); of the two wrappers only twixtor `tools/apply.py` exits 130, `sapphire` has no Ctrl+C handler; `tools/handoff.py` exits 130 too |

Under frames/1 the wrapper exits keep their old meaning: 0 done or dry run, 1 any refusal or failure, 2 the argument parser rejected the command line, 130 interrupted.

A stage process (the program behind `--stage`, `--pre-stage` or `--post-stage`) exits:

| Stage exit | Code | Meaning |
|---|---|---|
| 0 | none | every input frame was processed and written |
| 2 | `usage` | the stage rejected its command line |
| 3 | `unsupported` | the stage refuses the request (effect, parameter, depth, geometry) |
| 4 | `bad_input` | the stage could not read its input or a layer, or the input ended inside a frame |
| any other | `stage_failed` | any other status, a crash or a kill |

`sapphire stage apply` exits with these stage exits under both contracts; the frames/1 exits 0, 1, 2 belong to wrappers only. Under frames/2 both wrappers map the exits 2, 3 and 4 of every process stage; under frames/1 twixtor `tools/apply.py` maps them for the `vfx` step only.

Stall watch: `--stage-timeout` judges the pre and post stages, decode and encode, so a stalled ffmpeg is killed instead of hanging the wrapper. Decode is judged while the wrapper waits for its output; encode only while it refuses data it is offered, so an encode that finishes its file after the last frame is never judged. The `vfx` effect step stays unjudged, because a temporal effect is silent while it spools its input. twixtor `tools/apply.py` judges decode and encode under frames/2 only.

## Reports

A successful wrapper `--json` result carries `ok`, `contract` and `stages`: one entry per process in pipe order.

| Key | Meaning |
|---|---|
| `role` | `decode`, `pre`, `effect`, `post`, `encode` or `verify` |
| `index` | 1 for the first step of its role, counting in pipe order |
| `name` | a label for people, not an identifier: both twixtor `tools/apply.py` and `sapphire` name the stages before the effect `pre1`, `pre2` and those after it `post1`, `post2` (`sapphire` counts from its `vfx` marker) |
| `command` | the argv that ran, without a shell |
| `exit_code` | the status, or null in a dry run or for a step that did not start |
| `stderr_tail` | the last lines of the step's standard error |

Under frames/2 the result also carries `geometry` (width, height, depth, fps, frames after the last declared stage). Scripts select a step by `role` and `index`, never by `name`.

## Discovery

Tools. `contracts/tool.json` lists every registered tool with an `id` (lower case, no hyphen), `markers` (relative paths that all exist in that tool's repository root), `binaries` (name, environment override, build folders, version arguments) and `providers` (which binary lists stages, with which arguments and adapter).

| Question | Answer, in order |
|---|---|
| Which tool is this root? | `--self ID`, `HANDOFF_SELF`, else the one registry entry whose markers all exist under the root |
| Where is peer ID? | `--peer-dir ID=DIR`, `HANDOFF_<ID>_DIR` (ID in upper case), else the one sibling folder holding all of the peer's markers |
| Where is a built binary? | its environment override (`VFX_PATH`, `SAPPHIRE_PATH`), `$CARGO_TARGET_DIR/release`, the registry `dirs` under the tool's root, `PATH` |
| Where are ffmpeg and ffprobe? | `MEDIA_FFMPEG_PATH` or `FFMPEG` (`MEDIA_FFPROBE_PATH` or `FFPROBE`), `PATH`, the WinGet links folder, `%ProgramFiles%\ffmpeg\bin` |

Folder names are never used: a repository can be renamed or moved. A folder given by flag or variable that does not exist is an error, not a fall through; two sibling folders with the same markers are an error that names both.

Stages. A stage manifest (`contracts/stage-manifest.schema.json`) names the provider (`id`, `version`, `command`: the argv prefix that `list --json` and `apply STAGE ...` are appended to) and its stages (`id`, `kind` frame or temporal, `spool`, `geometry`, `depths`, `params`, `layers`). Providers are read from:

| Source | Form |
|---|---|
| `--provider FILE` | a `*.stage.json` manifest; a relative program path is relative to the file's folder |
| `HANDOFF_PROVIDERS` | folders (separated like `PATH`) whose `*.stage.json` files are manifests |
| registry, adapter `native` | `sapphire stage list --json` prints a manifest |
| registry, adapter `vfx-list` | `vfx list --json --all` prints a bare array without a contract field; `handoff.py` wraps it as a frames/1 manifest (every effect keeps its geometry, a temporal effect spools) and `vfx` stays unchanged |

`python -I tools/handoff.py stages --json` lists every stage of every provider with a ready command line in its `stage` key, for example `vfx apply retime --width {width} --height {height} --depth {depth} --fps {fps}`, to pass as `--stage`, `--pre-stage` or `--post-stage`; the wrapper fills the placeholders. A provider that cannot be listed is reported under `problems`, not as a failure.

## Geometry

| Value | Meaning |
|---|---|
| `preserve` | width, height, fps and pixel format of the output equal the input; the frame count may change only for a temporal stage that says so through its parameters |
| `declared` | frames/2: the stage answers `apply STAGE --describe --width W --height H --depth D --fps N/D [--frames N] [--param K=V]...` without reading standard input, with its output width, height, depth, fps and frame count (`contracts/describe.schema.json`) |

Only width, height and the frame count may change; depth and fps of the answer equal the request, and a stage that would change the rate refuses with `unsupported`. The wrapper asks every declared stage in chain order, feeds each answer into the next question, and passes the last answer to encode and verify. A consumer refuses a stage whose geometry value it does not know (`handoff.py stages` lists it under `problems`).

On a wrapper command line a stage is declared with `--geometry declared` placed right after it: after its `--pre-stage` or `--post-stage` in twixtor `tools/apply.py`, after its `--stage` in `sapphire`. `preserve` is the default, any other value exits 3 (`unsupported`), and `--geometry` or `--describe` without frames/2 is refused.

`--describe` on a wrapper (frames/2) reads and writes no frames and prints the geometry after every declared stage as one describe answer (`contracts/describe.schema.json`) whose `stage` is the effect id, such as `retime`; `sapphire` run without an effect names it `stages`.

## The ffmpeg filter stage

ffmpeg stays upstream and unmodified; it runs as a child process. The stage `ffmpeg-filter` runs one filter graph on raw frames:

| Part | Rule |
|---|---|
| Parameters | `graph` (an ffmpeg filter graph with one input and one output), `expect` (`preserve`, the default, or `declared`), `unsafe` (`1` allows filters outside the safe list below) |
| Implementations | Rust `crates/neutral-handoff/src/ffstage.rs` on `neutral_ffmpeg::Filter` in sapphire (`sapphire stage apply ffmpeg-filter`, in a wrapper also the built in `@ffmpeg-filter`); the Python twin twixtor `tools/ffstage.py` |
| `expect preserve` | bytes out are the same count as bytes in, frame for frame; a graph that changes the size, rate or count is an error |
| `expect declared` | `--describe` measures the output geometry with a bounded dry run (at most 2 seconds and 60 frames) |
| Safe filters | without `unsafe=1` a graph may only use the frame filters `boxblur`, `colorbalance`, `colorchannelmixer`, `colorlevels`, `copy`, `crop`, `deflicker`, `drawbox`, `drawgrid`, `edgedetect`, `eq`, `fade`, `format`, `fps`, `framerate`, `gblur`, `geq`, `hflip`, `hue`, `lut`, `lutrgb`, `lutyuv`, `minterpolate`, `negate`, `noise`, `null`, `pad`, `reverse`, `rotate`, `scale`, `select`, `setdar`, `setparams`, `setpts`, `setsar`, `smartblur`, `sobel`, `tblend`, `tmix`, `transpose`, `trim`, `unsharp`, `vflip`, `vignette`, which read and write nothing but the frames; any other filter is refused with exit 3 (`unsupported`), and a filter this ffmpeg lacks is refused first as `has no filter` |
| Version | ffmpeg 9.0.2 is the measured version; other versions run conformance live and are reported as unverified |
| Probing | adapters run `ffmpeg -version`, `-pix_fmts` and `-filters` before use |
| Truncation | ffmpeg exits 0 on truncated input, so the stage counts frames and bytes itself |
| Rates and colour | rates are exact fractions; the colour matrix rule of twixtor `tools/apply.py` applies (an untagged source of 1280 wide or 720 high and up is bt709, a smaller one smpte170m) |

## Packets v2

A packet is one JSON file that carries a task from one tool to another: inputs with hashes and geometry, the stage chain, what a correct result looks like, and later the result with the commands that ran.

| Rule | Detail |
|---|---|
| Place | `handoff/<id>.json` in the RECEIVING repository (git ignored); the id is `<UTC YYYYMMDDTHHMMSSZ>-<from>-<to>-<6 hex digits>` |
| Route | `from` and `to` are tool ids of `contracts/tool.json`; a packet naming an unregistered tool is invalid |
| Status | `new`, then `accepted`, then `done` or `failed`, then `verified`; `accept` on an accepted packet is a no-op, `done` needs an accepted packet |
| Paths | every path in the packet is a relative POSIX path (no leading slash, drive, backslash or `..` segment) from the root of the repository holding it; `new` copies an input or layer from elsewhere into `handoff/<id>/inputs/` or `handoff/<id>/layers/` |
| Revision | `new` writes revision 1; every later write is a compare and swap: under the lock the writer checks that the file still holds the revision it read and writes revision + 1; `--revision N` refuses unless the file holds N; a conflict is `bad_input` |
| Lock | `handoff/<id>.lock`, created exclusively, holds pid, host and time; a lock of a process that is gone on this host is broken, an unreadable lock older than 60 seconds too; a lock of another host is honoured; waiting longer than `--lock-wait` (default 10 seconds) is `stage_timeout` |
| Size | a packet file is at most 1 MiB |
| Atomic | every write goes through a temporary name in the same folder and a rename |
| Replay | `verify --replay` runs the recorded commands again with every output redirected to a scratch folder and compares bytes (exact, tolerance or none); without `--trust` only ffmpeg, ffprobe and registered providers are replayed, each through the local program of that name in the scratch folder, reaching no path but the packet inputs and the scratch folder; anything else is `unsupported` |
| Version 1 | `list`, `show` and `verify` read version 1 packets; every other write needs `handoff.py migrate ID` first, which makes the paths relative and refuses a packet that names a file outside the repository |

## Commands

| Command | Does |
|---|---|
| `new --to ID --task TEXT [--input PATH ...] [--stage ID ...]` | write a packet into the receiver's `handoff/` |
| `list [--other \| --peer ID] [--status S]` | list packets here or in a peer |
| `show ID` | print one packet |
| `accept ID [--revision N]` | take a new packet |
| `done ID --output PATH --run SPEC` | run or record commands and record the outputs |
| `verify ID [--replay [--trust]]` | recompute hashes and frame counts, check `expect`, optionally replay |
| `migrate ID \| --all` | rewrite version 1 packets as version 2 |
| `stages [--provider FILE]` | every stage of every provider with a ready `--stage` command |
| `sync [--to ID] [--check] [PATH ...]` | copy the shared files to the peers and confirm SHA-256 equality |
| `contracts` | check every contract file and every schema's examples |
| `doctor [--conformance]` | peers, ffmpeg, ffprobe, binaries and their contracts, contract files, shared file equality, optionally `tools/conformance.py --quick` |

Every command takes `--json`, `--root DIR`, `--self ID`, `--peer-dir ID=DIR` and `--lock-wait SECONDS`; `--json` results carry `"contract": "frames/2"`.

## Platforms

| Platform | Rule |
|---|---|
| Windows | a wrapper ends a stage and its children with `taskkill /T`, and `handoff.py` ends a command that runs past its timeout the same way; `handoff.py` checks a lock's pid through `OpenProcess` |
| POSIX | a wrapper starts each stage in its own process group (Python `start_new_session`, Rust `CommandExt::process_group`) and ends the group with `kill` (twixtor's Python tools signal the group even after its leader has exited); `handoff.py` does the same for a command that runs past its timeout and checks a lock's pid with signal 0 |
| CI | the stage, handoff and conformance suites run on Windows, Linux and macOS |
| Python | 3.12, standard library only, `python -I` |

The POSIX branches count as unproven until a CI run has passed them.

## How a third plugin joins

1. Implement `list --json` (a stage manifest) and `apply STAGE --width W --height H --depth D --fps N/D [--frames N] [--param K=V]... [--layer ID=PATH]...` with frames on standard input and output, the stage exits above and, for declared geometry, `--describe`. A plugin that cannot print a manifest ships a `*.stage.json` file instead.
2. Vendor `contracts/` and `tools/conformance.py` byte for byte.
3. Add one entry to `tools` in `contracts/tool.json` (id, markers, binaries, providers) and run `python -I tools/handoff.py sync` so every peer holds the same registry.
4. Pass `python -I tools/conformance.py --stage "<command>"`.
5. Check with `python -I tools/handoff.py stages --json` and `python -I tools/handoff.py doctor`.

A plugin that only provides stages and never exchanges packets can skip step 3 and be found through `--provider FILE` or `HANDOFF_PROVIDERS`.

## Known gaps

| Gap | State |
|---|---|
| `vfx list --json` | a bare array without a contract field; the `vfx-list` adapter wraps it, `vfx` stays unchanged |
| Error kinds under frames/1 | twixtor `tools/apply.py` gives `usage`, `unsupported` and `bad_input` to the `vfx` step only; under frames/2 every stage exit 2, 3 and 4 maps to them |
| POSIX | the process group branches have not run yet; the first Linux or macOS CI run covers them |
| Lock across hosts | a lock left by another host is never broken automatically; remove it by hand after checking that host |
| Replay trust | a trusted program is matched by name and runs as the local program of that name, so a replay proves the local build, not the recording machine's |
| Version 1 packets | a packet that names files outside its repository stays version 1 and read only |
| Interrupt in `sapphire` | the crate installs no Ctrl+C handler (standard library only, no unsafe code), so it never exits 130 |
| Declared stage before the effect | twixtor `tools/apply.py` with `--contract frames/2` accepts `--pre-stage CMD --geometry declared`: the stage's `--describe` answer becomes the size that `vfx` and the layer decodes run at; `sapphire` refuses `--stage CMD --geometry declared` placed before its `vfx` marker with `unsupported` (exit 3, stage `vfx`, `exit_code` null), because its effect runs at the input size; the conformance suite (`level5` in `tools/conformance.py`) tests declared stages after the effect only |
| Safe filters | `unsafe=1` lifts the safe list and runs any filter the graph names, so it suits only a trusted graph |
| `ffmpeg-filter` in `stages` | twixtor ships no `*.stage.json` for `tools/ffstage.py`, so `python -I tools/handoff.py stages` does not list twixtor's `ffmpeg-filter`; `python tools/ffstage.py list --json` prints its manifest |
| ffmpeg exit status | ffmpeg exits 0 on truncated input, on partial frames fed to the encoder and for `-n` with an existing output; frame and byte counts decide |
| Lossy output | libx264 output is not byte reproducible between runs; compare with `--lossless` or a tolerance |

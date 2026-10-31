# Emulator harness

Renders fx-parity test sets with the four original After Effects plugins (`Twixtor_64.aex`, `TwixtorPro_64.aex`, `CreateMotionVectors_64.aex`, `TwixtorProVectorsIn_64.aex`) under Qiling (x8664 Windows) with a mock After Effects host, so that `fx-parity run` can grade the rewrite against them. The method and the results are in [docs/parity.md](../../docs/parity.md).

**State.** All four plugins boot, complete `GLOBAL_SETUP` and `PARAMS_SETUP`, and render through `SMART_PRE_RENDER` and `SMART_RENDER`. The references of the six module grids were made with this harness. A plugin boots in about 40 s, and a small case then renders in about a second.

The harness loads the plugins unmodified and drives them through the host interface. It records what the loader reports, every host callback and Windows API call the plugin makes, and the address and registers of a fault. It never patches a plugin.

## Files

| File | Does |
|---|---|
| `ae_host.py` | Qiling start-up, guest memory, host functions behind trampolines, calls into the guest, API hooks, heap, floating point state, call record |
| `ae_mock.py` | The After Effects side: `in_data`/`out_data`, callbacks, suites, params, worlds, PARAMS_SETUP, SEQUENCE_SETUP, RENDER or SMART_PRE_RENDER + SMART_RENDER, wire format conversion |
| `win_seh.py`, `win_threads.py`, `win_semaphore.py`, `win_crypt.py`, `win_rtl.py`, `win_path.py`, `win_io.py` | The Windows functions the plugins need that Qiling lacks (see Stand-ins) |
| `keys.py` | Keyframe interpolation of case params (a stand-in, see below) |
| `worker.py` | One emulator process: boots one plugin once, renders its share of cases, one JSON line per event |
| `run_harness.py` | Renders a whole fx-parity test set, `-j N` workers, restarts a worker that dies |
| `dump_params.py` | Boots a plugin and writes what it registers in PARAMS_SETUP |
| `make_param_map.py`, `param_map_<plugin>.tsv` | The plugin params paired with `vfx` ids, one map per plugin (see [Param map](#param-map)) |
| `smoke.py` | The smoke test |
| `callrecord_check.py` | Proves `VFX_CALL_RECORD` does not change a byte of output |
| `overlay_mask.py`, `speed_probe.py`, `probe_set.py`, `capture_msvcr120.py` | Diagnostics: an output mask, emulator speed, probe sets, a trace of the C runtime's start-up |
| `gen_layouts.py`, `sdk_probe.c`, `gen_aegp_suites.py` | Struct layouts, enums and suite function tables read from the SDK headers' debug info (gcc `-g -D__ANDROID__`, which gives the Windows x64 layout: 32 bit `A_long`, 8 byte pointers); the generated `sdk_layouts.json` is not in the repository, `sdk_layouts_extra.json` is |
| `make_hive.py` | The empty registry hives Qiling opens |
| `push_dlls.sh`, `explore_load.py`, `needed_dlls.txt` | Copies exactly the DLLs the loader reports missing |
| `results/` | `summary.json` (the harness's own statement, shown in [parity: per-module notes](../../docs/parity.md#per-module-notes)) and `callrecord_check.json` |
| `test_*.py` | `python -m unittest discover -s . -p 'test_*.py'`: the parts that need no emulator |

## Running it

On Linux, with a Python venv holding qiling 1.4.6, unicorn 2.1.4 and python-registry. The root file system is `rootfs/x8664_windows/`: `Windows/System32/` holds the DLLs listed in `needed_dlls.txt`, `Windows/registry/` the six empty hives, and `plugins/` copies of the four `.aex` files. Keep the originals elsewhere, untouched.

```sh
H=path/to/parity/harness; PY=path/to/venv/bin/python; R=path/to/target/release

# 1. a test set (an ffmpeg that can run geq on float formats is needed for the 32 bpc inputs)
$R/fx-parity gen --out ts --vfx $R/vfx --effects retime,retime-fgbg,create-vectors,vectors-in --depths 8 \
    --no-sweep --sizes 64x48 --frames 4 --fps 24 --clip-sources testsrc2 --max-cases-per-effect 4

# 2. the param map: the plugin's registered params paired with the rewrite's ids (one dump per plugin)
$R/vfx list --json --all > vfx_list.json
$PY make_param_map.py vfx_list.json param_map.tsv results/<out>_params.json   # reads parity/<effect>-params/pairing.tsv when present

# 3. render under emulation: refs/<case id>.<pix_fmt>, one call record per case in records/.
#    -j N workers (about 1.5 GB each); --case ID (repeatable), --effect ID and --max-cases N select cases.
cd $H && $PY run_harness.py --testset ts --refs refs -j3 --records records --summary run_summary.json

# 4. render with the rewrite, with a call record of what it reads (one case; the arguments fx-parity uses)
VFX_CALL_RECORD=rec.jsonl $R/vfx apply retime --width 64 --height 48 --depth 8 --fps 24 --frames 4 \
    --in ts/inputs/<clip> --layer alt_source_a=ts/inputs/<layer clip> > out.raw

# 5. grade bit for bit
$R/fx-parity run --testset ts --refs refs --vfx $R/vfx --max-ulp 0 --json
```

`callrecord_check.py --vfx $R/vfx --testset ts` renders cases with and without `VFX_CALL_RECORD` and compares the SHA-256 of the output. The module grids have their own drivers and graders in `parity/<module>/`.

## Assumptions

- **Time base**: the rewrite's default. An integer rate F gets `time_scale = F * 100`, `time_step = 100`; a fraction N/D gets `time_scale = N`, `time_step = D`. `current_time = frame * time_step`, `total_time = frames * time_step`.
- **Footage frames.** A footage frame is the one whose span holds the tick asked for: `floor(t / time_scale * fps)`, in exact rational arithmetic.
- **Input worlds.** 8 bpc ARGB straight; 16 bpc ARGB in the range 0..32768, converted with the rewrite's own formulas (`(v * 32768 + 32767) / 65535` in, `(v * 65535 + 16384) / 32768` out, clamped at 32768); 32 bpc ARGB float straight. The wire formats are the README's (`rgba`, `rgba64le`, planar `gbrapf32le`).
- **`in_data`.** Version 13.28, `appl_id` `FXTC`, quality high (low for a draft case), field frame, downsample 1/1, pixel aspect 1/1, full-frame extent hint, no GPU (the GPU suites are answered "suite not found", `what_gpu` is none).
- **The render path** is decided by the plugin's own `out_flags2` from `GLOBAL_SETUP`. `SUPPORTS_SMART_RENDER` gives `SMART_PRE_RENDER` + `SMART_RENDER`, otherwise `FRAME_SETUP`, `RENDER`, `FRAME_SETDOWN`. All four plugins take the smart path.
- **Params.** A case's `--param id=value` is written to the registered param the plugin's map pairs with `id`. Popups keep their 1-based number, points are fixed point pixels, and colours are `A=255, r, g, b`.

## Stand-ins

Everything the host (After Effects or Windows) computes and the harness supplies:

| Stand-in | Where | Why |
|---|---|---|
| Keyframe interpolation: linear, hold, ease (the cubic shapes of the rewrite's timing doc); Bezier handles refused. Ties of key and fixed point rounding go half away from zero on both sides | `keys.py`, `ae_mock.py` | After Effects interpolates keyframes, not the plugin |
| `PF_IsRenderEngine` answers false (`*out = 0`), as an interactive After Effects does | `ae_mock.py` | With an error answer the plugin drew a fixed pattern over its output |
| A layer checkout in `SMART_PRE_RENDER` outside the clip's time range: `result_rect` empty, `max_result_rect` the layer's size (the header allows an empty `result_rect`); the pixels checked out in `SMART_RENDER` are transparent and full size | `ae_mock.py` | With a size-0 answer the plugin painted its missing-layer fill |
| An unset layer param whose registered default is `PF_LayerDefault_MYSELF` gets the case's input clip; a default of `PF_LayerDefault_NONE` stays unset | `worker.py` | After Effects hands the effect's own layer |
| FG matte layers are handed in as ordinary layer params with the case's clips | `ae_mock.py` | Faithful for mattes of the frame's size, unmasked and not time remapped; a host with masks or offsets would need a comp model |
| AEGP suite functions the plugin calls: Utility v11 (`RegisterWithAEGP` hands out ids from 1, `HostIsActivated` true), PF Interface v1 (`GetEffectLayer`: one opaque handle), Layer v5 (`GetLayerParentComp`: one opaque handle), Comp v4 (`GetCompFramerate`: the case's rate); every other member is a logged stub returning A_Err_NONE. The PF Handle Suite uses the same handle functions as the utils callbacks | `ae_mock.py`, `sdk_layouts_extra.json` | Twixtor asks for them while setting up a sequence |
| PF Path Query Suite 1: `PF_NumPaths` writes 0 (the cases have no masks) | `ae_mock.py` | With an error answer `TwixtorPro_64` returned PF_Err 4 |
| Empty registry hives (a root key, no keys, no values) | `make_hive.py` | Qiling needs the files to exist |
| Floating point control state: MXCSR 0x1F80 and x87 control word 0x27F (every exception masked, round to nearest, no DAZ or FTZ, double precision), written before the runtime libraries' `DllMain` and every plugin call, as the Microsoft x64 calling convention gives them; Unicorn's reset leaves both 0. Guest threads share the one MXCSR | `ae_host.py` | With 0, a divide by zero in the C runtime's math error path raised an exception nothing catches |
| The main guest thread's stack is 1 MiB (Windows' default reserve; Qiling maps 256 KB), and each guest thread's own stack bounds are written into the thread information block on every switch. 1 MiB is a stand-in: the host's real value is unknown. Worker stacks have no guard page | `ae_host.py`, `win_threads.py` | The plugins need a deep main stack, and their stack probes read the bounds of the running thread |
| Guest threads: `CreateThread`, `ResumeThread`, `ExitThread`, `GetCurrentThreadId` and waits, scheduled cooperatively (a thread runs until it blocks or ends; deterministic); a deadlock stops the run | `win_threads.py` | Twixtor starts worker threads that wait on semaphores |
| C++ exception dispatch from the documented x64 exception model: `RaiseException`, `RtlLookupFunctionEntry`, `RtlVirtualUnwind`, `RtlUnwindEx`, `RtlCaptureContext` and `RtlRestoreContext` are Python; the language handlers (`__CxxFrameHandler3` and the funclets it runs) are the C runtime's own code. A throw nobody handles stops the run | `win_seh.py` | C++ throws in the plugins' setup |
| `CryptAcquireContextA/W`, `CryptReleaseContext` and `CryptGenRandom` follow the documented behaviour for a process with no key store; `CryptGenRandom` gives a deterministic byte stream, not random bytes. No machine or user identity is used | `win_crypt.py` | The plugins' start-up calls them |
| `RtlPcToFileHeader` from the loader's image list; counted semaphores; `PathAppendW`; console `WriteFile` counted and dropped | `win_rtl.py`, `win_semaphore.py`, `win_path.py`, `win_io.py` | The C runtime and the plugins call them |
| One Python allocator for the heap functions and `malloc`/`calloc`/`realloc`/`free`/`_msize` | `ae_host.py` | Qiling's versions disagree about sizes |
| A Windows library export that Qiling has no Python version of returns 0; only `msvcr120`, `msvcp120` and `ucrtbase` run their own code. This is a fidelity risk | `ae_host.py` `hook_apis` | The Windows system DLLs cannot run natively here |
| `_initterm`, `_initterm_e` and the `printf` family run the C runtime's own code; the libraries' `DllMain` is called by the harness | `ae_host.py` | The static initialisers must run, and Qiling's `printf` reads variadic arguments wrongly |
| `MessageBoxW/A` log and return OK; window-station queries return 0; `ExitProcess`, `exit`, `abort` and `TerminateProcess` stop the run and are reported | `ae_host.py` | No desktop |
| The AE math callbacks point at the `ucrtbase.dll` exports of the same name inside the emulator | `ae_mock.py` | Real C runtime code |

**Speed.** Code hooks with a finite address range go to Unicorn directly instead of through Qiling's single dispatch callback, and Qiling's per-instruction thread scheduler hook is removed. On `Twixtor_64` that took `GLOBAL_SETUP` from 6 Windows calls a second to about 3,400 and the boot from 110 s to about 40 s.

**Limits.** Each library `DllMain` gets 180 s of emulation. A `PluginError` (a returned PF_Err) is the plugin's answer, so the worker goes on with the next case on the same boot; a fault ends the worker, and `run_harness.py` restarts it.

## Call record

`Record` writes JSON lines per case (`records/<case id>.jsonl`):
- `cmd`/`cmd_done`: command, time, `PF_Err`, flags, seconds.
- `cb` for every host callback and suite function the plugin calls: name, decoded arguments and return value. Checkouts also carry `time`, `time_step`, `time_scale`, the footage frame and the output frame that asked. A callback without a handler is marked `unimplemented` and returns `A_Err_GENERIC`.
- `frame`: output frame and time.
- `render_path`, `load_dll`, `api_args` (Windows loader and exception calls), `dllmain`, `messagebox`, `process_exit`, `timeout`.
- `fault`: access kind, address, pc, and the last API or callback.

Suite acquisitions are `cb` lines of `SPBasic.AcquireSuite` with the suite name and version.

```json
{"seq": 41, "ev": "cb", "fn": "inter.checkout_param", "args": ["0x300000010", 0, 200, 100, 2400, "0x300000a00"], "ret": 0, "param_index": 0, "time": 200, "time_step": 100, "time_scale": 2400, "footage_frame": 2, "out_frame": 2}
```

The rewrite's side is `VFX_CALL_RECORD=<file>` on `vfx apply`, with `run`, `frame` and `read` lines (`crates/fx-cli/src/record.rs`). The modules' `phase_b.py calls` compares the two.

## Param map

`param_map_<plugin>.tsv` has one row per parameter the plugin registers in PARAMS_SETUP: index, name, type, range, slider range, default, popup choices and flags, with the rewrite's id, kind, name, default and choices. Its `match` column is `exact` (same display name), `normalized` (equal after lower casing and dropping everything but letters and digits), from the pairing evidence of `parity/<effect>-params/pairing.tsv`, or `unmatched` (group markers, host-side controls, mask paths). The mock host reads it to set a case's `--param` values.

**Per case status values:**
- `rendered`;
- `error`, with the text as `GuestFault: ...` or `PluginError: <CMD> returned PF_Err N`;
- `unsupported`: a param, key or layer the harness cannot set, or a 16 or 32 bpc case for a plugin that did not declare it;
- `not_run`.

A case with `"quality": "draft"` sends `PF_Quality_LO` (0) in `PF_InData.quality`, anything else `PF_Quality_HI` (1).

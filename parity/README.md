# parity

The tooling that checks the rewrite against the original Twixtor plugins. What parity means, how it is checked and what the checks found are in [docs/parity.md](../docs/parity.md).

| Folder | Holds |
|---|---|
| `harness/` | The emulator harness: it loads the original plugins unmodified under Qiling with a mock After Effects host and renders test sets with them ([README](harness/README.md)) |
| `retime-time/`, `retime-flow/` | The two modules of `Twixtor_64` (effect `retime`): the time model, and the motion path |
| `create-vectors/` | The module of `CreateMotionVectors_64` |
| `vectors-in/` | The module of `TwixtorProVectorsIn_64` |
| `fgbg-core/`, `fgbg-layers/` | The two modules of `TwixtorPro_64` (effect `retime-fgbg`): without and with FG matte layers |
| `retime-params/`, `retime-fgbg-params/`, `create-vectors-params/`, `vectors-in-params/` | `pairing.tsv`: every parameter a plugin registers, paired with the rewrite's control and the evidence for the pair |

Each module folder holds:
- the grid generator and the test set definition (`cases.json`, without the raw frames);
- `phase_b.py`, the grader, with its tests;
- any supplementary case sets;
- `results/`, the module's result files.

`results/summary.json` is the module's result, from which `tools/gen_docs.py` builds the status tables. The module's README gives its commands.

The original plugins, the Windows runtime files, the After Effects SDK, the reference outputs and the raw frames are not in the repository.

# Rust port, milestone 1: native Linux — design

- Author: Paul Snow
- Version: 0.0.0
- Date: 2026-11-09
- Status: draft, awaiting review
- Branch: `rust-port`

## 1. Goal and scope

Rewrite gputaskman in idiomatic Rust, starting with native Linux. The Python implementation stays in
place, unchanged, as the reference.

**Milestone 1 delivers** a `gputaskman` binary for native Linux with:

- `--once` (text), `--json` (schema v2), `--gpu N`, `--list-gpus`, `-i/--interval` (accepted and
  validated; used by the TUI in a later milestone), and `gputaskman diff BEFORE AFTER [--json]`.
- Backends: NVML (NVIDIA, including the container case where NVML reports PIDs from another PID
  namespace) and amdgpu DRM (sysfs device readings, DRM fdinfo per-process memory and engine time).
- The full model layer: runtime detection, memory-mapped model files, command-line models,
  installed-model candidates, model-store names (Ollama, LM Studio, Hugging Face), GGUF headers,
  KV-cache and weight estimates, buffer-log parsing, local server `GET` probes, and the Handy adapter.

**Out of scope for milestone 1** (each a later milestone with its own spec): the TUI (`ratatui`),
WSL (host counter stream and context matching), Windows (WDDM/PDH/DXGI, ADL, AMD SMI), macOS
(IOKit/Metal), portable release builds, and PyPI/crates.io publishing.

**Success criteria**

1. `cargo test --workspace` passes on Ubuntu, Windows and macOS (core crate everywhere; Linux crate
   on Ubuntu only), with `cargo clippy --workspace -- -D warnings` clean.
2. Parity tests pass (section 8): backend results match on every shared fixture, and Python's
   `to_dict` output, loaded through the Rust v1 reader, equals the Rust v2 snapshot (ignoring `taken`).
3. A compile-fail test proves an inferred, estimated or reported byte count cannot be added to the
   measured attribution total.
4. Every provenance rule in `AGENTS.md` maps to a named test (section 8).
5. On a machine with no supported GPU (the development machine: Intel `i915` only), every command
   exits cleanly with a stated reason and `--json` still emits a valid v2 document.

## 2. Layout

```
rust/
├── Cargo.toml                  workspace
├── README.md                   build, run, validation status, live-check checklist
├── crates/
│   ├── gputaskman-core/        platform-neutral: model, provenance, collector, model layer, report, diff
│   ├── gputaskman-linux/       NvmlBackend, DrmBackend, LinuxProcs, PID-namespace matching
│   └── gputaskman/             clap binary; links the Linux crate under cfg(target_os = "linux")
└── tests/fixtures/             shared /sys and /proc trees, GGUF headers, logs, API responses, snapshots
```

`gputaskman-core` contains no OS-specific code and no `cfg(target_os)` gates, so it builds and
tests on every platform. Later backends (`gputaskman-wsl`, `gputaskman-windows`, `gputaskman-macos`)
and the TUI (`gputaskman-tui`) are added as crates implementing the same traits.

**Dependencies:** `serde`, `serde_json`, `clap` (derive), `nvml-wrapper` (dynamic loading of the
installed `libnvidia-ml.so`; never bundled), `ureq` (blocking
HTTP, `GET` only), `thiserror`, `libc` (Linux crate: `sysconf(_SC_CLK_TCK)`). `/proc` and `/sys` are
read with `std::fs` from an injectable root (the fallback allowed in section 10, chosen up front so
tests build trees the same way the Python tests do). Dev: `trybuild`, `tempfile`, `assert_cmd`. No
async runtime.

## 3. Core types and provenance

Provenance is encoded in types so that the rules in `AGENTS.md` are enforced by the compiler where
possible, rather than by convention.

```rust
/// A byte quantity and how it is known. Unknown is never zero.
pub enum Bytes {
    Measured(u64),
    Inferred { bytes: u64, by: Confidence },      // exact size, inferred owner, already counted
                                                    // under another row (the WSL VM): never summed
    Estimated { bytes: u64, from: EstimateSource }, // computed (GGUF maths)
    Reported { bytes: u64, by: Reporter },          // a runtime said so (API, load log)
    Unavailable(Reason),
}
pub enum Confidence { Consistent, Order }          // renders "*" and "?"
```

- `Snapshot::attributed() -> Option<u64>` sums only `Bytes::Measured`. It returns `None` when any
  row's VRAM is `Unavailable` with a reason whose `voids_attribution()` is true. That is every reason
  except `nvml_hidden`: Python counts an NVML row with hidden memory as 0 in the total (it is part of
  the unattributed remainder) but voids the total for DRM's unattributed rows, and the port keeps
  that behaviour.
- A row whose bytes are measured but whose owner is inferred (a container matching NVML's host PIDs
  to local PIDs) has `vram: Measured` plus `owner: Some(Confidence)`. It is summed, as in Python,
  because no other row holds those bytes. `Bytes::Inferred` is only for bytes another row already
  holds. The sum is built through a `MeasuredTotal` accumulator whose
  only `add` method takes a `Measured` newtype, so adding any other variant does not compile.
- `ProcGpu::shown_vram()` returns `Measured` or `Inferred` bytes for display, sorting and scan
  priority only. It returns a `Shown` newtype that has no conversion into `MeasuredTotal`.
- Engine utilisation is a separate `Engines` value (`Measured { by_engine, busiest }` or
  `Unavailable(Reason)`), independent of VRAM availability.
- DRM resident VRAM and GTT are a `Resident` reference value, explicitly not summable (no `Add`
  implementation, and no path into `MeasuredTotal`).
- Model evidence uses an ordered `Evidence` enum (`Api > Mapped > Cmdline > Installed`). Name-based
  hints (`kind_hint`) are a separate field that the KV and weight maths never reads.
- `Reason` is a closed enum of stable string codes. The DRM codes reuse Python's values: `legacy`,
  `shared_client`, `shared_buffers`, `primary`, `incomplete`, `partial_scan`. New codes include
  `nvml_not_loaded`, `nvml_hidden`, `process_unreadable`, `not_on_backend`, `outside_namespace`,
  `kfd_untimed`, `no_baseline`, `torn_counter`, `unreadable` and `v1_unknown`. `not_on_backend` means
  the backend never offers the field; `unreadable` means it does but the read failed.

Other core types mirror the Python data model: `Adapter` (keyed by PCI bus ID on Linux, never name
or index), `DeviceTelemetry` (every reading `Option`), `ProcRow`, `ProcDetail`, `MappedModel`,
`GgufInfo`, `LlmView`, `Segment`, `ServerInfo`, `HandyEvidence`, `Snapshot`.

## 4. Traits

```rust
pub trait GpuBackend: Send {
    fn kind(&self) -> BackendKind;                       // Nvml | Drm
    fn adapters(&self) -> Vec<Adapter>;
    fn sample(&mut self, budget: &Budget) -> BackendSample;
    fn device(&mut self, adapter: &Adapter) -> Result<DeviceTelemetry, Reason>;
}

pub trait ProcessSource: Send {                          // replaces psutil; /proc metadata only
    fn detail(&self, pid: Pid) -> Result<ProcBasics, Reason>;  // name, exe, cmdline, start time
    fn parents(&self, pid: Pid, depth: usize) -> Vec<Pid>;
    fn memory_maps(&self, pid: Pid) -> Result<Vec<MapEntry>, Reason>;
    fn listening_ports(&self) -> Map<Pid, Vec<u16>>;
}

pub trait HttpGet: Send {                                // the only network surface; GET only
    fn get(&self, url: &LocalUrl, timeout: Duration) -> Result<String, Reason>;
}

pub trait Clock: Send { fn now(&self) -> Instant; fn wall(&self) -> f64; }

pub enum Budget { Deadline(Instant), Unbounded }         // Unbounded for --once / --json
```

`BackendSample { procs: Map<Pid, ProcGpu>, outside: Map<Pid, ProcGpu>, adapter: AdapterGpu, status: Option<String> }`.

`LocalUrl` can only be built for `127.0.0.1` or `::1` with an anchored path (one that cannot change
the host). Restricting the port to the process's own listening sockets (or those of its parents,
for front-end processes) is enforced by `LocalUrl`'s callers, not by the type. Together they enforce
rule 6.
`ProcessSource` has no method that reads another process's memory.

## 5. Sampling flow

`Collector::new(gpu_index, budget_policy)` enumerates adapters, orders them (NVIDIA first, then by
VRAM descending, then key, matching Python's `ordered_adapters`), selects one, and builds only the
backend for that adapter. DRM is never started while an NVIDIA GPU is selected. A DRM backend takes
its engine baseline at construction.

`Collector::sample()`:

1. **Backend sample:** `backend.sample(&budget)`.
2. **Device telemetry:** `backend.device(&adapter)`. On failure, the device is `None` and the reason
   goes into the snapshot status. `vram_total` comes from the telemetry, else from the adapter, else
   it is `Unavailable`.
3. **Rows:** keep live processes (Python's `_is_live` thresholds: not attributed, or dedicated
   ≥ 1 MiB, or GPU ≥ 0.5 %, or shared ≥ 16 MiB), sorted by `shown_vram` descending. Outside-namespace
   PIDs become rows with `Origin::Outside` and kind `external`.
4. **Deep scans:** the top 25 local rows plus any row at 2 % GPU or more are scanned (memory maps,
   runtimes, command-line models, installed candidates) while the budget lasts. Results are cached
   per `(pid, start_time)` so a reused PID never inherits stale data. Rows are re-classified after
   scanning.
5. **Model layer:** for AI-kind rows with model evidence (or kind `LLM`) and measured or inferred
   memory: probe exclusive ports, then parent ports; detect a single Ollama runner and pass its log
   path; then `build_view` (GGUF weights, KV per token, buffer-log segments, notes). On an APU
   (`shared_is_spill == false`), GTT counts as held memory. Handy executables get `inspect_handy`.
6. **Prune** per-PID caches for processes that have gone.

GGUF headers and store manifests are cached by `(path, size, mtime)`. `/proc` reads are bounded
the same way Python's are; there is no equivalent of `psutil.open_files()` (rule 7). The DRM fd walk
honours the same budget and reports `partial_scan` when it is cut short.

## 6. JSON schema v2

Top level:

```json
{
  "schema_version": 2,
  "taken": 1762646400.0,
  "backend": "nvml",
  "adapter": {"name": "…", "vendor": "NVIDIA", "vendor_id": 4318, "key": "0000:01:00.0",
              "dedicated_vram": 17171480576, "shared_system": 0, "integrated": false},
  "driver": "570.86.16", "cuda": "12.8",
  "memory": {
    "total":       {"status": "measured", "bytes": 17171480576},
    "used":        {"status": "measured", "bytes": 4294967296},
    "shared_used": {"status": "unavailable", "reason": "not_on_backend"},
    "attributed":  {"status": "measured", "bytes": 3221225472},
    "shared_is_spill": true
  },
  "status": {"process": null, "device": null, "counter": null, "host": null, "wsl_split": null},
  "device": { "…DeviceTelemetry fields, each null when unread…": null },
  "notes": ["NVML not loaded: …"],
  "processes": [ … ]
}
```

`status.device` is the reason code when device telemetry could not be read, else `null`.
`notes` lists why parts of the snapshot are missing (for example a vendor library that failed to
load); it is empty when nothing degraded.
`memory.attributed` is `unavailable` with a reason whenever any row's VRAM is unavailable,
matching Python's `Snapshot.attributed` returning `None`.

A process (an NVML row, so `resident` is `null`):

```json
{
  "pid": 4242, "origin": "local", "name": "llama-server", "kind": "LLM", "runtimes": ["CUDA"],
  "vram":      {"status": "measured", "bytes": 3221225472}, "owner_inferred_by": null,
  "shared":    {"status": "unavailable", "reason": "not_on_backend"},
  "committed": {"status": "unavailable", "reason": "not_on_backend"},
  "paged_out": {"status": "unavailable", "reason": "not_on_backend"},
  "resident":  null,
  "engines":   {"status": "measured", "busiest": 37.5, "by_engine": {"3D": 37.5}},
  "ports": [8080], "inside": "Qwen3 14B Q4_K_M",
  "model_files":      [{"path": "…", "size": 0, "resident": 0, "gguf": true, "evidence": ["api", "mapped"],
                        "store_name": null, "kind": "LLM", "kind_hint": ""}],
  "installed_models": [],
  "handy": null,
  "llm": {
    "models": [{"path": "…", "name": "…", "label": "…", "arch": "qwen3", "kind": "LLM", "kind_label": "LLM",
                "kind_hint": "", "decision_type": null, "params": 14000000000, "quant": "Q4_K_M",
                "layers": 40, "ctx_train": 40960,
                "gpu_weight": {"status": "estimated", "bytes": 0, "from": "gguf"},
                "evidence": ["api", "mapped"], "store_name": null}],
    "segments": [{"label": "weights", "kind": "weights",
                  "bytes": {"status": "reported", "bytes": 0, "by": "load_log"}}],
    "kv_per_token": {"status": "estimated", "bytes": 0, "from": "gguf"},
    "n_ctx": 8192, "flags": {}, "server": null, "log_source": null, "notes": []
  }
}
```

Rules:

- `origin` is `local` or `outside` (later milestones add `windows`).
- `resident` is `null` on rows whose backend has no resident counters (NVML). On DRM `local` rows
  it is present when fdinfo gave it, for example
  `{"vram": {"status": "measured", "bytes": 0}, "gtt": {"status": "measured", "bytes": 0}, "summable": false}`.
  On `outside` rows both values are `unavailable` with `outside_namespace`, since their fds cannot
  be read.
- Fields that are unknown on this backend (`committed` and `paged_out` on Linux) are present and
  `unavailable` with `not_on_backend`, never omitted and never zero.
- A segment is `reported` when Python marks it `exact` (it came from a runtime's buffer log), and
  `estimated` otherwise.
- `evidence` lists sources in ladder order (`api`, `mapped`, `cmdline`, `installed`).
- The text report (`--once`) keeps Python's wording and markers (`*`, `?`, `n/a`) and its notes
  (`DRM_NOTES`, `DRM_RESIDENT_NOTE`, `INFERRED_NOTE`, `ORDER_NOTE`).

### v1 reader

`core::schema::v1::load` turns a Python `--json` document (no `schema_version`) into the v2 model:

| v1 field(s) | v2 |
|---|---|
| `vram_bytes` non-null | `vram: measured`; `owner_inferred_by` from `vram_inferred_by` |
| else `vram_inferred_bytes` non-null | `vram: inferred`, `by` from `vram_inferred_by` |
| else | `vram: unavailable`, reason from `vram_unattributed_reason`, else `v1_unknown` |
| `shared_bytes`, `committed_bytes`, `paged_out_bytes` | `measured` if non-null, else `unavailable(v1_unknown)` |
| `vram_resident_bytes`, `gtt_resident_bytes` | `resident.vram`, `resident.gtt` |
| `gpu_pct` + `engines` | `engines: measured` if `gpu_pct` non-null, else `unavailable` |
| `host` (`"outside"`/`"windows"`/null) | `origin` |
| `model_files[].sources` | `evidence` |
| `attributed`, `vram_total`, `vram_used`, `shared_used` | `memory.*` with the same null rule |
| `device`, `process_status`, `counter_status`, `host_status`, `wsl_split` | `device`, `status.*` |

`gputaskman diff` accepts v1 and v2 in any combination. It compares measured values directly,
keeps reported and estimated values in their own sections, and reports an absent row as
`not listed`, not as zero (the same as Python's `diff.py`).

## 7. Degradation (rule 9)

No fallible path panics; `unwrap` and `expect` are banned outside tests by a clippy lint
(`clippy::unwrap_used`, `clippy::expect_used`).

| Condition | Behaviour |
|---|---|
| `libnvidia-ml.so` absent or fails to initialise | NVIDIA adapters are not listed from NVML; device is `None`; status `nvml_not_loaded` |
| No supported GPU | `--list-gpus` prints nothing and exits 0; `--once`/`--json` emit a snapshot with no adapter and a status saying why |
| A process's `/proc` entries unreadable | Row fields `unavailable(process_unreadable)`; the row is kept |
| DRM fd walk cut short by the budget | Affected rows `unavailable(partial_scan)`; `status.process` explains |
| HTTP probe fails or times out (≤ 0.5 s) | No server info; evidence falls back down the ladder |
| `--gpu N` out of range | Argument error before collection, the same message as Python |

## 8. Testing

TDD throughout: each Python test case is ported first as a failing Rust test, then made to pass.

1. **Unit tests per module**, ported from `tests/test_drm.py`, `tests/test_linux.py`,
   `tests/test_core.py` (CLI, device selection, memory, Handy, model tests) and `tests/test_diff.py`.
2. **Fixtures** in `rust/tests/fixtures/`: the `/sys/class/drm` and `/proc` trees Python's DRM
   tests build (regenerated by a small script from the Python test helpers, so both suites use the
   same data), GGUF header samples, llama.cpp and Ollama load logs, server API responses, and
   saved v1 snapshots.
3. **Parity tests** (`cargo test --features parity`, needs `uv`), at two levels. *Backend parity:*
   scenario files in `rust/tests/fixtures/parity/` describe DRM cards, process fds and fdinfo, and
   fake NVML data; a Python harness builds the tree, runs `DrmCounters`/`LinuxCounters` with the
   Python test fakes, and prints per-process results, which must equal the Rust backends' results on
   the same tree. *Report parity:* the same scenario snapshots go through Python's `to_dict`, then the
   Rust v1 reader, and must equal the Rust v2 snapshot, ignoring `taken`. The whole Python collector
   is not run, because it reads the live `/proc` through psutil; the collector's logic is covered by
   the ported unit tests.
4. **Compile-fail tests** (`trybuild`): adding an `Inferred`, `Estimated`, `Reported` or `Shown`
   value to `MeasuredTotal`; adding two `Resident` values; building a `LocalUrl` for a non-loopback
   host.
5. **Rule-to-test map**, kept in `rust/README.md`: each `AGENTS.md` rule (1–10) names the test or
   tests that cover it.
6. **CI:** a new `rust` job in `.github/workflows/ci.yml`: `cargo test --workspace` and
   `cargo clippy --workspace -- -D warnings` on Ubuntu, Windows and macOS (the Linux crate builds on
   Ubuntu only), plus the parity tests on Ubuntu. The existing Python jobs are unchanged.

## 9. Hardware validation (rule 10)

The development machine has only an Intel `i915` GPU, which gputaskman does not support, so no
backend can be validated live during milestone 1. `rust/README.md` labels every backend
**not validated on hardware** and gives a live-check checklist for an NVIDIA or amdgpu machine:

1. Run a bounded, known workload (a llama-server with a known model, or a fixed-size CUDA or HIP
   allocation).
2. Capture `gputaskman --json` from both Python and Rust.
3. Compare with `nvidia-smi` (NVIDIA) or Python's DRM output (amdgpu), and run
   `gputaskman diff python.json rust.json`; measured buckets must agree.
4. Record hardware, driver, kernel and results before removing the "not validated" label.

## 10. Risks

- **Hand-written `/proc` parsing:** `std::fs` is used instead of the `procfs` crate, so the parsers
  for `stat`, `smaps`, `net/tcp{,6}` and fd links are our own. Each has fixture tests taken from real
  kernel output.
- **`nvml-wrapper` mocking:** NVML sits behind a small internal `NvmlApi` trait so tests never load
  the real library.
- **Parity harness drift:** the Python harness imports the Python modules directly; if their
  internals change, the harness changes with them in the same commit.
- **Schema v2 is a new public format:** v2 is marked unstable in `rust/README.md` until a later
  milestone settles it.

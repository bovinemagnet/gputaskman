# Rust port, milestone 1 (native Linux) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone Rust `gputaskman` binary for native Linux (NVML and amdgpu DRM), with the full model-identification layer, `--once`/`--json` (schema v2) and `diff`, whose numbers obey the same provenance rules as the Python version.

**Architecture:** A Cargo workspace in `rust/` with three crates. `gputaskman-core` is platform-neutral: provenance types, data model, traits, the model layer, the collector, the reports and `diff`. `gputaskman-linux` implements the core traits for NVML, DRM and `/proc`. `gputaskman` is the clap binary. Provenance is encoded in types (`Bytes`, `MeasuredTotal`, `Shown`, `Resident`) so that summing anything other than measured bytes does not compile.

**Tech Stack:** Rust 1.97 (edition 2024), serde/serde_json, clap 4 (derive), nvml-wrapper, ureq, regex, glob, chrono, thiserror, libc. Dev: trybuild, tempfile, assert_cmd.

**Spec:** `docs/superpowers/specs/2026-11-09-rust-linux-port-design.md`. Read it before starting any task.

## Global Constraints

- Workspace root `rust/`. Every `Cargo.toml` uses `version = "0.0.0"`, `edition = "2024"`, `authors = ["Paul Snow"]`, `license = "MIT"`.
- The Python sources in `src/gputaskman/` are the reference implementation. When a step says "port", reproduce the behaviour exactly: the same constants, thresholds, ordering, regexes and user-facing wording. Do not change any Python file.
- Unknown is never zero: an unreadable value is `Bytes::Unavailable { reason }` or `None`, never `0`. A measured zero stays `Measured { bytes: 0 }`.
- Only `Bytes::Measured` values may reach `MeasuredTotal`. `Shown`, `Resident`, `Inferred`, `Estimated` and `Reported` must never be summed into the attribution total.
- Never read another process's memory. `ProcessSource` exposes only `/proc` metadata. Network access goes only through `HttpGet` with a `LocalUrl` (loopback only, `GET` only, timeout ≤ 0.5 s, body ≤ 4 MiB).
- Never call anything equivalent to `psutil.open_files()`. Deep scans honour the scan budget.
- Never bundle a vendor library. NVML is loaded at runtime from the installed driver by `nvml-wrapper`.
- No `unwrap`/`expect` outside tests: the workspace denies `clippy::unwrap_used` and `clippy::expect_used`, and `clippy.toml` allows them in tests. Every integration-test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]`.
- British spelling in comments and docs. Never mention Claude, Anthropic or AI assistance in code, comments, docs or commits.
- Commit messages follow the repo's style: a short imperative sentence (for example `Add DRM fdinfo parsing`).
- Run every command from `rust/` unless stated. Verify with `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings`.
- Add dependencies with `cargo add` (for example `cargo add -p gputaskman-core serde --features derive`) so `Cargo.lock` pins current versions. Commit `Cargo.lock`.
- Hardware: the development machine has only an Intel `i915` GPU, so every backend stays labelled **not validated on hardware**.

## File structure

```
rust/
├── Cargo.toml                         workspace, shared lints
├── Cargo.lock
├── clippy.toml                        allow-unwrap-in-tests / allow-expect-in-tests
├── .gitignore                         /target
├── README.md                          build, run, rule-to-test map, validation checklist (Task 19)
├── parity/
│   ├── run_python.py                  Python harness for parity scenarios (Task 19)
│   └── scenarios/*.json               shared scenario descriptions (Task 19)
└── crates/
    ├── gputaskman-core/
    │   ├── Cargo.toml                 feature `test-support` exposes fakes
    │   ├── src/lib.rs
    │   ├── src/provenance.rs          Bytes, Confidence, Reason, Measured, MeasuredTotal, Shown, Resident (Task 1)
    │   ├── src/model.rs               Adapter, DeviceTelemetry, AdapterGpu, Engines, ProcGpu, Evidence, MappedModel, ProcDetail (Task 2)
    │   ├── src/env.rs                 Paths (home, XDG dirs) (Task 3)
    │   ├── src/traits.rs              Clock, ScanPolicy, Deadline, HttpGet, LocalUrl, ProcessSource, GpuBackend (Task 3)
    │   ├── src/http.rs                UreqGet (Task 3)
    │   ├── src/testing.rs             FakeClock, FakeHttp, FakeProcs, FakeBackend (Task 3, cfg feature test-support)
    │   ├── src/gguf.rs                GGUF header reader, estimates, cache (Task 4)
    │   ├── src/stores.rs              Ollama / HF / LM Studio names (Task 5)
    │   ├── src/procinfo/mod.rs        re-exports
    │   ├── src/procinfo/runtimes.rs   runtime detection, lib names (Task 6)
    │   ├── src/procinfo/models.rs     model-file rules, cmdline models, make_model, name hints (Task 6)
    │   ├── src/procinfo/classify.rs   classify(), kind constants (Task 6)
    │   ├── src/procinfo/installed.rs  installed-model walk + cache (Task 7)
    │   ├── src/procinfo/inspector.rs  ProcInspector (Task 7)
    │   ├── src/llm/mod.rs             re-exports, gb()
    │   ├── src/llm/flags.rs           parse_flags (Task 8)
    │   ├── src/llm/logs.rs            parse_load_log, is_gpu_device, ollama_log_path (Task 8)
    │   ├── src/llm/servers.rs         ServerInfo, normalisers, probe_server, ServerProber (Task 8)
    │   ├── src/llm/view.rs            Segment, LlmView, merge_api, build_view (Task 9)
    │   ├── src/handy.rs               Handy adapter (Task 10)
    │   ├── src/collector.rs           ProcRow, Snapshot, Collector (Task 15)
    │   ├── src/schema/mod.rs          SCHEMA_VERSION
    │   ├── src/schema/v2.rs           v2 serialisation (Task 16)
    │   ├── src/schema/v1.rs           v1 reader (Task 17)
    │   ├── src/report.rs              text report (Task 16)
    │   ├── src/diff.rs                snapshot comparison (Task 17)
    │   └── tests/                     compile_fail.rs + ui/*.rs, per-module integration tests
    ├── gputaskman-linux/
    │   ├── Cargo.toml                 feature `test-support` exposes FakeNvml
    │   ├── src/lib.rs                 linux_backends()
    │   ├── src/sysfs.rs               read_trimmed, read_int helpers (Task 11)
    │   ├── src/drm/mod.rs
    │   ├── src/drm/fdinfo.rs          parse_size, parse_fdinfo (Task 11)
    │   ├── src/drm/cards.rs           DrmCard, cards(), adapters(), card names (Task 11)
    │   ├── src/drm/device.rs          read_device, hwmon, vcn_major, engine_key (Task 11)
    │   ├── src/drm/counters.rs        DrmCounters (Task 12)
    │   ├── src/drm/backend.rs         DrmBackend: GpuBackend (Task 12)
    │   ├── src/nvml/mod.rs
    │   ├── src/nvml/api.rs            NvmlApi trait, NvProcess, NvUtilSample, FakeNvml (Task 13)
    │   ├── src/nvml/real.rs           RealNvml over nvml-wrapper (Task 13)
    │   ├── src/nvml/backend.rs        NvmlBackend: GpuBackend (Task 13)
    │   ├── src/pidns.rs               PidMatcher, namespace detection, NVIDIA holders (Task 13)
    │   ├── src/procs.rs               LinuxProcs: ProcessSource (Task 14)
    │   └── tests/                     support/mod.rs (fixture builders) + per-module tests
    └── gputaskman/
        ├── Cargo.toml
        ├── src/main.rs                clap CLI (Task 18)
        └── tests/cli.rs               assert_cmd tests (Task 18)
```

---

### Task 1: Workspace scaffold and provenance types

**Files:**
- Create: `rust/Cargo.toml`, `rust/clippy.toml`, `rust/.gitignore`
- Create: `rust/crates/gputaskman-core/Cargo.toml`, `src/lib.rs`, `src/provenance.rs`
- Create: `rust/crates/gputaskman-linux/Cargo.toml`, `src/lib.rs` (empty crate doc comment only)
- Create: `rust/crates/gputaskman/Cargo.toml`, `src/main.rs` (`fn main() {}`)
- Test: `rust/crates/gputaskman-core/tests/provenance.rs`, `tests/compile_fail.rs`, `tests/ui/*.rs`

**Interfaces:**
- Produces (in `gputaskman_core::provenance`):
  - `enum Bytes { Measured { bytes: u64 }, Inferred { bytes: u64, by: Confidence }, Estimated { bytes: u64, from: EstimateSource }, Reported { bytes: u64, by: Reporter }, Unavailable { reason: Reason } }`, serde internally tagged `status`, snake_case.
  - `Bytes::measured(u64)`, `Bytes::unavailable(Reason)`, `Bytes::as_measured(&self) -> Option<Measured>`, `Bytes::value(&self) -> Option<u64>` (any known bytes, for display only), `Bytes::is_measured(&self) -> bool`.
  - `enum Confidence { Consistent, Order }` with `marker() -> &'static str` (`"*"`, `"?"`).
  - `enum EstimateSource { Gguf }`, `enum Reporter { LoadLog, LlamaServerApi, OllamaApi, LmStudioApi }`.
  - `enum Reason { Legacy, SharedClient, SharedBuffers, Primary, Incomplete, PartialScan, NvmlNotLoaded, NvmlHidden, ProcessUnreadable, NotOnBackend, OutsideNamespace, KfdUntimed, NoBaseline, TornCounter, V1Unknown }` with `code() -> &'static str` and `voids_attribution() -> bool` (false only for `NvmlHidden`).
  - `struct Measured(u64)` (private field) with `get(self) -> u64`.
  - `struct MeasuredTotal` with `new()`, `add(&mut self, Measured)`, `total(&self) -> u64`.
  - `struct Shown(u64)` (private field) with `pub(crate) fn new(u64)` and `get(self) -> u64`.
  - `struct Resident { pub vram: Bytes, pub gtt: Bytes }` with no `Add` impl.

- [ ] **Step 1: Create the workspace files**

`rust/Cargo.toml`:
```toml
[workspace]
resolver = "3"
members = ["crates/gputaskman-core", "crates/gputaskman-linux", "crates/gputaskman"]

[workspace.package]
version = "0.0.0"
edition = "2024"
authors = ["Paul Snow"]
license = "MIT"
rust-version = "1.97"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
```

`rust/clippy.toml`:
```toml
allow-unwrap-in-tests = true
allow-expect-in-tests = true
```

`rust/.gitignore`:
```
/target
```

`rust/crates/gputaskman-core/Cargo.toml`:
```toml
[package]
name = "gputaskman-core"
description = "Platform-neutral core of gputaskman: provenance, model layer, reports"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
rust-version.workspace = true

[features]
test-support = []

[lints]
workspace = true
```
Create `gputaskman-linux` and `gputaskman` the same way, with the names `gputaskman-linux` and `gputaskman` and the descriptions "Linux backends for gputaskman (NVML, amdgpu DRM, /proc)" and "An htop for what's inside your GPU". `gputaskman-linux` depends on `gputaskman-core = { path = "../gputaskman-core" }`. `gputaskman` depends on both, with the Linux crate under `[target.'cfg(target_os = "linux")'.dependencies]`.

Then run: `cargo add -p gputaskman-core serde --features derive && cargo add -p gputaskman-core serde_json thiserror && cargo add -p gputaskman-core --dev trybuild`

- [ ] **Step 2: Write the failing tests**

`rust/crates/gputaskman-core/tests/provenance.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::*;

#[test]
fn only_measured_bytes_reach_the_total() {
    let rows = [
        Bytes::measured(3),
        Bytes::Inferred { bytes: 100, by: Confidence::Consistent },
        Bytes::Estimated { bytes: 100, from: EstimateSource::Gguf },
        Bytes::Reported { bytes: 100, by: Reporter::LoadLog },
        Bytes::measured(0),
        Bytes::measured(4),
    ];
    let mut total = MeasuredTotal::new();
    for m in rows.iter().filter_map(Bytes::as_measured) {
        total.add(m);
    }
    assert_eq!(total.total(), 7);
}

#[test]
fn measured_zero_is_not_unknown() {
    let zero = Bytes::measured(0);
    assert_eq!(zero.value(), Some(0));
    assert!(zero.is_measured());
    let unknown = Bytes::unavailable(Reason::NvmlHidden);
    assert_eq!(unknown.value(), None);
    assert!(unknown.as_measured().is_none());
}

#[test]
fn serialises_with_status_tag() {
    let cases = [
        (Bytes::measured(5), r#"{"status":"measured","bytes":5}"#),
        (Bytes::Inferred { bytes: 5, by: Confidence::Order }, r#"{"status":"inferred","bytes":5,"by":"order"}"#),
        (Bytes::Estimated { bytes: 5, from: EstimateSource::Gguf }, r#"{"status":"estimated","bytes":5,"from":"gguf"}"#),
        (Bytes::Reported { bytes: 5, by: Reporter::LoadLog }, r#"{"status":"reported","bytes":5,"by":"load_log"}"#),
        (Bytes::unavailable(Reason::SharedClient), r#"{"status":"unavailable","reason":"shared_client"}"#),
    ];
    for (value, json) in cases {
        assert_eq!(serde_json::to_string(&value).unwrap(), json);
        assert_eq!(serde_json::from_str::<Bytes>(json).unwrap(), value);
    }
}

#[test]
fn reason_codes_match_python() {
    let codes = [
        (Reason::Legacy, "legacy"), (Reason::SharedClient, "shared_client"),
        (Reason::SharedBuffers, "shared_buffers"), (Reason::Primary, "primary"),
        (Reason::Incomplete, "incomplete"), (Reason::PartialScan, "partial_scan"),
        (Reason::NvmlHidden, "nvml_hidden"), (Reason::V1Unknown, "v1_unknown"),
    ];
    for (reason, code) in codes {
        assert_eq!(reason.code(), code);
    }
    assert!(!Reason::NvmlHidden.voids_attribution());
    assert!(Reason::SharedBuffers.voids_attribution());
}

#[test]
fn markers() {
    assert_eq!(Confidence::Consistent.marker(), "*");
    assert_eq!(Confidence::Order.marker(), "?");
}
```

`rust/crates/gputaskman-core/tests/compile_fail.rs`:
```rust
#[test]
fn provenance_misuse_does_not_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
```

`tests/ui/sum_shown.rs`:
```rust
use gputaskman_core::provenance::{MeasuredTotal, Shown};
fn f(s: Shown) { let mut t = MeasuredTotal::new(); t.add(s); }
fn main() {}
```
`tests/ui/sum_bytes.rs`:
```rust
use gputaskman_core::provenance::{Bytes, MeasuredTotal};
fn f(b: Bytes) { let mut t = MeasuredTotal::new(); t.add(b); }
fn main() {}
```
`tests/ui/forge_measured.rs`:
```rust
use gputaskman_core::provenance::Measured;
fn main() { let _m = Measured(5); }
```
`tests/ui/add_resident.rs`:
```rust
use gputaskman_core::provenance::Resident;
fn f(a: Resident, b: Resident) { let _ = a + b; }
fn main() {}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core`
Expected: compile errors, because the `provenance` module does not exist.

- [ ] **Step 4: Implement `provenance.rs`**

`rust/crates/gputaskman-core/src/lib.rs`:
```rust
//! Platform-neutral core of gputaskman.
pub mod provenance;
```

`rust/crates/gputaskman-core/src/provenance.rs`:
```rust
//! How every number is known. Only measured bytes are ever summed (AGENTS.md rule 1), and unknown is
//! never zero (rule 2). These types make the first rule a compile-time check.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Only one explanation fits, or every explanation gives the same bytes.
    Consistent,
    /// A heuristic owner: several fit, paired by order.
    Order,
}

impl Confidence {
    pub fn marker(self) -> &'static str {
        match self {
            Confidence::Consistent => "*",
            Confidence::Order => "?",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateSource {
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reporter {
    LoadLog,
    LlamaServerApi,
    OllamaApi,
    LmStudioApi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    Legacy,
    SharedClient,
    SharedBuffers,
    Primary,
    Incomplete,
    PartialScan,
    NvmlNotLoaded,
    NvmlHidden,
    ProcessUnreadable,
    NotOnBackend,
    OutsideNamespace,
    KfdUntimed,
    NoBaseline,
    TornCounter,
    V1Unknown,
}

impl Reason {
    pub fn code(self) -> &'static str {
        match self {
            Reason::Legacy => "legacy",
            Reason::SharedClient => "shared_client",
            Reason::SharedBuffers => "shared_buffers",
            Reason::Primary => "primary",
            Reason::Incomplete => "incomplete",
            Reason::PartialScan => "partial_scan",
            Reason::NvmlNotLoaded => "nvml_not_loaded",
            Reason::NvmlHidden => "nvml_hidden",
            Reason::ProcessUnreadable => "process_unreadable",
            Reason::NotOnBackend => "not_on_backend",
            Reason::OutsideNamespace => "outside_namespace",
            Reason::KfdUntimed => "kfd_untimed",
            Reason::NoBaseline => "no_baseline",
            Reason::TornCounter => "torn_counter",
            Reason::V1Unknown => "v1_unknown",
        }
    }

    /// Python counts an NVML row with hidden memory as 0 in the attribution total (it is part of the
    /// unattributed remainder) but voids the total when a counter is missing. Keep that behaviour.
    pub fn voids_attribution(self) -> bool {
        !matches!(self, Reason::NvmlHidden)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Bytes {
    Measured { bytes: u64 },
    /// Exact size, inferred owner, already counted under another row: never summed.
    Inferred { bytes: u64, by: Confidence },
    Estimated { bytes: u64, from: EstimateSource },
    Reported { bytes: u64, by: Reporter },
    Unavailable { reason: Reason },
}

impl Bytes {
    pub fn measured(bytes: u64) -> Self {
        Bytes::Measured { bytes }
    }
    pub fn unavailable(reason: Reason) -> Self {
        Bytes::Unavailable { reason }
    }
    pub fn as_measured(&self) -> Option<Measured> {
        match *self {
            Bytes::Measured { bytes } => Some(Measured(bytes)),
            _ => None,
        }
    }
    pub fn is_measured(&self) -> bool {
        matches!(self, Bytes::Measured { .. })
    }
    /// Any known byte count, for display only. Never sum this.
    pub fn value(&self) -> Option<u64> {
        match *self {
            Bytes::Measured { bytes }
            | Bytes::Inferred { bytes, .. }
            | Bytes::Estimated { bytes, .. }
            | Bytes::Reported { bytes, .. } => Some(bytes),
            Bytes::Unavailable { .. } => None,
        }
    }
    pub fn reason(&self) -> Option<Reason> {
        match *self {
            Bytes::Unavailable { reason } => Some(reason),
            _ => None,
        }
    }
}

/// A measured byte count. Only `Bytes::as_measured` makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measured(u64);

impl Measured {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Default)]
pub struct MeasuredTotal(u64);

impl MeasuredTotal {
    pub fn new() -> Self {
        Self(0)
    }
    pub fn add(&mut self, m: Measured) {
        self.0 = self.0.saturating_add(m.0);
    }
    pub fn total(&self) -> u64 {
        self.0
    }
}

/// Measured or inferred bytes for display, sorting and scan priority. Has no path into `MeasuredTotal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shown(u64);

impl Shown {
    pub(crate) fn new(bytes: u64) -> Self {
        Self(bytes)
    }
    pub fn get(self) -> u64 {
        self.0
    }
}

/// amdgpu: resident bytes in every buffer a process can reach, shared ones included. A reference only:
/// the same buffer can appear under several processes, so these are never added up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resident {
    pub vram: Bytes,
    pub gtt: Bytes,
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core`
Expected: PASS for all five `provenance.rs` tests and `provenance_misuse_does_not_compile`. On the first run trybuild writes `tests/ui/*.stderr` under `wip/`. Inspect each one: they must be type-mismatch (`sum_shown`, `sum_bytes`), private-field (`forge_measured`) and "cannot add" (`add_resident`) errors. Move them into `tests/ui/` and re-run until it passes.

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings.
```bash
git add rust/
git commit -m "Add Rust workspace and provenance types"
```

---

### Task 2: Core data model

**Files:**
- Create: `rust/crates/gputaskman-core/src/model.rs`
- Modify: `rust/crates/gputaskman-core/src/lib.rs` (add `pub mod model;`)
- Test: `rust/crates/gputaskman-core/tests/model.rs`

**Interfaces:**
- Consumes: `provenance::*` (Task 1).
- Produces (in `gputaskman_core::model`):
  - `type Pid = u32;`
  - `const VENDOR_NVIDIA: u16 = 0x10de; const VENDOR_AMD: u16 = 0x1002; const MIB: u64 = 1 << 20; const GIB: u64 = 1 << 30;`
  - `enum BackendKind { Nvml, Drm }` (serde snake_case).
  - `struct Adapter { key: u64, pci_bus: String, name: String, vendor_id: u16, dedicated_vram: u64, shared_system: u64, integrated: bool, backend: BackendKind }` with `vendor(&self) -> String` (`"NVIDIA"`, `"AMD"`, `"Intel"` for 0x8086, else `format!("{:#x}")`, matching `dxgi.VENDORS`).
  - `fn pci_key(bus_id: &str) -> u64` (port of `linux.py` `pci_key`, line 40).
  - `fn ordered_adapters(adapters: Vec<Adapter>) -> Vec<Adapter>` (NVIDIA first, then `dedicated_vram` descending, then `key`).
  - `enum SelectError { OutOfRange { index: usize, available: usize } }` implementing `Display` as `--gpu {index} is out of range; {available} GPU adapter(s) available`.
  - `fn select_adapter(adapters: &[Adapter], index: usize) -> Result<Option<Adapter>, SelectError>` (port of `collector.select_adapter`: no adapters and index 0 gives `Ok(None)`).
  - `struct DeviceTelemetry` with every field from `telemetry.py`, numeric ones as `Option`, `throttle: Vec<String>`, `compute_cap: String`, `source: String`, `power_kind: String`.
  - `struct AdapterGpu { vram_used: Bytes, shared_used: Bytes }`, whose `Default` is both `Unavailable(NotOnBackend)`.
  - `enum Engines { Measured { by_engine: BTreeMap<String, f64> }, Unavailable { reason: Reason } }` with `busiest(&self) -> Option<f64>` (max, or `Some(0.0)` when the map is empty) and `get(&self, name) -> f64`.
  - `struct ProcGpu { vram: Bytes, shared: Bytes, committed: Bytes, engines: Engines, owner: Option<Confidence>, resident: Option<Resident>, kfd: bool }` with `shown_vram(&self) -> Option<Shown>`, `paged_out(&self) -> Bytes`, `is_live(&self) -> bool`, and `ProcGpu::measured(vram: u64) -> Self` (shared and committed `NotOnBackend`, engines measured and empty).
  - `fn attributed_total<'a>(rows: impl IntoIterator<Item = &'a ProcGpu>) -> Bytes`.
  - `enum Evidence { Api, Mapped, Cmdline, Installed }` (derives `Ord` in that order; serde snake_case).
  - `struct MappedModel { path: String, size: u64, resident: u64, gguf: bool, sources: Vec<Evidence>, store_name: Option<String>, kind: String, kind_hint: String }` with `add_source`, `kind_label() -> String`, `evidence_label() -> String`.
  - `struct ProcDetail { pid, name, exe, cmdline: Vec<String>, start_time: f64, start_ticks: Option<u64>, runtimes: Vec<String>, models: Vec<MappedModel>, candidates: Vec<MappedModel>, ports: Vec<u16>, parents: Option<Vec<Pid>>, access_denied: bool, maps_time: f64, scan_mem: f64 }` with `ProcDetail::new(pid, name)`, `is_ai()`, `is_diffusion()`.

- [ ] **Step 1: Write the failing tests**

`rust/crates/gputaskman-core/tests/model.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::model::*;
use gputaskman_core::provenance::*;

fn adapter(key: u64, vendor_id: u16, vram: u64) -> Adapter {
    Adapter { key, pci_bus: String::new(), name: format!("gpu{key}"), vendor_id, dedicated_vram: vram,
              shared_system: 0, integrated: false, backend: BackendKind::Drm }
}

#[test]
fn pci_key_matches_python() {
    assert_eq!(pci_key("00000000:01:00.0"), 0x0000_0000_0100_0);
    assert_eq!(pci_key("0000:03:00.0"), 0x0000_0300_0);
    assert_eq!(pci_key(""), 0);
}

#[test]
fn nvidia_first_then_vram_then_key() {
    let ordered = ordered_adapters(vec![adapter(3, VENDOR_AMD, 16 * GIB), adapter(1, VENDOR_NVIDIA, 8 * GIB),
                                        adapter(2, VENDOR_AMD, 32 * GIB)]);
    assert_eq!(ordered.iter().map(|a| a.key).collect::<Vec<_>>(), vec![1, 2, 3]);
}

#[test]
fn adapter_indices_do_not_clamp_to_another_gpu() {
    let list = [adapter(1, VENDOR_NVIDIA, GIB)];
    assert_eq!(select_adapter(&list, 0).unwrap().unwrap().key, 1);
    let err = select_adapter(&list, 1).unwrap_err();
    assert_eq!(err.to_string(), "--gpu 1 is out of range; 1 GPU adapter(s) available");
    assert!(select_adapter(&[], 0).unwrap().is_none());
    assert!(select_adapter(&[], 1).is_err());
}

#[test]
fn best_vram_keeps_measured_and_inferred_zero_distinct_from_unknown() {
    // Port of test_core.py MemoryTests.test_best_vram_keeps_measured_and_inferred_zero_distinct_from_unknown
    let mut g = ProcGpu::measured(0);
    assert_eq!(g.shown_vram().map(Shown::get), Some(0));
    g.vram = Bytes::Inferred { bytes: 8 * GIB, by: Confidence::Consistent };
    assert_eq!(g.shown_vram().map(Shown::get), Some(8 * GIB));
    g.vram = Bytes::Inferred { bytes: 0, by: Confidence::Consistent };
    assert_eq!(g.shown_vram().map(Shown::get), Some(0));
    g.vram = Bytes::unavailable(Reason::NvmlHidden);
    assert_eq!(g.shown_vram(), None);
}

#[test]
fn paged_out_is_clamped_commit_difference_and_unknown_without_commit() {
    let mut g = ProcGpu::measured(4 * GIB);
    assert!(matches!(g.paged_out(), Bytes::Unavailable { .. }));
    g.shared = Bytes::measured(GIB);
    g.committed = Bytes::measured(6 * GIB);
    assert_eq!(g.paged_out(), Bytes::measured(GIB));
    g.committed = Bytes::measured(GIB);
    assert_eq!(g.paged_out(), Bytes::measured(0));
}

#[test]
fn live_thresholds_match_python() {
    assert!(!ProcGpu::measured(MIB - 1).is_live());
    assert!(ProcGpu::measured(MIB).is_live());
    let mut hidden = ProcGpu::measured(0);
    hidden.vram = Bytes::unavailable(Reason::NvmlHidden);
    assert!(hidden.is_live());
    let mut busy = ProcGpu::measured(0);
    busy.engines = Engines::Measured { by_engine: [("3d".to_string(), 0.5)].into() };
    assert!(busy.is_live());
    let mut spill = ProcGpu::measured(0);
    spill.shared = Bytes::measured(16 * MIB);
    assert!(spill.is_live());
}

#[test]
fn attribution_total_rules() {
    let hidden = ProcGpu { vram: Bytes::unavailable(Reason::NvmlHidden), ..ProcGpu::measured(0) };
    let rows = [ProcGpu::measured(3 * GIB), hidden];
    assert_eq!(attributed_total(&rows), Bytes::measured(3 * GIB)); // hidden NVML counts as 0, like Python
    let drm = ProcGpu { vram: Bytes::unavailable(Reason::SharedBuffers), ..ProcGpu::measured(0) };
    let rows = [ProcGpu::measured(3 * GIB), drm];
    assert_eq!(attributed_total(&rows), Bytes::unavailable(Reason::SharedBuffers));
    let inferred = ProcGpu { vram: Bytes::Inferred { bytes: GIB, by: Confidence::Order }, ..ProcGpu::measured(0) };
    let rows = [ProcGpu::measured(GIB), inferred];
    assert_eq!(attributed_total(&rows), Bytes::measured(GIB)); // inferred bytes are never summed
}

#[test]
fn evidence_ladder_and_labels() {
    assert!(Evidence::Api < Evidence::Mapped && Evidence::Mapped < Evidence::Cmdline
            && Evidence::Cmdline < Evidence::Installed);
    let mut m = MappedModel { path: "/m.gguf".into(), size: 1, resident: 0, gguf: true,
                              sources: vec![Evidence::Mapped], store_name: None, kind: "LLM".into(),
                              kind_hint: String::new() };
    assert_eq!(m.evidence_label(), "in use (mapped)");
    m.add_source(Evidence::Api);
    m.add_source(Evidence::Api);
    assert_eq!(m.sources, vec![Evidence::Mapped, Evidence::Api]);
    assert_eq!(m.evidence_label(), "confirmed (api+mapped)");
    m.sources = vec![Evidence::Installed];
    assert_eq!(m.evidence_label(), "installed, not proven loaded");
    m.kind_hint = "decision model".into();
    assert_eq!(m.kind_label(), "likely decision model (name)");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test model`
Expected: FAIL to compile (`model` not found).

- [ ] **Step 3: Implement `model.rs`**

Port these exactly:
- `pci_key`: strip every character that isn't `[0-9a-f]` from the lower-cased bus ID and parse what's left as hex. Empty gives 0, and so does overflow (use `u64::from_str_radix(...).unwrap_or(0)`).
- `ordered_adapters`: sort by `(vendor_id != VENDOR_NVIDIA, Reverse(dedicated_vram), key)`.
- `select_adapter`: port of `collector.py` lines 141-149.
- `ProcGpu::shown_vram`: `Measured` or `Inferred` gives `Some(Shown::new(bytes))`, anything else `None`.
- `ProcGpu::paged_out`: when `vram`, `shared` and `committed` are all `Measured`, return `Measured(committed.saturating_sub(vram + shared))`; otherwise `Unavailable(NotOnBackend)`, or the reason of the first unavailable input.
- `ProcGpu::is_live`: port of `collector._is_live` (line 136). "Not attributed" means `vram` is not `Measured`. Engines below the threshold, or unavailable, don't make a row live.
- `attributed_total`: walk the rows. If any `vram` is `Unavailable { reason }` with `reason.voids_attribution()`, return `Bytes::unavailable(reason)`. Otherwise sum `as_measured()` through `MeasuredTotal` and return `Bytes::measured(total)`.
- `MappedModel::evidence_label` and `kind_label`: port of `procinfo.py` lines 89-106.
- `ProcDetail::is_ai`: models non-empty, or any runtime is in `AI_RUNTIMES`. Define `pub const AI_RUNTIMES: &[&str]` here, copied from `procinfo.py` line 46. `is_diffusion` ports `procinfo.py` `ProcDetail.is_diffusion`, which calls `_is_diffusion_cmdline` (line 108): put that function in this module as `pub fn is_diffusion_cmdline(cmd: &[String]) -> bool`.

Derive `Debug, Clone, PartialEq, Serialize, Deserialize` on every struct. `Engines` uses `#[serde(tag = "status", rename_all = "snake_case")]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test model`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add rust/crates/gputaskman-core
git commit -m "Add core data model with provenance-typed GPU memory"
```

---

### Task 3: Traits, environment paths, HTTP and test fakes

**Files:**
- Create: `src/env.rs`, `src/traits.rs`, `src/http.rs`, `src/testing.rs` in `gputaskman-core`
- Modify: `src/lib.rs` (add `pub mod env; pub mod traits; pub mod http; #[cfg(any(test, feature = "test-support"))] pub mod testing;`)
- Test: `rust/crates/gputaskman-core/tests/traits.rs`, `tests/ui/local_url_any_host.rs`

**Interfaces:**
- Consumes: `model::{Adapter, AdapterGpu, BackendKind, DeviceTelemetry, Pid, ProcGpu}`, `provenance::Reason`.
- Produces:
  - `env::Paths { home: PathBuf, xdg_data: PathBuf, xdg_config: PathBuf }` with `Paths::from_env()` (XDG fallbacks `~/.local/share` and `~/.config`) and `Paths::rooted(dir: &Path)` (all three under `dir`, for tests).
  - `traits::Clock: Send + Sync { fn mono(&self) -> Duration; fn wall(&self) -> f64; }`, plus `SystemClock`.
  - `traits::ScanPolicy { Budgeted, Unbounded }`.
  - `traits::Deadline { at: Option<Duration> }` with `Deadline::after(clock, budget: Duration)`, `Deadline::none()`, `passed(&self, clock) -> bool`.
  - `traits::LocalUrl` (private fields `port: u16`, `path: String`) with `LocalUrl::loopback(port: u16, path: &str) -> Self`, `base(&self) -> String` (`http://127.0.0.1:{port}`) and `Display` (base + path).
  - `traits::HttpGet: Send + Sync { fn get(&self, url: &LocalUrl, timeout: Duration) -> Option<String>; }`: `None` unless the status is 200; the body is capped at 4 MiB.
  - `traits::StartTime { ticks: u64, wall: f64 }`.
  - `traits::ProcBasics { name: Option<String>, exe: Option<String>, cmdline: Option<Vec<String>>, start: Option<StartTime> }`.
  - `traits::MapEntry { path: String, rss: u64 }`.
  - `traits::ProcessSource: Send + Sync`, with: `pids(&self) -> Vec<Pid>`, `basics(&self, pid) -> Option<ProcBasics>` (`None` when the process can't be opened at all), `start_time(&self, pid) -> Option<StartTime>`, `parent(&self, pid) -> Option<Pid>`, `memory_maps(&self, pid) -> Result<Vec<MapEntry>, Reason>`, `listening_ports(&self) -> BTreeMap<Pid, Vec<u16>>`, `self_pid(&self) -> Pid`.
  - `traits::BackendSample { procs: BTreeMap<Pid, ProcGpu>, outside: BTreeMap<Pid, ProcGpu>, adapter: AdapterGpu, status: Option<String> }`.
  - `traits::GpuBackend: Send`, with: `kind(&self) -> BackendKind`, `adapters(&self) -> Vec<Adapter>`, `start(&mut self, adapter: &Adapter, policy: ScanPolicy)`, `sample(&mut self) -> BackendSample`, `device(&mut self, index: usize) -> Result<DeviceTelemetry, Reason>`, `driver(&self) -> Option<String>`, `cuda(&self) -> Option<String>`.
  - `http::UreqGet` implementing `HttpGet`: no proxy, `Accept: application/json`, the given timeout, `GET` only.
  - `testing::FakeClock`: `new(wall: f64)`, `advance(Duration)`, `auto_advance(Duration)` (every `mono()` call moves forward by that much, like the Python test `Clock`).
  - `testing::FakeHttp`: `default()`, `route(port, path, status, body) -> Self`, `requests() -> Vec<String>`.
  - `testing::FakeProcs`: `default()`, `with(pid, ProcBasics) -> Self`, `named(pid, name) -> Self` (basics with `exe = /opt/<name>`, `cmdline = [name]`, `start` ticks 1), `ports(pid, &[u16]) -> Self`, `maps(pid, Vec<MapEntry>) -> Self`, `parent(pid, ppid) -> Self`.
  - `testing::FakeBackend`: `new(kind, adapters) -> Self`, `script(Vec<BackendSample>) -> Self` (the last sample repeats), `starts() -> Arc<Mutex<usize>>`, `with_driver(Option<String>, Option<String>) -> Self`, `with_device(DeviceTelemetry) -> Self`.
  - `BackendSample` derives `Default`.

- [ ] **Step 1: Write the failing tests**

`rust/crates/gputaskman-core/tests/traits.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::http::UreqGet;
use gputaskman_core::testing::{FakeClock, FakeHttp};
use gputaskman_core::traits::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

#[test]
fn local_url_is_loopback_only() {
    let url = LocalUrl::loopback(8080, "/props");
    assert_eq!(url.to_string(), "http://127.0.0.1:8080/props");
    assert_eq!(url.base(), "http://127.0.0.1:8080");
}

#[test]
fn deadline_follows_the_clock() {
    let clock = FakeClock::new(1000.0);
    let d = Deadline::after(&clock, Duration::from_millis(150));
    assert!(!d.passed(&clock));
    clock.advance(Duration::from_millis(150));
    assert!(d.passed(&clock));
    assert!(!Deadline::none().passed(&clock));
}

#[test]
fn fake_http_records_requests() {
    let http = FakeHttp::default().route(8080, "/props", 200, "{}");
    assert_eq!(http.get(&LocalUrl::loopback(8080, "/props"), Duration::from_millis(500)).as_deref(), Some("{}"));
    assert_eq!(http.get(&LocalUrl::loopback(8080, "/nope"), Duration::from_millis(500)), None);
    assert_eq!(http.requests(), vec!["http://127.0.0.1:8080/props", "http://127.0.0.1:8080/nope"]);
}

fn serve_once(status: &'static str, body: &'static str) -> (u16, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 2048];
        let n = stream.read(&mut buf).unwrap();
        let request = String::from_utf8_lossy(&buf[..n]).to_string();
        let response = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        stream.write_all(response.as_bytes()).unwrap();
        request
    });
    (port, handle)
}

#[test]
fn ureq_get_is_a_get_and_only_accepts_200() {
    let (port, server) = serve_once("200 OK", r#"{"ok":true}"#);
    let body = UreqGet::default().get(&LocalUrl::loopback(port, "/props"), Duration::from_millis(500));
    assert_eq!(body.as_deref(), Some(r#"{"ok":true}"#));
    let request = server.join().unwrap();
    assert!(request.starts_with("GET /props HTTP/1.1"), "{request}");

    let (port, server) = serve_once("404 Not Found", "missing");
    assert_eq!(UreqGet::default().get(&LocalUrl::loopback(port, "/x"), Duration::from_millis(500)), None);
    server.join().unwrap();
}
```

`tests/ui/local_url_any_host.rs`:
```rust
use gputaskman_core::traits::LocalUrl;
fn main() { let _u = LocalUrl { port: 80, path: String::new() }; }
```

In `gputaskman-core/Cargo.toml`, make the tests see the fakes:
```toml
[dev-dependencies]
gputaskman-core = { path = ".", features = ["test-support"] }
```
Then run: `cargo add -p gputaskman-core ureq`

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test traits`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

- `traits.rs`: the definitions listed under Interfaces. `SystemClock::mono` is the elapsed time since a process-wide `OnceLock<Instant>`, and `wall` is `SystemTime::now()` seconds since the epoch as `f64`.
- `http.rs`: `UreqGet` holds a `ureq::Agent` configured with proxy disabled and `http_status_as_error(false)`. `get` sends a GET with `Accept: application/json` and the per-call timeout, returns `None` on any error or a status other than 200, and reads the body through `.into_reader().take(4 << 20)` (or the ureq 3 equivalent `body_mut().with_config().limit(4 << 20).read_to_string()`) into a `String`, replacing invalid UTF-8. It has no method that sends a body.
- `testing.rs`: implement the fakes. They use `Mutex` internally so they are `Send + Sync`. `FakeClock` keeps `mono` and `wall` in a `Mutex<(Duration, f64)>`, and `advance` moves both forward.
- `env.rs`: `Paths::from_env` reads `HOME`, `XDG_DATA_HOME` and `XDG_CONFIG_HOME`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core`
Expected: PASS, including the new `local_url_any_host` compile-fail case (accept its `.stderr`, which must be a private-field error).

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add backend, process and HTTP traits with test fakes"
```

---

### Task 4: GGUF header reader and estimates

**Files:**
- Create: `rust/crates/gputaskman-core/src/gguf.rs`
- Test: `rust/crates/gputaskman-core/tests/gguf.rs`, `tests/support/gguf_writer.rs`

**Interfaces:**
- Produces (in `gputaskman_core::gguf`):
  - `enum MetaValue { U(u64), I(i64), F(f64), Bool(bool), Str(String), Array(Vec<MetaValue>) }` (serde untagged).
  - `struct GgufInfo { path: String, file_size: u64, version: u32, metadata: BTreeMap<String, MetaValue>, layer_bytes: BTreeMap<u32, u64>, output_bytes: u64, token_embd_bytes: u64, tensor_bytes: u64, n_params: Option<u64>, error: Option<String> }`.
  - Methods ported from `gguf.py` lines 30-139: `meta(&self, key) -> Option<&MetaValue>`, `arch()`, `kind()`, `decision_type()`, `causal()`, `name()`, `quant()`, `n_layers()`, `ctx_train()`, `sliding_window()`, `params_label()`, `gpu_weight_bytes(n_gpu_layers: Option<i64>) -> u64`, `kv_bytes_per_token(type_bytes: f64) -> Option<u64>`.
  - `fn is_gguf(path: &Path) -> bool`.
  - `struct GgufCache` with `new()` and `read(&mut self, path: &Path) -> Option<GgufInfo>`, cached by `(path, size, mtime)` (port of `_read_cached` and `read_gguf`, lines 186-254).

- [ ] **Step 1: Write the GGUF writer test helper**

`tests/support/gguf_writer.rs` writes a minimal little-endian GGUF v3 file: magic `GGUF`, `u32` version 3, `u64` tensor count, `u64` kv count, then each kv (string key as `u64` length + bytes, `u32` type, value), then each tensor (name, `u32` n_dims, `u64` dims, `u32` ggml type, `u64` offset):
```rust
use std::io::Write;
pub enum V<'a> { U32(u32), Str(&'a str) }
pub fn write_gguf(path: &std::path::Path, kv: &[(&str, V)], tensors: &[(&str, &[u64], u32)]) {
    let mut f = std::fs::File::create(path).unwrap();
    let s = |f: &mut std::fs::File, t: &str| { f.write_all(&(t.len() as u64).to_le_bytes()).unwrap(); f.write_all(t.as_bytes()).unwrap(); };
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&(kv.len() as u64).to_le_bytes()).unwrap();
    for (key, value) in kv {
        s(&mut f, key);
        match value {
            V::U32(n) => { f.write_all(&4u32.to_le_bytes()).unwrap(); f.write_all(&n.to_le_bytes()).unwrap(); }
            V::Str(t) => { f.write_all(&8u32.to_le_bytes()).unwrap(); s(&mut f, t); }
        }
    }
    let mut offset = 0u64;
    for (name, dims, ty) in tensors {
        s(&mut f, name);
        f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
        for d in *dims { f.write_all(&d.to_le_bytes()).unwrap(); }
        f.write_all(&ty.to_le_bytes()).unwrap();
        f.write_all(&offset.to_le_bytes()).unwrap();
        offset += dims.iter().product::<u64>() * 4;
    }
}
```

- [ ] **Step 2: Write the failing tests**

`tests/gguf.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support { pub mod gguf_writer; }
use gputaskman_core::gguf::*;
use support::gguf_writer::{write_gguf, V};
use std::collections::BTreeMap;

fn info(meta: &[(&str, MetaValue)]) -> GgufInfo {
    GgufInfo { path: "model.gguf".into(), file_size: 1000, version: 3,
               metadata: meta.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
               layer_bytes: BTreeMap::from([(0, 100), (1, 100)]), output_bytes: 50, token_embd_bytes: 50,
               tensor_bytes: 300, n_params: None, error: None }
}

#[test]
fn layer_and_kv_estimates() {
    // Port of test_core.py ModelTests.test_layer_and_kv_estimates
    let m = info(&[("general.architecture", MetaValue::Str("test".into())), ("test.block_count", MetaValue::U(2)),
                   ("test.attention.head_count", MetaValue::U(4)), ("test.attention.head_count_kv", MetaValue::U(2)),
                   ("test.embedding_length", MetaValue::U(16))]);
    assert_eq!(m.gpu_weight_bytes(None), 250);
    assert_eq!(m.gpu_weight_bytes(Some(1)), 100);
    assert_eq!(m.gpu_weight_bytes(Some(0)), 0);
    assert_eq!(m.kv_bytes_per_token(2.0), Some(64));
}

#[test]
fn truncated_model_reports_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.gguf");
    std::fs::write(&path, b"GGUF").unwrap();
    assert!(GgufCache::new().read(&path).unwrap().error.is_some());
    assert!(is_gguf(&path));
}

#[test]
fn reads_header_written_by_helper() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.gguf");
    write_gguf(&path, &[("general.architecture", V::Str("llama")), ("llama.block_count", V::U32(2)),
                        ("general.name", V::Str("Tiny"))],
               &[("blk.0.attn_q.weight", &[4, 4], 0), ("blk.1.attn_q.weight", &[4, 4], 0),
                 ("output.weight", &[4, 2], 0), ("token_embd.weight", &[4, 2], 0)]);
    let g = GgufCache::new().read(&path).unwrap();
    assert_eq!(g.error, None);
    assert_eq!(g.arch(), "llama");
    assert_eq!(g.name(), "Tiny");
    assert_eq!(g.n_layers(), 2);
    assert_eq!(g.layer_bytes.get(&0), Some(&64));
    assert_eq!(g.kind(), "LLM");
}

#[test]
fn not_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.bin");
    std::fs::write(&path, b"NOPE....").unwrap();
    assert!(!is_gguf(&path));
}
```
Also port `test_core.py` `IdentityTests.test_gguf_kinds_come_from_header_keys` (line 682) and `test_specific_header_kinds_override_name_hints` (line 613) into `tests/gguf.rs`: build each header with `write_gguf`, using the keys and expected kinds from those Python tests, and assert `kind()` and `decision_type()` match.

Run: `cargo add -p gputaskman-core --dev tempfile`

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test gguf`
Expected: FAIL to compile.

- [ ] **Step 4: Implement `gguf.rs`**

Port `src/gputaskman/gguf.py` in full:
- the value types and string/array reading of `_Reader` (lines 140-185);
- tensor byte sizes per ggml type, using the same table as the Python;
- the per-layer accumulation by `blk.N.` prefix, plus `output`/`token_embd` bytes;
- the `kind`/`causal`/`decision_type` rules and `params_label`;
- `gpu_weight_bytes` and `kv_bytes_per_token` (lines 108-139).

A short or corrupt file returns `GgufInfo` with `error: Some(message)`, never panics. `is_gguf` reads only the first 4 bytes.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test gguf`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "Add GGUF header reader and KV/weight estimates"
```

---

### Task 5: Model-store names

**Files:**
- Create: `rust/crates/gputaskman-core/src/stores.rs`
- Test: `rust/crates/gputaskman-core/tests/stores.rs`

**Interfaces:**
- Consumes: `env::Paths`, `traits::Clock`.
- Produces: `struct StoreNames` with `new(paths: Paths)`, `ollama_names(&mut self, models_root: &Path, now: f64) -> &BTreeMap<String, Vec<String>>`, `lmstudio_dirs(&mut self, now: f64) -> Vec<PathBuf>`, `store_name(&mut self, path: &str, now: f64) -> Option<String>`. The cache TTL is 60 s (`stores.py` `_TTL`).

- [ ] **Step 1: Write the failing test**

`tests/stores.rs` ports `test_core.py` `IdentityTests.test_store_names` (line 697). Build the same Ollama manifest tree (`manifests/registry.ollama.ai/library/<model>/<tag>` JSON with `layers[].digest`), Hugging Face cache path and LM Studio layout under a `tempfile::tempdir()`, using `Paths::rooted(dir)` so `~/.lmstudio/models` resolves inside it:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{env::Paths, stores::StoreNames};

#[test]
fn store_names() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::rooted(dir.path());
    let digest = "a".repeat(64);
    let manifest = dir.path().join("ollama/manifests/registry.ollama.ai/library/qwen3");
    std::fs::create_dir_all(&manifest).unwrap();
    std::fs::write(manifest.join("14b"), format!(r#"{{"layers":[{{"digest":"sha256:{digest}"}}]}}"#)).unwrap();
    let blob = dir.path().join(format!("ollama/blobs/sha256-{digest}"));
    let mut names = StoreNames::new(paths.clone());
    assert_eq!(names.store_name(blob.to_str().unwrap(), 0.0).as_deref(), Some("qwen3:14b"));
    assert_eq!(names.store_name("/x/models--Qwen--Qwen3-8B/snapshots/abc/model.safetensors", 0.0).as_deref(),
               Some("Qwen/Qwen3-8B"));
    let lms = paths.home.join(".lmstudio/models/lmstudio-community/Qwen3-GGUF");
    std::fs::create_dir_all(&lms).unwrap();
    assert_eq!(names.store_name(lms.join("q.gguf").to_str().unwrap(), 0.0).as_deref(),
               Some("lmstudio-community/Qwen3-GGUF"));
    assert_eq!(names.store_name("/tmp/loose.gguf", 0.0), None);
}
```
Add the remaining Python assertions from `test_store_names` (a non-library namespace, a non-default registry, and `downloadsFolder` from `settings.json`) as further `assert_eq!` lines in the same test.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p gputaskman-core --test stores`
Expected: FAIL to compile.

- [ ] **Step 3: Implement `stores.rs`**

Port `stores.py` in full, using `regex` (`cargo add -p gputaskman-core regex`) with the same two patterns, case-insensitive. Keep the 60 s caches and the `downloadsFolder` precedence, and dedupe the LM Studio directories in order. Linux paths are case-sensitive, so the original casing is used directly.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p gputaskman-core --test stores`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add Ollama, Hugging Face and LM Studio store names"
```

---

### Task 6: Runtime detection, model-file rules and classification

**Files:**
- Create: `src/procinfo/mod.rs`, `src/procinfo/runtimes.rs`, `src/procinfo/models.rs`, `src/procinfo/classify.rs` in `gputaskman-core`
- Test: `rust/crates/gputaskman-core/tests/procinfo_rules.rs`

**Interfaces:**
- Consumes: `model::{MappedModel, ProcDetail, Evidence, AI_RUNTIMES, MIB}`, `gguf::GgufCache`, `stores::StoreNames`.
- Produces:
  - `runtimes::lib_name(base: &str) -> Option<String>` and `runtimes::match_runtimes(libs: &BTreeSet<String>) -> Vec<String>`, ported from `procinfo.py` lines 18-45 and 137-149. Use `glob::Pattern` (`cargo add -p gputaskman-core glob`) for the fnmatch patterns, keeping the `RUNTIMES` order.
  - `models::MODEL_EXTS`, `AMBIGUOUS_EXTS`, `NOT_MODELS`, `skip_path(path) -> bool`, `models_from_cmdline(cmd: &[String]) -> Vec<String>`, `is_model_file(ext, size, gguf, store: Option<&str>) -> bool`, `name_kind_hint(path, store) -> String`, and `struct ModelContext { pub gguf: GgufCache, pub stores: StoreNames }` with `make_model(&mut self, path, size, resident, gguf, source: Evidence, store: Option<String>, now: f64) -> MappedModel`.
  - `classify::classify(d: &ProcDetail, engines: &Engines) -> String`, plus `AI_KINDS`, `MODEL_KINDS`, `LLM_NAMES`, `BROWSERS`, `COMPOSITORS`, ported from `procinfo.py` lines 46-78 and 423-454. PID 4 is ordinary on Linux, so the `kernel` branch is dropped (rule from `test_pid_4_is_only_special_on_windows`).

- [ ] **Step 1: Write the failing tests**

`tests/procinfo_rules.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::model::*;
use gputaskman_core::procinfo::{classify::classify, models::*, runtimes::*};
use std::collections::BTreeSet;

fn libs(names: &[&str]) -> BTreeSet<String> { names.iter().map(|s| s.to_string()).collect() }

#[test]
fn versioned_shared_objects_are_libraries() {
    assert_eq!(lib_name("libcudart.so.12.4").as_deref(), Some("libcudart.so"));
    assert_eq!(lib_name("libllama.so").as_deref(), Some("libllama.so"));
    assert_eq!(lib_name("model.gguf"), None);
}

#[test]
fn linux_runtimes_are_detected() {
    assert_eq!(match_runtimes(&libs(&["libllama.so", "libggml-cuda.so", "libcuda.so", "libvulkan.so"])),
               vec!["llama.cpp", "ggml", "CUDA", "Vulkan"]);
    assert_eq!(match_runtimes(&libs(&["libamdhip64.so"])), vec!["HIP"]);
}

#[test]
fn ambiguous_extensions_need_size_or_store() {
    assert!(!is_model_file(".bin", 10 * MIB, false, None));
    assert!(is_model_file(".bin", 64 * MIB, false, None));
    assert!(is_model_file(".bin", 1, false, Some("org/repo")));
    assert!(is_model_file(".safetensors", 4 * MIB, false, None));
    assert!(!is_model_file(".safetensors", 4 * MIB - 1, false, None));
    assert!(is_model_file("", 1, true, None));
}

#[test]
fn linux_compositors_and_browsers_and_pid_4() {
    let engines = Engines::Measured { by_engine: Default::default() };
    for (name, kind) in [("Xwayland", "compositor"), ("gnome-shell", "compositor"), ("firefox", "browser")] {
        assert_eq!(classify(&ProcDetail::new(100, name), &engines), kind, "{name}");
    }
    assert_eq!(classify(&ProcDetail::new(4, "kworker"), &engines), "app");
}

#[test]
fn hip_runtime_is_compute_evidence() {
    let mut d = ProcDetail::new(7, "worker");
    d.runtimes = vec!["HIP".into()];
    assert_eq!(classify(&d, &Engines::Measured { by_engine: Default::default() }), "compute");
}
```
Also port these Python tests into the same file, keeping their names, inputs and assertions:
- `test_core.py` `ModelTests.test_diffusion_arguments_do_not_classify_text_encoder_as_llm` (line 505);
- `IdentityTests.test_decision_name_hints_use_model_names_not_substrings` (line 582);
- `IdentityTests.test_installed_kind_hints_do_not_classify_the_process` (line 660);
- `IdentityTests.test_repeated_model_arguments_do_not_duplicate_evidence` (line 748), using real temp files for `models_from_cmdline`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test procinfo_rules`
Expected: FAIL to compile.

- [ ] **Step 3: Implement the three modules**

Port line by line from `procinfo.py`, keeping constant lists in the same order. `models_from_cmdline` returns canonical absolute paths of existing files (`std::path::absolute`; the Python uses `os.path.abspath`, which doesn't resolve symlinks). `make_model` reads the GGUF kind only when `gguf` is true and the header has no error, sets the store name via `StoreNames`, and computes `name_kind_hint` only when the kind is `""`, `"model"` or `"LLM"` (Python line 176).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test procinfo_rules`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add runtime detection, model-file rules and classification"
```

---

### Task 7: Installed-model candidates and the process inspector

**Files:**
- Create: `src/procinfo/installed.rs`, `src/procinfo/inspector.rs` in `gputaskman-core`
- Test: `rust/crates/gputaskman-core/tests/inspector.rs`

**Interfaces:**
- Consumes: `traits::{ProcessSource, Clock}`, `env::Paths`, `procinfo::models::ModelContext`, `procinfo::runtimes::*`.
- Produces:
  - `installed::Installed` with `new(paths)` and `models(&mut self, exe: &str, ctx: &mut ModelContext, now: f64) -> Vec<MappedModel>`: depth 4, 3000 entries, at most 6 results, cached for 300 s per exe. This ports `installed_models` (lines 191-274), including `_walk`, `_SKIP_DIRS`, `_is_system_path` (Linux branch) and `_data_dirs` (Linux branch).
  - `inspector::ProcInspector` with `new()`, `get(&mut self, procs: &dyn ProcessSource, pid, now) -> &mut ProcDetail`, `needs_scan(&self, d: &ProcDetail, mem: f64, now) -> bool`, `scan(&mut self, pid, procs, ctx, installed, mem, now)`, `exclusive_ports(&mut self, procs, pid, now) -> Vec<u16>`, `parents(&mut self, procs, pid, depth) -> Vec<Pid>`, `prune(&mut self, live: &BTreeSet<Pid>)`. The constants are `SCAN_MIN_GAP = 5.0`, `SCAN_BACKSTOP = 120.0`, `SCAN_DELTA = 64 MiB` and `PORTS_INTERVAL = 5.0`.

- [ ] **Step 1: Write the failing tests**

`tests/inspector.rs`, using `FakeProcs` from Task 3:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::model::*;
use gputaskman_core::procinfo::inspector::ProcInspector;
use gputaskman_core::testing::FakeProcs;
use gputaskman_core::traits::{ProcBasics, StartTime};

fn basics(name: &str, ticks: u64) -> ProcBasics {
    ProcBasics { name: Some(name.into()), exe: Some(format!("/opt/{name}")), cmdline: Some(vec![name.into()]),
                 start: Some(StartTime { ticks, wall: ticks as f64 }) }
}

#[test]
fn reused_pid_gets_fresh_details() {
    let procs = FakeProcs::default().with(42, basics("old", 1));
    let mut inspector = ProcInspector::new();
    assert_eq!(inspector.get(&procs, 42, 0.0).name, "old");
    let procs = FakeProcs::default().with(42, basics("new", 2));
    assert_eq!(inspector.get(&procs, 42, 1.0).name, "new");
}

#[test]
fn shared_listen_port_is_not_used_for_server_attribution() {
    let procs = FakeProcs::default().with(1, basics("a", 1)).with(2, basics("b", 1))
        .ports(1, &[8080, 11434]).ports(2, &[8080]);
    let mut inspector = ProcInspector::new();
    assert_eq!(inspector.exclusive_ports(&procs, 1, 0.0), vec![11434]);
}

#[test]
fn scans_follow_memory_changes() {
    // Port of test_core.py IdentityTests.test_scans_follow_memory_changes (line 766)
    let procs = FakeProcs::default().with(9, basics("llama-server", 1));
    let mut inspector = ProcInspector::new();
    let d = inspector.get(&procs, 9, 0.0).clone();
    assert!(inspector.needs_scan(&d, 0.0, 0.0));
    let mut scanned = d.clone();
    scanned.maps_time = 100.0;
    scanned.scan_mem = (GIB) as f64;
    assert!(!inspector.needs_scan(&scanned, (GIB) as f64, 103.0)); // inside the minimum gap
    assert!(!inspector.needs_scan(&scanned, (GIB + 10 * MIB) as f64, 110.0)); // small change
    assert!(inspector.needs_scan(&scanned, (2 * GIB) as f64, 110.0)); // large change
    assert!(inspector.needs_scan(&scanned, (GIB) as f64, 221.0)); // backstop
}

#[test]
fn unreadable_process_is_access_denied_and_never_scanned() {
    let procs = FakeProcs::default().with(5, ProcBasics { name: Some("x".into()), exe: None, cmdline: None, start: None });
    let mut inspector = ProcInspector::new();
    let d = inspector.get(&procs, 5, 0.0).clone();
    assert!(d.access_denied);
    assert!(!inspector.needs_scan(&d, 0.0, 0.0));
}
```
Also port these, building real model files in a tempdir and listing them in `FakeProcs` memory maps:
- `test_core.py` `IdentityTests.test_installed_models_are_candidates_only` (line 809): candidates appear only for AI runtimes with no mapped models, and their source is `Installed`;
- `IdentityTests.test_reused_pid_gets_fresh_details` (line 737);
- a scan test where `libllama.so` plus a mapped `.gguf` give `runtimes == ["llama.cpp"]` and one model with `sources == [Mapped]`, and a `-m <same file>` argument adds `Cmdline` without duplicating the model.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test inspector`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Port `ProcInspector` (lines 282-415) and `installed_models`:
- `get` refreshes ports at most every 5 s, rebuilds the detail when the stored `start_ticks` differs from `procs.start_time(pid)`, and sets `ports`.
- `_new` maps onto `ProcBasics`: a missing `exe`, `cmdline` or `start` sets `access_denied`, and `name` falls back to `"?"`.
- `scan` follows Python lines 365-408 exactly: library detection before `skip_path`, the extension filters, the GGUF check when the extension is `.gguf` or empty or the size is over 64 MiB, the store lookup, and the command-line merge. Models are sorted by size descending. Candidates come only from AI runtimes with no models. A `memory_maps` error sets `access_denied`.
- File sizes come from `std::fs::metadata`. An error skips the file.
- `parents` walks `procs.parent()` up to `depth` times and caches the result in `ProcDetail::parents`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test inspector`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add process inspector and installed-model candidates"
```

---

### Task 8: LLM flags, load logs and local server probes

**Files:**
- Create: `src/llm/mod.rs` (with `pub fn gb(n: f64) -> String`), `src/llm/flags.rs`, `src/llm/logs.rs`, `src/llm/servers.rs` in `gputaskman-core`
- Test: `rust/crates/gputaskman-core/tests/llm_inputs.rs`

**Interfaces:**
- Consumes: `traits::{HttpGet, LocalUrl}`, `env::Paths`.
- Produces:
  - `flags::Flags { ctx: Option<i64>, ngl: Option<i64>, parallel: Option<i64>, port: Option<i64>, ctk: Option<String>, ctv: Option<String>, log_file: Option<String>, no_mmap: bool }` (serialises like the Python dict, without null keys) and `flags::parse_flags(cmd: &[String]) -> Flags` (`llm.py` lines 25-63).
  - `logs::is_gpu_device(dev: &str) -> bool`, `logs::parse_load_log(path: &Path, not_before: f64) -> Option<BTreeMap<String, BTreeMap<String, u64>>>` (lines 66-107) and `logs::ollama_log_path(paths: &Paths) -> PathBuf` (Linux branch).
  - `servers::ServerModel { name: Option<String>, r#type: Option<String>, quant: Option<String>, ctx: Option<i64>, size: Option<u64>, size_vram: Option<u64>, path: Option<String> }`.
  - `servers::ServerInfo { kind: String, url: String, n_ctx: Option<i64>, slots_total: Option<i64>, slots_busy: Option<i64>, gen_tok_s: Option<f64>, prompt_tok_s: Option<f64>, model_path: Option<String>, models: Vec<ServerModel> }`.
  - `servers::{llama_server_models, ollama_models, lmstudio_models}` taking `&serde_json::Value`.
  - `servers::probe_server(http: &dyn HttpGet, port: u16, prev: Option<&ServerInfo>) -> Option<ServerInfo>`.
  - `servers::ServerProber` with `get(&mut self, http, ports: &[u16], now: f64) -> Option<ServerInfo>` (TTL 3 s on a hit and 30 s on a miss).
  - `const PROBE_TIMEOUT: Duration = Duration::from_millis(500);`

- [ ] **Step 1: Write the failing tests**

`tests/llm_inputs.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::llm::{flags::parse_flags, logs::*, servers::*};
use gputaskman_core::testing::FakeHttp;

fn args(a: &[&str]) -> Vec<String> { a.iter().map(|s| s.to_string()).collect() }
const MIB: u64 = 1 << 20;

#[test]
fn flags_include_equals_and_gpu_layers() {
    let f = parse_flags(&args(&["server", "--ctx-size=8192", "-ngl", "all", "--cache-type-k=q8_0"]));
    assert_eq!((f.ctx, f.ngl, f.ctk.as_deref()), (Some(8192), Some(-1), Some("q8_0")));
    assert_eq!(serde_json::to_value(&f).unwrap(), serde_json::json!({"ctx": 8192, "ngl": -1, "ctk": "q8_0"}));
}

#[test]
fn gpu_buffers_exclude_host_and_cpu() {
    for d in ["CPU", "CPU_Mapped", "CUDA_Host", "Vulkan_Host"] { assert!(!is_gpu_device(d), "{d}"); }
    for d in ["CUDA0", "Vulkan0"] { assert!(is_gpu_device(d), "{d}"); }
}

#[test]
fn latest_load_and_multiple_kv_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.log");
    std::fs::write(&path, "load_tensors: CUDA0 model buffer size = 900 MiB\n\
        llama_context: CUDA0 KV buffer size = 800 MiB\nnew load\n\
        load_tensors: CPU_Mapped model buffer size = 30 MiB\n\
        load_tensors: CUDA0 model buffer size = 100 MiB\n\
        llama_context: CUDA0 KV buffer size = 20 MiB\n\
        llama_context: CUDA0 KV buffer size = 10 MiB\n").unwrap();
    let log = parse_load_log(&path, 0.0).unwrap();
    assert_eq!(log["CUDA0"]["weights"], 100 * MIB);
    assert_eq!(log["CUDA0"]["kv"], 30 * MIB);
    assert_eq!(log["CPU_Mapped"]["weights"], 30 * MIB);
}

#[test]
fn probes_are_get_only_on_the_given_port() {
    let http = FakeHttp::default()
        .route(8080, "/props", 200, r#"{"model_path":"/m/q.gguf","default_generation_settings":{"n_ctx":4096},"total_slots":1}"#)
        .route(8080, "/v1/models", 200, r#"{"data":[{"id":"q.gguf","meta":{"ftype":"Q4_K_M","n_ctx":4096}}]}"#)
        .route(8080, "/slots", 200, r#"[{"is_processing":true}]"#)
        .route(8080, "/metrics", 200, "llamacpp:predicted_tokens_seconds 42.5\n");
    let s = probe_server(&http, 8080, None).unwrap();
    assert_eq!((s.kind.as_str(), s.n_ctx, s.slots_busy, s.gen_tok_s), ("llama-server", Some(4096), Some(1), Some(42.5)));
    assert!(http.requests().iter().all(|u| u.starts_with("http://127.0.0.1:8080/")));
}
```
Also port `test_core.py` `IdentityTests.test_runtime_api_normalization` (line 713) and `test_lmstudio_falls_back_to_v0_when_v1_is_an_error` (line 729), building the JSON inputs with `serde_json::json!` from the Python literals.

Run: `cargo add -p gputaskman-core regex`, if Task 5 hasn't already.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test llm_inputs`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Port `llm.py` lines 25-260:
- `_get` becomes `http.get(&LocalUrl::loopback(port, path), PROBE_TIMEOUT)`, followed by `serde_json::from_str`. If that fails, keep the text (only `/metrics` uses text).
- `parse_load_log` reads at most the last 512 KiB and returns `None` when the file's mtime is before `not_before`.
- The LM Studio fallback is decided by response shape, not status, as in Python line 225.
- `gb` ports lines 301-304.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test llm_inputs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add LLM flag, load-log and local server probe parsing"
```

---

### Task 9: LLM breakdown view

**Files:**
- Create: `rust/crates/gputaskman-core/src/llm/view.rs`
- Test: `rust/crates/gputaskman-core/tests/llm_view.rs`

**Interfaces:**
- Consumes: Tasks 4, 6 and 8.
- Produces:
  - `view::SegmentKind { Weights, Kv, Compute, Runtime, Unknown }` (serde snake_case).
  - `view::Segment { label: String, bytes: Bytes, kind: SegmentKind }`. A segment from a load log is `Bytes::Reported { by: Reporter::LoadLog }`. An estimate is `Bytes::Estimated { from: Gguf }`. A remainder computed from a measured total minus known parts is `Bytes::Estimated { from: Gguf }` too, matching Python `exact=False`.
  - `view::LlmView { models: Vec<GgufInfo>, files: Vec<MappedModel>, other_files: Vec<MappedModel>, flags: Flags, server: Option<ServerInfo>, host: Option<ServerInfo>, log_source: Option<String>, segments: Vec<Segment>, notes: Vec<String>, kv_per_token: Option<u64>, kv_type: String, n_ctx: Option<i64> }` with `label(i) -> String` and `kind_label(i) -> String`.
  - `view::build_view(d: &ProcDetail, vram: f64, shared: f64, server: Option<ServerInfo>, log_path: Option<PathBuf>, host: Option<ServerInfo>, attributed: bool, ctx: &mut ModelContext, now: f64) -> LlmView`.

- [ ] **Step 1: Write the failing tests**

`tests/llm_view.rs` ports these Python tests, keeping their names. Build GGUF files with `support::gguf_writer` (copy the helper module path from Task 4: `mod support { pub mod gguf_writer; }`) and logs in a tempdir:
- `test_core.py` `IdentityTests.test_decision_model_has_no_kv_cache_and_api_evidence` (line 775);
- `test_vulkan_log_does_not_label_unknown_overhead_as_cuda` (line 788);
- `test_ollama_runner_shows_only_its_model` (line 831);
- `test_runner_and_host_apis_do_not_duplicate_evidence` (line 846);
- `test_openjev_hint_does_not_change_header_kv_or_loading_evidence` (line 626);
- `test_linux.py` `UnattributedReportTests.test_llm_view_does_not_claim_cpu_inference_when_vram_is_unknown` (line 374).

Plus these Rust-specific provenance checks:
```rust
#[test]
fn log_segments_are_reported_and_estimates_are_estimated() {
    // A llama.cpp log for this process: every GPU segment is runtime-reported.
    // (build ProcDetail with one mapped GGUF and a --log-file pointing at a log like the one in Task 8)
    let view = /* build_view(...) with log */;
    assert!(view.segments.iter().filter(|s| s.kind != SegmentKind::Runtime)
        .all(|s| matches!(s.bytes, Bytes::Reported { by: Reporter::LoadLog, .. })));
    // Without a log: weights and KV are estimates from the GGUF header.
    let view = /* build_view(...) without log */;
    assert!(view.segments.iter().all(|s| matches!(s.bytes, Bytes::Estimated { from: EstimateSource::Gguf, .. })));
}
```
Write these two calls out in full, using the same `ProcDetail` setup as the Vulkan-log test above, with and without `--log-file`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test llm_view`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Port `_merge_api` and `build_view` (`llm.py` lines 306-441) exactly, including every note string, the 1.03 tolerance, the sliding-window note, the 256 MiB shared note and the 64 MiB CPU-inference note. Clone `d.models` before merging so cached details stay untouched (Python line 339). Thousands separators in the KV label follow Python `f"{n:,}"`; write a small `group_thousands(i64) -> String` helper in `view.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test llm_view`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add LLM memory breakdown view"
```

---

### Task 10: Handy adapter

**Files:**
- Create: `rust/crates/gputaskman-core/src/handy.rs`
- Test: `rust/crates/gputaskman-core/tests/handy.rs`

**Interfaces:**
- Consumes: `env::Paths`.
- Produces: `HandyEvidence { selected_model: Option<String>, last_model: Option<String>, last_event: Option<String>, last_event_at: Option<String> }` with `last_event_local(&self) -> String`, and `HandyProbe` with `new(paths)` and `inspect(&mut self, process_started: f64) -> Option<HandyEvidence>`. Settings are capped at 256 KiB and the log at 512 KiB. Caches are keyed by `(mtime_ns, size)`. Linux paths: `<xdg_data>/com.pais.handy/settings_store.json` and `<xdg_data>/com.pais.handy/logs/handy.log`.

- [ ] **Step 1: Write the failing tests**

`tests/handy.rs` ports `test_core.py` `HandyTests` (lines 436-503): `test_selected_model_is_distinct_from_runtime_load_and_unload_events` and `test_selected_model_without_fresh_log_is_not_reported_as_loaded`. Write the settings JSON and log lines verbatim from the Python test into `Paths::rooted(tempdir)` locations. Assert the same `selected_model`, `last_model`, `last_event` and `last_event_at` values. `last_event_at` must be the ISO form `2026-01-02T03:04:05+00:00`, the same as Python `datetime.isoformat()`.

Run: `cargo add -p gputaskman-core chrono --no-default-features --features clock,std`

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test handy`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Port `handy.py` lines 1-146, Linux branch only, with the same regexes. Timestamps are parsed as UTC with `chrono::NaiveDateTime::parse_from_str(.., "%Y-%m-%d %H:%M:%S")`, and events more than 2 s before `process_started` are skipped. Never read transcription content: only lines matching the timestamp, model-event and unload regexes are inspected.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test handy`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add read-only Handy settings and log adapter"
```

---

### Task 11: amdgpu sysfs and fdinfo parsing

**Files:**
- Create: `src/sysfs.rs`, `src/drm/mod.rs`, `src/drm/fdinfo.rs`, `src/drm/cards.rs`, `src/drm/device.rs` in `gputaskman-linux`
- Test: `rust/crates/gputaskman-linux/tests/support/mod.rs`, `tests/drm_parse.rs`

**Interfaces:**
- Consumes: `gputaskman_core::model::{Adapter, BackendKind, DeviceTelemetry, pci_key, VENDOR_AMD}`.
- Produces:
  - `sysfs::{read_trimmed(path) -> Option<String>, read_int(path, radix) -> Option<i64>}`.
  - `drm::fdinfo::parse_size(&str) -> Option<u64>`.
  - `drm::fdinfo::FdInfo { driver: String, client_id: Option<u64>, pdev: String, vram: Option<u64>, gtt: Option<u64>, shared: Option<u64>, cpu: Option<u64>, engines: Option<BTreeMap<String, u64>> }` and `drm::fdinfo::parse_fdinfo(&str) -> Option<FdInfo>` (`drm.py` lines 75-135).
  - `drm::cards::DrmCard { key: u64, pdev: String, sysfs: PathBuf, driver: String, vendor_id: u16, device_id: u16, name: String, vram_total: u64, integrated: bool, unified_video: bool }`, `drm::cards::cards(root: &Path, pci_ids: Option<&Path>) -> Vec<DrmCard>` (lines 137-261) and `drm::cards::adapters(&[DrmCard]) -> Vec<Adapter>` (lines 262-276).
  - `drm::device::read_device(card: &DrmCard, index: usize, kernel_release: &str) -> DeviceTelemetry` (lines 277-366), `vcn_major(device: &Path) -> i64` and `engine_key(name: &str, unified_video: bool) -> String` (lines 367-396).
  - Test support: `support::{make_card, card, write_fd, write_stat, fdinfo}`, ports of the `test_drm.py` helpers at lines 46-97 with the same parameters, plus `pub const FDINFO: &str` copied verbatim from `test_drm.py` lines 14-44.

- [ ] **Step 1: Write the test fixture builders**

`tests/support/mod.rs`:
```rust
#![allow(dead_code, clippy::unwrap_used)]
use gputaskman_linux::drm::cards::DrmCard;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

pub const GIB: u64 = 1 << 30;
pub const FDINFO: &str = "pos:\t0\nflags:\t02100002\nmnt_id:\t28\nino:\t415\ndrm-driver:\tamdgpu\n\
drm-client-id:\t26\ndrm-pdev:\t0000:03:00.0\npasid:\t25\ndrm-total-cpu:\t0\ndrm-shared-cpu:\t0\n\
drm-resident-cpu:\t0\ndrm-purgeable-cpu:\t0\ndrm-total-gtt:\t2 MiB\ndrm-shared-gtt:\t0\n\
drm-resident-gtt:\t2060 KiB\ndrm-purgeable-gtt:\t0\ndrm-total-vram:\t12 KiB\ndrm-shared-vram:\t0\n\
drm-resident-vram:\t0\ndrm-purgeable-vram:\t0\ndrm-memory-vram:\t0 KiB\ndrm-memory-gtt: \t2060 KiB\n\
drm-memory-cpu: \t0 KiB\namd-evicted-vram:\t12 KiB\namd-requested-vram:\t12 KiB\n\
amd-requested-gtt:\t2048 KiB\ndrm-engine-gfx:\t5308400172 ns\ndrm-engine-compute:\t0 ns\n\
drm-engine-enc:\t1200 ns\ndrm-engine-capacity-gfx:\t1\n";

pub fn make_card(root: &Path, name: &str, pdev: &str, vendor: u16, device: u16, vram: u64,
                 driver: &str, gtt: u64, apu: bool) -> PathBuf {
    let dev = root.join("drm").join(name).join("device");
    std::fs::create_dir_all(&dev).unwrap();
    std::fs::write(dev.join("vendor"), format!("0x{vendor:04x}\n")).unwrap();
    std::fs::write(dev.join("device"), format!("0x{device:04x}\n")).unwrap();
    std::fs::write(dev.join("mem_info_vram_total"), vram.to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_total"), gtt.to_string()).unwrap();
    if !apu { std::fs::write(dev.join("mem_info_vram_vendor"), "samsung\n").unwrap(); }
    let driver_dir = root.join("drivers").join(driver);
    std::fs::create_dir_all(&driver_dir).unwrap();
    symlink(&driver_dir, dev.join("driver")).unwrap();
    std::fs::write(dev.join("uevent"), format!("DRIVER={driver}\nPCI_SLOT_NAME={pdev}\n")).unwrap();
    dev
}

pub fn card(dev: &Path, pdev: &str, name: &str, vram: u64) -> DrmCard {
    DrmCard { key: gputaskman_core::model::pci_key(pdev), pdev: pdev.into(), sysfs: dev.into(), driver: "amdgpu".into(),
              vendor_id: 0x1002, device_id: 0x7550, name: name.into(), vram_total: vram,
              integrated: vram <= 4 * GIB, unified_video: false }
}

pub fn write_fd(proc_root: &Path, pid: u32, fd: u32, text: &str, target: &str) {
    let fdinfo = proc_root.join(pid.to_string()).join("fdinfo");
    std::fs::create_dir_all(&fdinfo).unwrap();
    std::fs::write(fdinfo.join(fd.to_string()), text).unwrap();
    let fds = proc_root.join(pid.to_string()).join("fd");
    std::fs::create_dir_all(&fds).unwrap();
    let link = fds.join(fd.to_string());
    if std::fs::symlink_metadata(&link).is_err() { symlink(target, &link).unwrap(); }
}

/// `/proc/<pid>/stat` with field 22 (start time) set, as `DrmCounters` reads it to spot a reused pid.
pub fn write_stat(proc_root: &Path, pid: u32, start_ticks: u64) {
    let dir = proc_root.join(pid.to_string());
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("stat"),
                   format!("{pid} (p) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 {start_ticks} 0 0")).unwrap();
}

#[allow(clippy::too_many_arguments)]
pub fn fdinfo(client: u64, pdev: &str, gfx: u64, vram: u64, gtt: u64, shared_vram: u64, shared_gtt: u64,
              shared_cpu: u64) -> String {
    format!("pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client}\ndrm-pdev:\t{pdev}\n\
             drm-shared-vram:\t{shared_vram} KiB\ndrm-resident-vram:\t{vram} KiB\n\
             drm-shared-gtt:\t{shared_gtt} KiB\ndrm-resident-gtt:\t{gtt} KiB\n\
             drm-shared-cpu:\t{shared_cpu} KiB\ndrm-resident-cpu:\t0\n\
             drm-engine-gfx:\t{gfx} ns\ndrm-engine-capacity-gfx:\t1\n")
}
```

- [ ] **Step 2: Write the failing tests**

`tests/drm_parse.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;
use gputaskman_linux::drm::{cards::cards, device::engine_key, fdinfo::*};
use support::*;

#[test]
fn parse_size_units() {
    assert_eq!(parse_size("2060 KiB"), Some(2060 * 1024));
    assert_eq!(parse_size("12"), Some(12));
    assert_eq!(parse_size("2 MiB"), Some(2 << 20));
    assert_eq!(parse_size("1 kB"), Some(1000));
    assert_eq!(parse_size("x KiB"), None);
    assert_eq!(parse_size("5 parsecs"), None);
    assert_eq!(parse_size(""), None);
}

#[test]
fn parse_fdinfo_example() {
    let info = parse_fdinfo(FDINFO).unwrap();
    assert_eq!((info.driver.as_str(), info.client_id, info.pdev.as_str()), ("amdgpu", Some(26), "0000:03:00.0"));
    assert_eq!((info.vram, info.gtt, info.shared), (Some(0), Some(2060 * 1024), Some(0)));
    let engines = info.engines.unwrap();
    assert_eq!(engines["gfx"], 5_308_400_172);
    assert_eq!(engines["enc"], 1200);
    assert!(!engines.contains_key("capacity-gfx"));
}

#[test]
fn parse_fdinfo_non_drm() {
    assert!(parse_fdinfo("pos:\t0\nflags:\t02\n").is_none());
}

#[test]
fn torn_engine_counter_is_unknown() {
    let text = FDINFO.replace("drm-engine-gfx:\t5308400172 ns", "drm-engine-gfx:\t");
    assert!(parse_fdinfo(&text).unwrap().engines.is_none());
}

#[test]
fn cards_lists_amdgpu_discrete_first() {
    let dir = tempfile::tempdir().unwrap();
    make_card(dir.path(), "card0", "0000:05:00.0", 0x1002, 0x1681, 512 << 20, "amdgpu", 0, true);
    make_card(dir.path(), "card1", "0000:03:00.0", 0x1002, 0x7550, 16 * GIB, "amdgpu", 0, false);
    make_card(dir.path(), "card2", "0000:01:00.0", 0x10de, 0x2684, 24 * GIB, "nvidia", 0, false);
    let found = cards(&dir.path().join("drm"), None);
    assert_eq!(found.iter().map(|c| c.pdev.as_str()).collect::<Vec<_>>(), vec!["0000:03:00.0", "0000:05:00.0"]);
    assert!(!found[0].integrated && found[1].integrated);
}

#[test]
fn engine_names() {
    // Port of test_drm.py EngineAndSelectionTests.test_engine_names (line 782)
    assert_eq!(engine_key("gfx", false), "3d");
    assert_eq!(engine_key("enc", true), "video");
}
```
Then port the rest of `test_drm.py`, keeping test names, into `drm_parse.rs`:
- `ParseTests` (lines 112-192): `test_parse_fdinfo_resident_fallback`, `test_parse_fdinfo_prefers_resident_keys`, `test_parse_fdinfo_shared_needs_all_memory_regions`, `test_parse_fdinfo_shared_cpu_counts_toward_private_gate`, `test_parse_fdinfo_unparseable_is_unknown`;
- `CardsTests.test_cards` (line 194) and `ReadDeviceTests.test_read_device` (line 229);
- from `DrmEdgeCaseTests`: `test_unified_vcn_reports_video`, `test_missing_sysfs_memory_is_unknown`, `test_power_average_of_zero_falls_back_to_input`, `test_vcn_major_from_ip_discovery` (lines 442-480);
- complete `engine_names` with every assertion from line 782.

Run: `cargo add -p gputaskman-linux --dev tempfile` and `cargo add -p gputaskman-linux regex`

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-linux --test drm_parse`
Expected: FAIL to compile.

- [ ] **Step 4: Implement**

Port `drm.py` lines 1-396. `read_device` sets `source = "AMD sysfs"`, and `driver` is filled by the backend as `kernel <release>`. Card names use `pci_ids` when given; otherwise the `/usr/share/hwdata/pci.ids` lookup in `_pci_ids`/`_card_name` (lines 164-221).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-linux --test drm_parse`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "Add amdgpu sysfs and DRM fdinfo parsing"
```

---

### Task 12: DRM per-process counters and the DRM backend

**Files:**
- Create: `src/drm/counters.rs`, `src/drm/backend.rs` in `gputaskman-linux`
- Test: `rust/crates/gputaskman-linux/tests/drm_counters.rs`

**Interfaces:**
- Consumes: Task 11, `gputaskman_core::{traits::*, model::*, provenance::*}`.
- Produces:
  - `drm::counters::DrmCounters` with `new(cards: Vec<DrmCard>, proc_root: PathBuf, clock: Arc<dyn Clock>)`, `start(&mut self)`, `set_budget(&mut self, Option<Duration>)`, `sample(&mut self) -> (BTreeMap<u64, BTreeMap<Pid, ProcGpu>>, BTreeMap<u64, AdapterGpu>)` and `status(&self) -> &str`. The default budget is `Some(50 ms)` (`drm.SCAN_BUDGET`).
  - `drm::backend::DrmBackend` implementing `GpuBackend`, with `new(sys_root: PathBuf, proc_root: PathBuf, clock: Arc<dyn Clock>, kernel_release: String)`. `adapters()` lists cards without scanning. `start()` keeps only the selected card, applies the policy (`Unbounded` means no budget) and takes the engine baseline. `sample()` returns the selected card's procs and adapter totals, with `status` set from `DrmCounters::status` (`None` when empty). `driver()` returns `Some(format!("kernel {release}"))`. `cuda()` returns `None`.
- Mapping from Python `ProcGpu` to the Rust one:
  - attributed → `vram = Measured(resident_vram)`, `shared = Measured(resident_gtt)`;
  - a reason → `vram = shared = Unavailable(reason)`;
  - `committed = Unavailable(NotOnBackend)`;
  - `resident = Some(Resident { vram, gtt })`, each `Measured` or `Unavailable(Incomplete or Legacy)`;
  - engines unknown → `Engines::Unavailable` with `KfdUntimed` (kfd holder), `Incomplete`, `PartialScan`, `SharedClient`, `TornCounter` or `NoBaseline`, checked in that order, following the condition order on `drm.py` lines 637-643.

- [ ] **Step 1: Write the failing tests**

`tests/drm_counters.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;
use gputaskman_core::{model::*, provenance::*, testing::FakeClock};
use gputaskman_linux::drm::counters::DrmCounters;
use std::{sync::Arc, time::Duration};
use support::*;

const PDEV: &str = "0000:03:00.0";

fn setup() -> (tempfile::TempDir, DrmCounters, Arc<FakeClock>, u64) {
    let dir = tempfile::tempdir().unwrap();
    let dev = make_card(dir.path(), "card0", PDEV, 0x1002, 0x7550, 16 * GIB, "amdgpu", 0, false);
    std::fs::write(dev.join("mem_info_vram_used"), (2 * GIB).to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_used"), (GIB).to_string()).unwrap();
    let c = card(&dev, PDEV, "AMD GPU", 16 * GIB);
    let key = c.key;
    let clock = Arc::new(FakeClock::new(0.0));
    std::fs::create_dir_all(dir.path().join("proc")).unwrap();
    let counters = DrmCounters::new(vec![c], dir.path().join("proc"), clock.clone());
    (dir, counters, clock, key)
}

#[test]
fn private_render_client_is_measured_and_engine_load_needs_two_samples() {
    let (dir, mut counters, clock, key) = setup();
    let proc_root = dir.path().join("proc");
    write_fd(&proc_root, 100, 5, &fdinfo(1, PDEV, 0, 1024, 2048, 0, 0, 0), "/dev/dri/renderD128");
    write_stat(&proc_root, 100, 555);
    let (procs, adapters) = counters.sample();
    let g = &procs[&key][&100];
    assert_eq!(g.vram, Bytes::measured(1024 * 1024));
    assert_eq!(g.shared, Bytes::measured(2048 * 1024));
    assert!(matches!(g.engines, Engines::Unavailable { reason: Reason::NoBaseline }));
    assert_eq!(adapters[&key].vram_used, Bytes::measured(2 * GIB));

    clock.advance(Duration::from_secs(1));
    std::fs::write(proc_root.join("100/fdinfo/5"), fdinfo(1, PDEV, 500_000_000, 1024, 2048, 0, 0, 0)).unwrap();
    let (procs, _) = counters.sample();
    assert_eq!(procs[&key][&100].engines.get("3d"), 50.0);
}

#[test]
fn any_shared_buffer_leaves_the_process_unattributed() {
    let (dir, mut counters, _clock, key) = setup();
    write_fd(&dir.path().join("proc"), 100, 5, &fdinfo(1, PDEV, 0, 1024, 0, 4, 0, 0), "/dev/dri/renderD128");
    let (procs, _) = counters.sample();
    let g = &procs[&key][&100];
    assert_eq!(g.vram, Bytes::unavailable(Reason::SharedBuffers));
    assert_eq!(g.resident.unwrap().vram, Bytes::measured(1024 * 1024)); // reference only
}

#[test]
fn client_held_by_several_processes_is_charged_to_none() {
    let (dir, mut counters, _clock, key) = setup();
    let p = dir.path().join("proc");
    write_fd(&p, 100, 5, &fdinfo(7, PDEV, 0, 1024, 0, 0, 0, 0), "/dev/dri/renderD128");
    write_fd(&p, 101, 9, &fdinfo(7, PDEV, 0, 1024, 0, 0, 0, 0), "/dev/dri/renderD128");
    let (procs, _) = counters.sample();
    for pid in [100, 101] {
        assert_eq!(procs[&key][&pid].vram, Bytes::unavailable(Reason::SharedClient));
    }
}

#[test]
fn primary_node_is_not_charged() {
    let (dir, mut counters, _clock, key) = setup();
    write_fd(&dir.path().join("proc"), 100, 5, &fdinfo(1, PDEV, 0, 1024, 0, 0, 0, 0), "/dev/dri/card0");
    let (procs, _) = counters.sample();
    assert_eq!(procs[&key][&100].vram, Bytes::unavailable(Reason::Primary));
}
```
Every fixture process needs a `stat` file whose field 22 is its start time, as the Python `_start_time` reads it. Call `support::write_stat` (Task 11) for every pid in every test, including the three tests above (add `write_stat(&p, 100, 1)` and so on after their `write_fd` calls).

Then port the remaining `test_drm.py` tests, keeping their names, using `FakeClock` for both the engine clock and the budget timer:
- `DrmCountersTests` (lines 283-363): `test_drm_counters`, `test_constructor_does_not_scan`, `test_rocm_process_engine_load_is_unknown`;
- `DrmEdgeCaseTests`: `test_old_kernel_without_shared_keys_is_not_summed`, `test_unparseable_memory_is_unknown`, `test_engine_counter_stepping_back_is_not_counted_twice`;
- every `DrmScanTests` case (lines 496-750). The Python `Clock` helper at line 99 advances on each call; give `FakeClock` an `auto_advance(Duration)` option that does the same, so the budget tests behave identically. Unreadable fds and fdinfo in `test_unreadable_*` are made with `std::fs::set_permissions(.., 0o000)`. Skip those two when running as root (`libc::geteuid() == 0`), as permission bits don't apply to root.

Add a backend test: `DrmBackend::start` with `ScanPolicy::Unbounded` lists every process (a `partial_scan` reason never appears), and `sample().status` reports an unreadable `/proc` root.

Run: `cargo add -p gputaskman-linux libc`. Under `[dev-dependencies]` of `gputaskman-linux`, add `gputaskman-core = { path = "../gputaskman-core", features = ["test-support"] }` and `gputaskman-linux = { path = ".", features = ["test-support"] }`, and declare `[features] test-support = []` in its `Cargo.toml` (Task 13's `FakeNvml` and `FakeNamespace` sit behind it).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-linux --test drm_counters`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Port `DrmCounters` (`drm.py` lines 405-664) exactly:
- the cursor rotation, with new processes first;
- the first process of each refresh listed in full;
- the budget checked every 64 fds;
- cached fds reused only while field 22 of `stat` matches;
- the client key `(pdev, client_id)`, or `(pdev, pid, fd)` without an ID;
- the render-node flag;
- the reason order (lines 616-628);
- the high-water engine counters and the 100% clamp;
- the baseline prune (`_PRUNE_NS`);
- the status note wording.

`mono()` from `Clock` replaces both `time.monotonic_ns` and `time.perf_counter`. A `/proc/<pid>/fd` that returns `PermissionDenied` counts as denied. Any other error means the process is gone.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-linux --test drm_counters`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add DRM fdinfo per-process counters and backend"
```

---

### Task 13: NVML backend and PID-namespace matching

**Files:**
- Create: `src/nvml/mod.rs`, `src/nvml/api.rs`, `src/nvml/real.rs`, `src/nvml/backend.rs`, `src/pidns.rs` in `gputaskman-linux`
- Test: `rust/crates/gputaskman-linux/tests/nvml.rs`, `tests/pidns.rs`

**Interfaces:**
- Produces:
  - `nvml::api::NvProcess { pid: u32, used: Option<u64> }` and `NvUtilSample { pid: u32, sm: u32, enc: u32, dec: u32, timestamp: u64 }`.
  - `nvml::api::NvmlApi: Send`, with: `device_count() -> Result<u32, String>`, `pci_bus_id(i) -> Result<String, String>`, `name(i) -> Result<String, String>`, `memory(i) -> Result<(u64, u64), String>` (total, used), `compute_processes(i) -> Result<Vec<NvProcess>, String>`, `graphics_processes(i) -> Result<Vec<NvProcess>, String>`, `process_utilization(i, since: u64) -> Result<Vec<NvUtilSample>, String>`, `telemetry(i) -> DeviceTelemetry` (port of `nvml.py` `Nvml.sample`, lines 54-101, with `source = "NVML"`), `driver_version() -> Option<String>`, `cuda_version() -> Option<String>` (formatted `major.minor` as in Python line 47).
  - `nvml::api::FakeNvml` (behind feature `test-support`): a builder with `device(bus, name, total, used)`, `compute(i, Vec<NvProcess>)`, `graphics(i, ...)`, `samples(i, Vec<NvUtilSample>)`, `fail_handle(i)`, and `util_calls() -> Vec<u64>`.
  - `nvml::real::RealNvml::load() -> Result<RealNvml, String>` over `nvml-wrapper` (`cargo add -p gputaskman-linux nvml-wrapper`). `usedGpuMemory` of `UsedGpuMemory::Unavailable`, or any value ≥ 2^63, becomes `None` (`linux.py` `_value`).
  - `pidns::PidMatcher` with `update(&mut self, host: &[u32], local: &[u32]) -> (BTreeMap<u32, (u32, Confidence)>, Vec<u32>)` (lines 177-208).
  - `pidns::NamespaceProbe` trait: `in_child_pid_namespace() -> Option<bool>`, `nvidia_holders() -> BTreeSet<u32>`, `has_cuda(pid) -> bool`, `exists(pid) -> bool`, `self_pid() -> u32`. `ProcNamespace { proc_root }` implements it (lines 112-175). Tests get a `FakeNamespace`.
  - `nvml::backend::NvmlBackend` implementing `GpuBackend`, with `new(api: Box<dyn NvmlApi>, ns: Box<dyn NamespaceProbe>)`. When `RealNvml::load()` fails, `linux_backends()` (Task 14) builds no NVML backend and records `NvmlNotLoaded`.
- Mapping: `used = Some(n)` gives `vram = Measured(n)` (maximum over the compute and graphics lists); `None` gives `vram = Unavailable(NvmlHidden)`. Shared and committed are `Unavailable(NotOnBackend)`. Engines are `Measured` with keys `compute` or `3d`, plus `encode` and `decode`. Matched container rows keep `vram` as `Measured` and set `owner = Some(confidence)`. Unmatched host PIDs go to `BackendSample::outside`.

- [ ] **Step 1: Write the failing tests**

`tests/pidns.rs`, a direct port of `test_linux.py` `ContainerTests` (lines 213-289):
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::Confidence;
use gputaskman_linux::pidns::PidMatcher;

#[test]
fn host_pids_are_matched_to_container_pids_by_creation_order() {
    let mut m = PidMatcher::default();
    let (out, outside) = m.update(&[5000, 5100], &[20, 30]);
    assert_eq!(out[&5100], (30, Confidence::Consistent));
    assert_eq!(out[&5000], (20, Confidence::Consistent));
    assert!(outside.is_empty());
}

#[test]
fn extra_outside_compute_is_a_guess_until_a_start_event() {
    let mut m = PidMatcher::default();
    let (out, outside) = m.update(&[5000, 5100], &[30]);
    assert_eq!(out[&5100], (30, Confidence::Order));
    assert_eq!(outside, vec![5000]);
    let (out, _) = m.update(&[5000, 5100, 5200], &[30, 40]);
    assert_eq!(out[&5200], (40, Confidence::Consistent)); // appeared together
}
```
Port the remaining `ContainerTests` cases with their exact inputs: `test_bare_metal_is_unchanged`, `test_a_host_pid_equal_to_a_local_pid_is_still_a_host_pid`, `test_namespace_detection` (with a fake `/proc/self/ns/pid` symlink in a tempdir root) and `test_outside_rows_never_borrow_a_local_identity`.

`tests/nvml.rs` ports `test_linux.py` `LinuxCountersTests` for native Linux only (lines 124-212), using `FakeNvml` and `FakeNamespace`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{model::*, provenance::*, traits::*};
use gputaskman_linux::nvml::{api::*, backend::NvmlBackend};
use gputaskman_linux::pidns::FakeNamespace;

const BUS: &str = "00000000:01:00.0";

fn backend(fake: FakeNvml) -> NvmlBackend {
    let ns = FakeNamespace::bare_metal([100, 101, 200, 300, 555]);
    let mut b = NvmlBackend::new(Box::new(fake), Box::new(ns));
    let adapter = b.adapters().remove(0);
    b.start(&adapter, ScanPolicy::Unbounded);
    b
}

fn p(pid: u32, used: Option<u64>) -> NvProcess { NvProcess { pid, used } }
fn s(pid: u32, sm: u32, dec: u32, ts: u64) -> NvUtilSample { NvUtilSample { pid, sm, enc: 0, dec, timestamp: ts } }

#[test]
fn native_linux_attributes_vram_and_engines_per_pid() {
    let fake = FakeNvml::default().device(BUS, "NVIDIA GeForce RTX 3090 Ti", 24 * GIB, 4 * GIB)
        .compute(0, vec![p(100, Some(3 * GIB)), p(200, Some(GIB))])
        .graphics(0, vec![p(200, Some(GIB)), p(300, Some(256 << 20))])
        .samples(0, vec![s(100, 80, 0, 5), s(100, 90, 0, 9), s(300, 30, 20, 7)]);
    let calls = fake.util_calls_handle();
    let mut b = backend(fake);
    let first = b.sample();
    b.sample();
    assert_eq!(first.adapter.vram_used, Bytes::measured(4 * GIB));
    assert_eq!(first.procs[&100].vram, Bytes::measured(3 * GIB));
    assert_eq!(first.procs[&200].vram, Bytes::measured(GIB)); // in both lists: counted once
    assert_eq!(first.procs[&100].engines.get("compute"), 90.0);
    assert_eq!(first.procs[&300].engines.get("3d"), 30.0);
    assert_eq!(first.procs[&300].engines.get("decode"), 20.0);
    assert_eq!(calls.lock().unwrap().clone(), vec![0, 9]);
}

#[test]
fn hidden_per_process_memory_is_unknown_not_zero() {
    let fake = FakeNvml::default().device(BUS, "GPU", 24 * GIB, GIB).compute(0, vec![p(100, None), p(101, None)]);
    let sample = backend(fake).sample();
    for pid in [100, 101] {
        assert_eq!(sample.procs[&pid].vram, Bytes::unavailable(Reason::NvmlHidden));
    }
}

#[test]
fn nvml_device_matches_linux_pci_key() {
    let fake = FakeNvml::default().device(BUS, "GPU", 24 * GIB, GIB);
    let b = NvmlBackend::new(Box::new(fake), Box::new(FakeNamespace::bare_metal([])));
    assert_eq!(b.adapters()[0].key, pci_key(BUS));
}
```
`FakeNvml::util_calls_handle()` returns an `Arc<Mutex<Vec<u64>>>` that the fake pushes `since` into. Also port `test_utilization_for_unlisted_pids_is_ignored`, `test_adapters_come_from_nvml`, `test_a_failing_gpu_is_skipped_not_fatal` and `test_other_users_processes_stay_local_on_bare_metal`, plus a container-namespace test: with `FakeNamespace::child(...)`, two host PIDs become one matched local row with `owner == Some(Order)` and one `outside` row.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-linux --test pidns --test nvml`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Port `linux.py` lines 40-356 (non-WSL paths only) and `nvml.py`:
- `NvmlBackend::adapters` ports `_nvml_adapters`, skipping devices whose handle, name or memory read fails.
- `sample` ports `LinuxCounters.sample` and `_resolve_namespaces`, including the per-device `_last_ts` utilisation cursor.
- `device(index)` returns `telemetry(i)` for the selected device. It is selected by PCI bus key, never by name or index (`collector.select_nv_device`, Linux branch).
- `ProcNamespace` reads `/proc/self/ns/pid` (compared with `PROC_PID_INIT_INO = 0xEFFFFFFC`), fd links starting `/dev/nvidia`, and `maps` containing `/libcuda.so`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-linux --test pidns --test nvml`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add NVML backend and container PID matching"
```

---

### Task 14: Linux process source and backend discovery

**Files:**
- Create: `rust/crates/gputaskman-linux/src/procs.rs`
- Modify: `rust/crates/gputaskman-linux/src/lib.rs` (add `pub fn linux_backends(clock: Arc<dyn Clock>) -> (Vec<Box<dyn GpuBackend>>, Vec<String>)`)
- Test: `rust/crates/gputaskman-linux/tests/procs.rs`

**Interfaces:**
- Produces:
  - `procs::LinuxProcs` with `new(proc_root: PathBuf)` and `with_clk_tck(proc_root, ticks_per_sec: u64)`, implementing `ProcessSource`:
    - `start_time` is field 22 of `stat` plus `btime` from `<root>/stat`, giving `wall = btime + ticks / clk_tck`;
    - `name` follows psutil: `comm` from `stat`, and when it is 15 characters long, the basename of `cmdline[0]` if that starts with it;
    - `exe` is the `exe` link;
    - `cmdline` is split on NUL, with trailing empty strings removed;
    - `parent` is field 4 of `stat`;
    - `memory_maps` parses `smaps` and sums `Rss` per pathname (psutil's grouping);
    - `listening_ports` reads state `0A` from `<root>/net/tcp` and `net/tcp6` and maps socket inodes to pids via `fd` links `socket:[inode]`;
    - `self_pid` is `std::process::id()`.
  - `linux_backends(clock)` returns `(backends, notes)`: an NVML backend when `RealNvml::load()` succeeds (otherwise a note `"NVML not loaded: <error>"`), and a DRM backend when `/sys/class/drm` has amdgpu cards. DRM adapters whose key equals an NVML adapter's key are dropped (`linux.enumerate_adapters`).

- [ ] **Step 1: Write the failing tests**

`tests/procs.rs` builds a fake `/proc` in a tempdir:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::traits::ProcessSource;
use gputaskman_linux::procs::LinuxProcs;
use std::os::unix::fs::symlink;

fn proc_tree() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let r = dir.path();
    std::fs::write(r.join("stat"), "cpu  1 2 3\nbtime 1700000000\n").unwrap();
    let p = r.join("4242");
    std::fs::create_dir_all(p.join("fd")).unwrap();
    std::fs::write(p.join("stat"), "4242 (llama-server) S 1 4242 4242 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 12345 0 0").unwrap();
    std::fs::write(p.join("cmdline"), b"/opt/llama-server\0-m\0/m/q.gguf\0").unwrap();
    symlink("/opt/llama-server", p.join("exe")).unwrap();
    std::fs::write(p.join("smaps"), "7f00-7f10 r--p 00000000 00:00 1 /m/q.gguf\nRss:  400 kB\n\
        7f10-7f20 r--p 00000000 00:00 1 /m/q.gguf\nRss:  600 kB\n").unwrap();
    symlink("socket:[999]", p.join("fd/7")).unwrap();
    std::fs::create_dir_all(r.join("net")).unwrap();
    std::fs::write(r.join("net/tcp"), "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n\
   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 999 1\n").unwrap();
    std::fs::write(r.join("net/tcp6"), "  sl  local_address\n").unwrap();
    dir
}

#[test]
fn reads_basics_maps_and_ports() {
    let dir = proc_tree();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    let b = procs.basics(4242).unwrap();
    assert_eq!(b.name.as_deref(), Some("llama-server"));
    assert_eq!(b.exe.as_deref(), Some("/opt/llama-server"));
    assert_eq!(b.cmdline.unwrap(), vec!["/opt/llama-server", "-m", "/m/q.gguf"]);
    let start = b.start.unwrap();
    assert_eq!((start.ticks, start.wall), (12345, 1_700_000_000.0 + 123.45));
    assert_eq!(procs.parent(4242), Some(1));
    let maps = procs.memory_maps(4242).unwrap();
    assert_eq!((maps[0].path.as_str(), maps[0].rss), ("/m/q.gguf", 1000 * 1024));
    assert_eq!(procs.listening_ports()[&4242], vec![8080]);
    assert_eq!(procs.pids(), vec![4242]);
}

#[test]
fn process_names_with_spaces_and_parens_parse() {
    let dir = proc_tree();
    std::fs::write(dir.path().join("4242/stat"), "4242 (a (b) c) S 7 0 0 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 5 0 0").unwrap();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    assert_eq!(procs.parent(4242), Some(7));
    assert_eq!(procs.start_time(4242).unwrap().ticks, 5);
}

#[test]
fn missing_process_is_none_not_a_panic() {
    let dir = proc_tree();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    assert!(procs.basics(1).is_none());
    assert!(procs.memory_maps(1).is_err());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-linux --test procs`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

`stat` is parsed by splitting at the last `)` (as `drm._start_time` does). `LinuxProcs::new` reads `CLK_TCK` with `unsafe { libc::sysconf(libc::_SC_CLK_TCK) }`, falling back to 100 when the result is ≤ 0. Write a comment noting it is the only `unsafe` block in the crate. Local ports in `net/tcp` are hex in the second half of `local_address`. A `PermissionDenied` on `exe` leaves `exe: None` but keeps `name`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-linux --test procs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add /proc process source and Linux backend discovery"
```

---

### Task 15: Collector, rows and snapshots

**Files:**
- Create: `rust/crates/gputaskman-core/src/collector.rs`
- Test: `rust/crates/gputaskman-core/tests/collector.rs`

**Interfaces:**
- Consumes: Tasks 2-10.
- Produces:
  - `enum Origin { Local, Outside }` (serde snake_case).
  - `struct ProcRow { pid: Pid, name: String, kind: String, gpu: ProcGpu, detail: ProcDetail, llm: Option<LlmView>, handy: Option<HandyEvidence>, origin: Origin }` with `key() -> String`, `pid_label() -> String`, `runtime() -> String` and `inside() -> String` (port of `collector.py` lines 44-102, without the WSL and Windows branches).
  - `struct Snapshot { taken: f64, adapter: Option<Adapter>, adapters: Vec<Adapter>, device: Option<DeviceTelemetry>, driver: Option<String>, cuda: Option<String>, vram_total: Bytes, vram_used: Bytes, shared_used: Bytes, procs: Vec<ProcRow>, backend: Option<BackendKind>, process_status: Option<String>, shared_is_spill: bool, notes: Vec<String> }` with `attributed(&self) -> Bytes` (delegates to `attributed_total` over every row).
  - `struct CollectorDeps { backends: Vec<Box<dyn GpuBackend>>, procs: Box<dyn ProcessSource>, http: Box<dyn HttpGet>, clock: Arc<dyn Clock>, paths: Paths, notes: Vec<String> }`.
  - `struct Collector` with `new(deps: CollectorDeps, gpu_index: usize, policy: ScanPolicy) -> Result<Collector, SelectError>`, `sample(&mut self) -> Snapshot` and `adapters(&self) -> &[Adapter]`. The constants are `DEEP_TOP_N = 25` and `SCAN_BUDGET = 150 ms`.
  - `fn list_adapters(backends: &[Box<dyn GpuBackend>]) -> Vec<Adapter>` (ordered, with software adapters excluded).

- [ ] **Step 1: Write the failing tests**

`tests/collector.rs`, using `FakeBackend`, `FakeProcs`, `FakeHttp` and `FakeClock`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{collector::*, env::Paths, model::*, provenance::*, testing::*, traits::*};
use std::sync::Arc;

fn nv_adapter(key: u64) -> Adapter {
    Adapter { key, pci_bus: "00000000:01:00.0".into(), name: "RTX".into(), vendor_id: VENDOR_NVIDIA,
              dedicated_vram: 24 * GIB, shared_system: 0, integrated: false, backend: BackendKind::Nvml }
}
fn amd_adapter(key: u64) -> Adapter {
    Adapter { key, pci_bus: "0000:03:00.0".into(), name: "Radeon".into(), vendor_id: VENDOR_AMD,
              dedicated_vram: 16 * GIB, shared_system: 0, integrated: false, backend: BackendKind::Drm }
}

fn deps(backends: Vec<Box<dyn GpuBackend>>, procs: FakeProcs, dir: &std::path::Path) -> CollectorDeps {
    CollectorDeps { backends, procs: Box::new(procs), http: Box::new(FakeHttp::default()),
                    clock: Arc::new(FakeClock::new(1000.0)), paths: Paths::rooted(dir), notes: vec![] }
}

#[test]
fn only_the_selected_backend_is_started() {
    let dir = tempfile::tempdir().unwrap();
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]);
    let drm = FakeBackend::new(BackendKind::Drm, vec![amd_adapter(2)]);
    let (nv_starts, drm_starts) = (nv.starts(), drm.starts());
    let c = Collector::new(deps(vec![Box::new(nv), Box::new(drm)], FakeProcs::default(), dir.path()), 0,
                           ScanPolicy::Unbounded).unwrap();
    assert_eq!(c.adapters().len(), 2); // mixed vendors are selectable
    assert_eq!((*nv_starts.lock().unwrap(), *drm_starts.lock().unwrap()), (1, 0));
}

#[test]
fn rows_are_live_and_sorted_by_shown_vram_with_outside_rows_kept_apart() {
    let dir = tempfile::tempdir().unwrap();
    let mut sample = BackendSample::default();
    sample.procs.insert(10, ProcGpu::measured(GIB));
    sample.procs.insert(11, ProcGpu::measured(3 * GIB));
    sample.procs.insert(12, ProcGpu::measured(10)); // below 1 MiB: not live
    sample.outside.insert(5000, ProcGpu::measured(2 * GIB));
    sample.adapter.vram_used = Bytes::measured(6 * GIB);
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let procs = FakeProcs::default().named(10, "a").named(11, "b").named(12, "c");
    let mut c = Collector::new(deps(vec![Box::new(nv)], procs, dir.path()), 0, ScanPolicy::Unbounded).unwrap();
    let snap = c.sample();
    let pids: Vec<_> = snap.procs.iter().map(|r| (r.pid, r.origin)).collect();
    assert_eq!(pids, vec![(11, Origin::Local), (10, Origin::Local), (5000, Origin::Outside)]);
    assert_eq!(snap.procs[2].name, "(outside this container)");
    assert_eq!(snap.procs[2].kind, "external");
    assert_eq!(snap.attributed(), Bytes::measured(6 * GIB));
    assert_eq!(snap.procs[2].key(), "o5000");
    assert_eq!(snap.procs[2].pid_label(), "H:5000");
}

#[test]
fn no_supported_gpu_gives_an_empty_snapshot_with_a_reason() {
    let dir = tempfile::tempdir().unwrap();
    let mut d = deps(vec![], FakeProcs::default(), dir.path());
    d.notes.push("NVML not loaded: libnvidia-ml.so not found".into());
    let mut c = Collector::new(d, 0, ScanPolicy::Unbounded).unwrap();
    let snap = c.sample();
    assert!(snap.adapter.is_none() && snap.procs.is_empty());
    assert!(matches!(snap.vram_total, Bytes::Unavailable { .. }));
    assert_eq!(snap.notes, vec!["NVML not loaded: libnvidia-ml.so not found"]);
}

#[test]
fn out_of_range_gpu_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]);
    let err = Collector::new(deps(vec![Box::new(nv)], FakeProcs::default(), dir.path()), 3, ScanPolicy::Unbounded)
        .err().unwrap();
    assert_eq!(err.to_string(), "--gpu 3 is out of range; 1 GPU adapter(s) available");
}
```
Also add these tests:
- `amd_snapshot_does_not_show_cuda_header`: a DRM fake backend gives `cuda == None` and `driver == Some("kernel ...")`.
- `deep_scans_stop_at_the_budget`: a `FakeClock` with `auto_advance(100 ms)`, three live rows and `ScanPolicy::Budgeted`. Only the first rows reach `maps_time > 0`, and every row reaches it under `Unbounded`.
- `apu_gtt_counts_as_held_memory`: an integrated adapter gives `shared_is_spill == false`.
- `llm_view_built_for_llm_rows`: an `llama-server` row with mapped GGUF evidence gets `llm: Some(..)`, and `FakeHttp` saw only its exclusive ports.
- `handy_rows_get_evidence`: a row whose `exe` basename is `handy` gets `handy` from `HandyProbe`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test collector`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

`Collector::new`:
1. Build the adapter list from every backend, ordered by `ordered_adapters`, remembering which backend owns each one.
2. Select the adapter (`select_adapter`).
3. Keep only the owning backend, dropping the others.
4. Call `backend.start(&adapter, policy)`.

`sample` follows `collector.py` lines 240-334 without the WSL and Windows branches:
- `vram_total`: `device.mem_total` if known, else the adapter's `dedicated_vram`, else `Unavailable(NotOnBackend)`.
- Live rows sorted by `shown_vram` descending (`None` last), then the outside rows.
- Deep scans of the top 25 plus anything at 2% GPU or more, while the deadline lasts. `Unbounded` means no deadline.
- Re-classify after scanning, and detect Ollama runners.
- Handy, then LLM views (own exclusive ports, then the parents' exclusive ports).
- Prune.

`notes` carries `deps.notes` unchanged. `taken` is `clock.wall()`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test collector`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add collector, process rows and snapshots"
```

---

### Task 16: JSON schema v2 and the text report

**Files:**
- Create: `src/schema/mod.rs`, `src/schema/v2.rs`, `src/report.rs` in `gputaskman-core`
- Test: `rust/crates/gputaskman-core/tests/report.rs`

**Interfaces:**
- Produces:
  - `schema::SCHEMA_VERSION: u32 = 2`.
  - `schema::v2::{SnapshotV2, ProcessV2, MemoryV2, StatusV2, ModelFileV2, LlmV2, LlmModelV2, SegmentV2}`: serde structs matching spec section 6 field for field. Optional objects (`handy`, `llm`, `device`) serialise as `null`, never omitted.
  - `schema::v2::from_snapshot(&Snapshot) -> SnapshotV2` and `to_json(&Snapshot) -> String` (pretty, two-space indent).
  - `SnapshotV2` derives `Deserialize` and `PartialEq` (the parity tests compare values).
  - `report::to_text(&Snapshot, limit: usize) -> String`, a port of `report.py` `to_text` (lines 148-252) for the `nvml` and `drm` backends, including `vram_label` markers, `DRM_NOTES`, `DRM_RESIDENT_NOTE`, `INFERRED_NOTE` and `ORDER_NOTE`, verbatim.

Rules in `from_snapshot`:
- `resident` is set only for DRM local rows. Outside rows get `vram` and `gtt` as `Unavailable(OutsideNamespace)` with `summable: false`.
- `engines` and `gpu_pct`: `busiest` is rounded to one decimal; `by_engine` drops zero entries and rounds (`report.py` line 115).
- `owner_inferred_by` comes from `ProcGpu::owner`.
- `memory.attributed` comes from `Snapshot::attributed()`.

- [ ] **Step 1: Write the failing tests**

`tests/report.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{collector::*, model::*, provenance::*, report::to_text, schema::v2::*};
use serde_json::json;

fn snap(backend: BackendKind, rows: Vec<ProcRow>) -> Snapshot {
    Snapshot { taken: 1.0, adapter: None, adapters: vec![], device: None, driver: None, cuda: None,
               vram_total: Bytes::measured(24 * GIB), vram_used: Bytes::measured(7 * GIB),
               shared_used: Bytes::unavailable(Reason::NotOnBackend), procs: rows, backend: Some(backend),
               process_status: None, shared_is_spill: true, notes: vec![] }
}
fn row(pid: Pid, gpu: ProcGpu) -> ProcRow {
    ProcRow { pid, name: "python3".into(), kind: "AI".into(), gpu, detail: ProcDetail::new(pid, "python3"),
              llm: None, handy: None, origin: Origin::Local }
}

#[test]
fn measured_zero_and_unknown_stay_apart() {
    let hidden = ProcGpu { vram: Bytes::unavailable(Reason::NvmlHidden), ..ProcGpu::measured(0) };
    let v = serde_json::to_value(from_snapshot(&snap(BackendKind::Nvml, vec![row(1, ProcGpu::measured(0)), row(2, hidden)]))).unwrap();
    assert_eq!(v["schema_version"], 2);
    assert_eq!(v["processes"][0]["vram"], json!({"status": "measured", "bytes": 0}));
    assert_eq!(v["processes"][1]["vram"], json!({"status": "unavailable", "reason": "nvml_hidden"}));
    assert_eq!(v["processes"][0]["committed"], json!({"status": "unavailable", "reason": "not_on_backend"}));
    assert_eq!(v["processes"][0]["engines"], json!({"status": "measured", "busiest": 0.0, "by_engine": {}}));
    assert_eq!(v["memory"]["attributed"], json!({"status": "measured", "bytes": 0}));
}

#[test]
fn native_hidden_vram_does_not_hide_measured_engine_load() {
    let mut g = ProcGpu { vram: Bytes::unavailable(Reason::NvmlHidden), ..ProcGpu::measured(0) };
    g.engines = Engines::Measured { by_engine: [("compute".to_string(), 47.0)].into() };
    let s = snap(BackendKind::Nvml, vec![row(4242, g)]);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["processes"][0]["engines"]["busiest"], 47.0);
    let text = to_text(&s, 30);
    let line = text.lines().find(|l| l.contains("4242")).unwrap();
    assert!(line.contains("n/a") && line.contains("47"), "{line}");
}

#[test]
fn drm_unattributed_rows_void_the_total_and_explain_why() {
    let mut g = ProcGpu::measured(0);
    g.vram = Bytes::unavailable(Reason::SharedBuffers);
    g.shared = Bytes::unavailable(Reason::SharedBuffers);
    g.resident = Some(Resident { vram: Bytes::measured(GIB), gtt: Bytes::measured(0) });
    let s = snap(BackendKind::Drm, vec![row(9, g)]);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["memory"]["attributed"], json!({"status": "unavailable", "reason": "shared_buffers"}));
    assert_eq!(v["processes"][0]["resident"]["summable"], false);
    assert!(to_text(&s, 30).contains("shares buffers with other clients"));
}

#[test]
fn json_round_trips() {
    let s = snap(BackendKind::Nvml, vec![row(1, ProcGpu::measured(GIB))]);
    let v2 = from_snapshot(&s);
    let back: SnapshotV2 = serde_json::from_str(&to_json(&s)).unwrap();
    assert_eq!(back, v2);
}
```
Also port the report assertions from these Python tests, translating null checks into `unavailable` checks and keeping the text assertions verbatim:
- `test_drm.py` `DrmReportTests.test_json_and_text_keep_unknown_apart_from_zero` (line 752);
- `test_linux.py` `UnattributedReportTests.test_measured_zero_is_not_changed_to_unknown` (nvml part only) and `test_model_detail_header_uses_inferred_or_unknown_vram`;
- `test_core.py` `MemoryTests.test_text_shows_unavailable_device_metrics_without_fabricating_zeroes` (line 362);
- `test_core.py` `HandyTests.test_handy_evidence_is_visible_in_text_and_json` (line 492);
- `test_core.py` `IdentityTests.test_non_gguf_hints_are_visible_in_reports` (line 670).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test report`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Write the v2 structs exactly as in spec section 6 and the mapping in `from_snapshot`. Port `to_text` and the helper functions it uses from `report.py` (lines 20-90 and 148-252), dropping the WSL, Windows and macOS branches.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test report`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add JSON schema v2 and text report"
```

---

### Task 17: v1 reader and snapshot diff

**Files:**
- Create: `src/schema/v1.rs`, `src/diff.rs` in `gputaskman-core`
- Test: `rust/crates/gputaskman-core/tests/diff.rs`, `tests/v1.rs`

**Interfaces:**
- Produces:
  - `schema::v1::from_v1(&serde_json::Value) -> Result<SnapshotV2, String>`, following the mapping table in spec section 6.
  - `schema::load(path: &Path) -> Result<SnapshotV2, String>`: decodes UTF-16 LE/BE with a BOM and UTF-8 with or without a BOM (`diff.py` `load`, lines 23-33), then reads v2 when `schema_version == 2`, else v1. The error messages match Python.
  - `diff::{diff(before: &SnapshotV2, after: &SnapshotV2) -> Result<DiffResult, String>, to_text(&DiffResult) -> String}`, where `DiffResult` is serde-serialisable for `--json`. This ports `diff.py` lines 34-254: matching by PID, then name and ports, then a unique name; measured buckets only in `vram`/`shared`; runtime-reported and estimated segments in their own sections; an absent row is `not listed`; and the same noise threshold.

- [ ] **Step 1: Write the failing tests**

`tests/v1.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{provenance::*, schema::v1::from_v1};
use serde_json::json;

#[test]
fn v1_rows_map_to_provenance() {
    let v1 = json!({"taken": 1.0, "gpu": "RTX", "backend": "nvml", "vram_total": 100, "vram_used": 50,
        "shared_used": null, "attributed": 30, "shared_is_spill": true, "device": null, "processes": [
        {"pid": 1, "host": null, "name": "a", "kind": "AI", "runtimes": [], "vram_attributed": true,
         "vram_bytes": 30, "vram_inferred_bytes": null, "vram_inferred_by": "order", "shared_bytes": null,
         "committed_bytes": null, "paged_out_bytes": null, "gpu_pct": 5.0, "engines": {"3d": 5.0},
         "ports": [], "inside": "", "model_files": [{"path": "/m", "size": 1, "resident": 0, "gguf": true,
         "sources": ["mapped", "api"], "store_name": null, "kind": "LLM", "kind_hint": ""}], "installed_models": []},
        {"pid": 2, "host": "outside", "name": "x", "kind": "external", "runtimes": [], "vram_attributed": false,
         "vram_bytes": null, "vram_inferred_bytes": 7, "vram_inferred_by": "consistent", "shared_bytes": null,
         "committed_bytes": null, "paged_out_bytes": null, "gpu_pct": null, "engines": null,
         "vram_unattributed_reason": null, "ports": [], "inside": "", "model_files": [], "installed_models": []}]});
    let s = from_v1(&v1).unwrap();
    assert_eq!(s.processes[0].vram, Bytes::measured(30));
    assert_eq!(s.processes[0].owner_inferred_by, Some(Confidence::Order));
    assert_eq!(s.processes[0].model_files[0].evidence.len(), 2);
    assert_eq!(s.processes[1].vram, Bytes::Inferred { bytes: 7, by: Confidence::Consistent });
    assert_eq!(s.processes[0].shared, Bytes::unavailable(Reason::V1Unknown));
    assert_eq!(s.memory.attributed, Bytes::measured(30));
    assert_eq!(s.memory.shared_used, Bytes::unavailable(Reason::V1Unknown));
}
```
`tests/diff.rs` ports all of `tests/test_diff.py`: the `proc`, `snap` and `llm` helpers become functions that build v1 `serde_json::Value`s with the same fields, and every test goes through `from_v1` and then `diff`. Keep each test name and assertion. Add one Rust-only test, `mixed_v1_and_v2_inputs_compare`: a v2 snapshot from Task 16's `from_snapshot` diffed against the same data written as v1 reports no measured change.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman-core --test v1 --test diff`
Expected: FAIL to compile.

- [ ] **Step 3: Implement**

Write `from_v1` following the spec table, with `exact: true` segments mapped to `Reported { by: LoadLog }` and others to `Estimated { from: Gguf }`. Then port `diff.py`, operating on `SnapshotV2`. Only `Measured` values feed the measured buckets; a bucket with `Unavailable` on either side reports `unknown` as Python does for `None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman-core --test v1 --test diff`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "Add v1 snapshot reader and snapshot diff"
```

---

### Task 18: Command-line interface

**Files:**
- Modify: `rust/crates/gputaskman/src/main.rs`
- Test: `rust/crates/gputaskman/tests/cli.rs`

**Interfaces:**
- Consumes: `Collector`, `linux_backends`, `LinuxProcs`, `UreqGet`, `SystemClock`, `schema`, `report`, `diff`.
- Produces: the binary `gputaskman` with `--once`, `--json`, `--gpu N`, `--list-gpus`, `-i/--interval SECONDS` (finite and > 0), and the subcommand `diff BEFORE AFTER [--json]`. Help and descriptions are copied from `cli.py`. A hidden `--proc-root`/`--sys-root` pair is read from the environment variables `GPUTASKMAN_PROC_ROOT` and `GPUTASKMAN_SYS_ROOT`, for tests only. Without `--once`/`--json` it prints `the TUI is not available yet in the Rust build; use --once or --json` to stderr and exits 2.

- [ ] **Step 1: Write the failing tests**

`tests/cli.rs`:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used)]
use assert_cmd::Command;

fn cmd(dir: &std::path::Path) -> Command {
    let mut c = Command::cargo_bin("gputaskman").unwrap();
    std::fs::create_dir_all(dir.join("proc")).unwrap();
    std::fs::write(dir.join("proc/stat"), "btime 1700000000\n").unwrap();
    std::fs::create_dir_all(dir.join("sys/class/drm")).unwrap();
    c.env("GPUTASKMAN_PROC_ROOT", dir.join("proc")).env("GPUTASKMAN_SYS_ROOT", dir.join("sys"))
     .env("HOME", dir).env("LD_LIBRARY_PATH", dir); // no libnvidia-ml.so can be found
    c
}

#[test]
fn help_needs_no_gpu() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path()).arg("--help").assert().success().stdout(predicates::str::contains("--json"));
}

#[test]
fn invalid_options_fail_before_collecting() {
    let dir = tempfile::tempdir().unwrap();
    for bad in [["--interval", "0"], ["--interval", "nan"], ["--gpu", "-1"]] {
        cmd(dir.path()).args(bad).arg("--once").assert().code(2);
    }
}

#[test]
fn json_without_a_supported_gpu_is_valid_v2() {
    let dir = tempfile::tempdir().unwrap();
    let out = cmd(dir.path()).arg("--json").assert().success().get_output().stdout.clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["schema_version"], 2);
    assert!(v["adapter"].is_null());
    assert!(v["processes"].as_array().unwrap().is_empty());
}

#[test]
fn out_of_range_gpu_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path()).args(["--gpu", "2", "--once"]).assert().code(2)
        .stderr(predicates::str::contains("--gpu 2 is out of range; 0 GPU adapter(s) available"));
}

#[test]
fn list_gpus_with_none_is_empty_and_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path()).arg("--list-gpus").assert().success().stdout("");
}

#[test]
fn diff_text_json_and_errors() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    std::fs::write(&a, r#"{"taken":1,"gpu":"RTX","backend":"nvml","vram_used":0,"shared_used":0,"attributed":0,"processes":[]}"#).unwrap();
    cmd(dir.path()).args(["diff"]).arg(&a).arg(&a).assert().success();
    cmd(dir.path()).args(["diff", "--json"]).arg(&a).arg(&a).assert().success();
    cmd(dir.path()).args(["diff"]).arg(&a).arg(dir.path().join("missing.json")).assert().code(2);
}
```
Run: `cargo add -p gputaskman clap --features derive && cargo add -p gputaskman serde_json && cargo add -p gputaskman --dev assert_cmd predicates tempfile serde_json`

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p gputaskman --test cli`
Expected: FAIL.

- [ ] **Step 3: Implement `main.rs`**

- Parse with clap derive. The `diff` subcommand mirrors `cli.diff_main`. Argument errors exit with code 2, as argparse does.
- `--list-gpus` prints `{i}: {name} ({vendor}, key 0x{key:016x})` for each ordered adapter.
- `--once`/`--json` build `CollectorDeps` from `linux_backends`, `LinuxProcs::new(proc_root)`, `UreqGet::default()`, `SystemClock` and `Paths::from_env()`, with `ScanPolicy::Unbounded`. Then sleep 1 s (engine rates need two samples, as `cli.py` line 80 does), call `sample()`, and print `to_json` or `to_text(.., 30)`.
- On a target other than Linux, the binary prints `gputaskman's Rust build supports native Linux only in this milestone` and exits 1.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p gputaskman --test cli`
Expected: PASS.

- [ ] **Step 5: Smoke-test on this machine**

Run: `cargo run -p gputaskman -- --once; cargo run -p gputaskman -- --json | head -20; cargo run -p gputaskman -- --list-gpus`
Expected: exit code 0. The Intel-only machine has no supported adapter, so the text output says the GPU is unavailable and why (an `NVML not loaded` note), and `--list-gpus` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "Add gputaskman command-line interface for Linux"
```

---

### Task 19: Parity harness, README and CI

**Files:**
- Create: `rust/parity/run_python.py`, `rust/parity/scenarios/{drm_private,drm_shared,nvml_bare_metal,nvml_container,report_mixed}.json`
- Create: `rust/crates/gputaskman-linux/tests/parity.rs`, `rust/crates/gputaskman-core/tests/parity_report.rs`
- Modify: `rust/crates/gputaskman-linux/Cargo.toml` and `gputaskman-core/Cargo.toml` (feature `parity = []`)
- Create: `rust/README.md`
- Modify: `.github/workflows/ci.yml` (add a `rust` job)

**Interfaces:**
- Scenario format (JSON):
  - `{"kind": "drm", "cards": [{name, pdev, vendor, device, vram, apu}], "samples": [[{pid, start, fds: [{fd, target, fdinfo: {client, gfx, vram, gtt, shared_vram, shared_gtt, shared_cpu}}]}]], "advance_ms": 1000}`
  - `{"kind": "nvml", "devices": [{bus, name, total, used}], "compute": [[pid, used|null]], "graphics": [...], "samples": [[pid, sm, enc, dec, ts]], "namespace": "bare"|"child", "holders": [...], "cuda": [...]}`
  - `{"kind": "report", "backend": "nvml"|"drm", "rows": [{pid, name, kind, dedicated, attributed, engines, drm_reason, resident_vram, resident_gtt, owner}]}`
- `run_python.py <scenario.json>` prints JSON on stdout.
  - drm and nvml scenarios: the per-sample per-pid results `{pid: {vram, shared, reason, engines}}`, using `null` for unknown.
  - report scenarios: `report.to_dict(snapshot)`.
  - It imports `gputaskman` from `src/` and the fakes from `tests/test_drm.py` and `tests/test_linux.py` (`_make_card`, `_write_fd`, `_fdinfo`, `FakeNvml`), and patches exactly as those tests do.

- [ ] **Step 1: Write the harness and scenarios**

`rust/parity/run_python.py` (run with `uv run --locked python rust/parity/run_python.py <file>` from the repo root):
```python
"""Run one parity scenario through the Python implementation and print the result as JSON."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from gputaskman import drm, linux  # noqa: E402
from gputaskman.collector import ProcRow, Snapshot  # noqa: E402
from gputaskman.pdh import ProcGpu  # noqa: E402
from gputaskman.procinfo import ProcDetail  # noqa: E402
from gputaskman.report import to_dict  # noqa: E402
from tests.test_drm import _make_card, _write_fd, _fdinfo  # noqa: E402
from tests.test_linux import FakeNvml, proc, sample  # noqa: E402


def gpu_result(g: ProcGpu) -> dict:
    return {"vram": int(g.dedicated) if g.attributed and "dedicated" in g.available else None,
            "shared": int(g.shared) if g.attributed and "shared" in g.available else None,
            "reason": g.drm_reason or None, "owner": g.inferred_by or None,
            "engines": {k: round(v, 1) for k, v in g.engines.items() if v} if "engine" in g.available else None}


def run_drm(s: dict) -> list:
    out = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for c in s["cards"]:
            _make_card(root, c["name"], c["pdev"], c["vendor"], c["device"], c["vram"], apu=c.get("apu", False))
        found = drm.cards(str(root / "drm"))
        ns = [0]
        counters = drm.DrmCounters(found, proc=str(root / "proc"), clock=lambda: ns[0], timer=lambda: 0.0)
        counters.scan_budget = None
        for step in s["samples"]:
            for p in step:
                (root / "proc" / str(p["pid"])).mkdir(parents=True, exist_ok=True)
                (root / "proc" / str(p["pid"]) / "stat").write_text(
                    f"{p['pid']} (p) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 {p['start']} 0 0")
                for fd in p["fds"]:
                    f = fd["fdinfo"]
                    _write_fd(root / "proc", p["pid"], fd["fd"],
                              _fdinfo(f["client"], found[0].pdev, f.get("gfx", 0), f.get("vram", 0), f.get("gtt", 0),
                                      f.get("shared_vram", 0), f.get("shared_gtt", 0), f.get("shared_cpu", 0)),
                              target=fd["target"])
            procs, _ = counters.sample()
            out.append({str(pid): gpu_result(g) for pid, g in procs[found[0].key].items()})
            ns[0] += s.get("advance_ms", 1000) * 1_000_000
    return out


def run_nvml(s: dict) -> list:
    dev = s["devices"][0]
    fake = FakeNvml(compute=[proc(p, u if u is not None else None) for p, u in s.get("compute", [])],
                    graphics=[proc(p, u) for p, u in s.get("graphics", [])],
                    samples=[sample(p, sm, enc=enc, dec=dec, ts=ts) for p, sm, enc, dec, ts in s.get("samples", [])],
                    used=dev["used"])
    child = s.get("namespace") == "child"
    with patch.object(linux, "N", fake), patch.object(linux, "is_wsl", return_value=False), \
            patch.object(linux, "in_child_pid_namespace", return_value=child), \
            patch.object(linux, "nvidia_holders", return_value=set(s.get("holders", []))), \
            patch.object(linux, "has_cuda", side_effect=lambda pid: pid in set(s.get("cuda", []))):
        c = linux.LinuxCounters()
        procs, _ = c.sample()
    key = next(iter(procs), None)
    local = {str(pid): gpu_result(g) for pid, g in procs.get(key, {}).items()} if key is not None else {}
    outside = {str(pid): gpu_result(g) for pid, g in c.outside.get(key, {}).items()} if key is not None else {}
    return [{"local": local, "outside": outside}]


def run_report(s: dict) -> dict:
    rows = []
    for r in s["rows"]:
        g = ProcGpu(dedicated=r.get("dedicated", 0), attributed=r.get("attributed", True),
                    available={"dedicated", "shared", "engine"} if s["backend"] == "drm" else
                    {"dedicated", "shared", "committed", "engine"})
        for k, v in (r.get("engines") or {}).items():
            g.engines[k] = v
        if r.get("drm_reason"):
            g.drm_reason = r["drm_reason"]
            g.attributed = False
            g.available -= {"dedicated", "shared"}
        g.resident_vram, g.resident_gtt = r.get("resident_vram"), r.get("resident_gtt")
        g.inferred_by = r.get("owner") or ""
        rows.append(ProcRow(r["pid"], r["name"], r["kind"], g, ProcDetail(r["pid"], r["name"])))
    snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=s.get("vram_total", 0),
                    vram_used=s.get("vram_used"), shared_used=s.get("shared_used"), procs=rows, backend=s["backend"])
    return to_dict(snap)


if __name__ == "__main__":
    scenario = json.loads(Path(sys.argv[1]).read_text())
    run = {"drm": run_drm, "nvml": run_nvml, "report": run_report}[scenario["kind"]]
    print(json.dumps(run(scenario), sort_keys=True))
```
Check each imported helper's signature against `tests/test_drm.py` and `tests/test_linux.py` (for example, whether `proc()` accepts `None` for hidden memory) and adjust the call sites so they match.

Write the five scenario files:
- `drm_private.json`: one private render client, two samples with a 500 ms gfx delta;
- `drm_shared.json`: a shared client held by two pids, plus a process with a shared buffer;
- `nvml_bare_metal.json`: the inputs of `test_native_linux_attributes_vram_and_engines_per_pid`;
- `nvml_container.json`: child namespace, two host pids, one local CUDA process;
- `report_mixed.json`: one NVML-hidden row, one measured zero row and one engine-only row.

- [ ] **Step 2: Write the failing parity tests**

`rust/crates/gputaskman-linux/tests/parity.rs` (compiled only with `--features parity`):
```rust
#![cfg(feature = "parity")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap() }

fn python(scenario: &Path) -> serde_json::Value {
    let out = Command::new("uv").args(["run", "--locked", "python", "rust/parity/run_python.py"]).arg(scenario)
        .current_dir(repo_root()).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn backends_match_python() {
    for name in ["drm_private", "drm_shared", "nvml_bare_metal", "nvml_container"] {
        let path = repo_root().join(format!("rust/parity/scenarios/{name}.json"));
        let expected = python(&path);
        let actual = gputaskman_linux::parity::run_scenario(&path); // same JSON shape as gpu_result()
        assert_eq!(actual, expected, "{name}");
    }
}
```
Add `pub mod parity` (behind `#[cfg(feature = "parity")]`) to `gputaskman-linux` with `run_scenario(&Path) -> serde_json::Value`. It builds the same tree (reusing the fixture builders from `tests/support`, moved into `src/parity.rs` as functions), runs `DrmCounters` with an unbounded budget or `NvmlBackend` with `FakeNvml`/`FakeNamespace`, and converts each `ProcGpu` into the `gpu_result` shape: `vram`/`shared` are the `Measured` value or null, `reason` is the reason code, `owner` is the confidence, and `engines` is the rounded non-zero map or null.

`rust/crates/gputaskman-core/tests/parity_report.rs` (`#![cfg(feature = "parity")]`) runs `report_mixed.json` through Python, converts the output with `from_v1`, builds the same rows in Rust (one `ProcGpu` per scenario row, with the mapping from Task 2), and asserts `from_v1(python) == from_snapshot(rust)` after setting both `taken` values to `0.0`.

- [ ] **Step 3: Run the parity tests**

Run: `cargo test --workspace --features gputaskman-linux/parity,gputaskman-core/parity`
Expected: PASS. If a scenario differs, the Rust side is wrong unless the Python output contradicts a provenance rule; in that case stop and report it rather than copying the bug.

- [ ] **Step 4: Write `rust/README.md`**

The README covers:
- **Build and run** (`cargo build --release`, `target/release/gputaskman --once`).
- **Status:** milestone 1, native Linux only, schema v2 unstable.
- **Validation:** a table with one row per backend (NVML bare metal, NVML in a container, amdgpu DRM), all marked **not validated on hardware**, with "development machine: Intel i915 only (unsupported)".
- **Live-check checklist**, copied from spec section 9.
- **Rule-to-test map:** one row per `AGENTS.md` rule 1-10 naming the Rust tests that cover it, for example: rule 1 → `only_measured_bytes_reach_the_total`, `attribution_total_rules`, `provenance_misuse_does_not_compile`; rule 2 → `measured_zero_is_not_unknown`, `hidden_per_process_memory_is_unknown_not_zero`, `measured_zero_and_unknown_stay_apart`; rule 6 → `local_url_is_loopback_only`, `ureq_get_is_a_get_and_only_accepts_200`, `probes_are_get_only_on_the_given_port`; rule 8 → "NVML is loaded at runtime by nvml-wrapper; no library is vendored". Fill every rule from the tests that exist by then.
- **Parity:** how to run the parity tests (`uv` required).

Write in British English, with no mention of tooling vendors beyond the crates used.

- [ ] **Step 5: Add the CI job**

Append to `.github/workflows/ci.yml` under `jobs:`:
```yaml
  rust:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, ubuntu-latest, macos-latest]
    defaults:
      run:
        working-directory: rust
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - name: Test core (all platforms)
        run: cargo test -p gputaskman-core
      - name: Test Linux crate and CLI
        if: runner.os == 'Linux'
        run: cargo test --workspace
      - name: Clippy
        if: runner.os == 'Linux'
        run: cargo clippy --workspace --all-targets -- -D warnings
      - name: Clippy (core)
        if: runner.os != 'Linux'
        run: cargo clippy -p gputaskman-core --all-targets -- -D warnings
      - uses: astral-sh/setup-uv@v6
        if: runner.os == 'Linux'
        with:
          version: "0.9.26"
          python-version: "3.13"
      - name: Parity with Python
        if: runner.os == 'Linux'
        run: |
          (cd .. && uv sync --locked)
          cargo test --workspace --features gputaskman-linux/parity,gputaskman-core/parity
```

- [ ] **Step 6: Run everything locally**

Run, from `rust/`: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --features gputaskman-linux/parity,gputaskman-core/parity`
Then, from the repo root: `uv run --locked python -m unittest discover -s tests -q`
Expected: all pass, and the Python suite is unchanged (229 tests OK).

- [ ] **Step 7: Commit and push**

```bash
git add rust/ .github/workflows/ci.yml
git commit -m "Add Python parity tests, Rust README and CI job"
git push origin rust-port
```

---

## Self-review notes

- **Spec coverage:**
  - §1 scope → Tasks 11-18;
  - §2 layout → Task 1;
  - §3 types → Tasks 1-2;
  - §4 traits → Task 3;
  - §5 flow → Task 15;
  - §6 schema and the v1 reader → Tasks 16-17;
  - §7 degradation → Tasks 3, 12, 13, 14, 15, 18;
  - §8 testing → every task, plus parity in Task 19;
  - §9 validation → Task 19 README;
  - §10 risks → the `NvmlApi` trait in Task 13 and the `/proc` parsers in Task 14.
- **Type consistency:** `ProcGpu::measured`, `Bytes::measured`, `Engines::get`, `Shown::get`, `ScanPolicy::Unbounded`, `FakeClock::advance`/`auto_advance`, `FakeBackend::new(..).script(..)`/`starts()` and `FakeProcs::with`/`named`/`ports` are each defined once (Tasks 1-3) and used with the same signatures later. Add any fake builder method a task needs to `testing.rs` in that task.

//! Runs a parity scenario (`rust/parity/scenarios/*.json`) through the Rust backends and prints the
//! per-process results in the shape `rust/parity/run_python.py` prints for Python. The DRM tree is
//! built with the same builders as Python's `tests/test_drm.py`; NVML runs on `FakeNvml` and
//! `FakeNamespace`, the Rust ports of the fakes in `tests/test_linux.py`.
use crate::drm::cards::cards;
use crate::drm::counters::DrmCounters;
use crate::fixtures::{fdinfo, make_card, write_fd, write_stat};
use crate::nvml::api::{FakeNvml, NvProcess, NvUtilSample};
use crate::nvml::backend::NvmlBackend;
use crate::pidns::FakeNamespace;
use gputaskman_core::model::{Engines, Pid, ProcGpu};
use gputaskman_core::testing::FakeClock;
use gputaskman_core::traits::{GpuBackend, ScanPolicy};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[derive(Deserialize)]
struct DrmCardSpec {
    name: String,
    pdev: String,
    vendor: u16,
    device: u16,
    vram: u64,
    #[serde(default)]
    apu: bool,
}

#[derive(Deserialize)]
struct FdInfoSpec {
    client: u64,
    #[serde(default)]
    gfx: u64,
    #[serde(default)]
    vram: u64,
    #[serde(default)]
    gtt: u64,
    #[serde(default)]
    shared_vram: u64,
    #[serde(default)]
    shared_gtt: u64,
    #[serde(default)]
    shared_cpu: u64,
}

#[derive(Deserialize)]
struct FdSpec {
    fd: u32,
    target: String,
    fdinfo: FdInfoSpec,
}

#[derive(Deserialize)]
struct ProcSpec {
    pid: Pid,
    start: u64,
    fds: Vec<FdSpec>,
}

#[derive(Deserialize)]
struct DrmScenario {
    cards: Vec<DrmCardSpec>,
    samples: Vec<Vec<ProcSpec>>,
    #[serde(default = "default_advance_ms")]
    advance_ms: u64,
}

fn default_advance_ms() -> u64 {
    1000
}

#[derive(Deserialize)]
struct NvDeviceSpec {
    bus: String,
    name: String,
    total: u64,
    used: u64,
}

#[derive(Deserialize)]
struct NvmlScenario {
    devices: Vec<NvDeviceSpec>,
    #[serde(default)]
    compute: Vec<(u32, Option<u64>)>,
    #[serde(default)]
    graphics: Vec<(u32, Option<u64>)>,
    /// `[pid, sm, enc, dec, timestamp]`.
    #[serde(default)]
    samples: Vec<(u32, u32, u32, u32, u64)>,
    namespace: String,
    #[serde(default)]
    holders: Vec<u32>,
    #[serde(default)]
    cuda: Vec<u32>,
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// One row in the harness's `gpu_result` shape: measured bytes or null, the amdgpu reason code
/// (null on NVML, where Python has no reason field), the owner confidence, and the non-zero engines
/// rounded to one decimal, or null when engine load is unknown. `shared` is a per-process counter
/// only on amdgpu (GTT); NVML has none, so it is null there.
fn gpu_result(g: &ProcGpu, drm: bool) -> Value {
    let measured = |b: &gputaskman_core::provenance::Bytes| b.as_measured().map(|m| m.get());
    let engines = match &g.engines {
        Engines::Measured { by_engine } => {
            let map: Map<String, Value> = by_engine
                .iter()
                .filter(|(_, v)| **v != 0.0)
                .map(|(k, v)| (k.clone(), json!(round1(*v))))
                .collect();
            Value::Object(map)
        }
        Engines::Unavailable { .. } => Value::Null,
    };
    json!({
        "vram": measured(&g.vram),
        "shared": if drm { measured(&g.shared) } else { None },
        "reason": if drm { g.vram.reason().map(|r| r.code()) } else { None },
        "owner": g.owner,
        "engines": engines,
    })
}

fn rows(procs: &BTreeMap<Pid, ProcGpu>, drm: bool) -> Value {
    Value::Object(
        procs
            .iter()
            .map(|(pid, g)| (pid.to_string(), gpu_result(g, drm)))
            .collect(),
    )
}

fn io(e: std::io::Error) -> String {
    e.to_string()
}

fn run_drm(s: DrmScenario) -> Result<Value, String> {
    let dir = tempfile::tempdir().map_err(io)?;
    let root = dir.path();
    for c in &s.cards {
        make_card(
            root, &c.name, &c.pdev, c.vendor, c.device, c.vram, "amdgpu", 0, c.apu,
        )
        .map_err(io)?;
    }
    let found = cards(&root.join("drm"), None);
    let first = found.first().ok_or("the scenario has no amdgpu card")?;
    let (key, pdev) = (first.key, first.pdev.clone());
    let proc_root = root.join("proc");
    std::fs::create_dir_all(&proc_root).map_err(io)?;
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(found, proc_root.clone(), clock.clone());
    counters.set_budget(None);
    let mut out = Vec::new();
    for step in &s.samples {
        for p in step {
            write_stat(&proc_root, p.pid, p.start).map_err(io)?;
            for fd in &p.fds {
                let f = &fd.fdinfo;
                let text = fdinfo(
                    f.client,
                    &pdev,
                    f.gfx,
                    f.vram,
                    f.gtt,
                    f.shared_vram,
                    f.shared_gtt,
                    f.shared_cpu,
                );
                write_fd(&proc_root, p.pid, fd.fd, &text, &fd.target).map_err(io)?;
            }
        }
        let (procs, _) = counters.sample();
        out.push(procs.get(&key).map_or(json!({}), |p| rows(p, true)));
        clock.advance(Duration::from_millis(s.advance_ms));
    }
    Ok(Value::Array(out))
}

fn run_nvml(s: NvmlScenario) -> Result<Value, String> {
    let dev = s.devices.first().ok_or("the scenario has no NVML device")?;
    let procs = |list: &[(u32, Option<u64>)]| {
        list.iter()
            .map(|&(pid, used)| NvProcess { pid, used })
            .collect::<Vec<_>>()
    };
    let samples = s
        .samples
        .iter()
        .map(|&(pid, sm, enc, dec, timestamp)| NvUtilSample {
            pid,
            sm,
            enc,
            dec,
            timestamp,
        })
        .collect();
    let fake = FakeNvml::default()
        .device(&dev.bus, &dev.name, dev.total, dev.used)
        .compute(0, procs(&s.compute))
        .graphics(0, procs(&s.graphics))
        .samples(0, samples);
    let ns = match s.namespace.as_str() {
        "bare" => FakeNamespace::bare_metal(s.holders.clone()).with_cuda(s.cuda.clone()),
        "child" => FakeNamespace::child(s.holders.clone(), s.cuda.clone()),
        other => return Err(format!("unknown namespace '{other}'")),
    };
    let mut backend = NvmlBackend::new(Box::new(fake), Box::new(ns));
    let adapter = backend
        .adapters()
        .into_iter()
        .next()
        .ok_or("NVML lists no adapter")?;
    backend.start(&adapter, ScanPolicy::Unbounded);
    let sample = backend.sample();
    Ok(json!([{
        "local": rows(&sample.procs, false),
        "outside": rows(&sample.outside, false),
    }]))
}

fn parse<T: DeserializeOwned>(v: Value) -> Result<T, String> {
    serde_json::from_value(v).map_err(|e| e.to_string())
}

/// The per-sample, per-pid results of one scenario, in the shape `run_python.py` prints.
pub fn run_scenario(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    match value.get("kind").and_then(Value::as_str) {
        Some("drm") => run_drm(parse(value)?),
        Some("nvml") => run_nvml(parse(value)?),
        other => Err(format!("not a backend scenario: kind {other:?}")),
    }
}

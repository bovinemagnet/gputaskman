#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{model::*, provenance::*, traits::*};
use gputaskman_linux::nvml::real::RealNvml;
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

fn p(pid: u32, used: Option<u64>) -> NvProcess {
    NvProcess { pid, used }
}
fn s(pid: u32, sm: u32, dec: u32, ts: u64) -> NvUtilSample {
    NvUtilSample {
        pid,
        sm,
        enc: 0,
        dec,
        timestamp: ts,
    }
}

#[test]
fn native_linux_attributes_vram_and_engines_per_pid() {
    let fake = FakeNvml::default()
        .device(BUS, "NVIDIA GeForce RTX 3090 Ti", 24 * GIB, 4 * GIB)
        .compute(0, vec![p(100, Some(3 * GIB)), p(200, Some(GIB))])
        .graphics(0, vec![p(200, Some(GIB)), p(300, Some(256 << 20))])
        .samples(
            0,
            vec![s(100, 80, 0, 5), s(100, 90, 0, 9), s(300, 30, 20, 7)],
        );
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
    assert_eq!(
        first.procs[&100].shared,
        Bytes::unavailable(Reason::NotOnBackend)
    );
    assert_eq!(
        first.procs[&100].committed,
        Bytes::unavailable(Reason::NotOnBackend)
    );
    assert_eq!(calls.lock().unwrap().clone(), vec![0, 9]);
}

#[test]
fn hidden_per_process_memory_is_unknown_not_zero() {
    let fake = FakeNvml::default()
        .device(BUS, "GPU", 24 * GIB, GIB)
        .compute(0, vec![p(100, None), p(101, None)]);
    let sample = backend(fake).sample();
    for pid in [100, 101] {
        assert_eq!(
            sample.procs[&pid].vram,
            Bytes::unavailable(Reason::NvmlHidden)
        );
    }
}

#[test]
fn nvml_device_matches_linux_pci_key() {
    let fake = FakeNvml::default().device(BUS, "GPU", 24 * GIB, GIB);
    let b = NvmlBackend::new(Box::new(fake), Box::new(FakeNamespace::bare_metal([])));
    assert_eq!(b.adapters()[0].key, pci_key(BUS));
}

#[test]
fn utilization_for_unlisted_pids_is_ignored() {
    let fake = FakeNvml::default()
        .device(BUS, "GPU", 24 * GIB, GIB)
        .compute(0, vec![p(100, Some(GIB))])
        .samples(0, vec![s(555, 50, 0, 1)]);
    let sample = backend(fake).sample();
    assert!(!sample.procs.contains_key(&555));
}

#[test]
fn adapters_come_from_nvml() {
    let fake = FakeNvml::default().device(BUS, "NVIDIA GeForce RTX 3090 Ti", 24 * GIB, 4 * GIB);
    let b = NvmlBackend::new(Box::new(fake), Box::new(FakeNamespace::bare_metal([])));
    let a = &b.adapters()[0];
    assert_eq!(
        (
            a.key,
            a.name.as_str(),
            a.dedicated_vram,
            a.vendor_id,
            a.backend
        ),
        (
            pci_key(BUS),
            "NVIDIA GeForce RTX 3090 Ti",
            24 * GIB,
            VENDOR_NVIDIA,
            BackendKind::Nvml
        )
    );
    assert_eq!(a.pci_bus, BUS);
    assert_eq!(b.kind(), BackendKind::Nvml);
}

#[test]
fn a_failing_gpu_is_skipped_not_fatal() {
    let fake = FakeNvml::default()
        .device("00000000:02:00.0", "lost", 8 * GIB, 0)
        .device(BUS, "GPU", 24 * GIB, GIB)
        .fail_handle(0);
    let b = NvmlBackend::new(Box::new(fake), Box::new(FakeNamespace::bare_metal([])));
    assert_eq!(b.adapters().len(), 1);
    assert_eq!(b.adapters()[0].key, pci_key(BUS));
    let b = NvmlBackend::new(
        Box::new(FakeNvml::default().device(BUS, "GPU", GIB, 0).fail_count()),
        Box::new(FakeNamespace::bare_metal([])),
    );
    assert!(b.adapters().is_empty());
}

#[test]
fn other_users_processes_stay_local_on_bare_metal() {
    let fake = FakeNvml::default()
        .device(BUS, "GPU", 24 * GIB, 20 * GIB)
        .compute(0, vec![p(6000, Some(20 * GIB))]);
    let ns = FakeNamespace::bare_metal([100]).with_cuda([100]);
    let mut b = NvmlBackend::new(Box::new(fake), Box::new(ns));
    let a = b.adapters().remove(0);
    b.start(&a, ScanPolicy::Unbounded);
    let sample = b.sample();
    assert_eq!(sample.procs[&6000].vram, Bytes::measured(20 * GIB));
    assert_eq!(sample.procs[&6000].owner, None);
    assert!(!sample.procs.contains_key(&100));
    assert!(sample.outside.is_empty());
}

#[test]
fn container_host_pids_match_local_rows_and_leftovers_go_outside() {
    let fake = FakeNvml::default()
        .device(BUS, "GPU", 24 * GIB, 4 * GIB)
        .compute(0, vec![p(5000, Some(GIB)), p(5100, Some(3 * GIB))])
        .graphics(0, vec![p(1145, Some(58 << 20))])
        .samples(0, vec![s(5100, 95, 0, 3)]);
    // 564 holds the devices but has no CUDA context (nvidia-smi); only 30 is a local CUDA process.
    let ns = FakeNamespace::child([30, 564], [30]);
    let mut b = NvmlBackend::new(Box::new(fake), Box::new(ns));
    let a = b.adapters().remove(0);
    b.start(&a, ScanPolicy::Unbounded);
    let sample = b.sample();
    assert_eq!(sample.procs.len(), 1);
    assert_eq!(sample.procs[&30].vram, Bytes::measured(3 * GIB));
    assert_eq!(sample.procs[&30].owner, Some(Confidence::Order));
    assert_eq!(sample.procs[&30].engines.get("compute"), 95.0);
    let outside: Vec<u32> = sample.outside.keys().copied().collect();
    assert_eq!(outside, vec![1145, 5000]);
}

#[test]
fn device_telemetry_is_selected_by_the_started_adapter() {
    let fake = FakeNvml::default()
        .device("00000000:02:00.0", "Other", 8 * GIB, 0)
        .device(BUS, "Mine", 24 * GIB, GIB);
    let mut b = NvmlBackend::new(Box::new(fake), Box::new(FakeNamespace::bare_metal([])));
    assert_eq!(b.device(0), Err(Reason::NotOnBackend)); // nothing started yet
    let mine = b.adapters().into_iter().find(|a| a.name == "Mine").unwrap();
    b.start(&mine, ScanPolicy::Unbounded);
    let t = b.device(0).unwrap();
    assert_eq!((t.name.as_str(), t.source.as_str()), ("Mine", "NVML"));
    assert_eq!(t.pci_bus, BUS);
}

#[test]
fn loading_the_real_library_never_panics() {
    // This machine may have no NVIDIA driver: either outcome is fine, a panic is not.
    let _ = RealNvml::load();
}

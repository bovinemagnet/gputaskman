#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support {
    pub mod gguf_writer;
}
use gputaskman_core::{collector::*, env::Paths, model::*, provenance::*, testing::*, traits::*};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use support::gguf_writer::{V, write_gguf};

fn nv_adapter(key: u64) -> Adapter {
    Adapter {
        key,
        pci_bus: "00000000:01:00.0".into(),
        name: "RTX".into(),
        vendor_id: VENDOR_NVIDIA,
        dedicated_vram: 24 * GIB,
        shared_system: 0,
        integrated: false,
        backend: BackendKind::Nvml,
    }
}
fn amd_adapter(key: u64) -> Adapter {
    Adapter {
        key,
        pci_bus: "0000:03:00.0".into(),
        name: "Radeon".into(),
        vendor_id: VENDOR_AMD,
        dedicated_vram: 16 * GIB,
        shared_system: 0,
        integrated: false,
        backend: BackendKind::Drm,
    }
}

fn deps(
    backends: Vec<Box<dyn GpuBackend>>,
    procs: FakeProcs,
    dir: &std::path::Path,
) -> CollectorDeps {
    CollectorDeps {
        backends,
        procs: Box::new(procs),
        http: Box::new(FakeHttp::default()),
        clock: Arc::new(FakeClock::new(1000.0)),
        paths: Paths::rooted(dir),
        notes: vec![],
    }
}

/// A one-tensor causal LLM GGUF (two layers, 4096 context).
fn llm_gguf(dir: &Path, file: &str) -> PathBuf {
    let path = dir.join(file);
    write_gguf(
        &path,
        &[
            ("general.architecture", V::Str("qwen3")),
            ("qwen3.block_count", V::U32(2)),
            ("qwen3.context_length", V::U32(4096)),
        ],
        &[("blk.0.w", &[100], 0)],
    );
    let len = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len.div_ceil(32) * 32 + 400)
        .unwrap();
    path
}

fn mapped(path: &Path) -> Vec<MapEntry> {
    vec![MapEntry {
        path: path.to_string_lossy().into_owned(),
        rss: 400,
    }]
}

#[test]
fn only_the_selected_backend_is_started() {
    let dir = tempfile::tempdir().unwrap();
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]);
    let drm = FakeBackend::new(BackendKind::Drm, vec![amd_adapter(2)]);
    let (nv_starts, drm_starts) = (nv.starts(), drm.starts());
    let c = Collector::new(
        deps(
            vec![Box::new(nv), Box::new(drm)],
            FakeProcs::default(),
            dir.path(),
        ),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    assert_eq!(c.adapters().len(), 2); // mixed vendors are selectable
    assert_eq!(
        (*nv_starts.lock().unwrap(), *drm_starts.lock().unwrap()),
        (1, 0)
    );
}

#[test]
fn selecting_the_second_adapter_starts_its_own_backend() {
    let dir = tempfile::tempdir().unwrap();
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]);
    let drm = FakeBackend::new(BackendKind::Drm, vec![amd_adapter(2)]);
    let (nv_starts, drm_starts) = (nv.starts(), drm.starts());
    let mut c = Collector::new(
        deps(
            vec![Box::new(drm), Box::new(nv)],
            FakeProcs::default(),
            dir.path(),
        ),
        1,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    assert_eq!(
        (*nv_starts.lock().unwrap(), *drm_starts.lock().unwrap()),
        (0, 1)
    );
    let snap = c.sample();
    assert_eq!(snap.adapter.unwrap().key, 2);
    assert_eq!(snap.backend, Some(BackendKind::Drm));
    assert_eq!(
        snap.adapters.iter().map(|a| a.key).collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[test]
fn list_adapters_orders_nvidia_first_then_by_vram() {
    let mut small = amd_adapter(3);
    small.dedicated_vram = 8 * GIB;
    let backends: Vec<Box<dyn GpuBackend>> = vec![
        Box::new(FakeBackend::new(
            BackendKind::Drm,
            vec![small, amd_adapter(2)],
        )),
        Box::new(FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)])),
    ];
    let keys: Vec<u64> = list_adapters(&backends).iter().map(|a| a.key).collect();
    assert_eq!(keys, vec![1, 2, 3]);
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
    let procs = FakeProcs::default()
        .named(10, "a")
        .named(11, "b")
        .named(12, "c");
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], procs, dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    let pids: Vec<_> = snap.procs.iter().map(|r| (r.pid, r.origin)).collect();
    assert_eq!(
        pids,
        vec![
            (11, Origin::Local),
            (10, Origin::Local),
            (5000, Origin::Outside)
        ]
    );
    assert_eq!(snap.procs[2].name, "(outside this container)");
    assert_eq!(snap.procs[2].kind, "external");
    assert_eq!(snap.attributed(), Bytes::measured(6 * GIB));
    assert_eq!(snap.procs[2].key(), "o5000");
    assert_eq!(snap.procs[2].pid_label(), "H:5000");
    assert_eq!(snap.procs[0].key(), "11");
    assert_eq!(snap.procs[0].pid_label(), "11");
    assert_eq!(snap.taken, 1000.0);
    assert_eq!(snap.vram_used, Bytes::measured(6 * GIB));
    assert_eq!(snap.vram_total, Bytes::measured(24 * GIB));
    assert!(snap.shared_is_spill);
}

#[test]
fn unknown_vram_ties_with_measured_zero_and_keeps_pid_order() {
    // Python sorts by -(shown_vram or 0): unknown counts as 0 and the stable sort keeps pid order.
    let dir = tempfile::tempdir().unwrap();
    let mut sample = BackendSample::default();
    let mut hidden = ProcGpu::measured(0);
    hidden.vram = Bytes::unavailable(Reason::NvmlHidden);
    sample.procs.insert(5, hidden);
    let mut busy = ProcGpu::measured(0);
    busy.engines = Engines::Measured {
        by_engine: [("compute".to_string(), 50.0)].into_iter().collect(),
    };
    sample.procs.insert(6, busy);
    sample.procs.insert(7, ProcGpu::measured(2 * MIB));
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], FakeProcs::default(), dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let pids: Vec<_> = c.sample().procs.iter().map(|r| r.pid).collect();
    assert_eq!(pids, vec![7, 5, 6]);
}

#[test]
fn outside_rows_never_borrow_a_local_identity() {
    let dir = tempfile::tempdir().unwrap();
    let mut sample = BackendSample::default();
    sample.procs.insert(1145, ProcGpu::measured(GIB));
    sample.outside.insert(1145, ProcGpu::measured(58 * MIB));
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let procs = FakeProcs::default().named(1145, "llama-server");
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], procs, dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    assert_eq!(snap.procs.len(), 2);
    let (local, outside) = (&snap.procs[0], &snap.procs[1]);
    assert_eq!(
        (local.origin, local.name.as_str()),
        (Origin::Local, "llama-server")
    );
    assert_eq!(
        (local.key(), local.pid_label()),
        ("1145".into(), "1145".into())
    );
    assert_eq!(outside.origin, Origin::Outside);
    assert_eq!(
        (outside.key(), outside.pid_label()),
        ("o1145".into(), "H:1145".into())
    );
    assert_eq!(outside.name, "(outside this container)");
    assert_eq!(outside.kind, "external");
    assert_eq!(
        outside.detail,
        ProcDetail::new(1145, "(outside this container)")
    );
    assert!(outside.llm.is_none() && outside.handy.is_none());
    assert_eq!(
        outside.inside(),
        "a process in another PID namespace (host or another container)"
    );
    assert_eq!(outside.gpu.vram, Bytes::measured(58 * MIB));
}

#[test]
fn installed_kind_hints_do_not_classify_the_process() {
    let dir = tempfile::tempdir().unwrap();
    let m = gputaskman_core::procinfo::models::ModelContext::new(Paths::rooted(dir.path()))
        .make_model(
            "OpenJev.safetensors",
            1000,
            0,
            false,
            Evidence::Installed,
            Some(String::new()),
            0.0,
        );
    let mut d = ProcDetail::new(10, "python.exe");
    d.runtimes = vec!["PyTorch".into()];
    d.candidates = vec![m];
    let gpu = ProcGpu::measured(0);
    let kind = gputaskman_core::procinfo::classify::classify(&d, &gpu.engines);
    assert_eq!(kind, "AI");
    let row = ProcRow {
        pid: 10,
        name: d.name.clone(),
        kind,
        gpu,
        detail: d,
        llm: None,
        handy: None,
        origin: Origin::Local,
    };
    assert!(
        row.inside().contains("installed (unproven)"),
        "{}",
        row.inside()
    );
    assert!(row.inside().contains("likely decision model (name)"));
    assert_eq!(row.runtime(), "PyTorch");
}

#[test]
fn no_supported_gpu_gives_an_empty_snapshot_with_a_reason() {
    let dir = tempfile::tempdir().unwrap();
    let mut d = deps(vec![], FakeProcs::default(), dir.path());
    d.notes
        .push("NVML not loaded: libnvidia-ml.so not found".into());
    let mut c = Collector::new(d, 0, ScanPolicy::Unbounded).unwrap();
    let snap = c.sample();
    assert!(snap.adapter.is_none() && snap.procs.is_empty());
    assert!(matches!(snap.vram_total, Bytes::Unavailable { .. }));
    assert_eq!(
        snap.notes,
        vec![
            "NVML not loaded: libnvidia-ml.so not found",
            "no supported GPU found (NVIDIA through NVML, or an amdgpu card)"
        ]
    );
    assert_eq!(snap.backend, None);
}

#[test]
fn no_supported_gpu_is_explained_even_when_every_backend_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = Collector::new(
        deps(vec![], FakeProcs::default(), dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let first = c.sample();
    let second = c.sample();
    let want = vec!["no supported GPU found (NVIDIA through NVML, or an amdgpu card)"];
    assert_eq!(first.notes, want);
    assert_eq!(
        second.notes, want,
        "the note is not repeated across samples"
    );
}

#[test]
fn out_of_range_gpu_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]);
    let err = Collector::new(
        deps(vec![Box::new(nv)], FakeProcs::default(), dir.path()),
        3,
        ScanPolicy::Unbounded,
    )
    .err()
    .unwrap();
    assert_eq!(
        err.to_string(),
        "--gpu 3 is out of range; 1 GPU adapter(s) available"
    );
}

#[test]
fn amd_snapshot_does_not_show_cuda_header() {
    let dir = tempfile::tempdir().unwrap();
    let drm = FakeBackend::new(BackendKind::Drm, vec![amd_adapter(2)])
        .with_driver(Some("kernel 6.17.0".into()), None)
        .with_device(DeviceTelemetry {
            source: "AMD sysfs".into(),
            ..Default::default()
        });
    let mut c = Collector::new(
        deps(vec![Box::new(drm)], FakeProcs::default(), dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    assert_eq!(snap.cuda, None);
    assert!(snap.driver.as_deref().unwrap().starts_with("kernel "));
    assert_eq!(snap.backend, Some(BackendKind::Drm));
}

#[test]
fn vram_total_prefers_device_telemetry() {
    let dir = tempfile::tempdir().unwrap();
    let device = DeviceTelemetry {
        mem_total: Some(20 * GIB),
        ..Default::default()
    };
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)])
        .with_driver(Some("570.86.16".into()), Some("12.8".into()))
        .with_device(device);
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], FakeProcs::default(), dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    assert_eq!(snap.vram_total, Bytes::measured(20 * GIB));
    assert_eq!(snap.device.unwrap().mem_total, Some(20 * GIB));
    assert_eq!(
        (snap.driver.as_deref(), snap.cuda.as_deref()),
        (Some("570.86.16"), Some("12.8"))
    );
}

#[test]
fn device_failure_is_recorded_and_hides_driver_versions() {
    let dir = tempfile::tempdir().unwrap();
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)])
        .with_driver(Some("570.86.16".into()), Some("12.8".into()));
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], FakeProcs::default(), dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    assert!(snap.device.is_none());
    assert_eq!(snap.device_status, Some(Reason::NvmlNotLoaded));
    assert_eq!((snap.driver, snap.cuda), (None, None));
    assert_eq!(snap.vram_total, Bytes::measured(24 * GIB)); // from the adapter
}

/// Ollama runners (exe under `/ollama/`) with mapped GGUFs, and a load log at Ollama's log path.
fn ollama_runners(pids: &[Pid], dir: &Path) -> Snapshot {
    let paths = Paths::rooted(dir);
    let log = gputaskman_core::llm::logs::ollama_log_path(&paths);
    std::fs::create_dir_all(log.parent().unwrap()).unwrap();
    std::fs::write(
        &log,
        "msg=\"model weights\" device=CUDA0 size=\"1.5 GiB\"\n\
         msg=\"kv cache\" device=CUDA0 size=\"512.0 MiB\"\n",
    )
    .unwrap();
    let model = llm_gguf(dir, "qwen3.gguf");
    let mut sample = BackendSample::default();
    let mut procs = FakeProcs::default();
    for &pid in pids {
        sample.procs.insert(pid, ProcGpu::measured(3 * GIB));
        procs = procs
            .with(
                pid,
                ProcBasics {
                    name: Some("llama-server".into()),
                    exe: Some("/usr/local/lib/ollama/llama-server".into()),
                    cmdline: Some(vec!["llama-server".into()]),
                    start: Some(StartTime {
                        ticks: 1,
                        wall: 1.0,
                    }),
                },
            )
            .maps(pid, mapped(&model));
    }
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], procs, dir),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    c.sample()
}

fn reported(row: &ProcRow) -> usize {
    let view = row.llm.as_ref().expect("an LLM view");
    view.segments
        .iter()
        .filter(|s| matches!(s.bytes, Bytes::Reported { .. }))
        .count()
}

#[test]
fn a_single_ollama_runner_reads_the_ollama_log() {
    let dir = tempfile::tempdir().unwrap();
    let snap = ollama_runners(&[10], dir.path());
    let log = gputaskman_core::llm::logs::ollama_log_path(&Paths::rooted(dir.path()));
    let view = snap.procs[0].llm.as_ref().unwrap();
    assert_eq!(
        view.log_source.as_deref(),
        Some(log.to_string_lossy().as_ref())
    );
    assert_eq!(reported(&snap.procs[0]), 2);
}

#[test]
fn two_ollama_runners_do_not_share_the_ollama_log() {
    let dir = tempfile::tempdir().unwrap();
    let snap = ollama_runners(&[10, 11], dir.path());
    assert_eq!(snap.procs.len(), 2);
    for row in &snap.procs {
        assert_eq!(row.llm.as_ref().unwrap().log_source, None);
        assert_eq!(reported(row), 0);
    }
}

fn three_rows(policy: ScanPolicy, dir: &Path) -> Vec<bool> {
    let mut sample = BackendSample::default();
    for (pid, gib) in [(10, 3), (11, 2), (12, 1)] {
        sample.procs.insert(pid, ProcGpu::measured(gib * GIB));
    }
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let procs = FakeProcs::default()
        .named(10, "a")
        .named(11, "b")
        .named(12, "c");
    let mut d = deps(vec![Box::new(nv)], procs, dir);
    d.clock = Arc::new(FakeClock::new(1000.0).auto_advance(Duration::from_millis(100)));
    let mut c = Collector::new(d, 0, policy).unwrap();
    c.sample()
        .procs
        .iter()
        .map(|r| r.detail.maps_time > 0.0)
        .collect()
}

#[test]
fn deep_scans_stop_at_the_budget() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        three_rows(ScanPolicy::Budgeted(SCAN_BUDGET), dir.path()),
        vec![true, false, false]
    );
    assert_eq!(
        three_rows(ScanPolicy::Unbounded, dir.path()),
        vec![true, true, true]
    );
}

#[test]
fn a_larger_budget_scans_more_rows() {
    // The fake clock moves 100 ms per reading, so 1 s leaves room for every row.
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        three_rows(ScanPolicy::Budgeted(Duration::from_secs(1)), dir.path()),
        vec![true, true, true]
    );
}

#[test]
fn collector_can_move_to_another_thread() {
    fn assert_send<T: Send>() {}
    assert_send::<Collector>();
}

/// The estimated segments of the LLM view for one `llama-server` row holding `vram` and `gtt`.
fn llm_segments(integrated: bool, dir: &Path) -> (bool, u64) {
    let model = llm_gguf(dir, "qwen3.gguf");
    let mut adapter = amd_adapter(2);
    adapter.integrated = integrated;
    let mut g = ProcGpu::measured(GIB);
    g.shared = Bytes::measured(2 * GIB);
    let mut sample = BackendSample::default();
    sample.procs.insert(10, g);
    let drm = FakeBackend::new(BackendKind::Drm, vec![adapter]).script(vec![sample]);
    let procs = FakeProcs::default()
        .named(10, "llama-server")
        .maps(10, mapped(&model));
    let mut c = Collector::new(
        deps(vec![Box::new(drm)], procs, dir),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    let view = snap.procs[0].llm.as_ref().unwrap();
    let held = view.segments.iter().filter_map(|s| s.bytes.value()).sum();
    (snap.shared_is_spill, held)
}

#[test]
fn apu_gtt_counts_as_held_memory() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(llm_segments(true, dir.path()), (false, 3 * GIB));
    assert_eq!(llm_segments(false, dir.path()), (true, GIB));
}

#[test]
fn llm_view_built_for_llm_rows() {
    let dir = tempfile::tempdir().unwrap();
    let model = llm_gguf(dir.path(), "qwen3.gguf");
    let mut sample = BackendSample::default();
    sample.procs.insert(10, ProcGpu::measured(GIB));
    sample.procs.insert(11, ProcGpu::measured(GIB / 2));
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let procs = FakeProcs::default()
        .named(10, "llama-server")
        .maps(10, mapped(&model))
        .named(11, "app")
        .named(50, "front-end")
        .parent(10, 50)
        .ports(10, &[8080, 9000])
        .ports(11, &[9000]) // shared with another process: not 10's to probe
        .ports(50, &[7000]);
    let http = Arc::new(FakeHttp::default());
    let mut d = deps(vec![Box::new(nv)], procs, dir.path());
    d.http = Box::new(SharedHttp(Arc::clone(&http)));
    let mut c = Collector::new(d, 0, ScanPolicy::Unbounded).unwrap();
    let snap = c.sample();
    let row = &snap.procs[0];
    assert_eq!((row.pid, row.kind.as_str()), (10, "LLM"));
    let view = row.llm.as_ref().expect("an LLM view");
    assert_eq!(view.models.len(), 1);
    assert!(snap.procs[1].llm.is_none());
    let seen = http.requests();
    assert!(seen.iter().any(|u| u.starts_with("http://127.0.0.1:8080/")));
    assert!(seen.iter().any(|u| u.starts_with("http://127.0.0.1:7000/")));
    assert!(
        seen.iter()
            .all(|u| u.starts_with("http://127.0.0.1:8080/")
                || u.starts_with("http://127.0.0.1:7000/")),
        "{seen:?}"
    );
}

#[test]
fn no_llm_view_when_the_backend_cannot_attribute_the_row() {
    let dir = tempfile::tempdir().unwrap();
    let model = llm_gguf(dir.path(), "qwen3.gguf");
    let mut g = ProcGpu::measured(0);
    g.vram = Bytes::unavailable(Reason::SharedClient);
    g.shared = Bytes::unavailable(Reason::SharedClient);
    let mut sample = BackendSample::default();
    sample.procs.insert(10, g);
    let drm = FakeBackend::new(BackendKind::Drm, vec![amd_adapter(2)]).script(vec![sample]);
    let procs = FakeProcs::default()
        .named(10, "llama-server")
        .maps(10, mapped(&model));
    let mut c = Collector::new(
        deps(vec![Box::new(drm)], procs, dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    assert_eq!(snap.procs[0].kind, "LLM");
    assert!(snap.procs[0].llm.is_none());
    assert!(matches!(snap.attributed(), Bytes::Unavailable { .. }));
}

#[test]
fn handy_rows_get_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::rooted(dir.path());
    let base = paths.xdg_data.join("com.pais.handy");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(
        base.join("settings_store.json"),
        r#"{"settings": {"selected_model": "handy-computer/nemotron/model-Q8_0.gguf"}}"#,
    )
    .unwrap();
    let mut sample = BackendSample::default();
    sample.procs.insert(20, ProcGpu::measured(GIB));
    let nv = FakeBackend::new(BackendKind::Nvml, vec![nv_adapter(1)]).script(vec![sample]);
    let procs = FakeProcs::default().with(
        20,
        ProcBasics {
            name: Some("handy".into()),
            exe: Some("/usr/bin/handy".into()),
            cmdline: Some(vec!["handy".into()]),
            start: Some(StartTime {
                ticks: 1,
                wall: 1.0,
            }),
        },
    );
    let mut c = Collector::new(
        deps(vec![Box::new(nv)], procs, dir.path()),
        0,
        ScanPolicy::Unbounded,
    )
    .unwrap();
    let snap = c.sample();
    let ev = snap.procs[0].handy.as_ref().expect("Handy evidence");
    assert_eq!(
        ev.selected_model.as_deref(),
        Some("handy-computer/nemotron/model-Q8_0.gguf")
    );
    assert_eq!(
        snap.procs[0].inside(),
        "Handy selected (not load proof): model-Q8_0.gguf"
    );
}

/// Lets the test keep a handle on the fake while the collector owns the boxed client.
struct SharedHttp(Arc<FakeHttp>);

impl HttpGet for SharedHttp {
    fn get(&self, url: &LocalUrl, timeout: Duration) -> Option<String> {
        self.0.get(url, timeout)
    }
}

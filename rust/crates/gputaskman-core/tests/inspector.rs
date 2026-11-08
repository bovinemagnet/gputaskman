#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use gputaskman_core::env::Paths;
use gputaskman_core::model::*;
use gputaskman_core::procinfo::inspector::ProcInspector;
use gputaskman_core::procinfo::installed::Installed;
use gputaskman_core::procinfo::models::ModelContext;
use gputaskman_core::testing::FakeProcs;
use gputaskman_core::traits::{MapEntry, ProcBasics, StartTime};

fn basics(name: &str, ticks: u64) -> ProcBasics {
    ProcBasics {
        name: Some(name.into()),
        exe: Some(format!("/opt/{name}")),
        cmdline: Some(vec![name.into()]),
        start: Some(StartTime {
            ticks,
            wall: ticks as f64,
        }),
    }
}

fn file(path: &Path, size: u64) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    File::create(path).unwrap().set_len(size).unwrap();
}

fn ctx(root: &Path) -> (ModelContext, Installed) {
    let paths = Paths::rooted(root);
    (ModelContext::new(paths.clone()), Installed::new(paths))
}

fn entry(path: &Path, rss: u64) -> MapEntry {
    MapEntry {
        path: path.to_string_lossy().into_owned(),
        rss,
    }
}

fn proc_with(pid: u32, exe: &str, cmdline: Vec<String>, maps: Vec<MapEntry>) -> FakeProcs {
    FakeProcs::default()
        .with(
            pid,
            ProcBasics {
                name: Some("llama-server".into()),
                exe: Some(exe.into()),
                cmdline: Some(cmdline),
                start: Some(StartTime {
                    ticks: 1,
                    wall: 1.0,
                }),
            },
        )
        .maps(pid, maps)
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
    let procs = FakeProcs::default()
        .with(1, basics("a", 1))
        .with(2, basics("b", 1))
        .ports(1, &[8080, 11434])
        .ports(2, &[8080]);
    let mut inspector = ProcInspector::new();
    assert_eq!(inspector.exclusive_ports(&procs, 1, 0.0), vec![11434]);
}

#[test]
fn scans_follow_memory_changes() {
    let procs = FakeProcs::default().with(9, basics("llama-server", 1));
    let mut inspector = ProcInspector::new();
    let d = inspector.get(&procs, 9, 0.0).clone();
    assert!(inspector.needs_scan(&d, 0.0, 0.0));
    let mut scanned = d.clone();
    scanned.maps_time = 100.0;
    scanned.scan_mem = (GIB) as f64;
    assert!(!inspector.needs_scan(&scanned, (GIB) as f64, 103.0));
    assert!(!inspector.needs_scan(&scanned, (GIB + 10 * MIB) as f64, 110.0));
    assert!(inspector.needs_scan(&scanned, (2 * GIB) as f64, 110.0));
    assert!(inspector.needs_scan(&scanned, (GIB) as f64, 221.0));
}

#[test]
fn unreadable_process_is_access_denied_and_never_scanned() {
    let procs = FakeProcs::default().with(
        5,
        ProcBasics {
            name: Some("x".into()),
            exe: None,
            cmdline: None,
            start: None,
        },
    );
    let mut inspector = ProcInspector::new();
    let d = inspector.get(&procs, 5, 0.0).clone();
    assert!(d.access_denied);
    assert!(!inspector.needs_scan(&d, 0.0, 0.0));
}

#[test]
fn ports_refresh_at_most_every_five_seconds() {
    let first = FakeProcs::default().with(1, basics("a", 1)).ports(1, &[80]);
    let mut inspector = ProcInspector::new();
    assert_eq!(inspector.get(&first, 1, 0.0).ports, vec![80]);
    let second = FakeProcs::default().with(1, basics("a", 1)).ports(1, &[81]);
    assert_eq!(inspector.get(&second, 1, 2.0).ports, vec![80]);
    assert_eq!(inspector.get(&second, 1, 6.0).ports, vec![81]);
}

#[test]
fn parents_are_walked_and_cached() {
    let procs = FakeProcs::default()
        .with(3, basics("c", 1))
        .parent(3, 2)
        .parent(2, 1);
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 3, 0.0);
    assert_eq!(inspector.parents(&procs, 3, 5), vec![2, 1]);
    assert_eq!(inspector.get(&procs, 3, 0.0).parents, Some(vec![2, 1]));
    assert_eq!(inspector.parents(&procs, 3, 1), vec![2, 1]); // cached
}

#[test]
fn prune_drops_dead_processes() {
    let procs = FakeProcs::default().with(1, basics("a", 1));
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 1, 0.0).name = "marked".into();
    inspector.prune(&BTreeSet::new());
    assert_eq!(inspector.get(&procs, 1, 0.0).name, "a");
}

#[test]
fn mapped_gguf_with_llama_library_is_one_mapped_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.gguf");
    file(&model, 5 * MIB);
    let procs = proc_with(
        7,
        "/opt/llama-server",
        vec!["llama-server".into()],
        vec![
            entry(Path::new("/opt/lib/libllama.so.1"), 0),
            entry(&model, 3 * MIB),
            entry(Path::new("[heap]"), 0),
        ],
    );
    let (mut c, mut installed) = ctx(dir.path());
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 7, 0.0);
    inspector.scan(7, &procs, &mut c, &mut installed, 0.0, 10.0);
    let d = inspector.get(&procs, 7, 10.0);
    assert_eq!(d.runtimes, vec!["llama.cpp".to_string()]);
    assert_eq!(d.models.len(), 1);
    assert_eq!(d.models[0].sources, vec![Evidence::Mapped]);
    assert_eq!(d.models[0].resident, 3 * MIB);
    assert!(d.candidates.is_empty());
    assert_eq!(d.maps_time, 10.0);
}

#[test]
fn model_argument_adds_cmdline_without_duplicating() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.gguf");
    file(&model, 5 * MIB);
    let m = model.to_string_lossy().into_owned();
    let procs = proc_with(
        7,
        "/opt/llama-server",
        vec![
            "llama-server".into(),
            "-m".into(),
            m.clone(),
            "--mmproj".into(),
            m,
        ],
        vec![entry(&model, 0)],
    );
    let (mut c, mut installed) = ctx(dir.path());
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 7, 0.0);
    inspector.scan(7, &procs, &mut c, &mut installed, 0.0, 10.0);
    let d = inspector.get(&procs, 7, 10.0);
    assert_eq!(d.models.len(), 1);
    assert_eq!(
        d.models[0].sources,
        vec![Evidence::Mapped, Evidence::Cmdline]
    );
}

#[test]
fn repeated_cmdline_arguments_do_not_duplicate_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.gguf");
    std::fs::write(&model, b"not a GGUF").unwrap();
    let m = model.to_string_lossy().into_owned();
    let procs = proc_with(
        6,
        "/opt/llama-server",
        vec![
            "server".into(),
            "--model".into(),
            m.clone(),
            "--mmproj".into(),
            m,
        ],
        vec![],
    );
    let (mut c, mut installed) = ctx(dir.path());
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 6, 0.0);
    inspector.scan(6, &procs, &mut c, &mut installed, 0.0, 10.0);
    let d = inspector.get(&procs, 6, 10.0);
    assert_eq!(d.models.len(), 1);
    assert_eq!(d.models[0].sources, vec![Evidence::Cmdline]);
}

#[test]
fn dotdot_cmdline_path_matches_mapped_path() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.gguf");
    file(&model, 5 * MIB);
    std::fs::create_dir_all(dir.path().join("sub")).unwrap();
    let odd = format!("{}/sub/../m.gguf", dir.path().display());
    let procs = proc_with(
        7,
        "/opt/llama-server",
        vec!["llama-server".into(), "-m".into(), odd],
        vec![entry(&model, 0)],
    );
    let (mut c, mut installed) = ctx(dir.path());
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 7, 0.0);
    inspector.scan(7, &procs, &mut c, &mut installed, 0.0, 10.0);
    let d = inspector.get(&procs, 7, 10.0);
    assert_eq!(d.models.len(), 1);
    assert_eq!(
        d.models[0].sources,
        vec![Evidence::Mapped, Evidence::Cmdline]
    );
}

#[test]
fn unreadable_maps_set_access_denied() {
    let procs = FakeProcs::default().with(5, basics("x", 1));
    let gone = FakeProcs::default();
    let (mut c, mut installed) = ctx(Path::new("/nonexistent"));
    let mut inspector = ProcInspector::new();
    inspector.get(&procs, 5, 0.0);
    inspector.scan(5, &gone, &mut c, &mut installed, 0.0, 10.0);
    assert!(inspector.get(&procs, 5, 10.0).access_denied);
}

#[test]
fn installed_models_are_candidates_only() {
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("Speechy");
    for (p, size) in [
        ("models/whisper-large.bin", 70 * MIB),
        ("models/parakeet/encoder.onnx", 8 * MIB),
        ("models/parakeet/decoder.onnx", 5 * MIB),
        ("models/tiles.bin", 5 * MIB),
        ("cache/big.onnx", 9 * MIB),
        ("speechy", 0),
    ] {
        file(&app.join(p), size);
    }
    let exe = app.join("speechy").to_string_lossy().into_owned();
    let (mut c, mut installed) = ctx(dir.path());
    let found = installed.models(&exe, &mut c, 1.0);
    let names: Vec<(String, u64)> = found
        .iter()
        .map(|m| {
            (
                Path::new(&m.path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                m.size,
            )
        })
        .collect();
    assert_eq!(
        names,
        vec![
            ("whisper-large.bin".to_string(), 70 * MIB),
            ("parakeet".to_string(), 13 * MIB)
        ]
    );
    assert_eq!(found[0].sources, vec![Evidence::Installed]);
    assert_eq!(found[0].evidence_label(), "installed, not proven loaded");
    // cached for 300 s even after the files vanish
    std::fs::remove_dir_all(&app).unwrap();
    assert_eq!(installed.models(&exe, &mut c, 2.0), found);
    assert!(installed.models(&exe, &mut c, 400.0).is_empty());
}

#[test]
fn candidates_only_for_ai_runtimes_without_models() {
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("Speechy");
    file(&app.join("models/whisper-large.bin"), 70 * MIB);
    file(&app.join("speechy"), 0);
    let exe = app.join("speechy").to_string_lossy().into_owned();
    let (mut c, mut installed) = ctx(dir.path());
    let mut inspector = ProcInspector::new();

    let ai = proc_with(
        1,
        &exe,
        vec!["speechy".into()],
        vec![entry(Path::new("/x/libonnxruntime.so.1.17"), 0)],
    );
    inspector.get(&ai, 1, 0.0);
    inspector.scan(1, &ai, &mut c, &mut installed, 0.0, 10.0);
    let d = inspector.get(&ai, 1, 10.0);
    assert_eq!(d.runtimes, vec!["ONNX RT".to_string()]);
    assert!(d.models.is_empty());
    assert_eq!(d.candidates.len(), 1);
    assert_eq!(d.candidates[0].sources, vec![Evidence::Installed]);

    let plain = proc_with(2, &exe, vec!["speechy".into()], vec![]);
    inspector.get(&plain, 2, 0.0);
    inspector.scan(2, &plain, &mut c, &mut installed, 0.0, 10.0);
    assert!(inspector.get(&plain, 2, 10.0).candidates.is_empty());
}

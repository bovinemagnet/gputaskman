#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::diff::{Change, Matched, diff, match_rows, to_text};
use gputaskman_core::model::Evidence;
use gputaskman_core::schema::load;
use gputaskman_core::schema::v1::from_v1;
use gputaskman_core::schema::v2::{ProcessV2, SnapshotV2};
use serde_json::{Value, json};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn proc(pid: Option<u32>, name: &str) -> Value {
    json!({"pid": pid, "host": null, "name": name, "vram_bytes": null, "shared_bytes": 0,
        "committed_bytes": null, "paged_out_bytes": null, "vram_inferred_bytes": null,
        "ports": [], "model_files": []})
}

fn set(mut p: Value, key: &str, v: Value) -> Value {
    p[key] = v;
    p
}

fn vram(p: Value, b: u64) -> Value {
    set(p, "vram_bytes", json!(b))
}

fn host(p: Value, h: &str) -> Value {
    set(p, "host", json!(h))
}

fn ports(p: Value, ps: &[u16]) -> Value {
    set(p, "ports", json!(ps))
}

fn with_llm(p: Value, l: Value) -> Value {
    set(p, "llm", l)
}

fn files(p: Value, fs: &[&str]) -> Value {
    let list: Vec<Value> = fs
        .iter()
        .map(|f| json!({"path": f, "sources": ["mapped"]}))
        .collect();
    set(p, "model_files", json!(list))
}

fn snap(procs: Vec<Value>, used: Option<u64>, attributed: Option<u64>) -> Value {
    snap_with(procs, used, attributed, 100.0, "RTX")
}

fn snap_with(
    procs: Vec<Value>,
    used: Option<u64>,
    attributed: Option<u64>,
    taken: f64,
    gpu: &str,
) -> Value {
    json!({"taken": taken, "gpu": gpu, "backend": "nvml", "vram_used": used, "shared_used": 0,
        "attributed": attributed, "processes": procs})
}

fn llm(n_ctx: i64, kv: u64, exact: bool) -> Value {
    json!({"n_ctx": n_ctx, "segments": [
        {"label": "weights", "bytes": 4 * GIB, "kind": "weights", "exact": exact},
        {"label": "KV", "bytes": kv, "kind": "kv", "exact": exact}]})
}

fn v2(v: &Value) -> SnapshotV2 {
    from_v1(v).unwrap()
}

fn rows(ps: Vec<Value>) -> Vec<ProcessV2> {
    v2(&snap(ps, None, None)).processes
}

#[test]
fn pid_then_name_and_ports_then_unique_name() {
    let before = rows(vec![
        proc(Some(1), "a.exe"),
        ports(proc(Some(2), "llama-server"), &[8080]),
        proc(Some(3), "solo.exe"),
    ]);
    let after = rows(vec![
        proc(Some(1), "a.exe"),
        ports(proc(Some(20), "llama-server"), &[8080]),
        proc(Some(30), "solo.exe"),
    ]);
    let got: Vec<_> = match_rows(&before, &after)
        .iter()
        .map(|(b, a, how)| (b.unwrap().pid, a.unwrap().pid, *how))
        .collect();
    assert_eq!(
        got,
        vec![
            (Some(1), Some(1), Matched::Pid),
            (Some(2), Some(20), Matched::NamePorts),
            (Some(3), Some(30), Matched::Name)
        ]
    );
}

#[test]
fn reused_pid_ambiguous_names_and_host_pid_space_are_not_paired() {
    let before = rows(vec![
        proc(Some(5), "old.exe"),
        proc(Some(7), "w.exe"),
        proc(Some(8), "w.exe"),
        host(proc(Some(9), "x"), "windows"),
    ]);
    let after = rows(vec![
        proc(Some(5), "new.exe"),
        proc(Some(17), "w.exe"),
        proc(Some(18), "w.exe"),
        proc(Some(9), "x"),
    ]);
    let mut got: Vec<_> = match_rows(&before, &after)
        .iter()
        .map(|(b, a, how)| (b.and_then(|r| r.pid), a.and_then(|r| r.pid), *how))
        .collect();
    got.sort_by_key(|g| format!("{g:?}"));
    let mut want = vec![
        (None, Some(5), Matched::New),
        (None, Some(17), Matched::New),
        (None, Some(18), Matched::New),
        (None, Some(9), Matched::New),
        (Some(5), None, Matched::Gone),
        (Some(7), None, Matched::Gone),
        (Some(8), None, Matched::Gone),
        (Some(9), None, Matched::Gone),
    ];
    want.sort_by_key(|g| format!("{g:?}"));
    assert_eq!(got, want);
}

#[test]
fn placeholder_rows_and_missing_pids_are_never_paired_or_dropped() {
    let outside = "(outside this container)";
    let before_v1 = vec![
        host(proc(Some(7), outside), "outside"),
        proc(None, "a"),
        proc(None, "b"),
        host(proc(Some(40), "?"), "windows"),
    ];
    let after_v1 = vec![
        host(proc(Some(9), outside), "outside"),
        proc(None, "a"),
        host(proc(Some(41), "?"), "windows"),
    ];
    let before = rows(before_v1.clone());
    let after = rows(after_v1.clone());
    let mut got: Vec<_> = match_rows(&before, &after)
        .iter()
        .map(|(b, a, how)| (b.map(|r| r.name.clone()), a.map(|r| r.name.clone()), *how))
        .collect();
    got.sort_by_key(|g| format!("{g:?}"));
    let s = |x: &str| Some(x.to_string());
    let mut want = vec![
        (s(outside), None, Matched::Gone),
        (None, s(outside), Matched::New),
        (s("?"), None, Matched::Gone),
        (None, s("?"), Matched::New),
        (s("a"), s("a"), Matched::Name),
        (s("b"), None, Matched::Gone),
    ];
    want.sort_by_key(|g| format!("{g:?}"));
    assert_eq!(got, want);
    let d = diff(
        &v2(&snap(before_v1, Some(0), Some(0))),
        &v2(&snap(after_v1, Some(0), Some(0))),
    )
    .unwrap();
    assert!(to_text(&d).contains("H:9"));
    assert!(to_text(&d).contains("W:41"));
}

#[test]
fn measured_runtime_and_mapped_evidence_stay_separate() {
    let before = snap(
        vec![
            vram(proc(Some(1), "dwm.exe"), 2 * GIB),
            with_llm(
                ports(vram(proc(Some(2), "llama-server"), 5 * GIB), &[8080]),
                llm(4096, GIB / 2, false),
            ),
        ],
        Some(8 * GIB),
        Some(7 * GIB),
    );
    let after = snap_with(
        vec![
            vram(proc(Some(1), "dwm.exe"), 2 * GIB),
            files(
                with_llm(
                    ports(vram(proc(Some(22), "llama-server"), 9 * GIB), &[8080]),
                    llm(32768, 4 * GIB, false),
                ),
                &[r"D:\models\q.gguf"],
            ),
            vram(proc(Some(3), "python.exe"), GIB),
        ],
        Some(13 * GIB),
        Some(12 * GIB),
        142.5,
        "RTX",
    );
    let d = diff(&v2(&before), &v2(&after)).unwrap();
    assert_eq!(d.seconds, Some(42.5));
    assert_eq!(d.device.vram_used.delta, Some(5 * GIB as i64));
    assert_eq!(
        d.device.unattributed,
        Change {
            before: Some(GIB as i64),
            after: Some(GIB as i64),
            delta: Some(0)
        }
    );
    let [server, new, dwm] = &d.processes[..] else {
        panic!("expected three rows")
    };
    assert_eq!(server.matched, Matched::NamePorts);
    assert_eq!(server.measured.vram_bytes.delta, Some(4 * GIB as i64));
    let rt = server.runtime.as_ref().unwrap();
    assert_eq!(
        rt.n_ctx,
        Change {
            before: Some(4096),
            after: Some(32768),
            delta: Some(28672)
        }
    );
    let kv = rt.segments.get("kv").unwrap();
    assert_eq!(kv.change.delta, Some((3584 * MIB) as i64));
    assert_eq!(kv.source, "estimated");
    assert_eq!(server.model_files.added.len(), 1);
    assert_eq!(server.model_files.added[0].path, r"D:\models\q.gguf");
    assert_eq!(server.model_files.added[0].sources, vec![Evidence::Mapped]);
    assert_eq!(new.matched, Matched::New);
    assert_eq!(
        new.measured.vram_bytes,
        Change {
            before: None,
            after: Some(GIB as i64),
            delta: None
        }
    );
    assert_eq!(dwm.name, "dwm.exe");

    let text = to_text(&d);
    assert!(text.contains("2->22"));
    assert!(text.contains("context length  4096 -> 32768"));
    assert!(text.contains("+3.50 GB  estimated"));
    assert!(text.contains("not listed"));
    assert!(text.contains("not what is resident in GPU memory"));
    assert!(!text.contains("dwm.exe"));
    assert!(text.contains("1 process(es) changed by less than 16 MB"));
}

#[test]
fn runtime_reported_segments_and_unknown_measurements() {
    let before = snap(
        vec![with_llm(proc(Some(1), "llama"), llm(4096, GIB, true))],
        None,
        None,
    );
    let after = snap(
        vec![with_llm(proc(Some(1), "llama"), llm(8192, 2 * GIB, true))],
        None,
        None,
    );
    let d = diff(&v2(&before), &v2(&after)).unwrap();
    assert_eq!(d.device.unattributed.delta, None);
    assert_eq!(d.processes[0].measured.vram_bytes.delta, None);
    assert_eq!(
        d.processes[0]
            .runtime
            .as_ref()
            .unwrap()
            .segments
            .get("kv")
            .unwrap()
            .source,
        "runtime-reported"
    );
    assert!(to_text(&d).contains("N/A"));
}

#[test]
fn rejects_snapshots_from_different_gpus() {
    let a = v2(&snap_with(vec![], Some(0), Some(0), 100.0, "A"));
    let b = v2(&snap_with(vec![], Some(0), Some(0), 100.0, "B"));
    let err = diff(&a, &b).unwrap_err();
    assert!(err.contains("different GPUs"), "{err}");
}

#[test]
fn remainder_segments_follow_the_noise_threshold() {
    let server = |runtime: u64| {
        let view = json!({"n_ctx": 4096, "segments": [
            {"label": "rest", "bytes": runtime, "kind": "runtime", "exact": false}]});
        with_llm(vram(proc(Some(1), "llama"), GIB + runtime), view)
    };
    let d = diff(
        &v2(&snap(vec![server(GIB)], Some(3 * GIB), Some(2 * GIB))),
        &v2(&snap(
            vec![server(GIB + 4096)],
            Some(3 * GIB),
            Some(2 * GIB),
        )),
    )
    .unwrap();
    assert!(to_text(&d).contains("(no process changed by 16 MB or more)"));
    assert!(!to_text(&d).contains("llama (1)"));
}

fn write_encoded(
    dir: &tempfile::TempDir,
    name: &str,
    data: &Value,
    enc: &str,
) -> std::path::PathBuf {
    let text = data.to_string();
    let bytes: Vec<u8> = match enc {
        "utf-16" => {
            let mut b = vec![0xff, 0xfe];
            for u in text.encode_utf16() {
                b.extend(u.to_le_bytes());
            }
            b
        }
        "utf-16be" => {
            let mut b = vec![0xfe, 0xff];
            for u in text.encode_utf16() {
                b.extend(u.to_be_bytes());
            }
            b
        }
        "utf-8-sig" => [&[0xef, 0xbb, 0xbf][..], text.as_bytes()].concat(),
        _ => text.into_bytes(),
    };
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn reads_powershell_utf16_and_utf8_bom() {
    let dir = tempfile::tempdir().unwrap();
    let data = snap(vec![vram(proc(Some(1), "a"), MIB)], Some(MIB), Some(MIB));
    let want = from_v1(&data).unwrap();
    for (name, enc) in [
        ("a.json", "utf-16"),
        ("b.json", "utf-8-sig"),
        ("d.json", "utf-16be"),
        ("e.json", "utf-8"),
    ] {
        assert_eq!(load(&write_encoded(&dir, name, &data, enc)).unwrap(), want);
    }
    let bad = write_encoded(&dir, "c.json", &json!({"hello": 1}), "utf-8");
    let err = load(&bad).unwrap_err();
    assert!(err.contains("not a gputaskman --json snapshot"), "{err}");
    let missing = load(&dir.path().join("missing.json")).unwrap_err();
    assert!(missing.contains("missing.json"), "{missing}");
}

#[test]
fn v2_file_round_trips_through_load() {
    let v1 = snap(
        vec![vram(proc(Some(1), "a"), GIB)],
        Some(2 * GIB),
        Some(GIB),
    );
    let from_v1_side = v2(&v1);
    // The same data written out as v2 JSON and read back through `load`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2.json");
    std::fs::write(&path, serde_json::to_string_pretty(&from_v1_side).unwrap()).unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded, from_v1_side);
    let d = diff(&from_v1_side, &loaded).unwrap();
    assert!(
        d.processes
            .iter()
            .all(|p| p.measured.vram_bytes.delta == Some(0))
    );
    assert_eq!(d.device.vram_used.delta, Some(0));
    assert!(to_text(&d).contains("(no process changed by 16 MB or more)"));
}

#[test]
fn mixed_v1_and_v2_inputs_compare() {
    use gputaskman_core::collector::{Origin, ProcRow, Snapshot};
    use gputaskman_core::model::{BackendKind, ProcDetail, ProcGpu};
    use gputaskman_core::provenance::{Bytes, Reason};
    use gputaskman_core::schema::v2::from_snapshot;

    let live = Snapshot {
        taken: 100.0,
        adapter: None,
        adapters: vec![],
        device: None,
        device_status: None,
        driver: None,
        cuda: None,
        vram_total: Bytes::measured(8 * GIB),
        vram_used: Bytes::measured(2 * GIB),
        shared_used: Bytes::unavailable(Reason::NotOnBackend),
        procs: vec![ProcRow {
            pid: 1,
            name: "a".into(),
            kind: "AI".into(),
            gpu: ProcGpu::measured(GIB),
            detail: ProcDetail::new(1, "a"),
            llm: None,
            handy: None,
            origin: Origin::Local,
        }],
        backend: Some(BackendKind::Nvml),
        process_status: None,
        shared_is_spill: true,
        notes: vec![],
    };
    let v2_side = from_snapshot(&live);
    let v1_side = v2(
        &json!({"taken": 100.0, "backend": "nvml", "vram_total": 8 * GIB, "vram_used": 2 * GIB,
        "attributed": GIB, "processes": [vram(proc(Some(1), "a"), GIB)]}),
    );
    let d = diff(&v1_side, &v2_side).unwrap();
    assert_eq!(d.device.vram_used.delta, Some(0));
    assert_eq!(d.device.unattributed.delta, Some(0));
    assert_eq!(d.processes[0].matched, Matched::Pid);
    assert_eq!(d.processes[0].measured.vram_bytes.delta, Some(0));
    assert!(to_text(&d).contains("(no process changed by 16 MB or more)"));
}

#[test]
fn rejects_wddm_against_wsl_with_the_backend_names() {
    let on = |backend: &str| {
        let mut s = snap(vec![], Some(0), Some(0));
        s["backend"] = json!(backend);
        v2(&s)
    };
    let err = diff(&on("wddm"), &on("wsl")).unwrap_err();
    assert_eq!(
        err,
        "snapshots are from different GPUs or backends: RTX (wddm) vs RTX (wsl)"
    );
    assert!(diff(&on("metal"), &on("metal")).is_ok());
}

#[test]
fn missing_taken_gives_no_interval() {
    let mut a = snap(vec![], Some(0), Some(0));
    a.as_object_mut().unwrap().remove("taken");
    let (a, b) = (v2(&a), v2(&snap(vec![], Some(0), Some(0))));
    assert_eq!(a.taken, None);
    assert_eq!(a, a.clone());
    assert_eq!(diff(&a, &b).unwrap().seconds, None);
}

#[test]
fn schema_version_absent_is_v1_two_is_v2_anything_else_errors() {
    let dir = tempfile::tempdir().unwrap();
    let mut data = snap(vec![], Some(0), Some(0));
    assert!(load(&write_encoded(&dir, "a.json", &data, "utf-8")).is_ok());
    data["schema_version"] = json!(3);
    let err = load(&write_encoded(&dir, "b.json", &data, "utf-8")).unwrap_err();
    assert!(err.contains("unsupported schema_version 3"), "{err}");
    data["schema_version"] = json!(1);
    assert!(load(&write_encoded(&dir, "c.json", &data, "utf-8")).is_err());
}

#[test]
fn huge_segment_sums_saturate_instead_of_overflowing() {
    let huge = 1u64 << 63;
    let view = json!({"n_ctx": 4096, "segments": [
        {"label": "a", "bytes": huge, "kind": "weights", "exact": true},
        {"label": "b", "bytes": huge, "kind": "weights", "exact": true}]});
    let s = snap(vec![with_llm(proc(Some(1), "llama"), view)], None, None);
    let d = diff(&v2(&s), &v2(&s)).unwrap();
    let weights = d.processes[0]
        .runtime
        .as_ref()
        .unwrap()
        .segments
        .get("weights")
        .unwrap()
        .change;
    assert_eq!(weights.after, Some(i64::MAX));
}

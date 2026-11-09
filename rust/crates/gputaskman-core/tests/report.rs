#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support {
    pub mod gguf_writer;
}
use std::path::{Path, PathBuf};

use gputaskman_core::env::Paths;
use gputaskman_core::handy::HandyEvidence;
use gputaskman_core::llm::flags::Flags;
use gputaskman_core::llm::servers::ServerInfo;
use gputaskman_core::llm::view::{LlmView, build_view};
use gputaskman_core::procinfo::classify::classify;
use gputaskman_core::procinfo::models::ModelContext;
use gputaskman_core::{collector::*, model::*, provenance::*, report::to_text, schema::v2::*};
use serde_json::json;
use support::gguf_writer::{V, write_gguf};

fn snap(backend: BackendKind, rows: Vec<ProcRow>) -> Snapshot {
    Snapshot {
        taken: 1.0,
        adapter: None,
        adapters: vec![],
        device: None,
        device_status: None,
        driver: None,
        cuda: None,
        vram_total: Bytes::measured(24 * GIB),
        vram_used: Bytes::measured(7 * GIB),
        shared_used: Bytes::unavailable(Reason::NotOnBackend),
        procs: rows,
        backend: Some(backend),
        process_status: None,
        shared_is_spill: true,
        notes: vec![],
    }
}

fn row(pid: Pid, gpu: ProcGpu) -> ProcRow {
    ProcRow {
        pid,
        name: "python3".into(),
        kind: "AI".into(),
        gpu,
        detail: ProcDetail::new(pid, "python3"),
        llm: None,
        handy: None,
        origin: Origin::Local,
    }
}

fn empty_view() -> LlmView {
    LlmView {
        models: vec![],
        files: vec![],
        other_files: vec![],
        flags: Flags::default(),
        server: None,
        host: None,
        log_source: None,
        segments: vec![],
        notes: vec![],
        kv_per_token: None,
        kv_type: "f16".into(),
        n_ctx: None,
    }
}

/// Writes a one-tensor GGUF (400 bytes in layer 0). Keys other than `general.*` and `stt.*` get the arch prefix.
fn gguf_file(dir: &Path, file: &str, arch: &str, extra: &[(&str, V)]) -> PathBuf {
    let path = dir.join(file);
    let mut kv: Vec<(String, V)> = vec![("general.architecture".into(), V::Str(arch))];
    for (k, v) in extra {
        let key = if k.starts_with("general.") || k.starts_with("stt.") {
            (*k).to_string()
        } else {
            format!("{arch}.{k}")
        };
        let v = match v {
            V::U32(n) => V::U32(*n),
            V::Bool(b) => V::Bool(*b),
            V::Str(s) => V::Str(s),
            V::StrArray(n) => V::StrArray(*n),
        };
        kv.push((key, v));
    }
    let refs: Vec<(&str, V)> = kv
        .iter()
        .map(|(k, v)| {
            let v = match v {
                V::U32(n) => V::U32(*n),
                V::Bool(b) => V::Bool(*b),
                V::Str(s) => V::Str(s),
                V::StrArray(n) => V::StrArray(*n),
            };
            (k.as_str(), v)
        })
        .collect();
    write_gguf(&path, &refs, &[("blk.0.w", &[100], 0)]);
    // The tensor table gives no sizes; the last tensor runs to the end of the file. Pad to the
    // 32-byte data start, then 400 bytes of data.
    let len = std::fs::metadata(&path).unwrap().len();
    let padded = len.div_ceil(32) * 32 + 400;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(padded)
        .unwrap();
    path
}

#[test]
fn measured_zero_and_unknown_stay_apart() {
    let hidden = ProcGpu {
        vram: Bytes::unavailable(Reason::NvmlHidden),
        ..ProcGpu::measured(0)
    };
    let v = serde_json::to_value(from_snapshot(&snap(
        BackendKind::Nvml,
        vec![row(1, ProcGpu::measured(0)), row(2, hidden)],
    )))
    .unwrap();
    assert_eq!(v["schema_version"], 2);
    assert_eq!(
        v["processes"][0]["vram"],
        json!({"status": "measured", "bytes": 0})
    );
    assert_eq!(
        v["processes"][1]["vram"],
        json!({"status": "unavailable", "reason": "nvml_hidden"})
    );
    assert_eq!(
        v["processes"][0]["committed"],
        json!({"status": "unavailable", "reason": "not_on_backend"})
    );
    assert_eq!(
        v["processes"][0]["paged_out"],
        json!({"status": "unavailable", "reason": "not_on_backend"})
    );
    assert_eq!(
        v["processes"][0]["shared"],
        json!({"status": "unavailable", "reason": "not_on_backend"})
    );
    assert_eq!(
        v["processes"][0]["engines"],
        json!({"status": "measured", "busiest": 0.0, "by_engine": {}})
    );
    assert_eq!(v["processes"][0]["resident"], json!(null));
    assert_eq!(v["processes"][0]["handy"], json!(null));
    assert_eq!(v["processes"][0]["llm"], json!(null));
    assert_eq!(v["device"], json!(null));
    assert_eq!(
        v["memory"]["attributed"],
        json!({"status": "measured", "bytes": 0})
    );
}

#[test]
fn native_hidden_vram_does_not_hide_measured_engine_load() {
    let mut g = ProcGpu {
        vram: Bytes::unavailable(Reason::NvmlHidden),
        ..ProcGpu::measured(0)
    };
    g.engines = Engines::Measured {
        by_engine: [("compute".to_string(), 47.0)].into(),
    };
    let s = snap(BackendKind::Nvml, vec![row(4242, g)]);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["processes"][0]["engines"]["busiest"], 47.0);
    assert_eq!(
        v["processes"][0]["engines"]["by_engine"],
        json!({"compute": 47.0})
    );
    let text = to_text(&s, 30);
    let line = text.lines().find(|l| l.contains("4242")).unwrap();
    assert!(line.contains("n/a") && line.contains("47"), "{line}");
}

#[test]
fn drm_unattributed_rows_void_the_total_and_explain_why() {
    let mut g = ProcGpu::measured(0);
    g.vram = Bytes::unavailable(Reason::SharedBuffers);
    g.shared = Bytes::unavailable(Reason::SharedBuffers);
    g.resident = Some(Resident {
        vram: Bytes::measured(GIB),
        gtt: Bytes::measured(0),
    });
    let s = snap(BackendKind::Drm, vec![row(9, g)]);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(
        v["memory"]["attributed"],
        json!({"status": "unavailable", "reason": "shared_buffers"})
    );
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

#[test]
fn json_is_pretty_with_two_space_indent() {
    let s = snap(BackendKind::Nvml, vec![]);
    assert!(to_json(&s).starts_with("{\n  \"schema_version\": 2,"));
}

#[test]
fn drm_json_and_text_keep_unknown_apart_from_zero() {
    let mut private = ProcGpu::measured(GIB);
    private.shared = Bytes::measured(0);
    private.resident = Some(Resident {
        vram: Bytes::measured(GIB),
        gtt: Bytes::measured(0),
    });
    let mut kwin = ProcGpu::measured(0);
    kwin.vram = Bytes::unavailable(Reason::SharedBuffers);
    kwin.shared = Bytes::unavailable(Reason::SharedBuffers);
    kwin.resident = Some(Resident {
        vram: Bytes::measured(2 * GIB),
        gtt: Bytes::unavailable(Reason::Unreadable),
    });
    let mut kwin_row = row(200, kwin);
    kwin_row.name = "kwin_wayland".into();
    kwin_row.kind = "compositor".into();
    let mut llama = row(100, private);
    llama.name = "llama-server".into();
    llama.kind = "LLM".into();
    let mut s = snap(BackendKind::Drm, vec![llama, kwin_row]);
    s.vram_total = Bytes::measured(16 * GIB);
    s.vram_used = Bytes::measured(4 * GIB);
    s.shared_used = Bytes::measured(0);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["memory"]["shared_is_spill"], true);
    let (llama, kwin) = (&v["processes"][0], &v["processes"][1]);
    assert_eq!(llama["vram"], json!({"status": "measured", "bytes": GIB}));
    assert_eq!(llama["shared"], json!({"status": "measured", "bytes": 0}));
    assert_eq!(
        llama["resident"]["vram"],
        json!({"status": "measured", "bytes": GIB})
    );
    assert_eq!(kwin["vram"]["status"], "unavailable");
    assert_eq!(kwin["vram"]["reason"], "shared_buffers");
    assert_eq!(kwin["shared"]["status"], "unavailable");
    assert_eq!(
        kwin["resident"]["vram"],
        json!({"status": "measured", "bytes": 2 * GIB})
    );
    assert_eq!(
        kwin["resident"]["gtt"],
        json!({"status": "unavailable", "reason": "unreadable"})
    );
    let text = to_text(&s, 30);
    let line = text
        .lines()
        .find(|x| x.trim_start().starts_with("200"))
        .unwrap();
    let cols: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(&cols[3..5], ["n/a", "N/A"], "{line}");
    assert!(text.contains(DRM_SHARED_BUFFERS));
    assert!(!text.contains("incomplete or unparseable drm-shared"));
}

const DRM_SHARED_BUFFERS: &str = "VRAM n/a: this process's DRM client shares buffers with other clients (dma-buf, e.g. windows and video frames). fdinfo counts shared buffers by where they prefer to live, not where they are, so its private part can't be measured.";

#[test]
fn outside_rows_have_unavailable_resident_and_are_labelled() {
    let mut r = row(77, ProcGpu::measured(GIB));
    r.origin = Origin::Outside;
    let s = snap(BackendKind::Nvml, vec![r]);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    let p = &v["processes"][0];
    assert_eq!(p["origin"], "outside");
    let gone = json!({"status": "unavailable", "reason": "outside_namespace"});
    assert_eq!(
        p["resident"],
        json!({"vram": gone, "gtt": gone, "summable": false})
    );
    assert!(to_text(&s, 30).contains("H:77"));
}

#[test]
fn engines_round_to_one_decimal_and_drop_zero_entries() {
    let mut g = ProcGpu::measured(0);
    g.engines = Engines::Measured {
        by_engine: [
            ("3d".to_string(), 37.46),
            ("copy".to_string(), 0.0),
            ("video".to_string(), 12.0),
        ]
        .into(),
    };
    g.owner = Some(Confidence::Order);
    let s = snap(BackendKind::Nvml, vec![row(5, g)]);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["processes"][0]["engines"]["busiest"], 37.5);
    assert_eq!(
        v["processes"][0]["engines"]["by_engine"],
        json!({"3d": 37.5, "video": 12.0})
    );
    assert_eq!(v["processes"][0]["owner_inferred_by"], "order");
    let mut un = ProcGpu::measured(0);
    un.engines = Engines::Unavailable {
        reason: Reason::Incomplete,
    };
    let v = serde_json::to_value(from_snapshot(&snap(BackendKind::Drm, vec![row(6, un)]))).unwrap();
    assert_eq!(
        v["processes"][0]["engines"],
        json!({"status": "unavailable", "reason": "incomplete"})
    );
}

#[test]
fn inferred_markers_and_notes_appear_in_text() {
    let mut a = ProcGpu::measured(0);
    a.owner = Some(Confidence::Consistent);
    a.vram = Bytes::Inferred {
        bytes: GIB,
        by: Confidence::Consistent,
    };
    let mut b = ProcGpu::measured(0);
    b.owner = Some(Confidence::Order);
    b.vram = Bytes::Inferred {
        bytes: 2 * GIB,
        by: Confidence::Order,
    };
    let text = to_text(&snap(BackendKind::Nvml, vec![row(1, a), row(2, b)]), 30);
    assert!(
        text.contains("1.00 GB*") && text.contains("2.00 GB?"),
        "{text}"
    );
    assert!(text.contains("* matched: the size is exact"));
    assert!(text.contains("? order guess: the size is exact"));
}

#[test]
fn snapshot_notes_and_process_status_reach_json_and_text() {
    let mut s = snap(BackendKind::Nvml, vec![]);
    s.notes = vec!["NVML not loaded: libnvidia-ml.so not found".into()];
    s.process_status = Some("1 processes whose fds or fdinfo could not be read".into());
    s.device_status = Some(Reason::NvmlNotLoaded);
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(
        v["notes"],
        json!(["NVML not loaded: libnvidia-ml.so not found"])
    );
    assert_eq!(
        v["status"],
        json!({"process": "1 processes whose fds or fdinfo could not be read",
        "device": "nvml_not_loaded", "counter": null, "host": null, "wsl_split": null})
    );
    let text = to_text(&s, 30);
    assert!(text.contains("NVML not loaded: libnvidia-ml.so not found"));
    assert!(text.contains("DRM process scan: 1 processes whose fds or fdinfo could not be read."));
}

#[test]
fn text_shows_unavailable_device_metrics_without_fabricating_zeroes() {
    let mut s = snap(BackendKind::Nvml, vec![]);
    s.device = Some(DeviceTelemetry {
        util_gpu: Some(0.0),
        power_w: Some(2.4),
        pcie_tx_kbs: Some(0),
        video_clock: Some(656),
        ..DeviceTelemetry::default()
    });
    let text = to_text(&s, 30);
    for want in [
        "GPU 0%",
        "mem-bandwidth N/A",
        "2/N/A W",
        "P N/A",
        "video 656 MHz",
        "PCIe rx N/A MB/s tx 0 MB/s",
    ] {
        assert!(text.contains(want), "{want}\n{text}");
    }
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["device"]["power_limit_w"], json!(null));
    assert_eq!(v["device"]["video_clock"], 656);
}

#[test]
fn handy_evidence_is_visible_in_text_and_json() {
    let mut r = row(10, ProcGpu::measured(GIB));
    r.name = "handy".into();
    r.detail = ProcDetail::new(10, "handy");
    r.handy = Some(HandyEvidence {
        selected_model: Some("handy-computer/nemotron/model.gguf".into()),
        last_model: Some("model.gguf".into()),
        last_event: Some("loaded".into()),
        last_event_at: Some("2026-11-02T23:01:00+00:00".into()),
    });
    assert!(r.inside().contains("Handy last log (loaded): model.gguf"));
    let s = snap(BackendKind::Nvml, vec![r]);
    assert!(to_text(&s, 30).contains("not current GPU residency"));
    let v = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(v["processes"][0]["handy"]["last_event"], "loaded");
}

#[test]
fn model_detail_header_uses_inferred_or_unknown_vram() {
    for (inferred, label) in [
        (None, "n/a"),
        (Some(0), "0 MB*"),
        (Some(3 * GIB), "3.00 GB*"),
    ] {
        let mut g = ProcGpu::measured(0);
        g.owner = Some(Confidence::Consistent);
        g.vram = match inferred {
            Some(bytes) => Bytes::Inferred {
                bytes,
                by: Confidence::Consistent,
            },
            None => Bytes::unavailable(Reason::Incomplete),
        };
        let mut r = row(4242, g);
        r.llm = Some(empty_view());
        let s = snap(BackendKind::Nvml, vec![r]);
        assert!(
            to_text(&s, 30).contains(&format!("== python3 (4242)  VRAM {label}")),
            "{label}"
        );
        let v = serde_json::to_value(from_snapshot(&s)).unwrap();
        assert_ne!(v["processes"][0]["vram"]["status"], "measured");
    }
}

#[test]
fn non_gguf_hints_are_visible_in_reports() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = ModelContext::new(Paths::rooted(dir.path()));
    let m = c.make_model(
        "OpenJev.safetensors",
        1000,
        0,
        false,
        Evidence::Mapped,
        Some(String::new()),
        0.0,
    );
    let mut d = ProcDetail::new(10, "python.exe");
    d.runtimes = vec!["PyTorch".into()];
    d.models = vec![m.clone()];
    let gpu = ProcGpu::measured(1000);
    let kind = classify(&d, &gpu.engines);
    let v = build_view(&d, 1000.0, 0.0, None, None, None, true, &mut c, 0.0);
    assert!(v.models.is_empty());
    assert_eq!(v.other_files, vec![m]);
    let r = ProcRow {
        pid: 10,
        name: d.name.clone(),
        kind,
        gpu,
        detail: d,
        llm: Some(v),
        handy: None,
        origin: Origin::Local,
    };
    assert!(r.inside().contains("likely decision model (name)"));
    let s = snap(BackendKind::Nvml, vec![r]);
    assert!(to_text(&s, 30).contains("likely decision model (name)"));
    let j = serde_json::to_value(from_snapshot(&s)).unwrap();
    assert_eq!(
        j["processes"][0]["model_files"][0]["kind_hint"],
        "decision model"
    );
    assert_eq!(
        j["processes"][0]["model_files"][0]["evidence"],
        json!(["mapped"])
    );
}

#[test]
fn openjev_hint_does_not_change_header_kv_or_loading_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let p = gguf_file(
        dir.path(),
        "OpenJev-Q4_K_M.gguf",
        "qwen35",
        &[
            ("block_count", V::U32(2)),
            ("context_length", V::U32(4096)),
            ("general.name", V::Str("Snapshot_Cfg")),
            ("attention.head_count", V::U32(4)),
            ("attention.head_count_kv", V::U32(2)),
            ("embedding_length", V::U32(16)),
        ],
    );
    let mut c = ModelContext::new(Paths::rooted(dir.path()));
    let m = c.make_model(
        p.to_str().unwrap(),
        1000,
        0,
        true,
        Evidence::Mapped,
        None,
        1_000_000.0,
    );
    let mut d = ProcDetail::new(10, "llama-server");
    d.models = vec![m.clone()];
    let gpu = ProcGpu::measured(1000);
    let kind = classify(&d, &gpu.engines);
    assert_eq!(kind, "decision?");
    let server = ServerInfo {
        kind: "llama-server".into(),
        url: "http://127.0.0.1:1".into(),
        n_ctx: Some(4),
        slots_total: None,
        slots_busy: None,
        gen_tok_s: None,
        prompt_tok_s: None,
        model_path: Some(m.path.clone()),
        models: vec![],
    };
    let v = build_view(
        &d,
        1000.0,
        0.0,
        Some(server),
        None,
        None,
        true,
        &mut c,
        1_000_000.0,
    );
    let r = ProcRow {
        pid: 10,
        name: d.name.clone(),
        kind,
        gpu,
        detail: d,
        llm: Some(v),
        handy: None,
        origin: Origin::Local,
    };
    let s = snap(BackendKind::Nvml, vec![r]);
    let j = serde_json::to_value(from_snapshot(&s)).unwrap();
    let data = &j["processes"][0];
    assert_eq!(data["kind"], "decision?");
    assert_eq!(data["model_files"][0]["kind_hint"], "decision model");
    assert_eq!(data["llm"]["models"][0]["kind"], "LLM");
    assert_eq!(data["llm"]["models"][0]["kind_hint"], "decision model");
    assert_eq!(
        data["llm"]["models"][0]["kind_label"],
        "likely decision model (name)"
    );
    assert_eq!(
        data["llm"]["models"][0]["evidence"],
        json!(["api", "mapped"])
    );
    assert_eq!(
        data["llm"]["models"][0]["gpu_weight"]["status"],
        "estimated"
    );
    assert_eq!(data["llm"]["models"][0]["gpu_weight"]["from"], "gguf");
    assert_eq!(
        data["llm"]["kv_per_token"],
        json!({"status": "estimated", "bytes": 64, "from": "gguf"})
    );
    assert_eq!(data["llm"]["n_ctx"], 4);
    assert_eq!(data["llm"]["server"]["kind"], "llama-server");
    let text = to_text(&s, 30);
    assert!(text.contains("likely decision model (name)"), "{text}");
    assert!(text.contains("confirmed (api+mapped)"), "{text}");
}

#[test]
fn llm_segments_keep_their_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let p = gguf_file(
        dir.path(),
        "m.gguf",
        "llama",
        &[("block_count", V::U32(2)), ("context_length", V::U32(4096))],
    );
    let mut c = ModelContext::new(Paths::rooted(dir.path()));
    let m = c.make_model(
        p.to_str().unwrap(),
        1000,
        0,
        true,
        Evidence::Mapped,
        None,
        1.0,
    );
    let mut d = ProcDetail::new(10, "llama-server");
    d.models = vec![m];
    let v = build_view(&d, 1000.0, 0.0, None, None, None, true, &mut c, 1.0);
    let r = ProcRow {
        llm: Some(v),
        ..row(10, ProcGpu::measured(1000))
    };
    let j = serde_json::to_value(from_snapshot(&snap(BackendKind::Nvml, vec![r]))).unwrap();
    let seg = &j["processes"][0]["llm"]["segments"][0];
    assert_eq!(seg["kind"], "weights");
    assert_eq!(seg["bytes"]["status"], "estimated");
}

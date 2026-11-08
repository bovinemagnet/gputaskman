#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support {
    pub mod gguf_writer;
}
use std::path::{Path, PathBuf};

use gputaskman_core::env::Paths;
use gputaskman_core::llm::servers::{ServerInfo, ServerModel};
use gputaskman_core::llm::view::{LlmView, SegmentKind, build_view};
use gputaskman_core::model::*;
use gputaskman_core::procinfo::models::ModelContext;
use gputaskman_core::provenance::{Bytes, EstimateSource, Reporter};
use support::gguf_writer::{V, write_gguf};

const NOW: f64 = 1_000_000.0;

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

fn laya() -> Vec<(&'static str, V<'static>)> {
    vec![
        ("block_count", V::U32(30)),
        ("context_length", V::U32(8192)),
        ("decision.type", V::Str("laya")),
        ("attention.causal", V::Bool(false)),
    ]
}

fn mapped(path: &Path, store: Option<&str>, kind: &str, sources: Vec<Evidence>) -> MappedModel {
    MappedModel {
        path: path.to_str().unwrap().to_string(),
        size: 1000,
        resident: 0,
        gguf: true,
        sources,
        store_name: store.map(String::from),
        kind: kind.to_string(),
        kind_hint: String::new(),
    }
}

fn llama_server(n_ctx: Option<i64>, model_path: Option<&str>) -> ServerInfo {
    ServerInfo {
        kind: "llama-server".into(),
        url: "http://127.0.0.1:1".into(),
        n_ctx,
        slots_total: None,
        slots_busy: None,
        gen_tok_s: None,
        prompt_tok_s: None,
        model_path: model_path.map(String::from),
        models: vec![],
    }
}

fn ollama(models: Vec<ServerModel>) -> ServerInfo {
    ServerInfo {
        kind: "ollama".into(),
        url: "http://127.0.0.1:2".into(),
        models,
        ..llama_server(None, None)
    }
}

fn ctx(dir: &Path) -> ModelContext {
    ModelContext::new(Paths::rooted(dir))
}

fn labels(v: &LlmView) -> Vec<String> {
    v.segments.iter().map(|s| s.label.clone()).collect()
}

const MIB_: u64 = 1 << 20;
const GIB_: f64 = (1u64 << 30) as f64;

#[test]
fn test_decision_model_has_no_kv_cache_and_api_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let p = gguf_file(dir.path(), "laya.gguf", "modern-bert", &laya());
    let mut d = ProcDetail::new(5, "llama-server");
    d.models = vec![mapped(&p, None, "decision model", vec![Evidence::Mapped])];
    let server = llama_server(Some(8192), p.to_str());
    let v = build_view(
        &d,
        1000.0,
        0.0,
        Some(server),
        None,
        None,
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    assert_eq!(v.files[0].evidence_label(), "confirmed (api+mapped)");
    assert_eq!(d.models[0].sources, vec![Evidence::Mapped]);
    assert_eq!(v.kv_per_token, None);
    assert_eq!(labels(&v), vec!["weights", "compute buffers + runtime"]);
    assert!(v.notes.iter().any(|n| n.contains("laya decision model")));
}

fn vulkan_detail(dir: &Path) -> (ProcDetail, PathBuf) {
    let p = gguf_file(dir, "laya.gguf", "modern-bert", &laya());
    let mut d = ProcDetail::new(5, "llama-server");
    d.models = vec![mapped(&p, None, "decision model", vec![Evidence::Mapped])];
    (d, p)
}

const VULKAN_LOG: &str = "load_tensors: Vulkan0 model buffer size = 10 MiB\n\
load_tensors: Vulkan_Host model buffer size = 3 MiB\n\
llama_context: Vulkan0 compute buffer size = 2 MiB\n";

#[test]
fn test_vulkan_log_does_not_label_unknown_overhead_as_cuda() {
    let dir = tempfile::tempdir().unwrap();
    let (d, _) = vulkan_detail(dir.path());
    let log = dir.path().join("server.log");
    std::fs::write(&log, VULKAN_LOG).unwrap();
    let v = build_view(
        &d,
        (20 * MIB_) as f64,
        0.0,
        None,
        Some(log),
        None,
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    let got: Vec<(String, Option<u64>)> = v
        .segments
        .iter()
        .map(|s| (s.label.clone(), s.bytes.value()))
        .collect();
    assert_eq!(
        got,
        vec![
            ("weights (Vulkan0)".to_string(), Some(10 * MIB_)),
            ("compute buffers (Vulkan0)".to_string(), Some(2 * MIB_)),
            ("runtime and other overhead".to_string(), Some(8 * MIB_)),
        ]
    );
}

#[test]
fn test_ollama_runner_shows_only_its_model() {
    let dir = tempfile::tempdir().unwrap();
    let p = gguf_file(
        dir.path(),
        "blob",
        "qwen3",
        &[("block_count", V::U32(1)), ("context_length", V::U32(4096))],
    );
    let mut d = ProcDetail::new(6, "llama-server");
    d.models = vec![mapped(&p, Some("qwen3:14b"), "LLM", vec![Evidence::Mapped])];
    let om = |name: &str, vram: u64, c: i64| ServerModel {
        name: Some(name.into()),
        size: Some(1 << 30),
        size_vram: Some(vram),
        ctx: Some(c),
        ..ServerModel::default()
    };
    let host = ollama(vec![
        om("qwen3:14b", 1 << 30, 4096),
        om("other:1b", 0, 2048),
    ]);
    let v = build_view(
        &d,
        GIB_,
        0.0,
        Some(host),
        None,
        None,
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    assert_eq!(v.label(0), "qwen3:14b");
    assert!(v.files[0].sources.contains(&Evidence::Api));
    assert_eq!(v.n_ctx, Some(4096));
    let notes: Vec<&String> = v.notes.iter().filter(|n| n.starts_with("Ollama")).collect();
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("qwen3:14b"));
}

#[test]
fn test_runner_and_host_apis_do_not_duplicate_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let p = gguf_file(
        dir.path(),
        "blob",
        "qwen3",
        &[("block_count", V::U32(1)), ("context_length", V::U32(4096))],
    );
    let model = mapped(&p, Some("qwen3:4b"), "LLM", vec![Evidence::Cmdline]);
    let mut d = ProcDetail::new(6, "llama-server");
    d.models = vec![model];
    let runner = llama_server(None, p.to_str());
    let host = ollama(vec![ServerModel {
        name: Some("qwen3:4b".into()),
        ctx: Some(4096),
        ..ServerModel::default()
    }]);
    let v = build_view(
        &d,
        GIB_,
        0.0,
        Some(runner),
        None,
        Some(host),
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    assert_eq!(v.files[0].sources, vec![Evidence::Cmdline, Evidence::Api]);
    assert_eq!(d.models[0].sources, vec![Evidence::Cmdline]);
}

#[test]
fn test_openjev_hint_does_not_change_header_kv_or_loading_evidence() {
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
    let mut c = ctx(dir.path());
    let m = c.make_model(
        p.to_str().unwrap(),
        1000,
        0,
        true,
        Evidence::Mapped,
        None,
        NOW,
    );
    assert_eq!(m.kind, "LLM");
    assert_eq!(m.kind_label(), "likely decision model (name)");
    let mut d = ProcDetail::new(10, "llama-server");
    d.models = vec![m.clone()];
    let server = llama_server(Some(4), Some(&m.path));
    let v = build_view(&d, 1000.0, 0.0, Some(server), None, None, true, &mut c, NOW);
    assert_eq!(v.label(0), "OpenJev-Q4_K_M.gguf");
    assert_eq!(v.kind_label(0), "likely decision model (name)");
    assert_eq!(v.files[0].evidence_label(), "confirmed (api+mapped)");
    assert_eq!(m.sources, vec![Evidence::Mapped]);
    assert_eq!(v.models[0].kind(), "LLM");
    assert_eq!(v.models[0].decision_type(), None);
    assert_eq!(v.kv_per_token, Some(64));
    assert!(v.segments.iter().any(|s| s.kind == SegmentKind::Kv));
    assert!(!v.notes.iter().any(|n| n.contains("Non-causal")));
}

#[test]
fn test_llm_view_does_not_claim_cpu_inference_when_vram_is_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let p = gguf_file(dir.path(), "x.gguf", "x", &[]);
    let mut d = ProcDetail::new(4242, "llama-server");
    d.cmdline = ["llama-server", "-c", "4096"].map(String::from).to_vec();
    d.models = vec![mapped(&p, None, "", vec![Evidence::Mapped])];
    let known = build_view(
        &d,
        0.0,
        0.0,
        None,
        None,
        None,
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    let unknown = build_view(
        &d,
        0.0,
        0.0,
        None,
        None,
        None,
        false,
        &mut ctx(dir.path()),
        NOW,
    );
    assert!(known.notes.iter().any(|n| n.contains("CPU inference")));
    assert!(!unknown.notes.iter().any(|n| n.contains("CPU inference")));
    assert!(unknown.notes.iter().any(|n| n.contains("unknown")));
}

#[test]
fn log_segments_are_reported_and_estimates_are_estimated() {
    let dir = tempfile::tempdir().unwrap();
    let (mut d, _) = vulkan_detail(dir.path());
    let log = dir.path().join("server.log");
    std::fs::write(&log, VULKAN_LOG).unwrap();
    d.cmdline = vec![
        "llama-server".into(),
        "--log-file".into(),
        log.to_str().unwrap().into(),
    ];
    let v = build_view(
        &d,
        (20 * MIB_) as f64,
        0.0,
        None,
        None,
        None,
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    assert_eq!(v.log_source.as_deref(), log.to_str());
    assert!(
        v.segments
            .iter()
            .filter(|s| s.kind != SegmentKind::Runtime)
            .all(|s| matches!(
                s.bytes,
                Bytes::Reported {
                    by: Reporter::LoadLog,
                    ..
                }
            ))
    );
    d.cmdline.clear();
    let v = build_view(
        &d,
        (20 * MIB_) as f64,
        0.0,
        None,
        None,
        None,
        true,
        &mut ctx(dir.path()),
        NOW,
    );
    assert!(!v.segments.is_empty());
    assert!(v.segments.iter().all(|s| matches!(
        s.bytes,
        Bytes::Estimated {
            from: EstimateSource::Gguf,
            ..
        }
    )));
}

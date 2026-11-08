#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support {
    pub mod gguf_writer;
}
use gputaskman_core::gguf::*;
use std::collections::BTreeMap;
use support::gguf_writer::{V, write_gguf};

fn info(meta: &[(&str, MetaValue)]) -> GgufInfo {
    GgufInfo {
        path: "model.gguf".into(),
        file_size: 1000,
        version: 3,
        metadata: meta
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        layer_bytes: BTreeMap::from([(0, 100), (1, 100)]),
        output_bytes: 50,
        token_embd_bytes: 50,
        tensor_bytes: 300,
        n_params: None,
        error: None,
    }
}

#[test]
fn layer_and_kv_estimates() {
    let m = info(&[
        ("general.architecture", MetaValue::Str("test".into())),
        ("test.block_count", MetaValue::U(2)),
        ("test.attention.head_count", MetaValue::U(4)),
        ("test.attention.head_count_kv", MetaValue::U(2)),
        ("test.embedding_length", MetaValue::U(16)),
    ]);
    assert_eq!(m.gpu_weight_bytes(None), 250);
    assert_eq!(m.gpu_weight_bytes(Some(1)), 100);
    assert_eq!(m.gpu_weight_bytes(Some(0)), 0);
    assert_eq!(m.kv_bytes_per_token(2.0), Some(64));
    assert_eq!(m.meta("block_count"), Some(&MetaValue::U(2)));
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
    write_gguf(
        &path,
        &[
            ("general.architecture", V::Str("llama")),
            ("llama.block_count", V::U32(2)),
            ("general.name", V::Str("Tiny")),
        ],
        &[
            ("blk.0.attn_q.weight", &[4, 4], 0),
            ("blk.1.attn_q.weight", &[4, 4], 0),
            ("output.weight", &[4, 2], 0),
            ("token_embd.weight", &[4, 2], 0),
        ],
    );
    let g = GgufCache::new().read(&path).unwrap();
    assert_eq!(g.error, None);
    assert_eq!(g.arch(), "llama");
    assert_eq!(g.name(), "Tiny");
    assert_eq!(g.n_layers(), 2);
    assert_eq!(g.layer_bytes.get(&0), Some(&64));
    assert_eq!(g.n_params, Some(16 + 16 + 8 + 8));
    assert_eq!(g.params_label(), "48");
    assert_eq!(g.kind(), "model"); // no context length, so not an LLM
}

#[test]
fn llm_kind_needs_layers_and_context() {
    let g = info(&[
        ("general.architecture", MetaValue::Str("llama".into())),
        ("llama.block_count", MetaValue::U(2)),
        ("llama.context_length", MetaValue::I(4096)),
    ]);
    assert_eq!(g.kind(), "LLM");
    assert_eq!(g.ctx_train(), Some(4096));
}

#[test]
fn not_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.bin");
    std::fs::write(&path, b"NOPE....").unwrap();
    assert!(!is_gguf(&path));
    let g = GgufCache::new().read(&path).unwrap();
    assert_eq!(g.error.as_deref(), Some("not a GGUF file"));
    assert!(GgufCache::new().read(&dir.path().join("missing")).is_none());
}

#[test]
fn corrupt_lengths_do_not_allocate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evil.gguf");
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(u64::MAX.to_le_bytes()); // key length
    std::fs::write(&path, b).unwrap();
    assert!(GgufCache::new().read(&path).unwrap().error.is_some());
}

fn header(arch: &str, extra: &[(&str, V)], dir: &std::path::Path) -> GgufInfo {
    let path = dir.join(format!("{arch}-{}.gguf", extra.len()));
    let mut kv: Vec<(String, V)> = vec![("general.architecture".into(), V::Str(arch))];
    for (k, v) in extra {
        let key = if k.starts_with("general.") || k.starts_with("stt.") {
            k.to_string()
        } else {
            format!("{arch}.{k}")
        };
        kv.push((
            key,
            match v {
                V::U32(n) => V::U32(*n),
                V::Bool(b) => V::Bool(*b),
                V::Str(s) => V::Str(s),
            },
        ));
    }
    let kv_ref: Vec<(&str, V)> = kv
        .iter()
        .map(|(k, v)| {
            (
                k.as_str(),
                match v {
                    V::U32(n) => V::U32(*n),
                    V::Bool(b) => V::Bool(*b),
                    V::Str(s) => V::Str(s),
                },
            )
        })
        .collect();
    write_gguf(&path, &kv_ref, &[("blk.0.w", &[10], 0)]);
    GgufCache::new().read(&path).unwrap()
}

fn laya() -> Vec<(&'static str, V<'static>)> {
    vec![
        ("block_count", V::U32(30)),
        ("context_length", V::U32(8192)),
        ("decision.type", V::Str("laya")),
        ("attention.causal", V::Bool(false)),
    ]
}

#[test]
fn gguf_kinds_come_from_header_keys() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let cases: Vec<(&str, GgufInfo)> = vec![
        (
            "LLM",
            header(
                "qwen3",
                &[
                    ("block_count", V::U32(28)),
                    ("context_length", V::U32(40960)),
                ],
                d,
            ),
        ),
        ("decision model", header("modern-bert", &laya(), d)),
        (
            "embedding model",
            header("bert", &[("pooling_type", V::U32(1))], d),
        ),
        (
            "reranker",
            header("bert", &[("pooling_type", V::U32(4))], d),
        ),
        (
            "speech model",
            header("parakeet", &[("stt.variant", V::Str("tdt"))], d),
        ),
        ("vision projector", header("clip", &[], d)),
        (
            "model",
            header(
                "modern-bert",
                &[
                    ("block_count", V::U32(30)),
                    ("context_length", V::U32(8192)),
                    ("attention.causal", V::Bool(false)),
                ],
                d,
            ),
        ),
    ];
    for (kind, g) in &cases {
        assert_eq!(g.kind(), *kind);
    }
    assert_eq!(cases[1].1.decision_type(), Some("laya"));
}

#[test]
fn specific_header_kinds_override_name_hints() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let adapter = header("qwen3", &[("general.type", V::Str("adapter"))], d);
    assert_eq!(adapter.kind(), "LoRA adapter");
    let kinds = [
        header("modern-bert", &laya(), d),
        header("clip", &[], d),
        header("bert", &[("pooling_type", V::U32(1))], d),
        header("bert", &[("pooling_type", V::U32(4))], d),
        header("parakeet", &[("stt.variant", V::Str("tdt"))], d),
        adapter,
    ];
    let expected = [
        "decision model",
        "vision projector",
        "embedding model",
        "reranker",
        "speech model",
        "LoRA adapter",
    ];
    for (g, e) in kinds.iter().zip(expected) {
        assert_eq!(g.kind(), e);
    }
}

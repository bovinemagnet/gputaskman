#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::env::Paths;
use gputaskman_core::llm::{flags::parse_flags, gb, logs::*, servers::*};
use gputaskman_core::testing::FakeHttp;
use serde_json::json;

fn args(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| s.to_string()).collect()
}
const MIB: u64 = 1 << 20;

#[test]
fn flags_include_equals_and_gpu_layers() {
    let f = parse_flags(&args(&[
        "server",
        "--ctx-size=8192",
        "-ngl",
        "all",
        "--cache-type-k=q8_0",
    ]));
    assert_eq!(
        (f.ctx, f.ngl, f.ctk.as_deref()),
        (Some(8192), Some(-1), Some("q8_0"))
    );
    assert_eq!(
        serde_json::to_value(&f).unwrap(),
        json!({"ctx": 8192, "ngl": -1, "ctk": "q8_0"})
    );
}

#[test]
fn flags_bool_and_bad_values() {
    let f = parse_flags(&args(&["--no-mmap", "-c", "abc", "--port", "80"]));
    assert_eq!((f.ctx, f.port, f.no_mmap), (None, Some(80), true));
    assert_eq!(
        serde_json::to_value(&f).unwrap(),
        json!({"port": 80, "no_mmap": true})
    );
}

#[test]
fn gpu_buffers_exclude_host_and_cpu() {
    for d in ["CPU", "CPU_Mapped", "CUDA_Host", "Vulkan_Host"] {
        assert!(!is_gpu_device(d), "{d}");
    }
    for d in ["CUDA0", "Vulkan0"] {
        assert!(is_gpu_device(d), "{d}");
    }
}

#[test]
fn latest_load_and_multiple_kv_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.log");
    std::fs::write(
        &path,
        "load_tensors: CUDA0 model buffer size = 900 MiB\n\
        llama_context: CUDA0 KV buffer size = 800 MiB\nnew load\n\
        load_tensors: CPU_Mapped model buffer size = 30 MiB\n\
        load_tensors: CUDA0 model buffer size = 100 MiB\n\
        llama_context: CUDA0 KV buffer size = 20 MiB\n\
        llama_context: CUDA0 KV buffer size = 10 MiB\n",
    )
    .unwrap();
    let log = parse_load_log(&path, 0.0).unwrap();
    assert_eq!(log["CUDA0"]["weights"], 100 * MIB);
    assert_eq!(log["CUDA0"]["kv"], 30 * MIB);
    assert_eq!(log["CPU_Mapped"]["weights"], 30 * MIB);
    assert!(parse_load_log(&path, 1e18).is_none());
    assert!(parse_load_log(&dir.path().join("missing"), 0.0).is_none());
}

#[test]
fn ollama_log_lines_and_units() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.log");
    std::fs::write(
        &path,
        "msg=\"model weights\" device=CUDA0 size=\"1.5 GiB\"\n\
         msg=\"kv cache\" device=CUDA0 size=\"512.0 MiB\"\n",
    )
    .unwrap();
    let log = parse_load_log(&path, 0.0).unwrap();
    assert_eq!(log["CUDA0"]["weights"], 1536 * MIB);
    assert_eq!(log["CUDA0"]["kv"], 512 * MIB);
}

#[test]
fn ollama_log_path_is_under_home() {
    let p = Paths::rooted(std::path::Path::new("/h"));
    assert_eq!(
        ollama_log_path(&p),
        std::path::PathBuf::from("/h/.ollama/logs/server.log")
    );
}

#[test]
fn gb_formats_like_python() {
    assert_eq!(gb((3u64 << 30) as f64), "3.00 GB");
    assert_eq!(gb(1.5 * MIB as f64), "2 MB");
    assert_eq!(gb(2.5 * MIB as f64), "2 MB");
    assert_eq!(gb(0.0), "0 MB");
    assert_eq!(gb(-((2u64 << 30) as f64)), "-2.00 GB");
}

#[test]
fn probes_are_get_only_on_the_given_port() {
    let http = FakeHttp::default()
        .route(8080, "/props", 200, r#"{"model_path":"/m/q.gguf","default_generation_settings":{"n_ctx":4096},"total_slots":1}"#)
        .route(8080, "/v1/models", 200, r#"{"data":[{"id":"q.gguf","meta":{"ftype":"Q4_K_M","n_ctx":4096}}]}"#)
        .route(8080, "/slots", 200, r#"[{"is_processing":true}]"#)
        .route(8080, "/metrics", 200, "llamacpp:predicted_tokens_seconds 42.5\n");
    let s = probe_server(&http, 8080, None).unwrap();
    assert_eq!(
        (s.kind.as_str(), s.n_ctx, s.slots_busy, s.gen_tok_s),
        ("llama-server", Some(4096), Some(1), Some(42.5))
    );
    assert_eq!(s.models[0].name.as_deref(), Some("q.gguf"));
    assert!(
        http.requests()
            .iter()
            .all(|u| u.starts_with("http://127.0.0.1:8080/"))
    );
}

#[test]
fn runtime_api_normalization() {
    let v1 = json!({"models": [
        {"type": "decision", "key": "laya", "decision_type": "laya", "quantization": {"name": "Q8_0"},
         "size_bytes": 449, "loaded_instances": [{"id": "laya", "config": {"context_length": 8192}}]},
        {"type": "llm", "key": "idle", "loaded_instances": []},
    ]});
    assert_eq!(
        lmstudio_models(Some(&v1), None),
        Some(vec![ServerModel {
            name: Some("laya".into()),
            r#type: Some("decision (laya)".into()),
            quant: Some("Q8_0".into()),
            ctx: Some(8192),
            size: Some(449),
            size_vram: None,
            path: None
        }])
    );
    let v0 = json!({"data": [
        {"id": "a", "type": "llm", "state": "loaded", "quantization": "Q4_K_M", "loaded_context_length": 4096},
        {"id": "b", "state": "not-loaded"}]});
    let got = lmstudio_models(None, Some(&v0)).unwrap();
    assert_eq!(
        got.iter()
            .map(|m| (m.name.clone().unwrap(), m.ctx))
            .collect::<Vec<_>>(),
        vec![("a".to_string(), Some(4096))]
    );
    assert_eq!(lmstudio_models(None, None), None);
    let llama = json!({"data": [{"id": r"C:\m\Laya-Q8_0.gguf", "architecture": {"output_modalities": ["decisions"]},
        "meta": {"ftype": "Q8_0", "n_ctx": 8192, "size": 447}}]});
    assert_eq!(
        llama_server_models(&llama, Some(r"C:\m\Laya-Q8_0.gguf"))[0]
            .r#type
            .as_deref(),
        Some("decision")
    );
    let ps = json!({"models": [{"name": "m:1", "details": {"quantization_level": "Q4"}, "context_length": 2048, "size": 10, "size_vram": 0}]});
    let o = ollama_models(&ps);
    assert_eq!(
        (
            o[0].size_vram,
            o[0].quant.as_deref(),
            o[0].r#type.as_deref()
        ),
        (Some(0), Some("Q4"), Some("llm"))
    );
}

#[test]
fn lmstudio_falls_back_to_v0_when_v1_is_an_error() {
    let http = FakeHttp::default()
        .route(
            1234,
            "/api/v1/models",
            200,
            r#"{"error": "Unexpected endpoint or method. (GET /api/v1/models)"}"#,
        )
        .route(
            1234,
            "/api/v0/models",
            200,
            r#"{"data": [{"id": "qwen", "state": "loaded", "max_context_length": 8192}]}"#,
        );
    let s = probe_server(&http, 1234, None).unwrap();
    assert_eq!(
        (s.kind.as_str(), s.models[0].name.as_deref()),
        ("lmstudio", Some("qwen"))
    );
    assert_eq!(s.url, "http://127.0.0.1:1234");
}

#[test]
fn prober_caches_hits_and_misses_per_port() {
    let http = FakeHttp::default().route(1, "/api/ps", 200, r#"{"models": []}"#);
    let mut p = ServerProber::default();
    assert!(p.get(&http, &[2, 1], 100.0).is_some());
    let n = http.requests().len();
    assert!(p.get(&http, &[2, 1], 101.0).is_some()); // both cached
    assert_eq!(http.requests().len(), n);
    assert!(p.get(&http, &[2, 1], 104.0).is_some()); // hit expired, miss for port 2 not
    assert!(http.requests().len() > n);
    let n = http.requests().len();
    p.get(&http, &[2], 131.0); // miss expired after 30 s
    assert!(http.requests().len() > n);
}

#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::env::Paths;
use gputaskman_core::model::*;
use gputaskman_core::procinfo::classify::{AI_KINDS, classify};
use gputaskman_core::procinfo::models::*;
use gputaskman_core::procinfo::runtimes::*;
use std::collections::BTreeSet;

fn libs(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn measured() -> Engines {
    Engines::Measured {
        by_engine: Default::default(),
    }
}

fn context(dir: &tempfile::TempDir) -> ModelContext {
    ModelContext::new(Paths::rooted(dir.path()))
}

fn model(kind: &str) -> MappedModel {
    MappedModel {
        path: "m.gguf".into(),
        size: 0,
        resident: 0,
        gguf: true,
        sources: vec![Evidence::Mapped],
        store_name: None,
        kind: kind.into(),
        kind_hint: String::new(),
    }
}

#[test]
fn versioned_shared_objects_are_libraries() {
    assert_eq!(
        lib_name("libcudart.so.12.4").as_deref(),
        Some("libcudart.so")
    );
    assert_eq!(lib_name("libllama.so").as_deref(), Some("libllama.so"));
    assert_eq!(lib_name("model.gguf"), None);
}

#[test]
fn linux_runtimes_are_detected() {
    assert_eq!(
        match_runtimes(&libs(&[
            "libllama.so",
            "libggml-cuda.so",
            "libcuda.so",
            "libvulkan.so"
        ])),
        vec!["llama.cpp", "ggml", "CUDA", "Vulkan"]
    );
    assert_eq!(match_runtimes(&libs(&["libamdhip64.so"])), vec!["HIP"]);
}

#[test]
fn ambiguous_extensions_need_size_or_store() {
    assert!(!is_model_file(".bin", 10 * MIB, false, None));
    assert!(is_model_file(".bin", 10 * MIB, false, Some("org/repo")));
    assert!(is_model_file(".bin", 64 * MIB, false, None));
    assert!(is_model_file(".bin", 1, false, Some("org/repo")));
    assert!(is_model_file(".safetensors", 4 * MIB, false, None));
    assert!(!is_model_file(".safetensors", 4 * MIB - 1, false, None));
    assert!(is_model_file("", 1, true, None));
    assert!(!is_model_file(".txt", GIB, false, None));
}

#[test]
fn linux_compositors_and_browsers_and_pid_4() {
    let engines = measured();
    for (name, kind) in [
        ("Xwayland", "compositor"),
        ("gnome-shell", "compositor"),
        ("firefox", "browser"),
    ] {
        assert_eq!(
            classify(&ProcDetail::new(100, name), &engines),
            kind,
            "{name}"
        );
    }
    assert_eq!(classify(&ProcDetail::new(4, "kworker"), &engines), "app");
}

#[test]
fn hip_runtime_is_compute_evidence() {
    let mut d = ProcDetail::new(7, "worker");
    d.runtimes = vec!["HIP".into()];
    assert_eq!(classify(&d, &measured()), "compute");
}

#[test]
fn skip_path_ignores_non_files() {
    assert!(skip_path("[heap]"));
    assert!(skip_path("/dev/nvidia0"));
    assert!(skip_path("/tmp/x.gguf (deleted)"));
    assert!(!skip_path("/home/u/model.gguf"));
}

#[test]
fn diffusion_arguments_do_not_classify_text_encoder_as_llm() {
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<String> = ["diffusion.gguf", "encoder.gguf", "vae.safetensors"]
        .iter()
        .map(|n| {
            let p = dir.path().join(n);
            std::fs::File::create(&p).unwrap();
            p.to_string_lossy().into_owned()
        })
        .collect();
    let cmd: Vec<String> = [
        "sd-cli.exe".to_string(),
        "--diffusion-model".into(),
        paths[0].clone(),
        format!("--llm={}", paths[1]),
        "--vae".into(),
        paths[2].clone(),
    ]
    .to_vec();
    assert_eq!(models_from_cmdline(&cmd), paths);
    let one = vec![
        "sd-cli.exe".to_string(),
        format!("--diffusion-model={}", paths[0]),
    ];
    assert_eq!(models_from_cmdline(&one), vec![paths[0].clone()]);
    let mut d = ProcDetail::new(1, "sd-cli.exe");
    d.cmdline = cmd;
    d.runtimes = vec!["ggml".into()];
    d.models = vec![model("LLM")];
    assert_eq!(classify(&d, &measured()), "diffusion");
    d.models[0].kind_hint = "decision model".into();
    assert_eq!(classify(&d, &measured()), "diffusion");
    assert!(AI_KINDS.contains(&"diffusion"));
}

#[test]
fn decision_name_hints_use_model_names_not_substrings() {
    let dir = tempfile::tempdir().unwrap();
    let mut ctx = context(&dir);
    let matches: &[(&str, Option<&str>)] = &[
        ("/models/OpenJev-Q4_K_M.gguf", None),
        ("/models/Laya-Q8_0.gguf", None),
        ("/models/julia-1.gguf", None),
        ("/models/LEV-Q4_K_M.gguf", None),
        ("/models/kev.gguf", None),
        ("/models/OpenJev/model.safetensors", None),
        ("/models/blob", Some("ggml-org/OpenJev-GGUF")),
        ("/models/blob", Some("openjev:latest")),
        ("/models/blob", Some("qwen3:14b, openjev:latest")),
    ];
    let misses: &[(&str, Option<&str>)] = &[
        ("/models/clever.gguf", None),
        ("/models/kevin.gguf", None),
        ("/models/openjevish.gguf", None),
        ("/home/laya/models/qwen.gguf", None),
        ("/models/model.gguf", Some("openjev/qwen3")),
        ("/models/model.gguf", Some("qwen3:openjev")),
        ("/models/model.gguf", Some("unsloth/Qwen3-GGUF")),
        ("/models/jev.gguf", None),
    ];
    for (cases, expected) in [(matches, "decision model"), (misses, "")] {
        for (path, store) in cases {
            let m = ctx.make_model(
                path,
                1000,
                0,
                false,
                Evidence::Mapped,
                Some(store.unwrap_or("").to_string()),
                0.0,
            );
            assert_eq!(m.kind_hint, expected, "{path} {store:?}");
            assert_eq!(m.sources, vec![Evidence::Mapped]);
        }
    }
}

#[test]
fn installed_kind_hints_do_not_classify_the_process() {
    let dir = tempfile::tempdir().unwrap();
    let m = context(&dir).make_model(
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
    d.candidates = vec![m.clone()];
    assert_eq!(m.kind_label(), "likely decision model (name)");
    assert_eq!(m.evidence_label(), "installed, not proven loaded");
    assert_eq!(classify(&d, &measured()), "AI");
}

#[test]
fn repeated_model_arguments_do_not_duplicate_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b"not a GGUF").unwrap();
    let p = path.to_string_lossy().into_owned();
    let cmd: Vec<String> = ["server", "--model", &p, "--mmproj", &p]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let found = models_from_cmdline(&cmd);
    assert_eq!(found, vec![p.clone(), p.clone()]);
    let mut ctx = context(&dir);
    let mut models: Vec<MappedModel> = Vec::new();
    for f in &found {
        match models.iter_mut().find(|m| &m.path == f) {
            Some(m) => m.add_source(Evidence::Cmdline),
            None => models.push(ctx.make_model(f, 10, 0, true, Evidence::Cmdline, None, 0.0)),
        }
    }
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].sources, vec![Evidence::Cmdline]);
    assert_eq!(models[0].kind, "");
}

#[test]
fn gguf_kind_is_not_read_when_not_gguf_and_llm_kind_classifies() {
    let mut d = ProcDetail::new(1, "python");
    d.models = vec![model("LLM")];
    assert_eq!(classify(&d, &measured()), "LLM");
    d.models[0].kind = "reranker".into();
    assert_eq!(classify(&d, &measured()), "embedding");
    d.models[0].kind = "model".into();
    d.models[0].kind_hint = "decision model".into();
    assert_eq!(classify(&d, &measured()), "decision?");
}

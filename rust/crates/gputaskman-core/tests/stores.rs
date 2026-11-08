#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::{env::Paths, stores::StoreNames};
use std::path::Path;

fn manifest(root: &Path, tag_dir: &str, tag: &str, digest: &str) {
    let folder = tag_dir
        .split('/')
        .fold(root.join("manifests"), |p, s| p.join(s));
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(
        folder.join(tag),
        format!(r#"{{"layers":[{{"digest":"sha256:{digest}"}}]}}"#),
    )
    .unwrap();
}

#[test]
fn store_names() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::rooted(dir.path());
    let digest = "ab".repeat(32);
    let root = dir.path().join("ollama");
    manifest(&root, "registry.ollama.ai/library/qwen3", "14b", &digest);
    manifest(&root, "hf.co/org/thing", "latest", &digest);
    manifest(&root, "registry.ollama.ai/someone/tool", "7b", &digest);
    let blob = root.join(format!("blobs/sha256-{digest}"));
    let mut names = StoreNames::new(paths.clone());
    assert_eq!(
        names.store_name(blob.to_str().unwrap(), 0.0).as_deref(),
        Some("hf.co/org/thing:latest, qwen3:14b, someone/tool:7b")
    );
    let other = root.join(format!("blobs/sha256-{}", "cd".repeat(32)));
    assert_eq!(names.store_name(other.to_str().unwrap(), 0.0), None);
    assert_eq!(
        names
            .store_name(
                "/x/models--Qwen--Qwen3-8B/snapshots/abc/model.safetensors",
                0.0
            )
            .as_deref(),
        Some("Qwen/Qwen3-8B")
    );
    assert_eq!(
        names
            .store_name(
                r"C:\x\hub\models--ggml-org--Laya-GGUF\snapshots\0123\Laya-Q8_0.gguf",
                0.0
            )
            .as_deref(),
        Some("ggml-org/Laya-GGUF")
    );
    let lms = paths
        .home
        .join(".lmstudio/models/lmstudio-community/Qwen3-GGUF");
    std::fs::create_dir_all(&lms).unwrap();
    assert_eq!(
        names
            .store_name(lms.join("q.gguf").to_str().unwrap(), 0.0)
            .as_deref(),
        Some("lmstudio-community/Qwen3-GGUF")
    );
    assert_eq!(names.store_name("/tmp/loose.gguf", 0.0), None);
    let elsewhere = paths.home.join(".lmstudio/models/a/b.gguf");
    assert_eq!(names.store_name(elsewhere.to_str().unwrap(), 0.0), None);
}

#[test]
fn downloads_folder_takes_precedence_and_dirs_are_deduped() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::rooted(dir.path());
    let custom = dir.path().join("custom");
    std::fs::create_dir_all(paths.home.join(".lmstudio")).unwrap();
    std::fs::write(
        paths.home.join(".lmstudio/settings.json"),
        format!(r#"{{"downloadsFolder":"{}"}}"#, custom.display()),
    )
    .unwrap();
    let mut names = StoreNames::new(paths.clone());
    let dirs = names.lmstudio_dirs(0.0);
    assert_eq!(
        dirs,
        vec![
            custom.clone(),
            paths.home.join(".lmstudio/models"),
            paths.home.join(".cache/lm-studio/models"),
        ]
    );
    let file = custom.join("pub/repo/m.gguf");
    assert_eq!(
        names.store_name(file.to_str().unwrap(), 1.0).as_deref(),
        Some("pub/repo")
    );

    // A settings file pointing at a default directory is not listed twice.
    let dup = paths.home.join(".lmstudio/models");
    std::fs::write(
        paths.home.join(".lmstudio/settings.json"),
        format!(r#"{{"downloadsFolder":"{}"}}"#, dup.display()),
    )
    .unwrap();
    let mut fresh = StoreNames::new(paths.clone());
    assert_eq!(fresh.lmstudio_dirs(0.0).len(), 2);
}

#[test]
fn caches_last_sixty_seconds() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::rooted(dir.path());
    let digest = "ab".repeat(32);
    let root = dir.path().join("ollama");
    let mut names = StoreNames::new(paths);
    assert!(names.ollama_names(&root, 100.0).is_empty());
    manifest(&root, "registry.ollama.ai/library/qwen3", "14b", &digest);
    assert!(names.ollama_names(&root, 159.0).is_empty());
    assert_eq!(
        names.ollama_names(&root, 160.0).get(&digest).unwrap(),
        &vec!["qwen3:14b".to_string()]
    );
}

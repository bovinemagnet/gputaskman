//! Friendly model names from where a file lives on disk (Ollama, Hugging Face cache, LM Studio).
//!
//! Pure path work plus small cached manifest reads: no runtime is contacted and no model is
//! loaded. Names are friendly labels only, never evidence of what a process has loaded.
use crate::env::Paths;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};

const TTL: f64 = 60.0;
const OLLAMA_BLOB: &str = r"(?i)^(.*)[\\/]blobs[\\/]sha256-([0-9a-f]{64})$";
const HF_REPO: &str = r"(?i)[\\/]models--([^\\/]+?)--([^\\/]+)[\\/]snapshots[\\/]";

type OllamaMap = BTreeMap<String, Vec<String>>;

pub struct StoreNames {
    paths: Paths,
    ollama_blob: Option<Regex>,
    hf_repo: Option<Regex>,
    ollama_cache: HashMap<PathBuf, (f64, OllamaMap)>,
    lms_cache: Option<(f64, Vec<PathBuf>)>,
}

/// Lexical absolute path, like Python's `os.path.abspath` (no symlink resolution).
fn abspath(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn walk_manifests(dir: &Path, depth: usize, parts: &mut Vec<String>, names: &mut OllamaMap) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if depth < 3 {
                parts.push(name);
                walk_manifests(&path, depth + 1, parts, names);
                parts.pop();
            }
        } else if depth == 3 {
            add_manifest(&path, &name, parts, names);
        }
    }
}

fn add_manifest(path: &Path, tag: &str, parts: &[String], names: &mut OllamaMap) {
    let (registry, namespace, model) = (&parts[0], &parts[1], &parts[2]);
    let mut prefix = String::new();
    if registry != "registry.ollama.ai" {
        prefix.push_str(&format!("{registry}/"));
    }
    if namespace != "library" {
        prefix.push_str(&format!("{namespace}/"));
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return;
    };
    let Some(layers) = value.get("layers").and_then(|l| l.as_array()) else {
        return;
    };
    for layer in layers {
        let digest = layer
            .get("digest")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .trim_start_matches("sha256:")
            .to_lowercase();
        if !digest.is_empty() {
            names
                .entry(digest)
                .or_default()
                .push(format!("{prefix}{model}:{tag}"));
        }
    }
}

impl StoreNames {
    pub fn new(paths: Paths) -> Self {
        Self {
            paths,
            ollama_blob: Regex::new(OLLAMA_BLOB).ok(),
            hf_repo: Regex::new(HF_REPO).ok(),
            ollama_cache: HashMap::new(),
            lms_cache: None,
        }
    }

    /// `{sha256 hex: ["qwen3:14b", ...]}` from `<root>/manifests/<registry>/<namespace>/<model>/<tag>`.
    pub fn ollama_names(&mut self, models_root: &Path, now: f64) -> &OllamaMap {
        let fresh = matches!(self.ollama_cache.get(models_root), Some((t, _)) if now - t < TTL);
        if !fresh {
            let mut names = OllamaMap::new();
            walk_manifests(
                &models_root.join("manifests"),
                0,
                &mut Vec::new(),
                &mut names,
            );
            self.ollama_cache
                .insert(models_root.to_path_buf(), (now, names));
        }
        &self.ollama_cache[models_root].1
    }

    pub fn lmstudio_dirs(&mut self, now: f64) -> Vec<PathBuf> {
        if let Some((t, dirs)) = &self.lms_cache
            && now - t < TTL
        {
            return dirs.clone();
        }
        let home = &self.paths.home;
        let mut dirs = vec![
            home.join(".lmstudio/models"),
            home.join(".cache/lm-studio/models"),
        ];
        for settings in [
            home.join(".lmstudio/settings.json"),
            home.join(".cache/lm-studio/settings.json"),
        ] {
            let folder = std::fs::read_to_string(settings)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v.get("downloadsFolder")?.as_str().map(str::to_owned))
                .filter(|f| !f.is_empty());
            if let Some(folder) = folder {
                dirs.insert(0, PathBuf::from(folder));
            }
        }
        let mut seen = BTreeSet::new();
        let dirs: Vec<PathBuf> = dirs
            .iter()
            .map(|d| abspath(d))
            .filter(|d| seen.insert(d.clone()))
            .collect();
        self.lms_cache = Some((now, dirs.clone()));
        dirs
    }

    /// `qwen3:14b` (Ollama), `org/repo` (Hugging Face cache) or `publisher/repo` (LM Studio).
    pub fn store_name(&mut self, path: &str, now: f64) -> Option<String> {
        if let Some(m) = self.ollama_blob.as_ref().and_then(|r| r.captures(path)) {
            let (root, digest) = (m[1].to_string(), m[2].to_lowercase());
            let names = self.ollama_names(Path::new(&root), now).get(&digest)?;
            let unique: BTreeSet<&String> = names.iter().collect();
            return Some(
                unique
                    .into_iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        if let Some(m) = self.hf_repo.as_ref().and_then(|r| r.captures(path)) {
            return Some(format!("{}/{}", &m[1], &m[2]));
        }
        let full = abspath(Path::new(path));
        for root in self.lmstudio_dirs(now) {
            if let Ok(rel) = full.strip_prefix(&root) {
                let parts: Vec<_> = rel.components().collect();
                if parts.len() >= 3 {
                    return Some(format!(
                        "{}/{}",
                        parts[0].as_os_str().to_string_lossy(),
                        parts[1].as_os_str().to_string_lossy()
                    ));
                }
            }
        }
        None
    }
}

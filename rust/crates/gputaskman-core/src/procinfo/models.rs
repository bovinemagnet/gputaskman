//! Model-file rules: which mapped or named files count as models (`procinfo.py`).
use std::path::{Component, Path, PathBuf};

use crate::env::Paths;
use crate::gguf::GgufCache;
use crate::model::{Evidence, MIB, MappedModel, is_diffusion_cmdline};
use crate::stores::StoreNames;

pub const MODEL_EXTS: &[&str] = &[
    ".gguf",
    ".ggml",
    ".safetensors",
    ".onnx",
    ".pt",
    ".pth",
    ".bin",
    ".engine",
    ".plan",
    ".tflite",
    ".ckpt",
];
/// Also used for caches and tiles, so these require a size or a model store.
pub const AMBIGUOUS_EXTS: &[&str] = &[".bin", ".pt", ".pth", ".ckpt"];
pub const NOT_MODELS: &[&str] = &[
    ".dll", ".exe", ".sys", ".mui", ".ttf", ".ttc", ".otf", ".dat", ".db", ".nls", ".pak", ".mun",
];

/// Paths in `/proc/<pid>/maps` that are not regular files: never stat or open them.
const SPECIAL_PREFIXES: &[&str] = &[
    "[",
    "/dev/",
    "/proc/",
    "/sys/",
    "/memfd:",
    "/SYSV",
    "anon_inode:",
];
/// macOS: other apps' sandbox containers are privacy-protected and open() on them can block.
const MAC_SKIP_PREFIXES: &[&str] = &[
    "/System/",
    "/usr/lib/",
    "/usr/share/",
    "/private/var/db/",
    "/Library/Apple/",
];
const MAC_SKIP_PARTS: &[&str] = &["/Library/Containers/", "/Library/Group Containers/"];

const DECISION_NAMES: &[&str] = &["openjev", "laya", "julia-1", "lev", "kev"];

#[must_use]
pub fn skip_path(path: &str) -> bool {
    if SPECIAL_PREFIXES.iter().any(|p| path.starts_with(p)) || path.ends_with(" (deleted)") {
        return true;
    }
    cfg!(target_os = "macos")
        && (MAC_SKIP_PREFIXES.iter().any(|p| path.starts_with(p))
            || MAC_SKIP_PARTS.iter().any(|p| path.contains(p)))
}

/// Like Python's `os.path.abspath`: absolute against the current directory with `.` and `..`
/// collapsed lexically. The filesystem is not consulted and symlinks are not resolved.
fn abspath(path: &Path) -> Option<PathBuf> {
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut out = PathBuf::new();
    for part in full.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    Some(out)
}

/// Existing files named by model flags on the command line, as absolute paths.
#[must_use]
pub fn models_from_cmdline(cmd: &[String]) -> Vec<String> {
    let mut flags = vec![
        "-m",
        "--model",
        "--mmproj",
        "-md",
        "--model-draft",
        "--lora",
    ];
    if is_diffusion_cmdline(cmd) {
        flags.extend(["--diffusion-model", "--vae", "--llm"]);
    }
    let mut paths = Vec::new();
    for (i, arg) in cmd.iter().enumerate() {
        let (key, value) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg.as_str(), None),
        };
        if !flags.contains(&key) {
            continue;
        }
        let val = match value {
            Some(v) => Some(v),
            None => cmd.get(i + 1).map(String::as_str),
        };
        if let Some(v) = val.filter(|v| !v.is_empty() && Path::new(v).is_file())
            && let Some(abs) = abspath(Path::new(v))
        {
            paths.push(abs.to_string_lossy().into_owned());
        }
    }
    paths
}

#[must_use]
pub fn is_model_file(ext: &str, size: u64, gguf: bool, store: Option<&str>) -> bool {
    if gguf {
        return true;
    }
    if !MODEL_EXTS.contains(&ext) {
        return false;
    }
    if AMBIGUOUS_EXTS.contains(&ext) {
        return size >= 64 * MIB || store.is_some();
    }
    size >= 4 * MIB
}

/// `(?<![a-z0-9])(?:openjev|laya|julia-1|lev|kev)(?![a-z0-9])`, case-insensitive.
fn has_decision_name(name: &str) -> bool {
    let b = name.to_ascii_lowercase().into_bytes();
    let word = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    (0..b.len()).any(|i| {
        (i == 0 || !word(b[i - 1]))
            && DECISION_NAMES.iter().any(|alt| {
                b[i..].starts_with(alt.as_bytes()) && b.get(i + alt.len()).is_none_or(|c| !word(*c))
            })
    })
}

/// Name-based suggestion only; never changes loading evidence or GGUF maths.
#[must_use]
pub fn name_kind_hint(path: &str, store: Option<&str>) -> String {
    let p = Path::new(path);
    let base = |q: Option<&Path>| {
        q.and_then(Path::file_name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let mut names = vec![base(Some(p)), base(p.parent())];
    for name in store.unwrap_or("").split(", ") {
        let last = name.rsplit('/').next().unwrap_or("");
        names.push(last.split(':').next().unwrap_or("").to_string());
    }
    if names.iter().any(|n| has_decision_name(n)) {
        "decision model".into()
    } else {
        String::new()
    }
}

/// Caches and lookups needed to build a [`MappedModel`].
pub struct ModelContext {
    pub gguf: GgufCache,
    pub stores: StoreNames,
}

impl ModelContext {
    #[must_use]
    pub fn new(paths: Paths) -> Self {
        Self {
            gguf: GgufCache::new(),
            stores: StoreNames::new(paths),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn make_model(
        &mut self,
        path: &str,
        size: u64,
        resident: u64,
        gguf: bool,
        source: Evidence,
        store: Option<String>,
        now: f64,
    ) -> MappedModel {
        let kind = if gguf {
            self.gguf
                .read(Path::new(path))
                .filter(|g| g.error.is_none())
                .map(|g| g.kind().to_string())
                .unwrap_or_default()
        } else {
            String::new()
        };
        let store = store.or_else(|| self.stores.store_name(path, now));
        let hint = if matches!(kind.as_str(), "" | "model" | "LLM") {
            name_kind_hint(path, store.as_deref())
        } else {
            String::new()
        };
        MappedModel {
            path: path.to_string(),
            size,
            resident,
            gguf,
            sources: vec![source],
            store_name: store,
            kind,
            kind_hint: hint,
        }
    }
}

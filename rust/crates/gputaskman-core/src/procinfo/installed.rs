//! Model files in an app's install and data folders (`procinfo.installed_models`). Fallback for
//! runtimes that read a model into RAM and close the file, which leaves no OS evidence of which
//! model is loaded. These are candidates only, never proof.
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::env::Paths;
use crate::gguf::is_gguf;
use crate::model::{Evidence, MappedModel};
use crate::procinfo::models::{MODEL_EXTS, ModelContext, is_model_file};
use crate::procinfo::runtimes::splitext;

const INSTALLED_TTL: f64 = 300.0;
const WALK_DEPTH: u32 = 4;
const WALK_BUDGET: i64 = 3000;
const MAX_RESULTS: usize = 6;
const SYSTEM_DIRS: &[&str] = &[
    "/usr/bin/",
    "/usr/sbin/",
    "/bin/",
    "/sbin/",
    "/usr/lib",
    "/lib",
    "/usr/libexec/",
    "/snap/",
];
const SKIP_DIRS: &[&str] = &[
    "ebwebview",
    "cache",
    "code cache",
    "gpucache",
    "node_modules",
    "logs",
    "crashpad",
    "locales",
    "__pycache__",
];

/// `/usr/bin/python3` is shared by every Python app: walking `/usr/bin` finds nothing about the process.
fn is_system_path(exe: &str) -> bool {
    SYSTEM_DIRS.iter().any(|d| exe.starts_with(d))
}

fn walk(root: &Path, depth: u32, out: &mut Vec<(String, u64)>, budget: &mut i64) {
    let Ok(rd) = fs::read_dir(root) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(fs::DirEntry::file_name);
    for e in entries {
        *budget -= 1;
        if *budget < 0 {
            return;
        }
        let Ok(kind) = e.file_type() else { continue };
        let name = e.file_name().to_string_lossy().to_lowercase();
        if kind.is_dir() {
            if depth > 0 && !SKIP_DIRS.contains(&name.as_str()) {
                walk(&e.path(), depth - 1, out, budget);
            }
            continue;
        }
        let ext = splitext(&name);
        if MODEL_EXTS.contains(&ext)
            && let Ok(meta) = fs::metadata(e.path())
            && is_model_file(ext, meta.len(), false, None)
        {
            out.push((e.path().to_string_lossy().into_owned(), meta.len()));
        }
    }
}

/// Per-exe cache of installed-model candidates.
pub struct Installed {
    paths: Paths,
    cache: BTreeMap<String, (f64, Vec<MappedModel>)>,
}

impl Installed {
    #[must_use]
    pub fn new(paths: Paths) -> Self {
        Self {
            paths,
            cache: BTreeMap::new(),
        }
    }

    /// Bounded walk (depth 4, 3000 entries, at most 6 results) cached per exe for 5 minutes.
    pub fn models(&mut self, exe: &str, ctx: &mut ModelContext, now: f64) -> Vec<MappedModel> {
        if exe.is_empty() {
            return Vec::new();
        }
        // Linux paths are case-sensitive, so the key keeps case (Python lower-cases on Windows only).
        if let Some((at, hit)) = self.cache.get(exe)
            && now - at < INSTALLED_TTL
        {
            return hit.clone();
        }
        let exe_path = Path::new(exe);
        let stem = exe_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        if !is_system_path(exe)
            && let Some(dir) = exe_path.parent()
        {
            roots.push(dir.to_path_buf());
        }
        if stem.chars().count() >= 4 {
            for base in [&self.paths.xdg_data, &self.paths.xdg_config] {
                let Ok(rd) = fs::read_dir(base) else { continue };
                for e in rd.flatten() {
                    let p = e.path();
                    if e.file_name()
                        .to_string_lossy()
                        .to_lowercase()
                        .contains(&stem)
                        && p.is_dir()
                    {
                        roots.push(p);
                    }
                }
            }
        }
        let mut seen: Vec<&std::path::PathBuf> = Vec::new();
        let mut files = Vec::new();
        let mut budget = WALK_BUDGET;
        for root in &roots {
            if !seen.contains(&root) {
                seen.push(root);
                walk(root, WALK_DEPTH, &mut files, &mut budget);
            }
        }
        let mut by_dir: Vec<(String, Vec<(String, u64)>)> = Vec::new();
        for (path, size) in files {
            let folder = Path::new(&path)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            match by_dir.iter_mut().find(|(f, _)| *f == folder) {
                Some((_, g)) => g.push((path, size)),
                None => by_dir.push((folder, vec![(path, size)])),
            }
        }
        let mut found = Vec::new();
        for (folder, group) in by_dir {
            if let [(path, size)] = group.as_slice() {
                // One file: name it by the file.
                let gguf = path.to_lowercase().ends_with(".gguf") && is_gguf(Path::new(path));
                found.push(ctx.make_model(path, *size, 0, gguf, Evidence::Installed, None, now));
            } else {
                // Several files (encoder + decoder, shards): name it by the folder.
                let total = group.iter().map(|(_, s)| s).sum();
                let store = ctx.stores.store_name(&group[0].0, now);
                found.push(ctx.make_model(
                    &folder,
                    total,
                    0,
                    false,
                    Evidence::Installed,
                    store,
                    now,
                ));
            }
        }
        found.sort_by_key(|m| std::cmp::Reverse(m.size));
        found.truncate(MAX_RESULTS);
        self.cache.insert(exe.to_string(), (now, found.clone()));
        found
    }
}

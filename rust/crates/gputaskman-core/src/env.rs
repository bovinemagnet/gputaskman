//! Locations of per-user directories, injectable so tests never touch the real home.
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub home: PathBuf,
    pub xdg_data: PathBuf,
    pub xdg_config: PathBuf,
}

fn non_empty(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

impl Paths {
    /// Reads `HOME`, `XDG_DATA_HOME` and `XDG_CONFIG_HOME`, with the XDG defaults.
    pub fn from_env() -> Self {
        let home = non_empty("HOME").unwrap_or_default();
        let xdg_data = non_empty("XDG_DATA_HOME").unwrap_or_else(|| home.join(".local/share"));
        let xdg_config = non_empty("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config"));
        Self {
            home,
            xdg_data,
            xdg_config,
        }
    }

    /// All three locations under `dir`, for tests.
    pub fn rooted(dir: &Path) -> Self {
        Self {
            home: dir.to_path_buf(),
            xdg_data: dir.join(".local/share"),
            xdg_config: dir.join(".config"),
        }
    }
}

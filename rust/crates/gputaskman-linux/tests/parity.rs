#![cfg(feature = "parity")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Backend parity with the Python implementation: the same scenario runs through Python's
//! `DrmCounters`/`LinuxCounters` (with the Python test fakes) and through the Rust backends, and the
//! per-process results must be equal. Needs `uv` and the synced Python environment.
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn python(scenario: &Path) -> serde_json::Value {
    let out = Command::new("uv")
        .args(["run", "--locked", "python", "rust/parity/run_python.py"])
        .arg(scenario)
        .current_dir(repo_root())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn backends_match_python() {
    for name in [
        "drm_private",
        "drm_shared",
        "nvml_bare_metal",
        "nvml_container",
    ] {
        let path = repo_root().join(format!("rust/parity/scenarios/{name}.json"));
        let expected = python(&path);
        let actual = gputaskman_linux::parity::run_scenario(&path).unwrap();
        assert_eq!(actual, expected, "{name}");
    }
}

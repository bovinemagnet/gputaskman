#![allow(clippy::unwrap_used, clippy::expect_used)]
use assert_cmd::Command;

fn cmd(dir: &std::path::Path) -> Command {
    let mut c = Command::cargo_bin("gputaskman").unwrap();
    std::fs::create_dir_all(dir.join("proc")).unwrap();
    std::fs::write(dir.join("proc/stat"), "btime 1700000000\n").unwrap();
    std::fs::create_dir_all(dir.join("sys/class/drm")).unwrap();
    c.env("GPUTASKMAN_PROC_ROOT", dir.join("proc"))
        .env("GPUTASKMAN_SYS_ROOT", dir.join("sys"))
        .env("HOME", dir)
        .env("LD_LIBRARY_PATH", dir); // no libnvidia-ml.so can be found from here
    c
}

/// True when this machine's NVIDIA driver loaded, so "no adapters" assertions do not apply.
fn nvml_loaded(dir: &std::path::Path) -> bool {
    let out = cmd(dir)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    let loaded = !v["notes"].to_string().contains("NVML not loaded");
    if loaded {
        eprintln!("note: NVML loaded on this machine; skipping the no-adapter assertions");
    }
    loaded
}

#[test]
fn help_needs_no_gpu() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path())
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("--json"));
}

#[test]
fn invalid_options_fail_before_collecting() {
    let dir = tempfile::tempdir().unwrap();
    for bad in [["--interval", "0"], ["--interval", "nan"], ["--gpu", "-1"]] {
        cmd(dir.path()).args(bad).arg("--once").assert().code(2);
    }
}

#[test]
fn interval_message_matches_python() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path())
        .args(["--interval", "0", "--once"])
        .assert()
        .code(2)
        .stderr(predicates::str::contains(
            "--interval must be a finite number greater than zero",
        ));
}

#[test]
fn the_tui_needs_a_terminal() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path())
        .assert()
        .code(2)
        .stderr(predicates::str::contains(
            "the TUI needs an interactive terminal; use --once or --json",
        ));
}

#[test]
fn json_without_a_supported_gpu_is_valid_v2() {
    let dir = tempfile::tempdir().unwrap();
    let out = cmd(dir.path())
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["schema_version"], 2);
    assert!(v["processes"].is_array());
    if v["notes"].to_string().contains("NVML not loaded") {
        assert!(v["adapter"].is_null());
        assert!(v["processes"].as_array().unwrap().is_empty());
    }
    if v["adapter"].is_null() {
        let notes: Vec<&str> = v["notes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect();
        let reason = "no supported GPU found (NVIDIA through NVML, or an amdgpu card)";
        assert_eq!(
            notes.iter().filter(|n| **n == reason).count(),
            1,
            "{notes:?}"
        );
    }
}

#[test]
fn out_of_range_gpu_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    if nvml_loaded(dir.path()) {
        return;
    }
    cmd(dir.path())
        .args(["--gpu", "2", "--once"])
        .assert()
        .code(2)
        .stderr(predicates::str::contains(
            "--gpu 2 is out of range; 0 GPU adapter(s) available",
        ));
}

#[test]
fn list_gpus_with_none_is_empty_and_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    if nvml_loaded(dir.path()) {
        return;
    }
    cmd(dir.path())
        .arg("--list-gpus")
        .assert()
        .success()
        .stdout("");
}

#[test]
fn diff_text_json_and_errors() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    std::fs::write(&a, r#"{"taken":1,"gpu":"RTX","backend":"nvml","vram_used":0,"shared_used":0,"attributed":0,"processes":[]}"#).unwrap();
    cmd(dir.path())
        .args(["diff"])
        .arg(&a)
        .arg(&a)
        .assert()
        .success();
    cmd(dir.path())
        .args(["diff", "--json"])
        .arg(&a)
        .arg(&a)
        .assert()
        .success();
    cmd(dir.path())
        .args(["diff"])
        .arg(&a)
        .arg(dir.path().join("missing.json"))
        .assert()
        .code(2);
}

#[test]
fn closed_stdout_pipe_exits_cleanly_without_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let base = cmd(dir.path());
    let mut child = std::process::Command::new(base.get_program())
        .envs(
            base.get_envs()
                .filter_map(|(k, v)| v.map(|v| (k.to_owned(), v.to_owned()))),
        )
        .arg("--json")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // Close the reading end before the snapshot is written (collection waits a second first).
    drop(child.stdout.take());
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
    assert_eq!(out.status.code(), Some(0), "stderr: {stderr}");
}

#[test]
fn a_huge_interval_is_a_usage_error_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    cmd(dir.path())
        .args(["-i", "1e30"])
        .assert()
        .code(2)
        .stderr(predicates::str::contains("--interval is too large"));
}

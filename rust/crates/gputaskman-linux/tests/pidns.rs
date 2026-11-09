#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::Confidence;
use gputaskman_linux::pidns::{NamespaceProbe, PidMatcher, ProcNamespace};
use std::os::unix::fs::symlink;
use std::path::Path;

#[test]
fn host_pids_are_matched_to_container_pids_by_creation_order() {
    let mut m = PidMatcher::default();
    let (out, outside) = m.update(&[5000, 5100], &[20, 30]);
    assert_eq!(out[&5100], (30, Confidence::Consistent));
    assert_eq!(out[&5000], (20, Confidence::Consistent));
    assert!(outside.is_empty());
}

#[test]
fn extra_outside_compute_is_a_guess_until_a_start_event() {
    let mut m = PidMatcher::default();
    let (out, outside) = m.update(&[5000, 5100], &[30]);
    assert_eq!(out[&5100], (30, Confidence::Order));
    assert_eq!(outside, vec![5000]);
    let (out, _) = m.update(&[5000, 5100, 5200], &[30, 40]);
    assert_eq!(out[&5200], (40, Confidence::Consistent)); // appeared together
}

#[test]
fn a_matched_pair_is_kept_while_both_sides_live() {
    let mut m = PidMatcher::default();
    m.update(&[5000], &[20]);
    let (out, outside) = m.update(&[5000, 5100], &[20, 30]);
    assert_eq!(out[&5000], (20, Confidence::Consistent));
    assert_eq!(out[&5100], (30, Confidence::Consistent));
    assert!(outside.is_empty());
    let (out, _) = m.update(&[5100], &[30]);
    assert_eq!(out.len(), 1);
}

fn root_with_ns(link: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    if let Some(target) = link {
        let ns = dir.path().join("self/ns");
        std::fs::create_dir_all(&ns).unwrap();
        symlink(target, ns.join("pid")).unwrap();
    }
    dir
}

#[test]
fn namespace_detection() {
    let host = root_with_ns(Some("pid:[4026531836]"));
    assert_eq!(
        ProcNamespace::new(host.path()).in_child_pid_namespace(),
        Some(false)
    );
    let child = root_with_ns(Some("pid:[4026532481]"));
    assert_eq!(
        ProcNamespace::new(child.path()).in_child_pid_namespace(),
        Some(true)
    );
    let none = root_with_ns(None);
    assert_eq!(
        ProcNamespace::new(none.path()).in_child_pid_namespace(),
        None
    );
}

fn fake_proc(root: &Path, pid: u32, fd_target: &str, maps: &str) {
    let dir = root.join(pid.to_string());
    std::fs::create_dir_all(dir.join("fd")).unwrap();
    symlink(fd_target, dir.join("fd/3")).unwrap();
    std::fs::write(dir.join("maps"), maps).unwrap();
}

#[test]
fn holders_cuda_and_existence_come_from_the_proc_root() {
    let dir = tempfile::tempdir().unwrap();
    fake_proc(
        dir.path(),
        547,
        "/dev/nvidia0",
        "7f00 r-xp 0 00:00 0 /usr/lib/libcuda.so.1\n",
    );
    fake_proc(
        dir.path(),
        564,
        "/dev/nvidiactl",
        "7f00 r-xp 0 00:00 0 /bin/nvidia-smi\n",
    );
    fake_proc(dir.path(), 700, "/dev/null", "");
    let ns = ProcNamespace::new(dir.path());
    assert_eq!(
        ns.nvidia_holders().into_iter().collect::<Vec<_>>(),
        vec![547, 564]
    );
    assert!(ns.has_cuda(547));
    assert!(!ns.has_cuda(564)); // NVML-only tools never load libcuda
    assert!(!ns.has_cuda(9999));
    assert!(ns.exists(700));
    assert!(!ns.exists(701));
}

#[test]
fn outside_rows_never_borrow_a_local_identity() {
    // Host PIDs that cannot be matched stay unmatched; they are never looked up locally.
    let mut m = PidMatcher::default();
    let (out, outside) = m.update(&[1145, 1360], &[]);
    assert!(out.is_empty());
    assert_eq!(outside, vec![1360, 1145]);
}

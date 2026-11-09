#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;
use gputaskman_core::model::{BackendKind, Engines, ProcGpu};
use gputaskman_core::traits::{GpuBackend, ScanPolicy};
use gputaskman_core::{provenance::*, testing::FakeClock};
use gputaskman_linux::drm::backend::DrmBackend;
use gputaskman_linux::drm::cards::DrmCard;
use gputaskman_linux::drm::counters::DrmCounters;
use gputaskman_linux::drm::device::read_device;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::{sync::Arc, time::Duration};
use support::*;

const PDEV: &str = "0000:03:00.0";

fn setup() -> (tempfile::TempDir, DrmCounters, Arc<FakeClock>, u64) {
    let dir = tempfile::tempdir().unwrap();
    let dev = make_card(
        dir.path(),
        "card0",
        PDEV,
        0x1002,
        0x7550,
        16 * GIB,
        "amdgpu",
        0,
        false,
    );
    std::fs::write(dev.join("mem_info_vram_used"), (2 * GIB).to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_used"), (GIB).to_string()).unwrap();
    let c = card(&dev, PDEV, "AMD GPU", 16 * GIB);
    let key = c.key;
    let clock = Arc::new(FakeClock::new(0.0));
    std::fs::create_dir_all(dir.path().join("proc")).unwrap();
    let counters = DrmCounters::new(vec![c], dir.path().join("proc"), clock.clone());
    (dir, counters, clock, key)
}

#[test]
fn private_render_client_is_measured_and_engine_load_needs_two_samples() {
    let (dir, mut counters, clock, key) = setup();
    let proc_root = dir.path().join("proc");
    write_fd(
        &proc_root,
        100,
        5,
        &fdinfo(1, PDEV, 0, 1024, 2048, 0, 0, 0),
        "/dev/dri/renderD128",
    );
    write_stat(&proc_root, 100, 555);
    let (procs, adapters) = counters.sample();
    let g = &procs[&key][&100];
    assert_eq!(g.vram, Bytes::measured(1024 * 1024));
    assert_eq!(g.shared, Bytes::measured(2048 * 1024));
    assert!(matches!(
        g.engines,
        Engines::Unavailable {
            reason: Reason::NoBaseline
        }
    ));
    assert_eq!(adapters[&key].vram_used, Bytes::measured(2 * GIB));

    clock.advance(Duration::from_secs(1));
    std::fs::write(
        proc_root.join("100/fdinfo/5"),
        fdinfo(1, PDEV, 500_000_000, 1024, 2048, 0, 0, 0),
    )
    .unwrap();
    let (procs, _) = counters.sample();
    assert_eq!(procs[&key][&100].engines.get("3d"), 50.0);
}

#[test]
fn any_shared_buffer_leaves_the_process_unattributed() {
    let (dir, mut counters, _clock, key) = setup();
    let p = dir.path().join("proc");
    write_fd(
        &p,
        100,
        5,
        &fdinfo(1, PDEV, 0, 1024, 0, 4, 0, 0),
        "/dev/dri/renderD128",
    );
    write_stat(&p, 100, 1);
    let (procs, _) = counters.sample();
    let g = &procs[&key][&100];
    assert_eq!(g.vram, Bytes::unavailable(Reason::SharedBuffers));
    assert_eq!(g.resident.unwrap().vram, Bytes::measured(1024 * 1024)); // reference only
}

#[test]
fn client_held_by_several_processes_is_charged_to_none() {
    let (dir, mut counters, _clock, key) = setup();
    let p = dir.path().join("proc");
    write_fd(
        &p,
        100,
        5,
        &fdinfo(7, PDEV, 0, 1024, 0, 0, 0, 0),
        "/dev/dri/renderD128",
    );
    write_fd(
        &p,
        101,
        9,
        &fdinfo(7, PDEV, 0, 1024, 0, 0, 0, 0),
        "/dev/dri/renderD128",
    );
    write_stat(&p, 100, 1);
    write_stat(&p, 101, 2);
    let (procs, _) = counters.sample();
    for pid in [100, 101] {
        assert_eq!(
            procs[&key][&pid].vram,
            Bytes::unavailable(Reason::SharedClient)
        );
    }
}

#[test]
fn primary_node_is_not_charged() {
    let (dir, mut counters, _clock, key) = setup();
    let p = dir.path().join("proc");
    write_fd(
        &p,
        100,
        5,
        &fdinfo(1, PDEV, 0, 1024, 0, 0, 0, 0),
        "/dev/dri/card0",
    );
    write_stat(&p, 100, 1);
    let (procs, _) = counters.sample();
    assert_eq!(procs[&key][&100].vram, Bytes::unavailable(Reason::Primary));
}

// ---- helpers shared by the ported Python tests ----

const RENDER: &str = "/dev/dri/renderD128";

/// `_fdinfo(client, pdev, gfx=, vram=)` with every other value 0.
fn fdi(client: u64, gfx: u64, vram: u64) -> String {
    fdinfo(client, PDEV, gfx, vram, 0, 0, 0, 0)
}

/// One fd plus the process's stat file (start time = pid), as every fixture process needs.
fn proc_fd(proc_root: &Path, pid: u32, fd: u32, text: &str, target: &str) {
    write_fd(proc_root, pid, fd, text, target);
    write_stat(proc_root, pid, u64::from(pid));
}

fn engines_unavailable(g: &ProcGpu) -> Option<Reason> {
    match g.engines {
        Engines::Unavailable { reason } => Some(reason),
        Engines::Measured { .. } => None,
    }
}

fn is_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

/// Restores a path's mode when dropped, so the tempdir can be removed even if an assertion fails.
struct Restore(PathBuf, u32);

impl Restore {
    fn chmod(path: PathBuf, mode: u32) -> Self {
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        Self(path, 0o755)
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(self.1));
    }
}

/// `DrmScanTests._setup`: a card with no sysfs counters and an empty `/proc`.
fn scan_setup(tmp: &Path) -> (DrmCard, PathBuf) {
    let dev = tmp.join("drm").join("card1").join("device");
    std::fs::create_dir_all(&dev).unwrap();
    let c = card(&dev, PDEV, "AMD GPU", 16 * GIB);
    let proc_root = tmp.join("proc");
    std::fs::create_dir_all(&proc_root).unwrap();
    (c, proc_root)
}

/// The Python budget tests' `timer`, which moves one second on every call.
fn ticking() -> Arc<FakeClock> {
    Arc::new(FakeClock::new(0.0).auto_advance(Duration::from_secs(1)))
}

struct Edge {
    card: DrmCard,
    proc_root: PathBuf,
    counters: DrmCounters,
    clock: Arc<FakeClock>,
}

/// `DrmEdgeCaseTests._counters`: fds written, baseline taken, clock moved on one second.
fn edge(tmp: &Path, fds: &[(u32, u32, String, &str)], unified_video: bool, sysfs: bool) -> Edge {
    let dev = tmp.join("drm").join("card1").join("device");
    std::fs::create_dir_all(&dev).unwrap();
    if sysfs {
        std::fs::write(dev.join("mem_info_vram_used"), GIB.to_string()).unwrap();
        std::fs::write(dev.join("mem_info_gtt_used"), "0").unwrap();
    }
    let mut c = card(&dev, PDEV, "AMD GPU", 16 * GIB);
    c.unified_video = unified_video;
    let proc_root = tmp.join("proc");
    std::fs::create_dir_all(&proc_root).unwrap();
    for (pid, fd, text, target) in fds {
        proc_fd(&proc_root, *pid, *fd, text, target);
    }
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c.clone()], proc_root.clone(), clock.clone());
    counters.start();
    clock.advance(Duration::from_secs(1));
    Edge {
        card: c,
        proc_root,
        counters,
        clock,
    }
}

// ---- DrmCountersTests ----

#[test]
fn test_drm_counters() {
    let tmp = tempfile::tempdir().unwrap();
    let tmp = tmp.path();
    let dev = tmp.join("drm").join("card1").join("device");
    std::fs::create_dir_all(&dev).unwrap();
    std::fs::write(dev.join("mem_info_vram_total"), (16 * GIB).to_string()).unwrap();
    std::fs::write(dev.join("mem_info_vram_used"), GIB.to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_total"), (8 * GIB).to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_used"), (64 * 1024).to_string()).unwrap();
    let c = card(&dev, PDEV, "AMD GPU", 16 * GIB);
    let key = c.key;

    let p = tmp.join("proc");
    proc_fd(
        &p,
        100,
        10,
        &fdinfo(26, PDEV, 1_000_000_000, 3, 2, 0, 0, 0),
        RENDER,
    );
    proc_fd(
        &p,
        300,
        12,
        &fdinfo(27, "0000:ff:00.0", 9_999_999_999, 4, 0, 0, 0, 0),
        RENDER,
    );
    // DRM-looking fdinfo behind a non-DRM fd: the /proc/<pid>/fd target rules it out.
    proc_fd(&p, 400, 13, &fdi(28, 9_999_999_999, 4), "socket:[123]");

    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c], p.clone(), clock.clone());
    counters.start();
    clock.advance(Duration::from_secs(1));

    let (procs, adapters) = counters.sample();
    assert_eq!(procs.keys().copied().collect::<Vec<_>>(), vec![key]);
    assert!(procs[&key].contains_key(&100));
    assert!(!procs[&key].contains_key(&300)); // unknown pdev
    assert!(!procs[&key].contains_key(&400)); // non-DRM fd
    let g = &procs[&key][&100];
    assert_eq!(g.vram, Bytes::measured(3 * 1024)); // private client, no shared buffers
    assert_eq!(g.resident.unwrap().vram, Bytes::measured(3 * 1024));
    assert_eq!(g.shared, Bytes::measured(2048));
    // No commit counter: unavailable, not 0.
    assert_eq!(g.committed, Bytes::unavailable(Reason::NotOnBackend));
    assert_eq!(engines_unavailable(g), None);
    assert_eq!(g.engines.get("3d"), 0.0); // idle since the baseline
    assert_eq!(adapters[&key].vram_used, Bytes::measured(GIB));
    assert_eq!(adapters[&key].shared_used, Bytes::measured(64 * 1024));

    std::fs::write(dev.join("mem_info_vram_used"), (2 * GIB).to_string()).unwrap();
    proc_fd(
        &p,
        100,
        10,
        &fdinfo(26, PDEV, 1_500_000_000, 1, 2, 0, 0, 0),
        RENDER,
    );
    clock.advance(Duration::from_secs(1));
    let (procs, adapters) = counters.sample();
    assert_eq!(procs[&key][&100].engines.get("3d"), 50.0);
    assert_eq!(engines_unavailable(&procs[&key][&100]), None);
    assert_eq!(adapters[&key].vram_used, Bytes::measured(2 * GIB));
}

#[test]
fn test_constructor_does_not_scan() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = tmp.path().join("drm").join("card1").join("device");
    std::fs::create_dir_all(&dev).unwrap();
    // A missing /proc would set the status on the first scan: an NVIDIA GPU may be selected, so no
    // fd walk happens until start().
    let counters = DrmCounters::new(
        vec![card(&dev, PDEV, "AMD GPU", 16 * GIB)],
        tmp.path().join("proc"),
        Arc::new(FakeClock::new(0.0)),
    );
    assert_eq!(counters.status(), "");
    assert!(counters.known_pids().is_empty());
}

#[test]
fn test_rocm_process_engine_load_is_unknown() {
    // A HIP process holds /dev/kfd next to its render node: its VRAM is in fdinfo, but its compute
    // runs on KFD user queues with no per-process busy time, so engines are unavailable.
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 3, "pos:\t0\nflags:\t02\n", "/dev/kfd");
    proc_fd(&p, 100, 7, &fdi(500, 0, 3891436), RENDER);
    proc_fd(&p, 200, 7, &fdi(501, 5, 4), RENDER); // a graphics client, for contrast
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c], p, clock.clone());
    counters.start();
    clock.advance(Duration::from_secs(1));
    let (procs, _) = counters.sample();
    assert_eq!(procs[&key][&100].vram, Bytes::measured(3891436 * 1024));
    assert!(procs[&key][&100].kfd);
    assert_eq!(
        engines_unavailable(&procs[&key][&100]),
        Some(Reason::KfdUntimed)
    );
    assert_eq!(engines_unavailable(&procs[&key][&200]), None);
}

// ---- DrmEdgeCaseTests ----

#[test]
fn test_old_kernel_without_shared_keys_is_not_summed() {
    let old =
        "drm-driver:\tamdgpu\ndrm-client-id:\t5\ndrm-pdev:\tPDEV\ndrm-memory-vram:\t4096 KiB\n"
            .replace("PDEV", PDEV);
    let tmp = tempfile::tempdir().unwrap();
    let mut e = edge(tmp.path(), &[(100, 5, old, RENDER)], false, true);
    let g = e.counters.sample().0[&e.card.key][&100].clone();
    // Resident may include imports: it would double count.
    assert_eq!(g.vram, Bytes::unavailable(Reason::Legacy));
    assert_eq!(g.shared, Bytes::unavailable(Reason::Legacy));
    assert_eq!(g.resident.unwrap().vram, Bytes::measured(4096 * 1024)); // still shown as a reference
}

#[test]
fn test_unparseable_memory_is_unknown() {
    let text = fdi(7, 0, 3).replace("drm-resident-vram:\t3 KiB", "drm-resident-vram:\tlots");
    let tmp = tempfile::tempdir().unwrap();
    let mut e = edge(tmp.path(), &[(100, 5, text, RENDER)], false, true);
    let g = e.counters.sample().0[&e.card.key][&100].clone();
    assert_eq!(g.resident.unwrap().vram, Bytes::unavailable(Reason::Legacy));
    assert_eq!(g.shown_vram(), None);
}

#[test]
fn test_engine_counter_stepping_back_is_not_counted_twice() {
    let tmp = tempfile::tempdir().unwrap();
    let mut e = edge(
        tmp.path(),
        &[(100, 5, fdi(7, 1_000_000_000, 0), RENDER)],
        false,
        true,
    );
    let path = e.proc_root.join("100").join("fdinfo").join("5");
    std::fs::write(&path, fdi(7, 900_000_000, 0)).unwrap(); // the DRM spec allows a brief step back
    let g = e.counters.sample().0[&e.card.key][&100].clone();
    assert_eq!(engines_unavailable(&g), None);
    assert_eq!(g.engines.get("3d"), 0.0);
    e.clock.advance(Duration::from_secs(1));
    std::fs::write(&path, fdi(7, 1_500_000_000, 0)).unwrap();
    let g = e.counters.sample().0[&e.card.key][&100].clone();
    assert_eq!(g.engines.get("3d"), 50.0); // from the high-water mark
}

#[test]
fn test_unified_vcn_reports_video() {
    let tmp = tempfile::tempdir().unwrap();
    let start = fdi(9, 0, 0).replace("drm-engine-gfx:\t0 ns", "drm-engine-enc:\t0 ns");
    let mut e = edge(tmp.path(), &[(100, 5, start, RENDER)], true, true);
    let later = fdi(9, 0, 0).replace("drm-engine-gfx:\t0 ns", "drm-engine-enc:\t1000000000 ns");
    std::fs::write(e.proc_root.join("100").join("fdinfo").join("5"), later).unwrap();
    let engines = e.counters.sample().0[&e.card.key][&100].engines.clone();
    let Engines::Measured { by_engine } = engines else {
        panic!("engines should be measured: {engines:?}");
    };
    assert_eq!(by_engine["video"], 100.0);
    assert!(!by_engine.contains_key("encode"));
}

#[test]
fn test_missing_sysfs_memory_is_unknown() {
    let tmp = tempfile::tempdir().unwrap();
    let mut e = edge(tmp.path(), &[], false, false);
    let a = e.counters.sample().1[&e.card.key];
    assert_eq!(a.vram_used, Bytes::unavailable(Reason::NotOnBackend));
    assert_eq!(a.shared_used, Bytes::unavailable(Reason::NotOnBackend));
    assert_eq!(read_device(&e.card, 0, "6.1.0").mem_used, None);
}

// ---- DrmScanTests ----

#[test]
fn test_new_client_engine_load_is_unknown_until_two_samples() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c], p.clone(), clock.clone());
    counters.start();
    clock.advance(Duration::from_secs(1));
    proc_fd(&p, 200, 5, &fdi(2, 500_000_000, 4096), RENDER);
    let procs = counters.sample().0.remove(&key).unwrap();
    assert_eq!(engines_unavailable(&procs[&100]), None); // in the baseline: a real, measured idle
    assert_eq!(
        engines_unavailable(&procs[&200]),
        Some(Reason::NoBaseline) // first seen now: unknown, not 0
    );
    assert_eq!(procs[&200].vram, Bytes::measured(4096 * 1024)); // memory doesn't need two samples
    assert!(!procs[&200].kfd);
    clock.advance(Duration::from_secs(1));
    proc_fd(&p, 200, 5, &fdi(2, 1_000_000_000, 4096), RENDER);
    assert_eq!(counters.sample().0[&key][&200].engines.get("3d"), 50.0);
}

#[test]
fn test_engine_first_reported_counts_from_zero() {
    // amdgpu omits an engine until it has run, so its whole counter is new work.
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(
        &p,
        100,
        5,
        &fdi(1, 0, 0).replace("drm-engine-gfx:\t0 ns\n", ""),
        RENDER,
    );
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c], p.clone(), clock.clone());
    counters.start();
    clock.advance(Duration::from_secs(1));
    proc_fd(&p, 100, 5, &fdi(1, 250_000_000, 0), RENDER);
    assert_eq!(counters.sample().0[&key][&100].engines.get("3d"), 25.0);
}

#[test]
fn test_client_held_by_several_processes_is_charged_to_none() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    proc_fd(&p, 200, 9, &fdi(1, 0, 4096), RENDER);
    proc_fd(&p, 300, 5, &fdi(2, 0, 8192), RENDER);
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c], p, clock.clone());
    counters.start();
    clock.advance(Duration::from_secs(1));
    let procs = counters.sample().0.remove(&key).unwrap();
    for pid in [100, 200] {
        assert_eq!(procs[&pid].vram, Bytes::unavailable(Reason::SharedClient));
        assert_eq!(procs[&pid].shown_vram(), None);
        assert_eq!(
            engines_unavailable(&procs[&pid]),
            Some(Reason::SharedClient)
        );
    }
    assert_eq!(procs[&300].shown_vram().map(|s| s.get()), Some(8192 * 1024));
    // The other rows still add up: the shared client's bytes are left out, not the whole total.
    let measured: u64 = procs
        .values()
        .filter_map(|g| g.vram.as_measured())
        .map(|m| m.get())
        .sum();
    assert_eq!(measured, 8192 * 1024);
}

#[test]
fn test_private_and_co_owned_clients_leave_the_whole_process_unattributed() {
    // pid 100 alone holds client 1 and shares client 2 with pid 200: charging client 1 to it would
    // depend on which other processes happened to be visible, so neither process gets bytes.
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 1024), RENDER);
    proc_fd(&p, 100, 6, &fdi(2, 0, 2048), RENDER);
    proc_fd(&p, 200, 5, &fdi(2, 0, 2048), RENDER);
    let mut counters = DrmCounters::new(vec![c], p, Arc::new(FakeClock::new(0.0)));
    counters.start();
    let procs = counters.sample().0.remove(&key).unwrap();
    for pid in [100, 200] {
        assert_eq!(procs[&pid].vram, Bytes::unavailable(Reason::SharedClient));
        assert_eq!(procs[&pid].shared, Bytes::unavailable(Reason::SharedClient));
    }
    assert_eq!(
        procs[&100].resident.unwrap().vram,
        Bytes::measured(3072 * 1024)
    );
}

#[test]
fn test_uninspectable_processes_are_reported() {
    if is_root() {
        eprintln!("skipped: permission bits don't apply to root");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    std::fs::create_dir_all(p.join("200").join("fd")).unwrap();
    write_stat(&p, 200, 200);
    let mut counters = DrmCounters::new(vec![c], p.clone(), Arc::new(FakeClock::new(0.0)));
    {
        let _locked = Restore::chmod(p.join("200").join("fd"), 0o000);
        assert!(counters.sample().0[&key].contains_key(&100));
    }
    assert!(
        counters.status().contains("1 processes not inspectable"),
        "{}",
        counters.status()
    );
    counters.sample();
    assert_eq!(counters.status(), ""); // readable now
}

#[test]
fn test_unreadable_fd_after_listing_is_incomplete() {
    if is_root() {
        eprintln!("skipped: permission bits don't apply to root");
        return;
    }
    // Python fails one readlink. Linux checks only the directory's search bit for readlink, so
    // here the fd directory stays listable (r--) but none of its links can be read.
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    proc_fd(&p, 100, 6, &fdi(2, 0, 4096), RENDER);
    let mut counters = DrmCounters::new(vec![c], p.clone(), Arc::new(FakeClock::new(0.0)));
    let procs = {
        let _locked = Restore::chmod(p.join("100").join("fd"), 0o400);
        counters.sample().0.remove(&key).unwrap()
    };
    // A client behind an unreadable link might be shared: nothing is charged.
    assert!(procs.get(&100).is_none_or(|g| g.shown_vram().is_none()));
    assert!(
        counters
            .status()
            .contains("1 processes whose fds or fdinfo couldn't all be read"),
        "{}",
        counters.status()
    );
}

#[test]
fn test_unreadable_fdinfo_is_incomplete() {
    if is_root() {
        eprintln!("skipped: permission bits don't apply to root");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    proc_fd(&p, 100, 6, &fdi(2, 0, 4096), RENDER);
    let mut counters = DrmCounters::new(vec![c], p.clone(), Arc::new(FakeClock::new(0.0)));
    let g = {
        let _locked = Restore::chmod(p.join("100").join("fdinfo").join("6"), 0o000);
        counters.sample().0[&key][&100].clone()
    };
    assert_eq!(g.vram, Bytes::unavailable(Reason::Incomplete));
    assert_eq!(g.shown_vram(), None);
}

#[test]
fn test_reused_pid_drops_cached_fds() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    write_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    std::fs::write(
        p.join("100").join("stat"),
        "100 (old) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 1000 0\n",
    )
    .unwrap();
    let mut counters = DrmCounters::new(vec![c], p.clone(), Arc::new(FakeClock::new(0.0)));
    counters.sample();
    assert!(counters.known_pids().contains(&100));
    std::fs::write(
        p.join("100").join("stat"),
        "100 (new) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 2000 0\n",
    )
    .unwrap();
    counters.set_budget(Some(Duration::ZERO)); // nothing is rescanned: only the cache could keep pid 100
    assert!(!counters.sample().0[&key].contains_key(&100));
    assert!(!counters.known_pids().contains(&100));
}

#[test]
fn test_scan_budget_keeps_rows_and_resumes() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    for pid in [100, 200, 300, 400] {
        proc_fd(&p, pid, 5, &fdi(u64::from(pid), 0, 4096), RENDER);
    }
    let mut counters = DrmCounters::new(vec![c], p.clone(), ticking());
    counters.start(); // the baseline is never limited
    assert_eq!(counters.known_pids(), vec![100, 200, 300, 400]);
    counters.set_budget(Some(Duration::from_millis(2500)));
    proc_fd(&p, 500, 5, &fdi(500, 0, 4096), RENDER);
    let rows = counters.sample().0.remove(&key).unwrap();
    for pid in [100, 200, 300, 400] {
        assert!(rows.contains_key(&pid)); // no row flickers while the scan is partial
    }
    assert!(rows.contains_key(&500)); // processes not seen before are scanned first
    assert!(
        counters.status().contains("scan budget reached"),
        "{}",
        counters.status()
    );
    counters.set_budget(None);
    counters.sample();
    assert_eq!(counters.status(), "");
}

#[test]
fn test_budget_is_checked_inside_one_process() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    proc_fd(&p, 50, 5, &fdi(1, 0, 4096), RENDER);
    for fd in 0..200 {
        // a process with many fds
        proc_fd(&p, 100, fd, "pos:\t0\n", &format!("socket:[{fd}]"));
    }
    let mut counters = DrmCounters::new(vec![c], p, ticking());
    counters.set_budget(Some(Duration::from_millis(3500)));
    counters.sample();
    let known = counters.known_pids();
    assert!(known.contains(&50)); // the first process of a refresh is listed in full
    assert!(!known.contains(&100)); // interrupted partway: deferred, not cached as fd-less
    assert!(
        counters.status().contains("scan budget reached"),
        "{}",
        counters.status()
    );
}

#[test]
fn test_one_huge_process_does_not_stall_the_rotation() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    for fd in 0..200 {
        proc_fd(&p, 100, fd, "pos:\t0\n", &format!("socket:[{fd}]"));
    }
    proc_fd(&p, 200, 5, &fdi(2, 0, 4096), RENDER);
    let mut counters = DrmCounters::new(vec![c], p, ticking());
    counters.set_budget(Some(Duration::from_millis(1500)));
    for _ in 0..3 {
        counters.sample();
    }
    assert_eq!(counters.known_pids(), vec![100, 200]);
}

#[test]
fn test_partial_scan_charges_no_one() {
    // A process the budget skipped may hold the same client, so single ownership isn't proven.
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    for pid in [100, 200] {
        proc_fd(&p, pid, 5, &fdi(u64::from(pid), 0, 4096), RENDER);
    }
    let mut counters = DrmCounters::new(vec![c], p, ticking());
    counters.start();
    counters.set_budget(Some(Duration::from_millis(1500)));
    let procs = counters.sample().0.remove(&key).unwrap();
    assert!(counters.partial());
    for pid in [100, 200] {
        assert_eq!(procs[&pid].vram, Bytes::unavailable(Reason::PartialScan));
        assert_eq!(procs[&pid].shown_vram(), None);
        assert_eq!(engines_unavailable(&procs[&pid]), Some(Reason::PartialScan));
        assert_eq!(
            procs[&pid].resident.unwrap().vram,
            Bytes::measured(4096 * 1024)
        );
    }
}

#[test]
fn test_rotation_resumes_at_a_known_process() {
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    for pid in [100, 200, 300] {
        proc_fd(&p, pid, 5, &fdi(u64::from(pid), 0, 4096), RENDER);
    }
    let mut counters = DrmCounters::new(vec![c], p.clone(), ticking());
    counters.start();
    for pid in [900, 901] {
        // new processes are scanned first and run out the budget
        proc_fd(&p, pid, 5, &fdi(u64::from(pid), 0, 4096), RENDER);
    }
    counters.set_cursor(200);
    counters.set_budget(Some(Duration::from_millis(1500)));
    counters.sample();
    assert_eq!(counters.cursor(), 200); // not 901: known processes 200 and 300 aren't skipped
}

#[test]
fn test_incomplete_process_has_no_partial_totals() {
    if is_root() {
        eprintln!("skipped: permission bits don't apply to root");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (c, p) = scan_setup(tmp.path());
    let key = c.key;
    proc_fd(&p, 100, 5, &fdi(1, 0, 4096), RENDER);
    proc_fd(&p, 100, 6, &fdi(2, 0, 4096), RENDER);
    let clock = Arc::new(FakeClock::new(0.0));
    let mut counters = DrmCounters::new(vec![c], p.clone(), clock.clone());
    let g = {
        let _locked = Restore::chmod(p.join("100").join("fdinfo").join("6"), 0o000);
        counters.start();
        clock.advance(Duration::from_secs(1));
        counters.sample().0[&key][&100].clone()
    };
    assert_eq!(
        g.resident.unwrap().vram,
        Bytes::unavailable(Reason::Incomplete)
    );
    assert_eq!(engines_unavailable(&g), Some(Reason::Incomplete));
    assert!(
        counters
            .status()
            .contains("1 processes whose fds or fdinfo"),
        "{}",
        counters.status()
    );
}

// ---- DrmBackend ----

fn backend_tree(tmp: &Path, pids: &[u32]) -> PathBuf {
    let dev = make_card(
        &tmp.join("sys").join("class"),
        "card0",
        PDEV,
        0x1002,
        0x7550,
        16 * GIB,
        "amdgpu",
        8 * GIB,
        false,
    );
    std::fs::write(dev.join("mem_info_vram_used"), (3 * GIB).to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_used"), "0").unwrap();
    let p = tmp.join("proc");
    std::fs::create_dir_all(&p).unwrap();
    for &pid in pids {
        proc_fd(&p, pid, 5, &fdi(u64::from(pid), 0, 4096), RENDER);
    }
    p
}

#[test]
fn backend_unbounded_policy_lists_every_process() {
    let tmp = tempfile::tempdir().unwrap();
    let pids = [100, 200, 300, 400, 500];
    let p = backend_tree(tmp.path(), &pids);

    // A clock that runs out any budget at once: a budgeted scan would leave rows partial.
    let mut budgeted =
        DrmBackend::new(tmp.path().join("sys"), p.clone(), ticking(), "6.1.0".into());
    let adapters = budgeted.adapters();
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0].backend, BackendKind::Drm);
    budgeted.start(&adapters[0], ScanPolicy::Budgeted);
    let partial = budgeted.sample();
    assert!(
        partial
            .procs
            .values()
            .any(|g| g.vram == Bytes::unavailable(Reason::PartialScan))
    );

    let mut backend = DrmBackend::new(tmp.path().join("sys"), p, ticking(), "6.1.0".into());
    backend.start(&adapters[0], ScanPolicy::Unbounded);
    let sample = backend.sample();
    assert_eq!(sample.procs.keys().copied().collect::<Vec<_>>(), pids);
    for g in sample.procs.values() {
        assert_eq!(g.vram, Bytes::measured(4096 * 1024));
        assert_ne!(engines_unavailable(g), Some(Reason::PartialScan));
    }
    assert_eq!(sample.adapter.vram_used, Bytes::measured(3 * GIB));
    assert_eq!(sample.status, None);
    assert_eq!(backend.driver(), Some("kernel 6.1.0".to_string()));
    assert_eq!(backend.cuda(), None);
    assert_eq!(backend.kind(), BackendKind::Drm);
    assert_eq!(backend.device(0).unwrap().mem_used, Some(3 * GIB));
}

#[test]
fn backend_status_reports_unreadable_proc_root() {
    let tmp = tempfile::tempdir().unwrap();
    backend_tree(tmp.path(), &[]);
    let missing = tmp.path().join("no-proc");
    let mut backend = DrmBackend::new(
        tmp.path().join("sys"),
        missing.clone(),
        Arc::new(FakeClock::new(0.0)),
        "6.1.0".into(),
    );
    let adapters = backend.adapters();
    backend.start(&adapters[0], ScanPolicy::Unbounded);
    let sample = backend.sample();
    assert_eq!(
        sample.status,
        Some(format!("{} is not readable", missing.display()))
    );
    assert!(sample.procs.is_empty());
}

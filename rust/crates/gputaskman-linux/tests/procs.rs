#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::traits::{Clock, ProcessSource, SystemClock};
use gputaskman_linux::linux_backends;
use gputaskman_linux::procs::LinuxProcs;
use std::os::unix::fs::symlink;
use std::sync::Arc;

fn proc_tree() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let r = dir.path();
    std::fs::write(r.join("stat"), "cpu  1 2 3\nbtime 1700000000\n").unwrap();
    let p = r.join("4242");
    std::fs::create_dir_all(p.join("fd")).unwrap();
    std::fs::write(
        p.join("stat"),
        "4242 (llama-server) S 1 4242 4242 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 12345 0 0",
    )
    .unwrap();
    std::fs::write(p.join("cmdline"), b"/opt/llama-server\0-m\0/m/q.gguf\0").unwrap();
    symlink("/opt/llama-server", p.join("exe")).unwrap();
    std::fs::write(
        p.join("smaps"),
        "7f00-7f10 r--p 00000000 00:00 1 /m/q.gguf\nRss:  400 kB\n\
        7f10-7f20 r--p 00000000 00:00 1 /m/q.gguf\nRss:  600 kB\n",
    )
    .unwrap();
    symlink("socket:[999]", p.join("fd/7")).unwrap();
    std::fs::create_dir_all(r.join("net")).unwrap();
    std::fs::write(r.join("net/tcp"), "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n\
   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 999 1\n").unwrap();
    std::fs::write(r.join("net/tcp6"), "  sl  local_address\n").unwrap();
    dir
}

#[test]
fn reads_basics_maps_and_ports() {
    let dir = proc_tree();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    let b = procs.basics(4242).unwrap();
    assert_eq!(b.name.as_deref(), Some("llama-server"));
    assert_eq!(b.exe.as_deref(), Some("/opt/llama-server"));
    assert_eq!(
        b.cmdline.unwrap(),
        vec!["/opt/llama-server", "-m", "/m/q.gguf"]
    );
    let start = b.start.unwrap();
    assert_eq!((start.ticks, start.wall), (12345, 1_700_000_000.0 + 123.45));
    assert_eq!(procs.parent(4242), Some(1));
    let maps = procs.memory_maps(4242).unwrap();
    assert_eq!(
        (maps[0].path.as_str(), maps[0].rss),
        ("/m/q.gguf", 1000 * 1024)
    );
    assert_eq!(procs.listening_ports()[&4242], vec![8080]);
    assert_eq!(procs.pids(), vec![4242]);
}

#[test]
fn process_names_with_spaces_and_parens_parse() {
    let dir = proc_tree();
    std::fs::write(
        dir.path().join("4242/stat"),
        "4242 (a (b) c) S 7 0 0 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 5 0 0",
    )
    .unwrap();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    assert_eq!(procs.parent(4242), Some(7));
    assert_eq!(procs.start_time(4242).unwrap().ticks, 5);
}

#[test]
fn missing_process_is_none_not_a_panic() {
    let dir = proc_tree();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    assert!(procs.basics(1).is_none());
    assert!(procs.memory_maps(1).is_err());
}

#[test]
fn truncated_comm_uses_cmdline_basename() {
    let dir = proc_tree();
    let p = dir.path().join("4242");
    std::fs::write(
        p.join("stat"),
        "4242 (llama-server-ab) S 1 0 0 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 5 0 0",
    )
    .unwrap();
    std::fs::write(p.join("cmdline"), b"/opt/llama-server-abcdef\0").unwrap();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    // comm here is 15 characters, so the longer cmdline basename wins
    assert_eq!(
        procs.basics(4242).unwrap().name.as_deref(),
        Some("llama-server-abcdef")
    );
}

#[test]
fn smaps_entries_without_a_path_are_skipped() {
    let dir = proc_tree();
    std::fs::write(dir.path().join("4242/smaps"),
        "7f00-7f10 rw-p 00000000 00:00 0\nRss:  50 kB\n7f10-7f20 r--p 00000000 00:00 1 /m/a.gguf\nRss:  4 kB\n").unwrap();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    let maps = procs.memory_maps(4242).unwrap();
    assert_eq!(maps.len(), 1);
    assert_eq!(maps[0].rss, 4096);
}

#[test]
fn linux_backends_does_not_panic_without_drivers() {
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let (_backends, _notes) = linux_backends(clock);
}

#[test]
fn smaps_pathname_after_padding_spaces_is_kept() {
    let dir = proc_tree();
    std::fs::write(
        dir.path().join("4242/smaps"),
        "7f00-7f10 r--p 00000000 00:00 1                  /m/my model.gguf\nRss:  4 kB\n",
    )
    .unwrap();
    let procs = LinuxProcs::with_clk_tck(dir.path().into(), 100);
    assert_eq!(procs.memory_maps(4242).unwrap()[0].path, "/m/my model.gguf");
}

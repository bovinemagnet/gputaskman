#![allow(
    dead_code,
    unused_imports,
    clippy::unwrap_used,
    clippy::too_many_arguments
)]
//! The tree builders live in `gputaskman_linux::fixtures`, shared with the parity harness. The
//! fallible ones are wrapped here so a failed write fails the test at once.
use std::path::{Path, PathBuf};

pub use gputaskman_linux::fixtures::{FDINFO, GIB, card, fdinfo};

pub fn make_card(
    root: &Path,
    name: &str,
    pdev: &str,
    vendor: u16,
    device: u16,
    vram: u64,
    driver: &str,
    gtt: u64,
    apu: bool,
) -> PathBuf {
    gputaskman_linux::fixtures::make_card(root, name, pdev, vendor, device, vram, driver, gtt, apu)
        .unwrap()
}

pub fn write_fd(proc_root: &Path, pid: u32, fd: u32, text: &str, target: &str) {
    gputaskman_linux::fixtures::write_fd(proc_root, pid, fd, text, target).unwrap();
}

pub fn write_stat(proc_root: &Path, pid: u32, start_ticks: u64) {
    gputaskman_linux::fixtures::write_stat(proc_root, pid, start_ticks).unwrap();
}

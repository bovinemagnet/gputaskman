#![allow(dead_code, clippy::unwrap_used, clippy::too_many_arguments)]
use gputaskman_linux::drm::cards::DrmCard;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

pub const GIB: u64 = 1 << 30;
pub const FDINFO: &str = "pos:\t0\nflags:\t02100002\nmnt_id:\t28\nino:\t415\ndrm-driver:\tamdgpu\n\
drm-client-id:\t26\ndrm-pdev:\t0000:03:00.0\npasid:\t25\ndrm-total-cpu:\t0\ndrm-shared-cpu:\t0\n\
drm-resident-cpu:\t0\ndrm-purgeable-cpu:\t0\ndrm-total-gtt:\t2 MiB\ndrm-shared-gtt:\t0\n\
drm-resident-gtt:\t2060 KiB\ndrm-purgeable-gtt:\t0\ndrm-total-vram:\t12 KiB\ndrm-shared-vram:\t0\n\
drm-resident-vram:\t0\ndrm-purgeable-vram:\t0\ndrm-memory-vram:\t0 KiB\ndrm-memory-gtt: \t2060 KiB\n\
drm-memory-cpu: \t0 KiB\namd-evicted-vram:\t12 KiB\namd-requested-vram:\t12 KiB\n\
amd-requested-gtt:\t2048 KiB\ndrm-engine-gfx:\t5308400172 ns\ndrm-engine-compute:\t0 ns\n\
drm-engine-enc:\t1200 ns\ndrm-engine-capacity-gfx:\t1\n";

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
    let dev = root.join("drm").join(name).join("device");
    std::fs::create_dir_all(&dev).unwrap();
    std::fs::write(dev.join("vendor"), format!("0x{vendor:04x}\n")).unwrap();
    std::fs::write(dev.join("device"), format!("0x{device:04x}\n")).unwrap();
    std::fs::write(dev.join("mem_info_vram_total"), vram.to_string()).unwrap();
    std::fs::write(dev.join("mem_info_gtt_total"), gtt.to_string()).unwrap();
    if !apu {
        std::fs::write(dev.join("mem_info_vram_vendor"), "samsung\n").unwrap();
    }
    let driver_dir = root.join("drivers").join(driver);
    std::fs::create_dir_all(&driver_dir).unwrap();
    symlink(&driver_dir, dev.join("driver")).unwrap();
    std::fs::write(
        dev.join("uevent"),
        format!("DRIVER={driver}\nPCI_SLOT_NAME={pdev}\n"),
    )
    .unwrap();
    dev
}

pub fn card(dev: &Path, pdev: &str, name: &str, vram: u64) -> DrmCard {
    DrmCard {
        key: gputaskman_core::model::pci_key(pdev),
        pdev: pdev.into(),
        sysfs: dev.into(),
        driver: "amdgpu".into(),
        vendor_id: 0x1002,
        device_id: 0x7550,
        name: name.into(),
        vram_total: vram,
        integrated: vram <= 4 * GIB,
        unified_video: false,
    }
}

pub fn write_fd(proc_root: &Path, pid: u32, fd: u32, text: &str, target: &str) {
    let fdinfo = proc_root.join(pid.to_string()).join("fdinfo");
    std::fs::create_dir_all(&fdinfo).unwrap();
    std::fs::write(fdinfo.join(fd.to_string()), text).unwrap();
    let fds = proc_root.join(pid.to_string()).join("fd");
    std::fs::create_dir_all(&fds).unwrap();
    let link = fds.join(fd.to_string());
    if std::fs::symlink_metadata(&link).is_err() {
        symlink(target, &link).unwrap();
    }
}

/// `/proc/<pid>/stat` with field 22 (start time) set, as `DrmCounters` reads it to spot a reused pid.
pub fn write_stat(proc_root: &Path, pid: u32, start_ticks: u64) {
    let dir = proc_root.join(pid.to_string());
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("stat"),
        format!("{pid} (p) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 {start_ticks} 0 0"),
    )
    .unwrap();
}

#[allow(clippy::too_many_arguments)]
pub fn fdinfo(
    client: u64,
    pdev: &str,
    gfx: u64,
    vram: u64,
    gtt: u64,
    shared_vram: u64,
    shared_gtt: u64,
    shared_cpu: u64,
) -> String {
    format!(
        "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client}\ndrm-pdev:\t{pdev}\n\
             drm-shared-vram:\t{shared_vram} KiB\ndrm-resident-vram:\t{vram} KiB\n\
             drm-shared-gtt:\t{shared_gtt} KiB\ndrm-resident-gtt:\t{gtt} KiB\n\
             drm-shared-cpu:\t{shared_cpu} KiB\ndrm-resident-cpu:\t0\n\
             drm-engine-gfx:\t{gfx} ns\ndrm-engine-capacity-gfx:\t1\n"
    )
}

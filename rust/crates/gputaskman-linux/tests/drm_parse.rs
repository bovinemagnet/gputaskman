#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;
use gputaskman_core::model::{BackendKind, pci_key};
use gputaskman_linux::drm::cards::adapters;
use gputaskman_linux::drm::device::{read_device, vcn_major};
use gputaskman_linux::drm::{cards::cards, device::engine_key, fdinfo::*};
use std::collections::BTreeMap;
use support::*;

const PCI_IDS: &str = "# fake\n1002  Advanced Micro Devices, Inc. [AMD/ATI]\n\t7550  Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]\n\t\t7550 1234  Subsystem line that must be skipped\n\t13c0  Granite Ridge [Radeon 610M]\n1003  Somebody else\n\t9999  Not this vendor\n";

#[test]
fn parse_size_units() {
    assert_eq!(parse_size("2060 KiB"), Some(2060 * 1024));
    assert_eq!(parse_size("12"), Some(12));
    assert_eq!(parse_size("2 MiB"), Some(2 << 20));
    assert_eq!(parse_size("1 GiB"), Some(1 << 30));
    assert_eq!(parse_size("0"), Some(0));
    assert_eq!(parse_size("  4 KiB  "), Some(4096));
    assert_eq!(parse_size("1 kB"), Some(1000));
    assert_eq!(parse_size("x KiB"), None);
    assert_eq!(parse_size("bogus"), None);
    assert_eq!(parse_size("5 parsecs"), None);
    assert_eq!(parse_size("12 bananas"), None);
    assert_eq!(parse_size(""), None);
}

#[test]
fn parse_fdinfo_example() {
    let info = parse_fdinfo(FDINFO).unwrap();
    assert_eq!(
        (info.driver.as_str(), info.client_id, info.pdev.as_str()),
        ("amdgpu", Some(26), "0000:03:00.0")
    );
    assert_eq!(
        (info.vram, info.gtt, info.shared, info.cpu),
        (Some(0), Some(2060 * 1024), Some(0), Some(0))
    );
    let engines = info.engines.unwrap();
    assert_eq!(engines["gfx"], 5_308_400_172);
    assert_eq!(engines["enc"], 1200);
    assert_eq!(engines["compute"], 0);
    assert!(!engines.contains_key("capacity-gfx"));
}

#[test]
fn parse_fdinfo_non_drm() {
    assert!(parse_fdinfo("pos:\t0\nflags:\t02\n").is_none());
}

#[test]
fn torn_engine_counter_is_unknown() {
    let text = FDINFO.replace("drm-engine-gfx:\t5308400172 ns", "drm-engine-gfx:\t");
    assert!(parse_fdinfo(&text).unwrap().engines.is_none());
}

#[test]
fn test_parse_fdinfo_resident_fallback() {
    let info = parse_fdinfo("drm-driver:\tamdgpu\ndrm-resident-vram:\t8 KiB\ndrm-resident-gtt:\t3 MiB\ndrm-resident-cpu:\t0\n").unwrap();
    assert_eq!(
        (info.vram, info.gtt, info.cpu),
        (Some(8 * 1024), Some(3 << 20), Some(0))
    );
    assert_eq!(info.engines, Some(BTreeMap::new()));
    assert_eq!(info.client_id, None);
    assert_eq!(info.shared, None);
}

#[test]
fn test_parse_fdinfo_prefers_resident_keys() {
    let info = parse_fdinfo(
        "drm-driver:\tamdgpu\ndrm-resident-vram:\t8 KiB\ndrm-memory-vram:\t9 KiB\n\
                             drm-shared-vram:\t0\ndrm-shared-gtt:\t4 KiB\ndrm-shared-cpu:\t0\n",
    )
    .unwrap();
    assert_eq!(info.vram, Some(8 * 1024));
    assert_eq!(info.shared, Some(4 * 1024));
}

#[test]
fn test_parse_fdinfo_shared_needs_all_memory_regions() {
    let info =
        parse_fdinfo("drm-driver:\tamdgpu\ndrm-memory-vram:\t300 MiB\ndrm-shared-vram:\t0\n")
            .unwrap();
    assert_eq!(info.shared, None);
    let info =
        parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n").unwrap();
    assert_eq!(info.shared, None);
}

#[test]
fn test_parse_fdinfo_shared_cpu_counts_toward_private_gate() {
    let info = parse_fdinfo(
        "drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\ndrm-shared-cpu:\t64 KiB\n",
    )
    .unwrap();
    assert_eq!(info.shared, Some(64 * 1024));
    let info = parse_fdinfo(
        "drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\ndrm-shared-cpu:\tlots\n",
    )
    .unwrap();
    assert_eq!(info.shared, None);
}

#[test]
fn test_parse_fdinfo_unparseable_is_unknown() {
    let info = parse_fdinfo(
        "drm-driver:\tamdgpu\ndrm-resident-vram:\tlots\ndrm-shared-vram:\t0\n\
                             drm-shared-gtt:\t?\ndrm-engine-gfx:\ttorn\n",
    )
    .unwrap();
    assert_eq!((info.vram, info.gtt, info.shared), (None, None, None));
    assert!(info.engines.is_none());
}

#[test]
fn cards_lists_amdgpu_discrete_first() {
    let dir = tempfile::tempdir().unwrap();
    make_card(
        dir.path(),
        "card0",
        "0000:05:00.0",
        0x1002,
        0x1681,
        512 << 20,
        "amdgpu",
        0,
        true,
    );
    make_card(
        dir.path(),
        "card1",
        "0000:03:00.0",
        0x1002,
        0x7550,
        16 * GIB,
        "amdgpu",
        0,
        false,
    );
    make_card(
        dir.path(),
        "card2",
        "0000:01:00.0",
        0x10de,
        0x2684,
        24 * GIB,
        "nvidia",
        0,
        false,
    );
    let found = cards(&dir.path().join("drm"), None);
    assert_eq!(
        found.iter().map(|c| c.pdev.as_str()).collect::<Vec<_>>(),
        vec!["0000:03:00.0", "0000:05:00.0"]
    );
    assert!(!found[0].integrated && found[1].integrated);
}

#[test]
fn test_cards() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    make_card(
        root,
        "card0",
        "0000:05:00.0",
        0x1002,
        0x13c0,
        8 * GIB,
        "amdgpu",
        3 * GIB,
        true,
    );
    make_card(
        root,
        "card1",
        "0000:03:00.0",
        0x1002,
        0x7550,
        16 * GIB,
        "amdgpu",
        32 * GIB,
        false,
    );
    make_card(
        root,
        "card2",
        "0000:0a:00.0",
        0x10de,
        0x9999,
        24 * GIB,
        "nvidia",
        0,
        false,
    );
    for junk in ["card1-DP-1", "renderD128"] {
        std::fs::create_dir_all(root.join("drm").join(junk)).unwrap();
    }
    let ids = root.join("pci.ids");
    std::fs::write(&ids, PCI_IDS).unwrap();

    let found = cards(&root.join("drm"), Some(&ids));
    assert_eq!(
        found.iter().map(|c| c.pdev.as_str()).collect::<Vec<_>>(),
        vec!["0000:03:00.0", "0000:05:00.0"]
    );
    assert_eq!(
        found.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        vec!["AMD Radeon RX 9070/9070 XT/9070 GRE", "AMD Radeon 610M"]
    );
    assert!(!found[0].integrated && found[1].integrated);
    assert_eq!(found[0].vram_total, 16 * GIB);
    assert_eq!(found[0].key, pci_key("0000:03:00.0"));
    assert_eq!((found[0].vendor_id, found[0].device_id), (0x1002, 0x7550));

    let ads = adapters(&found);
    assert_eq!(
        ads.iter().map(|a| a.key).collect::<Vec<_>>(),
        found.iter().map(|c| c.key).collect::<Vec<_>>()
    );
    assert_eq!(ads[0].dedicated_vram, 16 * GIB);
    assert_eq!(ads[0].shared_system, 32 * GIB);
    assert_eq!(ads[1].shared_system, 3 * GIB);
    assert!(ads.iter().all(|a| a.backend == BackendKind::Drm));
    assert_eq!(ads[0].pci_bus, "0000:03:00.0");
    assert!(!ads[0].integrated && ads[1].integrated);
}

#[test]
fn card_name_falls_back_to_vendor_and_device_id() {
    let dir = tempfile::tempdir().unwrap();
    make_card(
        dir.path(),
        "card0",
        "0000:03:00.0",
        0x1002,
        0x7550,
        16 * GIB,
        "amdgpu",
        0,
        false,
    );
    let ids = dir.path().join("empty.ids");
    std::fs::write(&ids, "# nothing\n").unwrap();
    let found = cards(&dir.path().join("drm"), Some(&ids));
    assert_eq!(found[0].name, "AMD GPU 7550");
}

#[test]
fn test_read_device() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = tmp.path().join("card0").join("device");
    let hwmon = dev.join("hwmon").join("hwmon0");
    std::fs::create_dir_all(&hwmon).unwrap();
    let w = |p: std::path::PathBuf, t: &str| std::fs::write(p, t).unwrap();
    w(dev.join("mem_info_vram_total"), &(16 * GIB).to_string());
    w(dev.join("mem_info_vram_used"), &(3 * GIB).to_string());
    w(dev.join("gpu_busy_percent"), "42\n");
    w(dev.join("mem_busy_percent"), "7\n");
    w(dev.join("current_link_speed"), "32.0 GT/s PCIe\n");
    w(dev.join("current_link_width"), "16\n");
    w(
        dev.join("pp_dpm_sclk"),
        "0: 500Mhz\n1: 2100Mhz\n2: 2570Mhz *\nS: 44Mhz *\n",
    );
    w(hwmon.join("temp1_label"), "edge\n");
    w(hwmon.join("temp1_input"), "57000\n");
    w(hwmon.join("temp2_label"), "junction\n");
    w(hwmon.join("temp2_input"), "65000\n");
    w(hwmon.join("power1_average"), "47000000\n");
    w(hwmon.join("power1_label"), "PPT\n");
    w(hwmon.join("power1_cap"), "340000000\n");
    w(hwmon.join("freq1_label"), "sclk\n");
    w(hwmon.join("freq1_input"), "27000000\n");
    w(hwmon.join("freq2_label"), "mclk\n");
    w(hwmon.join("freq2_input"), "1258000000\n");
    w(hwmon.join("fan1_input"), "1000\n");
    w(hwmon.join("fan1_max"), "2000\n");

    let c = card(&dev, "0000:03:00.0", "AMD Radeon RX 9070", 16 * GIB);
    let d = read_device(&c, 0, "6.1.0");
    assert_eq!(d.index, 0);
    assert_eq!(d.name, "AMD Radeon RX 9070");
    assert_eq!(d.pci_bus, "0000:03:00.0");
    assert_eq!(d.driver_model, "amdgpu");
    assert_eq!(d.mem_total, Some(16 * GIB));
    assert_eq!(d.mem_used, Some(3 * GIB));
    assert_eq!(d.util_gpu, Some(42.0));
    assert_eq!(d.util_mem, Some(7.0));
    assert_eq!(d.temp, Some(57.0));
    assert_eq!(d.power_w, Some(47.0));
    assert_eq!(d.power_limit_w, Some(340.0));
    assert_eq!(d.fan, Some(50));
    assert_eq!(d.gfx_clock, Some(27));
    assert_eq!(d.source, "AMD sysfs");
    assert_eq!(d.power_kind, "PPT");
    assert_eq!(d.mem_clock, Some(1258));
    assert_eq!(d.sm_clock_max, Some(2570));
    assert_eq!(d.pcie_gen, Some(5));
    assert_eq!(d.pcie_width, Some(16));
    assert_eq!((d.enc, d.dec), (None, None));
}

#[test]
fn test_missing_sysfs_memory_is_unknown() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = tmp.path().join("device");
    std::fs::create_dir_all(&dev).unwrap();
    let c = card(&dev, "0000:03:00.0", "AMD GPU", 16 * GIB);
    let d = read_device(&c, 0, "6.1.0");
    assert_eq!((d.mem_used, d.mem_total), (None, None));
    assert_eq!(adapters(&[c])[0].shared_system, 0);
}

#[test]
fn test_power_average_of_zero_falls_back_to_input() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = tmp.path().join("device");
    let hwmon = dev.join("hwmon").join("hwmon0");
    std::fs::create_dir_all(&hwmon).unwrap();
    std::fs::write(hwmon.join("power1_average"), "0\n").unwrap();
    std::fs::write(hwmon.join("power1_input"), "42000000\n").unwrap();
    let c = card(&dev, "0000:03:00.0", "AMD GPU", 16 * GIB);
    assert_eq!(read_device(&c, 0, "6.1.0").power_w, Some(42.0));
    std::fs::remove_file(hwmon.join("power1_average")).unwrap();
    std::fs::write(hwmon.join("power1_input"), "0\n").unwrap();
    assert_eq!(read_device(&c, 0, "6.1.0").power_w, Some(0.0));
}

#[test]
fn test_vcn_major_from_ip_discovery() {
    let tmp = tempfile::tempdir().unwrap();
    let uvd = tmp.path().join("ip_discovery/die/0/UVD/0");
    std::fs::create_dir_all(&uvd).unwrap();
    std::fs::write(uvd.join("major"), "5\n").unwrap();
    assert_eq!(vcn_major(tmp.path()), 5);
    assert_eq!(vcn_major(&tmp.path().join("none")), 0);
}

#[test]
fn test_unified_vcn_reports_video() {
    assert_eq!(engine_key("enc", true), "video");
    assert_eq!(engine_key("dec", true), "video");
    assert_eq!(engine_key("enc", false), "encode");
}

#[test]
fn engine_names() {
    assert_eq!(engine_key("gfx", false), "3d");
    assert_eq!(engine_key("compute", false), "compute");
    assert_eq!(engine_key("dma", false), "copy");
    assert_eq!(engine_key("enc_1", false), "encode");
    assert_eq!(engine_key("jpeg", false), "decode");
    assert_eq!(engine_key("vpe", false), "other");
    assert_eq!(engine_key("enc", true), "video");
    assert_eq!(engine_key("gfx", true), "3d");
}

"""Run one parity scenario through the Python implementation and print the result as JSON.

Usage, from the repository root: uv run --locked python rust/parity/run_python.py <scenario.json>

The trees and fakes come from the Python tests themselves (tests/test_drm.py, tests/test_linux.py),
patched exactly as those tests patch them, so both suites read the same data. If the Python modules'
internals change, this harness changes with them in the same commit.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from gputaskman import drm, linux  # noqa: E402
from gputaskman.collector import ProcRow, Snapshot  # noqa: E402
from gputaskman.pdh import ProcGpu  # noqa: E402
from gputaskman.procinfo import ProcDetail  # noqa: E402
from gputaskman.report import to_dict  # noqa: E402
from tests.test_drm import _fdinfo, _make_card, _write_fd  # noqa: E402
from tests.test_linux import FakeNvml, proc, sample  # noqa: E402


def gpu_result(g: ProcGpu, spill: bool) -> dict:
    """The per-process values a report can show, with null for unknown.

    `shared` is a per-process counter only on amdgpu (GTT): the report shows it only there
    (report.has_spill_counters), so NVML rows give null rather than ProcGpu's internal 0.
    """
    return {"vram": int(g.dedicated) if g.attributed and "dedicated" in g.available else None,
            "shared": int(g.shared) if spill and g.attributed and "shared" in g.available else None,
            "reason": g.drm_reason or None, "owner": g.inferred_by or None,
            "engines": {k: round(v, 1) for k, v in g.engines.items() if v} if "engine" in g.available else None}


def run_drm(s: dict) -> list:
    out = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for c in s["cards"]:
            _make_card(root, c["name"], c["pdev"], c["vendor"], c["device"], c["vram"], apu=c.get("apu", False))
        found = drm.cards(str(root / "drm"))
        proc_root = root / "proc"
        proc_root.mkdir()
        ns = [0]
        counters = drm.DrmCounters(found, proc=str(proc_root), clock=lambda: ns[0], timer=lambda: 0.0)
        counters.scan_budget = None
        for step in s["samples"]:
            for p in step:
                (proc_root / str(p["pid"])).mkdir(parents=True, exist_ok=True)
                (proc_root / str(p["pid"]) / "stat").write_text(
                    f"{p['pid']} (p) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 {p['start']} 0 0")
                for fd in p["fds"]:
                    f = fd["fdinfo"]
                    _write_fd(proc_root, p["pid"], fd["fd"],
                              _fdinfo(f["client"], found[0].pdev, gfx=f.get("gfx", 0), vram=f.get("vram", 0),
                                      gtt=f.get("gtt", 0), shared_vram=f.get("shared_vram", 0),
                                      shared_gtt=f.get("shared_gtt", 0), shared_cpu=f.get("shared_cpu", 0)),
                              target=fd["target"])
            procs, _ = counters.sample()
            out.append({str(pid): gpu_result(g, spill=True) for pid, g in procs[found[0].key].items()})
            ns[0] += s.get("advance_ms", 1000) * 1_000_000
    return out


def run_nvml(s: dict) -> list:
    dev = s["devices"][0]
    fake = FakeNvml(compute=[proc(p, u) for p, u in s.get("compute", [])],
                    graphics=[proc(p, u) for p, u in s.get("graphics", [])],
                    samples=[sample(p, sm, enc=enc, dec=dec, ts=ts) for p, sm, enc, dec, ts in s.get("samples", [])],
                    used=dev["used"])
    # The fake's other device fields are fixed; set them from the scenario on this instance only.
    fake.nvmlDeviceGetPciInfo = lambda h: SimpleNamespace(busId=dev["bus"].encode())
    fake.nvmlDeviceGetName = lambda h: dev["name"]
    fake.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(total=dev["total"], used=dev["used"])
    child = s["namespace"] == "child"
    holders, cuda = set(s.get("holders", [])), set(s.get("cuda", []))
    # As tests/test_linux.py: no real amdgpu cards, no WSL, and a scripted PID namespace.
    with patch("gputaskman.drm.cards", return_value=[]), patch.object(linux, "N", fake), \
            patch.object(linux, "is_wsl", return_value=False), \
            patch.object(linux, "in_child_pid_namespace", return_value=child), \
            patch.object(linux, "nvidia_holders", return_value=holders), \
            patch.object(linux, "has_cuda", side_effect=lambda pid: pid in cuda):
        c = linux.LinuxCounters()
        procs, _ = c.sample()
    key = next(iter(procs), None)
    local = {str(pid): gpu_result(g, spill=False) for pid, g in procs.get(key, {}).items()}
    outside = {str(pid): gpu_result(g, spill=False) for pid, g in c.outside.get(key, {}).items()}
    return [{"local": local, "outside": outside}]


def run_report(s: dict) -> dict:
    drm_backend = s["backend"] == "drm"
    rows = []
    for r in s["rows"]:
        g = ProcGpu(dedicated=r.get("dedicated", 0), shared=r.get("shared", 0), attributed=r.get("attributed", True),
                    available={"dedicated", "shared", "engine"} if drm_backend else
                    {"dedicated", "shared", "committed", "engine"})
        for k, v in (r.get("engines") or {}).items():
            g.engines[k] = v
        if r.get("drm_reason"):
            g.drm_reason = r["drm_reason"]
            g.attributed = False
            g.available -= {"dedicated", "shared"}
        g.resident_vram, g.resident_gtt = r.get("resident_vram"), r.get("resident_gtt")
        g.inferred_by = r.get("owner") or ""
        rows.append(ProcRow(r["pid"], r["name"], r["kind"], g, ProcDetail(r["pid"], r["name"])))
    snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=s.get("vram_total", 0),
                    vram_used=s.get("vram_used"), shared_used=s.get("shared_used"), procs=rows, backend=s["backend"])
    return to_dict(snap)


if __name__ == "__main__":
    scenario = json.loads(Path(sys.argv[1]).read_text())
    run = {"drm": run_drm, "nvml": run_nvml, "report": run_report}[scenario["kind"]]
    print(json.dumps(run(scenario), sort_keys=True))

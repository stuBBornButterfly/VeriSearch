"""P0 hardware and toolchain check.

Prints a report that goes verbatim into the README hardware section, and exits
non-zero if a hard requirement from PROJECT.md section 5 is not met.

Usage:
    python scripts/00_setup_check.py
    python scripts/00_setup_check.py --json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

# --- Requirements, from PROJECT.md section 5 -------------------------------
MIN_PYTHON: tuple[int, int] = (3, 11)
VRAM_GB_FOR_3B: float = 16.0
VRAM_GB_FOR_1_5B: float = 8.0
RECOMMENDED_RAM_GB: float = 32.0
# Reported physical RAM runs ~2% under nominal (a "16 GB" machine reports 15.7),
# so the floor is set just below nominal 16 GB.
MIN_RAM_GB: float = 15.0
RECOMMENDED_FREE_DISK_GB: float = 200.0


class Status(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    vram_gb: float
    driver: str


@dataclass
class Report:
    os: str
    python: str
    python_executable: str
    cpu_count: int
    ram_gb: float | None
    free_disk_gb: float | None
    gpus: list[GpuInfo] = field(default_factory=list)
    gpu_error: str | None = None
    torch: str | None = None
    torch_cuda: bool | None = None
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.status is Status.FAIL for c in self.checks)


def _total_ram_gb() -> float | None:
    """Total physical RAM in GB, or None if the platform is not handled."""
    system = platform.system()
    if system == "Windows":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) == 0:
            return None
        return round(stat.ullTotalPhys / 1024**3, 1)

    if system == "Linux":
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            return None
        return round(pages * page_size / 1024**3, 1)

    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return round(int(out.stdout.strip()) / 1024**3, 1)

    return None


def _free_disk_gb(path: Path) -> float | None:
    try:
        return round(shutil.disk_usage(path).free / 1024**3, 1)
    except OSError:
        return None


def _query_gpus() -> tuple[list[GpuInfo], str | None]:
    """Query nvidia-smi. Returns (gpus, error). Error is the raw first line."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found (no NVIDIA driver installed)"
    except subprocess.SubprocessError as exc:
        return [], f"nvidia-smi failed to run: {exc}"

    combined = f"{proc.stdout}\n{proc.stderr}".strip()
    gpus: list[GpuInfo] = []
    errors: list[str] = []

    for line in combined.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3 and parts[1].replace(".", "", 1).isdigit():
            gpus.append(
                GpuInfo(name=parts[0], vram_gb=round(float(parts[1]) / 1024, 1), driver=parts[2])
            )
        else:
            errors.append(row)

    return gpus, errors[0] if errors else None


def _torch_info() -> tuple[str | None, bool | None]:
    try:
        import torch
    except ImportError:
        return None, None
    return torch.__version__, bool(torch.cuda.is_available())


def build_report() -> Report:
    gpus, gpu_error = _query_gpus()
    torch_version, torch_cuda = _torch_info()

    report = Report(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=platform.python_version(),
        python_executable=sys.executable,
        cpu_count=os.cpu_count() or 0,
        ram_gb=_total_ram_gb(),
        free_disk_gb=_free_disk_gb(Path.cwd()),
        gpus=gpus,
        gpu_error=gpu_error,
        torch=torch_version,
        torch_cuda=torch_cuda,
    )

    checks: list[Check] = []

    # Python
    current = sys.version_info[:2]
    checks.append(
        Check(
            "python",
            Status.OK if current >= MIN_PYTHON else Status.FAIL,
            f"{platform.python_version()} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
        )
    )

    # GPU
    if gpu_error and not gpus:
        checks.append(Check("gpu", Status.FAIL, gpu_error))
    elif not gpus:
        checks.append(Check("gpu", Status.FAIL, "no NVIDIA GPU detected"))
    else:
        best = max(g.vram_gb for g in gpus)
        if best >= VRAM_GB_FOR_3B:
            detail = f"{best} GB VRAM - 1.5B and 3B both viable"
            status = Status.OK
        elif best >= VRAM_GB_FOR_1_5B:
            detail = f"{best} GB VRAM - 1.5B only; 3B needs 4-bit or a bigger card"
            status = Status.WARN
        else:
            detail = f"{best} GB VRAM - below {VRAM_GB_FOR_1_5B} GB, GRPO at group 8 will OOM"
            status = Status.FAIL
        checks.append(Check("gpu", status, detail))

    # RAM
    if report.ram_gb is None:
        checks.append(Check("ram", Status.WARN, "could not determine total RAM"))
    elif report.ram_gb >= RECOMMENDED_RAM_GB:
        checks.append(Check("ram", Status.OK, f"{report.ram_gb} GB"))
    elif report.ram_gb >= MIN_RAM_GB:
        checks.append(
            Check("ram", Status.WARN, f"{report.ram_gb} GB - FAISS index build will be tight")
        )
    else:
        checks.append(
            Check("ram", Status.FAIL, f"{report.ram_gb} GB - below {MIN_RAM_GB} GB minimum")
        )

    # Disk
    if report.free_disk_gb is None:
        checks.append(Check("disk", Status.WARN, "could not determine free disk"))
    elif report.free_disk_gb >= RECOMMENDED_FREE_DISK_GB:
        checks.append(Check("disk", Status.OK, f"{report.free_disk_gb} GB free"))
    else:
        checks.append(
            Check(
                "disk",
                Status.WARN,
                f"{report.free_disk_gb} GB free - corpus + index + checkpoints want "
                f"{RECOMMENDED_FREE_DISK_GB} GB",
            )
        )

    # torch (optional at P0; the PR pipeline does not install it)
    if torch_version is None:
        checks.append(Check("torch", Status.WARN, "not installed (expected at P0)"))
    else:
        checks.append(
            Check(
                "torch",
                Status.OK if torch_cuda else Status.WARN,
                f"{torch_version}, cuda_available={torch_cuda}",
            )
        )

    report.checks = checks
    return report


def print_report(report: Report) -> None:
    print("VeriSearch setup check")
    print("=" * 60)
    print(f"OS               {report.os}")
    print(f"Python           {report.python}  ({report.python_executable})")
    print(f"CPU cores        {report.cpu_count}")
    print(f"RAM              {report.ram_gb if report.ram_gb is not None else 'unknown'} GB")
    print(
        f"Free disk        "
        f"{report.free_disk_gb if report.free_disk_gb is not None else 'unknown'} GB"
    )
    if report.gpus:
        for i, gpu in enumerate(report.gpus):
            print(f"GPU {i}            {gpu.name}, {gpu.vram_gb} GB, driver {gpu.driver}")
    if report.gpu_error:
        print(f"GPU error        {report.gpu_error}")
    print(f"torch            {report.torch or 'not installed'}")
    print()
    print("Checks")
    print("-" * 60)
    for check in report.checks:
        print(f"[{check.status:<4}] {check.name:<8} {check.detail}")
    print()
    print("RESULT:", "FAIL" if report.failed else "PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    args = parser.parse_args()

    report = build_report()

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print_report(report)

    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

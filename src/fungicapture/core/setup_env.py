"""
Setting up the segmentation model, and working out why it is not running.

Why this module exists
----------------------
The application installs its interface and its science by default, but NOT the
deep-learning parts - those are hundreds of megabytes, and most of the pipeline
works without them. The consequence was a bad failure: with no PyTorch
installed, SAM 3 could never load, the application quietly fell back to the
classical CPU segmenter, and the user was left thinking "segmentation is broken
and it is stuck on CPU".

Two lessons are built in here:

1. **Never fail silently.** ``check_environment`` reports exactly which piece is
   missing and what to do about it.
2. **Install the right build.** ``pip install torch`` gives a CPU-only build on
   many systems. A GPU needs the wheel matching the installed driver, from
   PyTorch's own index. Getting that wrong is the single most common reason a
   machine with a working GPU still runs on CPU.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ProgressFn = Callable[[str], None]


def _noop(message: str) -> None:
    pass


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------


@dataclass
class GpuInfo:
    """What graphics hardware is present, as far as we can tell.

    Detection never raises and never blocks: every probe is wrapped, so on any
    machine the worst case is ``kind="none"`` and the classical segmenter, never
    a crash. ``detected_by`` records which probe succeeded, and ``notes`` carries
    anything worth telling the user (for example, a GPU whose driver is not
    responding).
    """

    present: bool = False
    """A GPU with a *working* driver, usable once the right PyTorch is installed."""

    name: str = ""
    driver_version: str = ""
    memory_mb: int = 0
    kind: str = "none"
    """``nvidia``, ``apple``, ``amd`` or ``none``."""

    hardware_seen: bool = False
    """An NVIDIA chip is physically present, even if the driver is not working.
    Distinguishes "no GPU" from "GPU here but driver missing" - very different
    advice."""

    detected_by: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def driver_major(self) -> int:
        match = re.match(r"(\d+)", self.driver_version or "")
        return int(match.group(1)) if match else 0


# --- individual probes, each best-effort and non-raising -------------------


def _find_nvidia_smi() -> str | None:
    """Find ``nvidia-smi`` even when it is not on PATH.

    A desktop-launched app can inherit a very short PATH that omits the NVIDIA
    directories, so ``shutil.which`` alone reports "no GPU" on a machine that
    plainly has one. Check the usual absolute locations too, across platforms.
    """
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidates = [
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",  # WSL2 GPU bridge
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return path
        except Exception:
            continue
    return None


def _query_nvidia_smi(smi: str) -> GpuInfo | None:
    """Ask nvidia-smi for the first GPU. Returns None if it is not usable."""
    try:
        output = subprocess.run(
            [
                smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if output.returncode != 0 or not output.stdout.strip():
        # nvidia-smi exists but cannot talk to the driver - a real state on a
        # machine whose driver is broken or not loaded.
        return None
    first = output.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    try:
        memory = int(float(parts[2])) if len(parts) > 2 and parts[2] else 0
    except ValueError:
        memory = 0
    return GpuInfo(
        present=True,
        name=parts[0] if parts and parts[0] else "NVIDIA GPU",
        driver_version=parts[1] if len(parts) > 1 else "",
        memory_mb=memory,
        kind="nvidia",
        detected_by="nvidia-smi",
    )


def _nvml_library_names() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return ["nvml.dll"]
    if system == "Darwin":
        return ["libnvidia-ml.dylib"]
    return ["libnvidia-ml.so.1", "libnvidia-ml.so"]


def _query_nvml() -> GpuInfo | None:
    """Query the NVIDIA Management Library directly, through ctypes.

    NVML is what ``nvidia-smi`` is built on. Loading it directly means the GPU
    is still detected when the ``nvidia-smi`` binary is missing (a driver-only
    install, an HPC module, a container) - as long as the driver itself is
    present. No third-party package is needed.
    """
    lib = None
    for name in _nvml_library_names():
        try:
            lib = ctypes.CDLL(name)
            break
        except Exception:
            continue
    if lib is None:
        return None

    try:
        if lib.nvmlInit_v2() != 0:
            if lib.nvmlInit() != 0:  # older driver entry point
                return None
    except Exception:
        try:
            if lib.nvmlInit() != 0:
                return None
        except Exception:
            return None

    info = GpuInfo(present=True, kind="nvidia", name="NVIDIA GPU", detected_by="nvml")
    try:
        buf = ctypes.create_string_buffer(96)
        try:
            lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(96))
            info.driver_version = buf.value.decode("utf-8", "ignore")
        except Exception:
            pass

        count = ctypes.c_uint(0)
        try:
            lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        except Exception:
            try:
                lib.nvmlDeviceGetCount(ctypes.byref(count))
            except Exception:
                count = ctypes.c_uint(0)

        if count.value > 0:
            handle = ctypes.c_void_p()
            got = False
            try:
                got = lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0
            except Exception:
                try:
                    got = lib.nvmlDeviceGetHandleByIndex(0, ctypes.byref(handle)) == 0
                except Exception:
                    got = False
            if got:
                name_buf = ctypes.create_string_buffer(96)
                try:
                    lib.nvmlDeviceGetName(handle, name_buf, ctypes.c_uint(96))
                    decoded = name_buf.value.decode("utf-8", "ignore")
                    if decoded:
                        info.name = decoded
                except Exception:
                    pass
                # Memory: nvmlMemory_t is {total, free, used} as 3x ulonglong.
                try:
                    mem = (ctypes.c_ulonglong * 3)()
                    if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0:
                        info.memory_mb = int(mem[0] / (1024 * 1024))
                except Exception:
                    pass
    finally:
        try:
            lib.nvmlShutdown()
        except Exception:
            pass
    return info


def _read_proc_driver_version() -> str:
    """Linux: the driver version string, even with no nvidia-smi installed."""
    try:
        text = Path("/proc/driver/nvidia/version").read_text(encoding="utf-8")
    except Exception:
        return ""
    match = re.search(r"Kernel Module\s+([0-9.]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{3}\.\d+(?:\.\d+)?)\b", text)
    return match.group(1) if match else ""


def _nvidia_hardware_present() -> str:
    """Is an NVIDIA GPU physically present, regardless of driver?

    Reads PCI vendor IDs straight from the kernel on Linux (vendor ``0x10de`` is
    NVIDIA), so it needs no external tool and works on a machine with the GPU
    but no driver installed - exactly the case where we must tell the user to
    install the driver rather than claim there is no GPU. Best-effort on other
    platforms.
    """
    system = platform.system()
    if system == "Linux":
        try:
            for vendor in Path("/sys/bus/pci/devices").glob("*/vendor"):
                try:
                    if vendor.read_text().strip().lower() == "0x10de":
                        return "NVIDIA GPU"
                except Exception:
                    continue
        except Exception:
            pass
        # lspci as a secondary check.
        smi = shutil.which("lspci")
        if smi:
            try:
                out = subprocess.run(
                    [smi], capture_output=True, text=True, timeout=10
                )
                for line in out.stdout.splitlines():
                    if "nvidia" in line.lower() and (
                        "vga" in line.lower() or "3d controller" in line.lower()
                    ):
                        return line.split(":")[-1].strip() or "NVIDIA GPU"
            except Exception:
                pass
    elif system == "Windows":
        try:
            out = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=15,
            )
            for line in out.stdout.splitlines():
                if "nvidia" in line.lower():
                    return line.strip()
        except Exception:
            pass
    return ""


def _rocm_present() -> bool:
    try:
        if shutil.which("rocminfo"):
            return True
        return os.path.isdir("/opt/rocm")
    except Exception:
        return False


def detect_gpu() -> GpuInfo:
    """
    Work out what graphics hardware is here, without needing PyTorch installed.

    Tries several independent probes so a single missing tool never hides a real
    GPU: nvidia-smi, then NVML through ctypes, then ``/proc`` for the driver
    version, then raw PCI enumeration to tell "no GPU" apart from "GPU present
    but driver not working". Every probe is wrapped; the function cannot raise.
    """
    # Apple Silicon: the chip has an integrated GPU, no separate driver to query.
    if sys.platform == "darwin":
        machine = os.uname().machine if hasattr(os, "uname") else ""
        if machine == "arm64":
            return GpuInfo(
                present=True, name="Apple Silicon (MPS)", kind="apple",
                detected_by="platform",
            )
        return GpuInfo(present=False, kind="none")

    # 1) nvidia-smi - the richest source when it works.
    smi = _find_nvidia_smi()
    if smi:
        info = _query_nvidia_smi(smi)
        if info is not None:
            if not info.driver_version:
                info.driver_version = _read_proc_driver_version()
            return info

    # 2) NVML directly, for a driver-only install with no nvidia-smi binary.
    info = _query_nvml()
    if info is not None:
        if not info.driver_version:
            info.driver_version = _read_proc_driver_version()
        return info

    # 3) Driver present in /proc but neither tool worked.
    driver = _read_proc_driver_version()
    if driver:
        return GpuInfo(
            present=True, name="NVIDIA GPU", driver_version=driver,
            kind="nvidia", detected_by="/proc",
        )

    # 4) AMD.
    if _rocm_present():
        return GpuInfo(
            present=True, name="AMD GPU (ROCm)", kind="amd", detected_by="rocm",
        )

    # 5) NVIDIA hardware physically present, but no working driver.
    hardware = _nvidia_hardware_present()
    if hardware:
        return GpuInfo(
            present=False, kind="none", hardware_seen=True, name=hardware,
            detected_by="pci",
            notes=[
                "An NVIDIA GPU is installed but its driver is not responding, "
                "so it cannot be used yet. Install or repair the NVIDIA driver "
                "for your system, then press 'Check again'. Until then "
                "segmentation runs on the processor.",
            ],
        )

    return GpuInfo(present=False, kind="none")


def recommend_torch_index(gpu: GpuInfo) -> tuple[str, str]:
    """
    Which PyTorch build to install, as ``(index_url, explanation)``.

    The CUDA runtime bundled in a PyTorch wheel needs a driver at least as new
    as itself. Installing a CUDA 12.4 build against a 2022-era driver produces
    a torch that imports fine and then reports ``cuda.is_available() == False``
    - which looks exactly like "stuck on CPU" and is very hard to diagnose. So
    the driver version chooses the wheel.
    """
    if gpu.kind == "apple":
        return (
            "",
            "Apple Silicon: the standard PyTorch build includes GPU support "
            "(MPS). No special index needed.",
        )
    if gpu.kind == "amd":
        return (
            "https://download.pytorch.org/whl/rocm6.2",
            "AMD GPU detected. Installing the ROCm build. Support is less "
            "widely tested than NVIDIA - if it misbehaves, the CPU build still "
            "works.",
        )
    if gpu.kind == "none" and gpu.hardware_seen:
        return (
            "https://download.pytorch.org/whl/cpu",
            f"An NVIDIA GPU ({gpu.name}) is present but its driver is not "
            "responding, so the GPU cannot be used yet. Installing the "
            "processor-only build now, which always works. Install or repair "
            "the NVIDIA driver, then reinstall to switch to the GPU build.",
        )
    if gpu.kind == "nvidia":
        major = gpu.driver_major
        if major >= 550:
            return (
                "https://download.pytorch.org/whl/cu124",
                f"NVIDIA driver {gpu.driver_version} supports CUDA 12.4.",
            )
        if major >= 525:
            return (
                "https://download.pytorch.org/whl/cu121",
                f"NVIDIA driver {gpu.driver_version} supports CUDA 12.1.",
            )
        if major >= 450:
            return (
                "https://download.pytorch.org/whl/cu118",
                f"NVIDIA driver {gpu.driver_version} is older, so CUDA 11.8 is "
                "the safe choice.",
            )
        return (
            "https://download.pytorch.org/whl/cpu",
            f"NVIDIA driver {gpu.driver_version} is too old for the current "
            "CUDA builds. Update your graphics driver to use the GPU; "
            "installing the CPU build for now.",
        )
    return (
        "https://download.pytorch.org/whl/cpu",
        "No GPU detected. Installing the CPU build, which is much smaller. "
        "Segmentation still works, just more slowly.",
    )


# --------------------------------------------------------------------------
# What is installed
# --------------------------------------------------------------------------


@dataclass
class EnvironmentReport:
    """A full picture of whether segmentation can run, and why not if it cannot."""

    gpu: GpuInfo
    torch_installed: bool = False
    torch_version: str = ""
    torch_cuda_build: str = ""
    cuda_available: bool = False
    mps_available: bool = False
    ultralytics_installed: bool = False
    ultralytics_version: str = ""
    weights_path: Path | None = None
    problems: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)

    @property
    def can_run_sam3(self) -> bool:
        return (
            self.torch_installed
            and self.ultralytics_installed
            and self.weights_path is not None
        )

    @property
    def effective_device(self) -> str:
        if self.cuda_available:
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"

    def summary(self) -> str:
        """One line for the status bar."""
        if not self.torch_installed:
            return "Segmentation model not installed - using the classical segmenter"
        if not self.ultralytics_installed:
            return "PyTorch present but Ultralytics missing - using the classical segmenter"
        if self.weights_path is None:
            return "SAM 3 weights not installed - using the classical segmenter"
        return f"SAM 3 ready on {self.effective_device.upper()}"

    def full_text(self) -> str:
        """The report shown in the setup window."""
        lines: list[str] = []

        lines.append("GRAPHICS HARDWARE")
        if self.gpu.present:
            detail = self.gpu.name
            if self.gpu.driver_version:
                detail += f"   driver {self.gpu.driver_version}"
            if self.gpu.memory_mb:
                detail += f"   {self.gpu.memory_mb / 1024:.1f} GB"
            lines.append(f"  found: {detail}")
            if self.gpu.detected_by:
                lines.append(f"  (detected via {self.gpu.detected_by})")
        elif self.gpu.hardware_seen:
            lines.append(
                f"  {self.gpu.name} is present, but its driver is not "
                "responding - the processor will be used until the driver is "
                "installed or repaired"
            )
        else:
            lines.append("  none found - the processor will be used instead")

        lines.append("")
        lines.append("SOFTWARE")
        lines.append(
            f"  PyTorch:     {self.torch_version or 'NOT INSTALLED'}"
            + (f"  (built for {self.torch_cuda_build})" if self.torch_cuda_build else "")
        )
        lines.append(
            f"  Ultralytics: {self.ultralytics_version or 'NOT INSTALLED'}"
        )
        lines.append(
            f"  SAM 3 model: {self.weights_path if self.weights_path else 'NOT INSTALLED'}"
        )

        lines.append("")
        lines.append("WHAT WILL BE USED")
        if self.can_run_sam3:
            lines.append(f"  SAM 3, running on {self.effective_device.upper()}")
            if self.gpu.present and self.effective_device == "cpu":
                lines.append(
                    "  WARNING: a GPU is present but is not being used - see below"
                )
        else:
            lines.append("  The classical segmenter (no model needed)")

        if self.problems:
            lines.append("")
            lines.append("PROBLEMS")
            for problem in self.problems:
                lines.append(f"  - {problem}")

        if self.advice:
            lines.append("")
            lines.append("WHAT TO DO")
            for item in self.advice:
                lines.append(f"  - {item}")

        return "\n".join(lines)


# The probe runs in a *separate* Python process. Importing a PyTorch build that
# does not match the graphics driver can hard-crash (segfault) the interpreter
# on ``torch.cuda.is_available()`` - and if that happened inside the GUI it
# would take the whole application down with no error. Isolating it means the
# worst case is "we could not read the report", never a crashed app.
_PROBE_SCRIPT = r"""
import json
info = {"torch": False, "torch_version": "", "cuda_build": "", "cuda": False,
        "mps": False, "cuda_name": "", "ultralytics": False, "ul_version": ""}
try:
    import torch
    info["torch"] = True
    info["torch_version"] = str(torch.__version__)
    b = getattr(torch.version, "cuda", None)
    info["cuda_build"] = ("CUDA " + b) if b else "CPU only"
    try:
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["cuda_name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        m = getattr(torch.backends, "mps", None)
        info["mps"] = bool(m and m.is_available())
    except Exception:
        pass
except Exception:
    pass
try:
    import ultralytics
    info["ultralytics"] = True
    info["ul_version"] = str(ultralytics.__version__)
except Exception:
    pass
print(json.dumps(info))
"""


def _probe_libraries(python_executable: str) -> tuple[dict | None, bool]:
    """
    Ask a separate process what PyTorch and Ultralytics can actually do.

    Returns ``(info, crashed)``. ``info`` is the parsed JSON, or ``None`` if the
    probe did not return usable output. ``crashed`` is True when the probe
    process died abnormally (a non-zero exit with no JSON) - which is exactly
    what a segfaulting torch looks like, and is reported as "installed but will
    not load" rather than "not installed".
    """
    import json

    try:
        result = subprocess.run(
            [python_executable, "-I", "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return None, False

    out = (result.stdout or "").strip()
    if out:
        try:
            return json.loads(out.splitlines()[-1]), False
        except Exception:
            pass
    # No parseable output. If the process died with a non-zero code, treat it
    # as a crash (segfault, abort) rather than a clean "nothing installed".
    return None, result.returncode != 0


def check_environment(python_executable: str | None = None) -> EnvironmentReport:
    """
    Work out whether SAM 3 can run, and if not, exactly why.

    Each missing piece produces both a problem and a matching instruction, so
    the user is never told "it does not work" without being told what to do.

    All PyTorch/GPU inspection happens in a child process (see
    :func:`_probe_libraries`), so a broken deep-learning install can never crash
    the running application - the historical cause of "it just closes when I
    open segmentation setup".
    """
    import importlib.util

    from .segment import find_model

    python_executable = python_executable or sys.executable
    gpu = detect_gpu()
    report = EnvironmentReport(gpu=gpu)

    # A GPU whose driver is not working is a problem worth stating up front,
    # with the fix, rather than silently running on the processor.
    if gpu.hardware_seen and not gpu.present:
        report.problems.append(
            f"{gpu.name} is present but its graphics driver is not responding."
        )
        report.advice.extend(gpu.notes)

    # Is each package even present? find_spec does not execute the package, so
    # it is safe to call in-process even for a torch build that would segfault
    # on import.
    torch_present = importlib.util.find_spec("torch") is not None
    ultra_present = importlib.util.find_spec("ultralytics") is not None

    info, crashed = _probe_libraries(python_executable)

    if info is not None:
        report.torch_installed = bool(info.get("torch"))
        report.torch_version = info.get("torch_version", "")
        report.torch_cuda_build = info.get("cuda_build", "")
        report.cuda_available = bool(info.get("cuda"))
        report.mps_available = bool(info.get("mps"))
        report.ultralytics_installed = bool(info.get("ultralytics"))
        report.ultralytics_version = info.get("ul_version", "")
        if info.get("cuda") and info.get("cuda_name") and not gpu.name:
            gpu.present = True
            gpu.name = info["cuda_name"]
    elif crashed and torch_present:
        # The probe process died - almost always a PyTorch build that does not
        # match the installed graphics driver, which segfaults on the CUDA
        # check. This is the real cause of the app "crashing" on setup.
        report.torch_installed = False
        report.problems.append(
            "PyTorch is installed but crashes when it loads. This is almost "
            "always a mismatch between the PyTorch build and your graphics "
            "driver (or a broken install)."
        )
        report.advice.append(
            "Tick 'Install the processor-only version' and press "
            "'Install PyTorch + Ultralytics' - the CPU build does not touch the "
            "graphics driver and will not crash. You can move back to the GPU "
            "build once the driver is updated."
        )
        report.ultralytics_installed = ultra_present

    if not report.torch_installed and not (crashed and torch_present):
        report.problems.append(
            "PyTorch is not installed, so the SAM 3 model cannot run at all."
        )
        report.advice.append(
            "Press 'Install PyTorch + Ultralytics' below. This downloads about "
            "2 GB and takes several minutes."
        )

    if report.torch_installed and not report.ultralytics_installed:
        report.problems.append(
            "Ultralytics is not installed. It provides the SAM 3 interface."
        )
        if not any("Install PyTorch + Ultralytics" in a for a in report.advice):
            report.advice.append("Press 'Install PyTorch + Ultralytics' below.")

    report.weights_path = find_model()
    if report.weights_path is None:
        report.problems.append("The SAM 3 weights file has not been installed.")
        report.advice.append(
            "Use 'Install weights...' to point the application at the file. "
            "Meta requires you to request access and accept their licence on "
            "their own page first."
        )

    # The subtle case: everything installed, GPU present, still on CPU.
    if report.torch_installed and gpu.present and not (
        report.cuda_available or report.mps_available
    ):
        if gpu.kind == "nvidia":
            report.problems.append(
                f"A GPU is present ({gpu.name}) but PyTorch cannot use it. "
                f"PyTorch reports it was built for: {report.torch_cuda_build}."
            )
            if "CPU only" in report.torch_cuda_build:
                report.advice.append(
                    "The CPU-only build of PyTorch is installed. Press "
                    "'Install PyTorch + Ultralytics' - it will detect your driver "
                    "and fetch the matching GPU build."
                )
            else:
                report.advice.append(
                    f"Your graphics driver ({gpu.driver_version}) may be too "
                    "old for this PyTorch build. Either update the driver, or "
                    "press 'Install PyTorch + Ultralytics' to fit the build to "
                    "the driver you have."
                )

    return report


# --------------------------------------------------------------------------
# Installing
# --------------------------------------------------------------------------


def _run_pip(
    python_executable: str, args: list[str], *, timeout: int = 3600
) -> tuple[bool, str]:
    """
    Run one ``python -m pip`` command.

    Returns ``(succeeded, detail)`` where ``detail`` is the last few lines of
    output, useful for showing the user *why* an install failed. Never raises -
    a broken install must leave the user with an explanation, not a traceback.
    """
    command = [python_executable, "-m", "pip", *args]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, (
            "The step took too long and was stopped. This is almost always a "
            "slow or dropped connection."
        )
    except Exception as error:  # pragma: no cover - pip should always run
        return False, f"Could not run pip: {error}"

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, "\n".join(tail[-8:]) if tail else "no output from pip"
    return True, ""


def install_segmentation(
    progress: ProgressFn = _noop,
    *,
    python_executable: str | None = None,
    force_cpu: bool = False,
) -> tuple[bool, str]:
    """
    Install PyTorch and Ultralytics, choosing the build that suits this machine.

    Returns ``(succeeded, message)``. Never raises: a failed install must leave
    the user with an explanation, not a stack trace.
    """
    python_executable = python_executable or sys.executable
    gpu = GpuInfo() if force_cpu else detect_gpu()
    index_url, explanation = recommend_torch_index(gpu)

    # pip must be reachable through this interpreter. In a frozen (PyInstaller)
    # build ``sys.executable`` is the app, not Python, and ``-m pip`` is absent.
    # Say so plainly rather than failing deep inside a pip subprocess.
    ok, _ = _run_pip(python_executable, ["--version"], timeout=60)
    if not ok:
        return False, (
            "Could not find 'pip' for this installation, so libraries cannot "
            "be installed automatically.\n\n"
            "If you started FungiCapture from the START launcher this should "
            "not happen - please reinstall using the launcher. If you are "
            "running a packaged (.exe/.app) build, install PyTorch and "
            "Ultralytics into a normal Python environment instead."
        )

    progress(explanation)
    progress("")

    # --- Step 1: PyTorch + torchvision (the required deep-learning libraries).
    progress("Installing PyTorch and torchvision (the big one - several minutes)...")
    torch_pkgs = ["torch", "torchvision"]
    torch_args = ["install", *torch_pkgs]
    if index_url:
        torch_args += ["--index-url", index_url]

    ok, detail = _run_pip(python_executable, torch_args)
    if not ok:
        # A GPU-specific build can fail for reasons the CPU build does not share
        # (an index without a matching wheel, a driver mismatch). Rather than
        # leave the user with nothing, fall back to the CPU build automatically.
        cpu_index = "https://download.pytorch.org/whl/cpu"
        if index_url and index_url != cpu_index and not force_cpu:
            progress("")
            progress(
                "  The GPU build did not install. Falling back to the "
                "processor-only build so segmentation still works..."
            )
            ok, detail = _run_pip(
                python_executable,
                ["install", *torch_pkgs, "--index-url", cpu_index],
            )
        if not ok:
            return False, (
                f"Installing PyTorch failed.\n\n{detail}\n\n"
                "The usual causes are a dropped internet connection or no "
                "space left on the disk."
            )
    progress("  PyTorch and torchvision done.")

    # --- Step 2: Ultralytics (the SAM 3 interface). 8.3.237 is the first
    # release with SAM 3, so the pin matters: an older 8.3.x installs cleanly
    # and then cannot load the model.
    #
    # torch is installed FIRST, above, precisely so this step keeps it: pip is
    # not given --upgrade, so the GPU build already present satisfies
    # ultralytics' ``torch>=1.8.0`` and is left untouched. Without that order
    # ultralytics would pull a plain-PyPI torch and quietly undo the GPU build.
    progress("Installing Ultralytics (the SAM 3 interface)...")
    ok, detail = _run_pip(
        python_executable, ["install", "ultralytics>=8.3.237"]
    )
    if not ok:
        return False, (
            f"Installing Ultralytics failed.\n\n{detail}\n\n"
            "The usual causes are a dropped internet connection or no "
            "space left on the disk."
        )

    # --- Step 3: undo Ultralytics' Qt-breaking OpenCV.
    # Ultralytics depends on ``opencv-python`` - the DESKTOP build, which ships
    # its own Qt plugins. Installed next to our ``opencv-python-headless`` and
    # PySide6, those bundled plugins hijack the Qt platform plugin and the whole
    # application crashes on the next launch with "could not load the Qt
    # platform plugin" - which looks exactly like "installing SAM broke the
    # program".
    #
    # This is done unconditionally, not "only if present": the two builds share
    # the same ``cv2`` folder, so uninstalling the desktop build also deletes
    # cv2 for the headless build, and pip will NOT rewrite the "already
    # installed" headless files without --force-reinstall. So: remove the
    # desktop builds, then force-reinstall headless to restore a clean,
    # GUI-safe cv2. --no-deps keeps it from disturbing numpy/torch.
    progress("Making the image library safe for the graphical interface...")
    _run_pip(
        python_executable,
        ["uninstall", "-y", "opencv-python", "opencv-contrib-python"],
        timeout=300,
    )
    ok, detail = _run_pip(
        python_executable,
        ["install", "--force-reinstall", "--no-deps", "opencv-python-headless>=4.8"],
        timeout=600,
    )
    if not ok:
        return False, (
            "SAM installed, but the image library could not be made safe for "
            f"the graphical interface.\n\n{detail}\n\n"
            "Run the START launcher again to repair, or reinstall from scratch."
        )
    progress("  Ultralytics done.")

    progress("")
    progress("Checking what was installed...")
    # Python caches which packages exist when it starts, so a package installed
    # a moment ago in this same process may not import until the caches are
    # cleared. Clear them before checking, so the report reflects reality.
    import importlib

    importlib.invalidate_caches()
    report = check_environment()

    if not report.torch_installed:
        # pip finished successfully (returncode 0 on every step above), so the
        # libraries are on disk. This running process simply cannot import the
        # brand-new install yet - which a restart fixes. That is success
        # pending a restart, NOT a failure.
        message = (
            "Installed successfully. PyTorch and Ultralytics are now on this "
            "computer.\n\n" + restart_note()
        )
        if report.weights_path is None:
            message += (
                "\n\nOne step left after restarting: the SAM 3 weights file. "
                "Use 'Install weights...' - Meta requires you to accept their "
                "licence on their own page before downloading it."
            )
        return True, message

    if report.cuda_available:
        message = f"Ready. SAM 3 will run on your GPU ({gpu.name})."
    elif report.mps_available:
        message = "Ready. SAM 3 will run on the Apple Silicon GPU."
    elif gpu.present and not force_cpu:
        message = (
            f"Installed, but PyTorch still cannot reach the GPU ({gpu.name}).\n\n"
            f"PyTorch build: {report.torch_cuda_build}\n"
            f"Graphics driver: {gpu.driver_version or 'unknown'}\n\n"
            "This nearly always means the graphics driver is older than the "
            "CUDA version in the build. Updating the driver is the fix. "
            "Segmentation will work on the processor in the meantime."
        )
    else:
        message = (
            "Ready. SAM 3 will run on the processor. That works, it is just "
            "slower than a GPU."
        )

    if report.weights_path is None:
        message += (
            "\n\nOne step left: the SAM 3 weights file. Use 'Install "
            "weights...' - Meta requires you to accept their licence on their "
            "own page before downloading it."
        )
    return True, message


def restart_note() -> str:
    """Why a restart is sometimes needed after installing."""
    return (
        "Please close and reopen FungiCapture. Python loads libraries once when "
        "it starts, so newly installed ones are only picked up on a fresh start."
    )

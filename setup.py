#!/usr/bin/env python3
"""
One-command setup for the Caspian Sea PINN project.
Auto-detects existing venv, installed deps, and downloaded dataset.
Skips what's already done.

Usage:
    python setup.py              # Interactive (asks: venv? CUDA?)
    uv run setup.py              # Use uv instead of pip
    python setup.py --yes        # Non-interactive, use defaults
    python setup.py --cpu        # Force CPU-only PyTorch
    python setup.py --deps-only  # Only install dependencies
    python setup.py --data-only  # Only download dataset
"""

import subprocess
import sys
import shutil
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
DATASET_DIR = REPO_ROOT / "dataset"
DATASET_OWNER = "chingchonghaha"
DATASET_SLUG = "caspian-sea-hydroclimatic-dataset-1993-2026"

TORCH_INDEX = "https://download.pytorch.org/whl/cu118"


def green(text): return f"\033[32m{text}\033[0m"
def bold(text):  return f"\033[1m{text}\033[0m"
def dim(text):   return f"\033[2m{text}\033[0m"


def detect_cuda():
    """Check for NVIDIA GPU."""
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return True, r.stdout.strip().split("\n")[0]
        except Exception:
            pass
    if Path("/proc/driver/nvidia/version").exists():
        return True, "NVIDIA driver detected"
    return False, None


def has_venv():
    """Check if venv exists and has python."""
    py = VENV_DIR / "bin" / "python"
    return py.exists()


def get_python():
    """Return path to venv python if it exists, else system python."""
    py = VENV_DIR / "bin" / "python"
    return str(py) if py.exists() else sys.executable


def pip_install(packages, index_url=None):
    """Install packages with pip."""
    cmd = [get_python(), "-m", "pip", "install", "--quiet"] + packages
    if index_url:
        cmd += ["--index-url", index_url]
    subprocess.run(cmd, check=True)


def check_package(pkg_name):
    """Check if a Python package is installed."""
    try:
        subprocess.run([get_python(), "-c", f"import {pkg_name}"],
                       capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def install_deps(force_cpu=False, yes=False):
    """Install Python dependencies. Auto-detect CUDA, skip if already installed."""

    # -- Check what's already installed --
    need_torch = not check_package("torch")
    need_numpy = not check_package("numpy")
    need_pandas = not check_package("pandas")
    need_xarray = not check_package("xarray")

    if not any([need_torch, need_numpy, need_pandas, need_xarray]):
        print(green("All Python dependencies already installed. Skipping."))
        subprocess.run([get_python(), "-c",
            "import torch; print(f'  PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"])
        return

    # -- Detect CUDA --
    has_cuda, gpu_name = detect_cuda()
    use_cuda = has_cuda and not force_cpu

    print(f"\n{bold('Installing dependencies')}")
    print(f"  {'GPU' if use_cuda else 'CPU'} mode")

    # Read requirements (skip torch line)
    deps = []
    if REQUIREMENTS.exists():
        with open(REQUIREMENTS) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("torch"):
                    deps.append(line)

    # Install torch separately (CUDA or CPU)
    if need_torch:
        if use_cuda:
            pip_install(["torch", "torchvision", "torchaudio"], index_url=TORCH_INDEX)
        else:
            pip_install(["torch"])

    # Install everything else
    if deps:
        missing = [d for d in deps if not check_package(d.split(">=")[0].split("==")[0])]
        if missing:
            pip_install(missing)
        else:
            print(dim("  Other packages already installed."))

    # Verify
    subprocess.run([get_python(), "-c",
        "import torch; import numpy; import pandas; import xarray; "
        "print(f'  PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"],
        check=True)
    print(green("Dependencies ready."))


def dataset_exists():
    """Check if dataset is already downloaded and extracted."""
    if DATASET_DIR.exists():
        nc_files = list(DATASET_DIR.glob("*.nc"))
        if nc_files:
            return True, nc_files
    return False, []


def download_dataset():
    """Download dataset from Kaggle, skip if already present."""
    exists, files = dataset_exists()
    if exists:
        print(green(f"Dataset already downloaded ({len(files)} NetCDF files). Skipping."))
        return

    print(f"\n{bold('Downloading dataset from Kaggle')}")
    print(f"  https://www.kaggle.com/datasets/{DATASET_OWNER}/{DATASET_SLUG}")

    # Try kaggle CLI
    if shutil.which("kaggle"):
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                subprocess.run(["kaggle", "datasets", "download", "-d",
                    f"{DATASET_OWNER}/{DATASET_SLUG}", "-p", str(REPO_ROOT)],
                    check=True)
                _extract_dataset()
                return
            except Exception as e:
                print(dim(f"  Kaggle CLI failed: {e}"))

    # Manual instructions
    print(f"\n  {'='*50}")
    print(f"  Manual download required:")
    print(f"  1. Visit: https://www.kaggle.com/datasets/{DATASET_OWNER}/{DATASET_SLUG}")
    print(f"  2. Download dataset.rar")
    print(f"  3. Place it at: {REPO_ROOT}/dataset.rar")
    print(f"  4. Run: python download_dataset.py")
    print(f"\n  Or install kaggle CLI: pip install kaggle")
    print(f"  Then create API key at https://www.kaggle.com/settings/account")
    print(f"  Place kaggle.json at ~/.kaggle/kaggle.json")
    print(f"  {'='*50}")


def _extract_dataset():
    """Extract dataset.rar."""
    rar = REPO_ROOT / "dataset.rar"
    if not rar.exists():
        return
    DATASET_DIR.mkdir(exist_ok=True)
    if shutil.which("unrar"):
        subprocess.run(["unrar", "x", "-y", str(rar), str(DATASET_DIR)], check=True)
        rar.unlink()
        print(green("Dataset extracted."))
    elif shutil.which("7z"):
        subprocess.run(["7z", "x", str(rar), f"-o{DATASET_DIR}", "-y"], check=True)
        rar.unlink()
        print(green("Dataset extracted."))


def create_venv(yes=False):
    """Create virtual environment if it doesn't exist."""
    if has_venv():
        print(green("Virtual environment already exists. Skipping."))
        return

    if not yes:
        choice = input(f"\nCreate virtual environment at {VENV_DIR}? [Y/n] ").strip().lower()
        if choice and choice not in ("y", "yes"):
            print(dim("Skipping venv. Installing packages globally."))
            return

    print(f"Creating venv at {VENV_DIR}...")
    venv.create(VENV_DIR, with_pip=True)
    print(green("Virtual environment created."))


def install_uv_deps(force_cpu=False):
    """Install using uv if available (faster)."""
    if not shutil.which("uv"):
        return False

    has_cuda, _ = detect_cuda()
    use_cuda = has_cuda and not force_cpu

    with open(REQUIREMENTS) as f:
        deps = [l.strip() for l in f if l.strip() and not l.startswith("#") and not l.startswith("torch")]

    if use_cuda:
        subprocess.run(["uv", "pip", "install"] + deps, check=True)
        subprocess.run(["uv", "pip", "install", "torch", "torchvision", "torchaudio",
                        "--index-url", TORCH_INDEX], check=True)
    else:
        subprocess.run(["uv", "pip", "install", "-r", str(REQUIREMENTS)], check=True)

    print(green("Dependencies installed via uv."))
    return True


def main():
    yes = "--yes" in sys.argv
    deps_only = "--deps-only" in sys.argv
    data_only = "--data-only" in sys.argv
    force_cpu = "--cpu" in sys.argv

    print(f"\n{bold('Caspian Sea PINN — Setup')}")
    print(f"  Paper ID 100 — IEEE DG 2026")
    print(f"  {REPO_ROOT}")

    if data_only:
        download_dataset()
        return

    # 1. Venv
    create_venv(yes)

    # 2. Dependencies
    if deps_only or not data_only:
        # Try uv first, fall back to pip
        installed = install_uv_deps(force_cpu) if shutil.which("uv") else False
        if not installed:
            install_deps(force_cpu, yes)

    if deps_only:
        return

    # 3. Dataset
    download_dataset()

    # 4. Done
    print(f"\n{bold('Setup complete!')}")
    print(f"  Run the pipeline:")
    print(f"    {get_python()} master_runner.py --stage all --resolution monthly")
    print(f"  Compile the paper:")
    print(f"    cd manuscript/latex && tectonic main.tex")


if __name__ == "__main__":
    main()

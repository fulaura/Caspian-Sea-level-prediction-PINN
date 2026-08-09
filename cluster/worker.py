import os
import sys
import time
import json
import zipfile
import subprocess
import shutil
import platform
from datetime import datetime

try:
    import requests
except ImportError:
    print("[INIT] Installing required dependency 'requests'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

SERVER_URL = os.environ.get("PINN_SERVER_URL", "http://34.170.167.48:8000")
WORKER_ID = f"worker_{os.getpid()}_{int(time.time())}"

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "xarray",
    "netcdf4",
    "matplotlib",
    "scipy",
    "seaborn",
    "requests"
]

def get_base_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == "cluster":
        return os.path.abspath(os.path.join(current_dir, ".."))
    return current_dir

def ensure_uv_installed():
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        try:
            print("[ENV SETUP] Installing 'uv' package manager for 10x faster environment setup...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
            return True
        except Exception:
            return False

def setup_venv_and_get_python(base_dir):
    venv_dir = os.path.join(base_dir, ".venv")
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    has_uv = ensure_uv_installed()

    # 1. Create venv if missing
    if not os.path.exists(venv_python):
        print(f"\n[ENV SETUP] Virtual environment not found. Creating '.venv' at {venv_dir}...")
        if has_uv:
            subprocess.check_call([sys.executable, "-m", "uv", "venv", venv_dir, "--python", sys.executable])
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print("[ENV SETUP] Virtual environment created successfully!")

    # 2. Check & Install required packages into venv
    flag_file = os.path.join(venv_dir, ".packages_installed_cuda")
    if not os.path.exists(flag_file):
        print("\n[ENV SETUP] Installing PyTorch & dependencies into '.venv'...")
        
        try:
            if has_uv:
                print("[ENV SETUP] Installing PyTorch CUDA 12.1 via ultra-fast 'uv'...")
                uv_torch_cmd = [sys.executable, "-m", "uv", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121", "--python", venv_python]
                subprocess.check_call(uv_torch_cmd)

                print("[ENV SETUP] Installing ML libraries (numpy, pandas, xarray, etc.) via 'uv'...")
                other_pkgs = [p for p in REQUIRED_PACKAGES if p != "torch"]
                uv_pkg_cmd = [sys.executable, "-m", "uv", "pip", "install"] + other_pkgs + ["--python", venv_python]
                subprocess.check_call(uv_pkg_cmd)
            else:
                print("[ENV SETUP] Attempting PyTorch installation with CUDA 12.1 support via pip...")
                torch_cmd = [venv_python, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"]
                subprocess.check_call(torch_cmd)
                other_pkgs = [p for p in REQUIRED_PACKAGES if p != "torch"]
                subprocess.check_call([venv_python, "-m", "pip", "install"] + other_pkgs)

            with open(flag_file, "w") as f:
                f.write("INSTALLED_CUDA_UV")
            print("[ENV SETUP] All dependencies installed successfully!\n")

        except Exception as e:
            print(f"[ENV SETUP WARNING] CUDA index install failed: {e}. Falling back to standard PyPI torch...")
            try:
                if has_uv:
                    subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "torch"] + REQUIRED_PACKAGES + ["--python", venv_python])
                else:
                    subprocess.check_call([venv_python, "-m", "pip", "install", "torch"] + REQUIRED_PACKAGES)
                with open(flag_file, "w") as f:
                    f.write("INSTALLED_FALLBACK")
            except Exception as e2:
                print("\n[ERROR] Package installation failed!")
                print("   Solution: Please run worker.py using Python 3.12 (e.g., `py -3.12 worker.py`).\n")
                sys.exit(1)

    # 3. Check CUDA status using venv python
    try:
        cuda_check_code = "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()} | Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
        res = subprocess.check_output([venv_python, "-c", cuda_check_code], text=True).strip()
        print(f"[GPU SETUP] {res}")
    except Exception as e:
        print(f"[GPU SETUP] Couldn't check CUDA: {e}")

    return venv_python

def prompt_worker_name():
    print("=" * 60)
    print("🚀 CASPIAN PINN DISTRIBUTED WORKER NODE")
    print("=" * 60)
    name = input("--> Enter your name / worker ID (e.g. Alex-PC, Beka-GPU): ").strip()
    if not name:
        name = f"Worker_{platform.node()}"
    return name

def check_dataset_exists(base_dir):
    dataset_path = os.path.join(base_dir, "dataset")
    if not os.path.exists(dataset_path):
        print("\n[WARNING] 'dataset/' folder not found in working directory!")
        print(f"   Expected path: {os.path.abspath(dataset_path)}")
        print("   Please place the extracted 'dataset/' folder inside this directory before training.\n")

import hashlib

def get_self_hash():
    current_file = os.path.abspath(__file__)
    hasher = hashlib.md5()
    with open(current_file, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def check_and_apply_auto_update(server_url, server_hash):
    if not server_hash:
        return
    local_hash = get_self_hash()
    if local_hash != server_hash:
        print("\n" + "🔄" * 30)
        print(f"[AUTO-UPDATE] New worker version detected on server!")
        print(f"   Local Hash:  {local_hash[:8]}")
        print(f"   Server Hash: {server_hash[:8]}")
        print("   Downloading updated worker.py...")
        
        try:
            resp = requests.get(f"{server_url}/api/worker/download_latest", timeout=15)
            if resp.status_code == 200:
                current_file = os.path.abspath(__file__)
                with open(current_file, "wb") as f:
                    f.write(resp.content)
                print("[AUTO-UPDATE] worker.py successfully updated!")
                print("[AUTO-UPDATE] Restarting worker process...\n")
                
                os.execv(sys.executable, [sys.executable, current_file] + sys.argv[1:])
        except Exception as e:
            print(f"[AUTO-UPDATE ERROR] Failed to update worker: {e}")

def main():
    base_dir = get_base_dir()
    venv_python = setup_venv_and_get_python(base_dir)

    server_url = input(f"--> Enter Server URL [default: {SERVER_URL}]: ").strip()
    if not server_url:
        server_url = SERVER_URL
    server_url = server_url.rstrip("/")

    worker_name = prompt_worker_name()
    check_dataset_exists(base_dir)

    print(f"\n[CONNECTED] Registered as '{worker_name}' to server: {server_url}")
    print(f"[ENV] Using Python Interpreter: {venv_python}")
    print("[STATUS] Polling for jobs... (Press Ctrl+C to stop)\n")

    while True:
        try:
            resp = requests.get(f"{server_url}/api/jobs/poll", params={"worker_id": WORKER_ID, "name": worker_name}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                
                # Check for worker code updates from server
                server_hash = data.get("latest_worker_hash")
                check_and_apply_auto_update(server_url, server_hash)

                if data.get("has_job"):
                    job_id = data["job_id"]
                    command = data["command"]
                    target_dir = data["target_dir"]

                    print("\n" + "=" * 60)
                    print(f"⚡ [JOB RECEIVED] ID: {job_id}")
                    print(f"   Command:    {command}")
                    print(f"   Target Dir: {target_dir}")
                    print("=" * 60)

                    # 1. Download payload
                    p_resp = requests.get(f"{server_url}/api/jobs/payload/{job_id}", timeout=30)
                    payload_zip = os.path.join(base_dir, f"_temp_{job_id}.zip")
                    with open(payload_zip, "wb") as f:
                        f.write(p_resp.content)

                    # 2. Extract code payload
                    with zipfile.ZipFile(payload_zip, 'r') as zip_ref:
                        zip_ref.extractall(base_dir)
                    os.remove(payload_zip)

                    # 3. Execute job & measure time
                    start_time = time.time()
                    start_iso = datetime.now().isoformat()
                    
                    full_target_path = os.path.join(base_dir, target_dir)
                    os.makedirs(full_target_path, exist_ok=True)

                    print(f"   Starting execution at {start_iso}...")
                    
                    # Run command using the venv python interpreter
                    cmd_parts = command.split()
                    if cmd_parts[0] in ["python", "python3"]:
                        cmd_parts[0] = venv_python

                    process = subprocess.Popen(
                        cmd_parts,
                        cwd=base_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )

                    log_lines = []
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log_lines.append(line)
                    process.wait()

                    end_time = time.time()
                    end_iso = datetime.now().isoformat()
                    elapsed_seconds = end_time - start_time
                    success = (process.returncode == 0)

                    minutes = int(elapsed_seconds // 60)
                    seconds = int(elapsed_seconds % 60)
                    formatted_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

                    print(f"\n[JOB COMPLETE] Status: {'SUCCESS' if success else 'FAILED'}")
                    print(f"   Total Time Spent: {formatted_time} ({elapsed_seconds:.2f} seconds)")

                    # 4. Save job_info.json inside target_dir
                    job_info = {
                        "job_id": job_id,
                        "worker_name": worker_name,
                        "worker_id": WORKER_ID,
                        "command": command,
                        "start_time": start_iso,
                        "end_time": end_iso,
                        "elapsed_seconds": round(elapsed_seconds, 2),
                        "elapsed_formatted": formatted_time,
                        "status": "SUCCESS" if success else "FAILED"
                    }
                    with open(os.path.join(full_target_path, "job_info.json"), "w") as f:
                        json.dump(job_info, f, indent=2)

                    # 5. Zip target_dir
                    results_zip_path = os.path.join(base_dir, f"_res_{job_id}.zip")
                    with zipfile.ZipFile(results_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, _, files in os.walk(full_target_path):
                            for file in files:
                                fpath = os.path.join(root, file)
                                arcname = os.path.relpath(fpath, full_target_path)
                                zipf.write(fpath, arcname)

                    # 6. Upload results back to server
                    with open(results_zip_path, "rb") as rf:
                        files = {"results_zip": rf}
                        form_data = {
                            "job_id": job_id,
                            "worker_id": WORKER_ID,
                            "success": str(success).lower(),
                            "elapsed_seconds": str(elapsed_seconds),
                            "log_output": "".join(log_lines[-100:])
                        }
                        requests.post(f"{server_url}/api/jobs/complete", data=form_data, files=files, timeout=60)

                    os.remove(results_zip_path)
                    print(f"[STATUS] Results uploaded to server. Waiting for next job...\n")

        except KeyboardInterrupt:
            print("\n[WORKER] Stopped by user.")
            break
        except Exception as e:
            print(f"[WORKER ERROR] {e}")

        time.sleep(3)

if __name__ == "__main__":
    main()

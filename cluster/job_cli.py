import os
import sys
import time
import zipfile
import argparse
import requests

SERVER_URL = os.environ.get("PINN_SERVER_URL", "http://34.170.167.48:8000")

def zip_code_payload(base_dir, zip_path):
    # Only include code files (.py, .json, .md, .sh) and exclude dataset/results/venv
    include_dirs = ["scripts", "model", "cluster", "manuscript"]
    include_files = ["master_runner.py", "handoff.md", "requirements.txt"]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in include_files:
            fp = os.path.join(base_dir, f)
            if os.path.exists(fp):
                zipf.write(fp, f)

        for d in include_dirs:
            dp = os.path.join(base_dir, d)
            if os.path.exists(dp):
                for root, _, files in os.walk(dp):
                    for file in files:
                        if file.endswith(('.py', '.json', '.md', '.sh', '.tex', '.bib', '.cls')):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, base_dir)
                            zipf.write(full_path, rel_path)

def sync_worker_code(base_dir, server_url):
    worker_file = os.path.join(base_dir, "cluster", "worker.py")
    if not os.path.exists(worker_file):
        worker_file = os.path.join(base_dir, "worker.py")
    if os.path.exists(worker_file):
        try:
            with open(worker_file, "rb") as wf:
                files = {"worker_file": wf}
                resp = requests.post(f"{server_url}/api/worker/upload_latest", files=files, timeout=10)
                if resp.status_code == 200:
                    print(f"🔄 Worker code synced with server (Hash: {resp.json().get('worker_hash')[:8]})")
        except Exception as e:
            print(f"⚠️ Could not sync worker code to server: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dispatch jobs to Caspian PINN Distributed Cluster")
    parser.add_argument("action", choices=["submit", "status", "update-worker"], help="Action to perform")
    parser.add_argument("--command", type=str, help="Python command to execute on worker")
    parser.add_argument("--target-dir", type=str, help="Relative output target directory (e.g., results/multiseed_evaluation/pinn1/1m/seed_42)")
    parser.add_argument("--job-id", type=str, help="Custom job ID (optional)")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="GCP Server URL")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion and auto-download results")

    args = parser.parse_args()
    server_url = args.server.rstrip("/")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.action == "update-worker":
        print("🔄 Uploading current worker.py to GCP server...")
        sync_worker_code(base_dir, server_url)
        print("✅ Worker code updated! Active worker nodes will auto-update on next poll.")
        sys.exit(0)

    if args.action == "submit":
        if not args.command or not args.target_dir:
            print("Error: --command and --target-dir are required for submission.")
            sys.exit(1)

        # Auto-sync worker code on submission
        sync_worker_code(base_dir, server_url)

        job_id = args.job_id or f"job_{int(time.time())}"
        payload_zip = os.path.join(base_dir, f"_payload_{job_id}.zip")

        print(f"📦 Packaging code payload into {payload_zip}...")
        zip_code_payload(base_dir, payload_zip)

        print(f"🚀 Submitting job '{job_id}' to server at {server_url}...")
        with open(payload_zip, "rb") as pf:
            form_data = {
                "job_id": job_id,
                "command": args.command,
                "target_dir": args.target_dir
            }
            files = {"code_payload": pf}
            resp = requests.post(f"{server_url}/api/jobs/submit", data=form_data, files=files)

        os.remove(payload_zip)

        if resp.status_code == 200:
            print(f"✅ [SUCCESS] Job queued successfully! Job ID: {job_id}")
            print(f"   Dashboard: {server_url}/")
        else:
            print(f"❌ [ERROR] Submission failed: {resp.text}")
            sys.exit(1)

        if args.wait:
            print(f"\n⏳ Waiting for job '{job_id}' to complete...")
            while True:
                time.sleep(4)
                status_resp = requests.get(f"{server_url}/api/jobs/status/{job_id}").json()
                st = status_resp.get("status")
                worker = status_resp.get("assigned_worker_name") or "Unassigned"
                print(f"   [STATUS] {st} (Worker: {worker})")

                if st in ["COMPLETED", "FAILED"]:
                    if st == "COMPLETED":
                        print(f"\n🎉 Job completed successfully by worker '{worker}' in {status_resp.get('elapsed_seconds', 0):.1f}s!")
                        # Download results zip
                        r_resp = requests.get(f"{server_url}/api/jobs/download_result/{job_id}")
                        if r_resp.status_code == 200:
                            out_zip = os.path.join(base_dir, f"_res_{job_id}.zip")
                            with open(out_zip, "wb") as rf:
                                rf.write(r_resp.content)

                            dest_path = os.path.join(base_dir, args.target_dir)
                            os.makedirs(dest_path, exist_ok=True)
                            with zipfile.ZipFile(out_zip, 'r') as zr:
                                zr.extractall(dest_path)
                            os.remove(out_zip)
                            print(f"📁 Extracted results directly into: {dest_path}")
                    else:
                        print(f"❌ Job failed on worker '{worker}'. Log excerpt:\n{status_resp.get('log_output')}")
                    break

    elif args.action == "status":
        if not args.job_id:
            print("Error: --job-id required for status check.")
            sys.exit(1)
        resp = requests.get(f"{server_url}/api/jobs/status/{args.job_id}").json()
        print(f"Job ID:  {resp['job_id']}")
        print(f"Status:  {resp['status']}")
        print(f"Worker:  {resp.get('assigned_worker_name')}")
        print(f"Elapsed: {resp.get('elapsed_seconds')}s")

if __name__ == "__main__":
    main()

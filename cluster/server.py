import os
import time
import zipfile
import shutil
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Caspian PINN Distributed Job Server")

# Storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "server_storage", "jobs")
RESULTS_DIR = os.path.join(BASE_DIR, "server_storage", "results")
WORKER_STORAGE_DIR = os.path.join(BASE_DIR, "server_storage", "worker")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WORKER_STORAGE_DIR, exist_ok=True)

LATEST_WORKER_PATH = os.path.join(WORKER_STORAGE_DIR, "latest_worker.py")
latest_worker_hash = ""

import hashlib

def compute_file_hash(filepath):
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

if os.path.exists(LATEST_WORKER_PATH):
    latest_worker_hash = compute_file_hash(LATEST_WORKER_PATH)

# In-memory database
workers_db: Dict[str, dict] = {}
jobs_db: Dict[str, dict] = {}

class JobSubmitRequest(BaseModel):
    job_id: str
    command: str
    target_dir: str
    model_name: Optional[str] = "unknown"
    seed: Optional[str] = "none"

@app.post("/api/workers/register")
def register_worker(worker_id: str = Form(...), name: str = Form(...), system_info: str = Form("Windows")):
    workers_db[worker_id] = {
        "worker_id": worker_id,
        "name": name,
        "system_info": system_info,
        "status": "IDLE",
        "current_job": None,
        "last_seen": time.time(),
        "completed_count": workers_db.get(worker_id, {}).get("completed_count", 0)
    }
    return {"status": "registered", "worker_id": worker_id, "name": name}

@app.post("/api/jobs/submit")
async def submit_job(
    job_id: str = Form(...),
    command: str = Form(...),
    target_dir: str = Form(...),
    code_payload: UploadFile = File(...)
):
    payload_path = os.path.join(JOBS_DIR, f"{job_id}_code.zip")
    with open(payload_path, "wb") as f:
        shutil.copyfileobj(code_payload.file, f)

    job_entry = {
        "job_id": job_id,
        "command": command,
        "target_dir": target_dir,
        "status": "QUEUED",  # QUEUED, RUNNING, COMPLETED, FAILED
        "payload_path": payload_path,
        "assigned_worker_id": None,
        "assigned_worker_name": None,
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "elapsed_seconds": None,
        "log_output": "",
        "result_path": None
    }
    jobs_db[job_id] = job_entry
    return {"status": "queued", "job_id": job_id}

@app.post("/api/worker/upload_latest")
async def upload_latest_worker(worker_file: UploadFile = File(...)):
    global latest_worker_hash
    with open(LATEST_WORKER_PATH, "wb") as f:
        shutil.copyfileobj(worker_file.file, f)
    latest_worker_hash = compute_file_hash(LATEST_WORKER_PATH)
    return {"status": "worker_updated", "worker_hash": latest_worker_hash}

@app.get("/api/worker/download_latest")
def download_latest_worker():
    if not os.path.exists(LATEST_WORKER_PATH):
        raise HTTPException(status_code=404, detail="No worker script uploaded yet")
    return FileResponse(LATEST_WORKER_PATH, filename="worker.py")

@app.get("/api/jobs/poll")
def poll_job(worker_id: str, name: str):
    # Update worker heartbeat
    now = time.time()
    if worker_id in workers_db:
        workers_db[worker_id]["last_seen"] = now
        workers_db[worker_id]["name"] = name
    else:
        workers_db[worker_id] = {
            "worker_id": worker_id,
            "name": name,
            "system_info": "Windows",
            "status": "IDLE",
            "current_job": None,
            "last_seen": now,
            "completed_count": 0
        }

    # Find oldest queued job
    for j_id, j_data in jobs_db.items():
        if j_data["status"] == "QUEUED":
            j_data["status"] = "RUNNING"
            j_data["assigned_worker_id"] = worker_id
            j_data["assigned_worker_name"] = name
            j_data["started_at"] = time.time()
            
            workers_db[worker_id]["status"] = "BUSY"
            workers_db[worker_id]["current_job"] = j_id
            
            return {
                "has_job": True,
                "job_id": j_id,
                "command": j_data["command"],
                "target_dir": j_data["target_dir"],
                "latest_worker_hash": latest_worker_hash
            }

    workers_db[worker_id]["status"] = "IDLE"
    workers_db[worker_id]["current_job"] = None
    return {
        "has_job": False,
        "latest_worker_hash": latest_worker_hash
    }

@app.get("/api/jobs/payload/{job_id}")
def download_payload(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    p_path = jobs_db[job_id]["payload_path"]
    if not os.path.exists(p_path):
        raise HTTPException(status_code=404, detail="Payload file missing")
    return FileResponse(p_path, filename=f"{job_id}_code.zip")

@app.post("/api/jobs/complete")
async def complete_job(
    job_id: str = Form(...),
    worker_id: str = Form(...),
    success: bool = Form(...),
    elapsed_seconds: float = Form(...),
    log_output: str = Form(""),
    results_zip: UploadFile = File(None)
):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    job["status"] = "COMPLETED" if success else "FAILED"
    job["completed_at"] = time.time()
    job["elapsed_seconds"] = elapsed_seconds
    job["log_output"] = log_output

    if results_zip and success:
        res_path = os.path.join(RESULTS_DIR, f"{job_id}_results.zip")
        with open(res_path, "wb") as f:
            shutil.copyfileobj(results_zip.file, f)
        job["result_path"] = res_path

    if worker_id in workers_db:
        workers_db[worker_id]["status"] = "IDLE"
        workers_db[worker_id]["current_job"] = None
        if success:
            workers_db[worker_id]["completed_count"] += 1

    return {"status": "recorded", "job_id": job_id}

@app.get("/api/jobs/status/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]

@app.get("/api/jobs/download_result/{job_id}")
def download_result(job_id: str):
    if job_id not in jobs_db or not jobs_db[job_id]["result_path"]:
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(jobs_db[job_id]["result_path"], filename=f"{job_id}_results.zip")

@app.get("/", response_class=HTMLResponse)
def web_dashboard():
    now = time.time()
    
    # Workers HTML table
    workers_rows = ""
    for w_id, w in workers_db.items():
        is_online = (now - w["last_seen"]) < 20
        status_badge = f'<span style="color: green; font-weight: bold;">ONLINE ({w["status"]})</span>' if is_online else '<span style="color: gray;">OFFLINE</span>'
        workers_rows += f"""
        <tr>
            <td><b>{w['name']}</b> ({w_id[:8]})</td>
            <td>{w['system_info']}</td>
            <td>{status_badge}</td>
            <td>{w['current_job'] or '-'}</td>
            <td>{w['completed_count']} jobs</td>
        </tr>
        """
    if not workers_rows:
        workers_rows = "<tr><td colspan='5' style='text-align:center; color: gray;'>No workers registered yet.</td></tr>"

    # Jobs HTML table
    jobs_rows = ""
    for j_id, j in reversed(list(jobs_db.items())):
        status_color = {"QUEUED": "orange", "RUNNING": "blue", "COMPLETED": "green", "FAILED": "red"}.get(j["status"], "black")
        elapsed_str = f"{j['elapsed_seconds']:.1f}s" if j['elapsed_seconds'] else "-"
        worker_name = j['assigned_worker_name'] or "-"
        jobs_rows += f"""
        <tr>
            <td><code>{j_id}</code></td>
            <td><code style="font-size: 11px;">{j['command']}</code></td>
            <td><b style="color: {status_color};">{j['status']}</b></td>
            <td>{worker_name}</td>
            <td>{elapsed_str}</td>
            <td>
                {'<a href="/api/jobs/download_result/' + j_id + '">Download Zip</a>' if j['result_path'] else '-'}
            </td>
        </tr>
        """
    if not jobs_rows:
        jobs_rows = "<tr><td colspan='6' style='text-align:center; color: gray;'>No jobs in queue.</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Caspian PINN Distributed Cluster Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f4f6f9; }}
            h1, h2 {{ color: #1e293b; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
            th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
            tr:hover {{ background: #f1f5f9; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; }}
        </style>
    </head>
    <body>
        <h1>🚀 Caspian PINN Distributed Cluster Server</h1>
        <p>Live Dashboard &bull; Auto-refreshes every 5 seconds</p>

        <div class="card">
            <h2>💻 Active Workers ({len(workers_db)})</h2>
            <table>
                <thead>
                    <tr><th>Worker Name</th><th>System</th><th>Status</th><th>Current Job</th><th>Completed</th></tr>
                </thead>
                <tbody>
                    {workers_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>📋 Training Jobs Queue ({len(jobs_db)})</h2>
            <table>
                <thead>
                    <tr><th>Job ID</th><th>Command</th><th>Status</th><th>Worker</th><th>Elapsed Time</th><th>Artifacts</th></tr>
                </thead>
                <tbody>
                    {jobs_rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

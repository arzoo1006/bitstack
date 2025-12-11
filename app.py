# app.py
"""
BitStack simple UI (uses user's supplied HTML)
- GET /        -> serves the upload UI (the HTML you provided)
- POST /report -> accepts form file 'file', saves to workspace/<run_id>/input.csv,
                  runs report_generator.process_csv, writes log.txt, redirects to report
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil, uuid, traceback, os

# import the processing function from your report_generator.py
from report_generator import process_csv

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE / "workspace"
WORKSPACE.mkdir(exist_ok=True)

app = FastAPI(title="BitStack Minimal UI")
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE)), name="workspace")


# The exact HTML UI you pasted (kept as a plain triple-quoted string, NOT an f-string)
HOME_HTML = """
<html>
  <head>
    <title>BitStack</title>
    <style>
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top left, #111827, #020617);
        color: #e5e7eb;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .wrapper {
        width: 100%;
        max-width: 480px;
        padding: 20px;
      }
      .card {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 16px;
        padding: 28px 24px 24px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.25);
      }
      .logo {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.12);
        color: #5eead4;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .logo-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #22c55e;
      }
      h1 {
        margin: 16px 0 4px;
        font-size: 28px;
        font-weight: 700;
        color: #f9fafb;
      }
      .subtitle {
        margin: 0 0 20px;
        font-size: 14px;
        color: #9ca3af;
      }
      .pill-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 18px;
      }
      .pill {
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: #9ca3af;
      }
      .field-label {
        text-align: left;
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 6px;
      }
      .file-input-wrapper {
        position: relative;
        margin-bottom: 14px;
      }
      input[type="file"] {
        width: 100%;
        padding: 40px 14px 40px;
        border-radius: 14px;
        border: 1px dashed rgba(148, 163, 184, 0.8);
        background: rgba(15, 23, 42, 0.7);
        color: #e5e7eb;
        cursor: pointer;
      }
      input[type="file"]::file-selector-button {
        display: none;
      }
      .file-input-overlay {
        position: absolute;
        inset: 0;
        pointer-events: none;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        color: #9ca3af;
      }
      .file-input-overlay span {
        display: block;
      }
      .btn {
        width: 100%;
        margin-top: 6px;
        padding: 12px;
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #22c55e, #0ea5e9);
        color: #020617;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
        box-shadow: 0 12px 25px rgba(34, 197, 94, 0.35);
      }
      .btn:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
        box-shadow: 0 16px 35px rgba(34, 197, 94, 0.45);
      }
      .footer {
        margin-top: 12px;
        text-align: center;
        font-size: 11px;
        color: #6b7280;
      }
      .footer strong {
        color: #e5e7eb;
      }
    </style>
  </head>
  <body>
    <div class="wrapper">
      <div class="card">
        <div class="logo">
          <div class="logo-dot"></div>
          BITSTACK • DATA LAB
        </div>
        <h1>Upload your dataset</h1>
        <p class="subtitle">BitStack will clean it, analyze it, and generate an interactive HTML report for you.</p>

        <div class="pill-row">
          <div class="pill">Auto-Cleaning</div>
          <div class="pill">EDA & Visuals</div>
          <div class="pill">Download Cleaned CSV</div>
        </div>

        <form action="/report" enctype="multipart/form-data" method="post">
          <div class="field-group">
            <div class="field-label">Select a CSV file</div>
            <div class="file-input-wrapper">
              <input name="file" type="file" accept=".csv" required />
              <div class="file-input-overlay">
                <span><strong>Click to browse</strong> or drag & drop</span>
                <span style="font-size: 11px; margin-top: 4px;">.csv files up to ~10 MB</span>
              </div>
            </div>
          </div>
          <button class="btn" type="submit">Generate Report</button>
        </form>

        <div class="footer">
          Built with <strong>FastAPI</strong>, <strong>pandas</strong> & <strong>BitStack magic</strong>.
        </div>
      </div>
    </div>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HOME_HTML)


@app.post("/report")
async def report(file: UploadFile = File(...)):
    # create run folder
    run_id = uuid.uuid4().hex[:10]
    run_dir = WORKSPACE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.csv"
    log_path = run_dir / "log.txt"

    # save uploaded file
    try:
        with input_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return HTMLResponse(f"<pre>Failed to save uploaded file: {e}</pre>", status_code=500)

    # call processing function (fast defaults)
    try:
        result = process_csv(str(input_path), str(run_dir), explicit_target=None, fast_mode=True, sample_max_rows=2000, reduced_estimators=30)
    except Exception as e:
        tb = traceback.format_exc()
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(tb)
        return HTMLResponse(f"<pre>Processing crashed: {e}\\nSee /workspace/{run_id}/log.txt</pre>", status_code=500)

    # write a brief log file for inspection
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("RESULT:\\n")
        lf.write(str(result))

    # if report created, redirect to it
    report_file = run_dir / "report.html"
    if result.get("report") and report_file.exists():
        return RedirectResponse(url=f"/workspace/{run_id}/report.html", status_code=303)
    else:
        err = result.get("error", "Unknown error")
        return HTMLResponse(f"<pre>Processing failed: {err}\\nSee /workspace/{run_id}/log.txt</pre>", status_code=500)



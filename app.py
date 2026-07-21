"""
AI Face Recognition — Flask web app (browser webcam, deployable on Render).

The browser captures webcam frames with JavaScript (getUserMedia + canvas)
and POSTs them as base64 JPEGs to this backend. The backend runs face
detection/recognition with OpenCV and returns bounding boxes + names,
which the browser draws back over the video feed.

NOTE on Render's free tier: web services there have ephemeral disk, so
known_faces/, data/, and attendance.csv are wiped on every redeploy or
restart. Fine for a demo; for real persistent attendance tracking you'd
need a paid Render plan with a persistent disk, or an external store
(e.g. S3 for face crops, a real database for attendance rows).
"""

import os
import csv
import base64
import uuid
import datetime

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, session

from face_utils import FaceDatabase, ConsistencyTracker

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me-in-production")

ATTENDANCE_FILE = "attendance.csv"
ENROLL_TARGET_SAMPLES = 10

db = FaceDatabase()

# Simple in-memory per-browser-session state (enrollment progress,
# consistency tracker, who's already been marked present today).
# Lives only as long as the process — resets on restart, same caveat
# as the file storage above.
SESSIONS = {}


def get_state():
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "enroll_name": None,
            "enroll_frames": [],
            "tracker": ConsistencyTracker(),
            "marked_today": _load_marked_today(),
        }
    return SESSIONS[sid]


def _ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])


def _load_marked_today():
    marked = set()
    today = datetime.date.today().isoformat()
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            for row in csv.DictReader(f):
                if row.get("Date") == today:
                    marked.add(row["Name"])
    return marked


def _mark_attendance(name, state):
    now = datetime.datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, now.date().isoformat(), now.strftime("%H:%M:%S")])
    state["marked_today"].add(name)
    return now.strftime("%H:%M:%S")


def _decode_frame(data_url):
    """data_url looks like 'data:image/jpeg;base64,/9j/4AAQ...'"""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    _ensure_attendance_file()
    return render_template("index.html")


@app.route("/api/people_count")
def people_count():
    return jsonify({"count": len(db.all_names()), "names": db.all_names()})


@app.route("/api/enroll/start", methods=["POST"])
def enroll_start():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400
    state = get_state()
    state["enroll_name"] = name
    state["enroll_frames"] = []
    return jsonify({"ok": True, "target": ENROLL_TARGET_SAMPLES})


@app.route("/api/enroll/frame", methods=["POST"])
def enroll_frame():
    state = get_state()
    if not state.get("enroll_name"):
        return jsonify({"error": "Enrollment not started."}), 400

    data = request.get_json(force=True)
    frame = _decode_frame(data["image"])

    if len(state["enroll_frames"]) < ENROLL_TARGET_SAMPLES:
        state["enroll_frames"].append(frame)

    count = len(state["enroll_frames"])
    done = count >= ENROLL_TARGET_SAMPLES
    result = {"count": count, "target": ENROLL_TARGET_SAMPLES, "done": False}

    if done:
        name = state["enroll_name"]
        ok = db.enroll(name, state["enroll_frames"])
        state["enroll_name"] = None
        state["enroll_frames"] = []
        result.update({"done": True, "success": ok, "name": name})

    return jsonify(result)


@app.route("/api/enroll/cancel", methods=["POST"])
def enroll_cancel():
    state = get_state()
    state["enroll_name"] = None
    state["enroll_frames"] = []
    return jsonify({"ok": True})


@app.route("/api/attendance/frame", methods=["POST"])
def attendance_frame():
    state = get_state()
    data = request.get_json(force=True)
    frame = _decode_frame(data["image"])
    h, w = frame.shape[:2]

    results = db.recognize(frame)
    faces = []
    newly_marked = []

    for r in results:
        top, right, bottom, left = r["box"]
        name = r["name"]
        status = "unknown"

        if name != "Unknown":
            state["tracker"].update(name)
            confirmed = state["tracker"].is_confirmed(name)
            status = "confirmed" if confirmed else "checking"
            if confirmed and name not in state["marked_today"]:
                marked_time = _mark_attendance(name, state)
                newly_marked.append({"name": name, "time": marked_time})

        faces.append({
            "name": name,
            "status": status,
            "box": {"top": top, "right": right, "bottom": bottom, "left": left},
        })

    return jsonify({"faces": faces, "width": w, "height": h, "newly_marked": newly_marked})


@app.route("/api/attendance/reset_tracker", methods=["POST"])
def reset_tracker():
    state = get_state()
    state["tracker"].reset()
    return jsonify({"ok": True})


@app.route("/api/attendance/log")
def attendance_log():
    _ensure_attendance_file()
    rows = []
    with open(ATTENDANCE_FILE, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.reverse()  # most recent first
    return jsonify({"rows": rows})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)




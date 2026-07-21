# AI Face Recognition — Web App (Flask, deployable on Render)

Browser-based version of the desktop app: your webcam is accessed via
JavaScript (`getUserMedia`), frames are sent to a Flask backend for
detection/recognition (OpenCV Haar Cascade + LBPH — same engine as the
desktop version, no dlib/cmake needed), and results are drawn back over
the video as bounding boxes.

## Project structure
```
face_recognition_web/
├── app.py                          # Flask backend + API routes
├── face_utils.py                    # face database, LBPH recognition, consistency tracker
├── haarcascade_frontalface_default.xml   # bundled, no download needed at deploy time
├── templates/index.html             # page shell
├── static/js/app.js                  # webcam capture, canvas overlay, API calls
├── static/css/style.css               # dark theme UI
├── requirements.txt
├── Procfile                          # tells Render how to run the app
├── known_faces/<name>/                 # saved face crops per enrolled person (auto-created)
├── data/                                # trained LBPH model + label map (auto-created)
└── attendance.csv                       # attendance log (auto-created)
```

## Run it locally

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser. It'll ask for camera
permission — allow it. (Chrome/Edge/Firefox all support this; Safari
works too but sometimes needs "Allow" clicked twice.)

## How to use it

1. **Enroll Face** — type a name, then look at the camera. It grabs 10
   samples over a few seconds.
2. **Start Attendance** — recognized faces show a name in amber
   ("checking...") until matched on several consecutive frames, then
   turn green ("confirmed") and get logged automatically (once per
   person per day).
3. **Attendance Log** — view everyone marked present.
4. **Stop Camera** — releases the webcam at any time.

## Deploying to Render

1. Push this folder to a GitHub repo (root of the repo should contain
   `app.py`, `requirements.txt`, `Procfile`, etc. directly — not nested
   in a subfolder, unless you set Render's "Root Directory" to match).
2. On [render.com](https://render.com): **New +** → **Web Service** →
   connect your GitHub repo.
3. Settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** leave blank (Render reads the `Procfile`) or
     set it explicitly to:
     `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Instance type:** Free tier works for a demo.
4. Deploy. Render gives you a public `https://your-app.onrender.com`
   URL — camera access requires HTTPS, which Render provides
   automatically, so this works out of the box (unlike plain HTTP).

### Important limitations on Render's free tier

- **Ephemeral disk.** `known_faces/`, `data/`, and `attendance.csv` are
  wiped on every redeploy, and on the free tier the service also spins
  down after inactivity, which can lose the same data. Fine for a demo
  or portfolio piece — **not** suitable for real attendance tracking
  unless you upgrade to a paid plan with a persistent disk, or swap the
  storage layer for something external (e.g. face crops in S3,
  attendance rows in a real database like Postgres).
- **One worker only.** The `Procfile` uses `--workers 1` on purpose —
  the face database and per-session state live in server memory, so
  multiple worker processes would each have their own inconsistent
  copy. This keeps things simple but means the app won't scale past
  one process; fine for single-user/demo use, not for concurrent heavy
  traffic.
- **Free-tier cold starts.** After 15 minutes of inactivity the free
  instance sleeps; the next request can take ~30-60 seconds to wake it
  up.
- **No real liveness/anti-spoofing.** Same caveat as the desktop
  version — LBPH gives no landmark data for blink detection, so this
  relies only on a multi-frame "consistency check," not true
  anti-spoofing. A printed photo or phone screen can still pass it.

## Tuning

In `face_utils.py`:
- `confidence_threshold` in `FaceDatabase.recognize()` (default `75`) —
  LBPH's "confidence" is actually a distance, so **lower = stricter**.
- `ConsistencyTracker.REQUIRED_STREAK` (default `8`) — consecutive
  matching frames needed before attendance is marked.

In `static/js/app.js`:
- `CAPTURE_INTERVAL_MS` (default `700`) — how often a frame is sent to
  the backend. Lower = more responsive but more server load/bandwidth.

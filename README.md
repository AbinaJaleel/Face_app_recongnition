# AI Face Recognition — Desktop App (VS Code, dlib-free version)

This version avoids `dlib`/`face_recognition` entirely, using only
OpenCV's built-in tools:
- **Detection:** Haar Cascade (`haarcascade_frontalface_default.xml`,
  ships inside OpenCV).
- **Recognition:** LBPH (Local Binary Patterns Histograms), via
  `cv2.face.LBPHFaceRecognizer_create()` — part of `opencv-contrib-python`.

Both come as **prebuilt wheels** — nothing to compile, no CMake, no
Visual Studio Build Tools. This is the trade-off for that simplicity:

| | dlib/face_recognition version | This version |
|---|---|---|
| Install difficulty | Hard (needs a C++ compiler) | Easy (`pip install` just works) |
| Recognition accuracy | High (deep-learning embeddings) | Lower (classic algorithm) |
| Liveness / anti-spoofing | Blink detection via landmarks | **None** — a printed photo can fool it |
| What replaces liveness | — | A "consistency check": needs several consecutive matching frames before marking attendance (filters flicker/misreads, not spoofing) |

## Project structure
```
face_recognition_app/
├── main.py            # Tkinter UI + app logic
├── face_utils.py       # Face database (Haar + LBPH), consistency tracker
├── requirements.txt
├── known_faces/<name>/  # saved face crops per enrolled person (auto-created)
├── data/                 # trained LBPH model + label map (auto-created)
└── attendance.csv        # attendance log (auto-created)
```

## Setup in VS Code

1. Open this folder in VS Code (`File > Open Folder...`).
2. Create and activate a virtual environment (any recent Python 3.9–3.13
   works fine here — no need to downgrade for dlib compatibility):
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Select that interpreter in VS Code (bottom-right corner, or
   `Ctrl+Shift+P` → "Python: Select Interpreter").
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   This should just work — `opencv-contrib-python` ships prebuilt binaries
   for all common platforms.
5. Run it:
   ```
   python main.py
   ```

## How to use it

1. Click **Enroll New Face**, type a name, and look at the camera — it
   grabs 10 samples automatically. Enroll in decent, even lighting; LBPH
   is more lighting-sensitive than the deep-learning version was.
2. Click **Start Attendance**. Recognized faces show a name in orange
   ("checking...") until they've been matched on several consecutive
   frames, then turn green ("confirmed") and get logged automatically
   (once per person per day).
3. Click **View Attendance Log** to see everyone marked present.
4. Click **Stop Camera** any time to release the webcam.

## Tuning

In `face_utils.py`:
- `confidence_threshold` in `FaceDatabase.recognize()` (default `75`) —
  LBPH's "confidence" is actually a distance, so **lower = stricter**.
  Raise it if real people are being marked Unknown; lower it if strangers
  are being matched to enrolled names.
- `ConsistencyTracker.REQUIRED_STREAK` (default `8`) — how many
  consecutive matching frames are needed before attendance is marked.
- If recognition is unreliable, enroll each person from a few different
  angles/distances (run Enroll New Face more than once for the same name)
  to give LBPH more data to work with.

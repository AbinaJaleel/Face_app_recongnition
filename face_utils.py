"""
Core face-recognition logic — dlib-free, GUI-free (safe for a server).

Uses OpenCV's built-in Haar Cascade for face detection and its built-in
LBPH (Local Binary Patterns Histograms) recognizer for identification.
Both ship inside opencv-contrib-python-headless as prebuilt wheels — no
dlib, no cmake, no compiler needed, no GUI dependencies (fine for Render).

Trade-off vs. a dlib/face_recognition version: LBPH is a lighter, older
algorithm (less accurate than deep-learning face embeddings), and there's
no facial-landmark data to run a blink/liveness check on. A printed photo
held up to the camera CAN fool this version — see ConsistencyTracker below
for what it does instead.
"""

import os
import glob
import pickle
import numpy as np
import cv2

DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "lbph_model.yml")
LABELS_FILE = os.path.join(DATA_DIR, "labels.pkl")
KNOWN_FACES_DIR = "known_faces"
FACE_SIZE = (200, 200)

# Shipped as a local file in the repo (no download-on-first-run needed,
# and no reliance on an internet fetch at deploy time on Render).
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def _cascade_path():
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CASCADE_FILENAME)
    if os.path.exists(local_path):
        return local_path
    # fall back to OpenCV's bundled copy if the local file is ever missing
    bundled_path = os.path.join(cv2.data.haarcascades, CASCADE_FILENAME)
    if os.path.exists(bundled_path):
        return bundled_path
    raise FileNotFoundError(
        f"Could not find {CASCADE_FILENAME} locally or in cv2's bundled data."
    )


CASCADE_PATH = _cascade_path()


class FaceDatabase:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)
        if self.detector.empty():
            raise RuntimeError(
                f"Failed to load face-detection cascade from {CASCADE_PATH}."
            )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_name = {}   # int label -> name
        self.name_to_label = {}   # name -> int label
        self._load()

    def _load(self):
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "rb") as f:
                self.label_to_name = pickle.load(f)
            self.name_to_label = {v: k for k, v in self.label_to_name.items()}
        if os.path.exists(MODEL_FILE):
            self.recognizer.read(MODEL_FILE)

    def detect_faces(self, gray_frame):
        return self.detector.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

    def enroll(self, name, frames):
        """frames: list of BGR numpy images containing the person's face."""
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        existing = len(glob.glob(os.path.join(person_dir, "*.jpg")))

        saved = 0
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            saved += 1
            cv2.imwrite(os.path.join(person_dir, f"{existing + saved}.jpg"), crop)

        if saved == 0:
            return False

        self._retrain()
        return True

    def _retrain(self):
        """LBPH has no easy 'add one person' update in cv2's Python API, so we
        just retrain on everything in known_faces/ — fast enough at this scale."""
        images, labels = [], []
        self.label_to_name = {}
        self.name_to_label = {}

        people = sorted(
            p for p in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, p))
        )
        for label, name in enumerate(people):
            files = glob.glob(os.path.join(KNOWN_FACES_DIR, name, "*.jpg"))
            if not files:
                continue
            self.label_to_name[label] = name
            self.name_to_label[name] = label
            for fpath in files:
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)

        if not images:
            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(images, np.array(labels))
        self.recognizer.save(MODEL_FILE)
        with open(LABELS_FILE, "wb") as f:
            pickle.dump(self.label_to_name, f)

    def all_names(self):
        return list(self.name_to_label.keys())

    def recognize(self, frame, confidence_threshold=75):
        """LBPH's 'confidence' is really a distance — LOWER means a better
        match. Anything above the threshold is treated as Unknown."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        results = []
        for (x, y, w, h) in faces:
            crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            name = "Unknown"
            conf = None
            if self.label_to_name:
                try:
                    label, conf = self.recognizer.predict(crop)
                    if conf <= confidence_threshold:
                        name = self.label_to_name.get(label, "Unknown")
                except cv2.error:
                    pass
            # box kept in (top, right, bottom, left) form
            results.append({"name": name, "box": (int(y), int(x + w), int(y + h), int(x)), "confidence": conf})
        return results


class ConsistencyTracker:
    """NOT a liveness/anti-spoofing check — LBPH gives us no landmark data to
    detect a blink with. This just requires a name to be recognized on several
    consecutive frames in a row before it's treated as 'confirmed', which
    filters out one-off misreads/flicker before attendance is marked. A
    printed photo held up to the camera can still pass this."""

    REQUIRED_STREAK = 8

    def __init__(self):
        self.streaks = {}
        self.confirmed = set()

    def update(self, name):
        for other in list(self.streaks.keys()):
            if other != name:
                self.streaks[other] = 0
        self.streaks[name] = self.streaks.get(name, 0) + 1
        if self.streaks[name] >= self.REQUIRED_STREAK:
            self.confirmed.add(name)
        return self.confirmed

    def is_confirmed(self, name):
        return name in self.confirmed

    def reset(self):
        self.streaks.clear()
        self.confirmed.clear()

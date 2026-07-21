"""
AI Face Recognition — desktop app (Tkinter UI), dlib-free version.

Uses OpenCV's built-in Haar Cascade + LBPH recognizer (no dlib/cmake
required). Additions over the original single-face notebook script:

  1. Multi-person database — enroll any number of people; face crops and
     the trained model persist under known_faces/ and data/.
  2. Attendance log (attendance.csv) with a consistency check that requires
     several consecutive matching frames before marking someone present —
     NOT a real liveness/anti-spoofing check (see face_utils.py), just
     protection against one-off misreads.

Run with:  python main.py
"""

import os
import csv
import datetime
import tkinter as tk
from tkinter import simpledialog, ttk
import cv2
from PIL import Image, ImageTk

from face_utils import FaceDatabase, ConsistencyTracker

ATTENDANCE_FILE = "attendance.csv"

# ---------------------------------------------------------------- palette --
BG = "#0f1117"
PANEL = "#171a23"
PANEL_ALT = "#1d2130"
BORDER = "#262b3d"
TEXT = "#e9ebf3"
SUBTEXT = "#8a90a6"
ACCENT = "#7c5cff"
ACCENT_HOVER = "#8f72ff"
ACCENT_DIM = "#2a2440"
SUCCESS = "#3ddc97"
WARNING = "#ffb648"
DANGER = "#ff5c7a"
VIDEO_BG = "#05060a"

FONT_TITLE = ("Segoe UI Semibold", 19)
FONT_SUB = ("Segoe UI", 10)
FONT_NAV = ("Segoe UI Semibold", 11)
FONT_STATUS = ("Segoe UI", 10)
FONT_BADGE = ("Segoe UI Semibold", 10)


class NavButton(tk.Frame):
    """A sidebar button with an icon glyph, label, hover state, and an
    active accent bar — styled by hand since ttk theming is limited."""

    def __init__(self, parent, icon, text, command, **kwargs):
        super().__init__(parent, bg=PANEL, **kwargs)
        self.command = command
        self.active = False

        self.bar = tk.Frame(self, bg=PANEL, width=3)
        self.bar.pack(side="left", fill="y")

        inner = tk.Frame(self, bg=PANEL)
        inner.pack(side="left", fill="both", expand=True, padx=(14, 10), pady=12)

        self.icon_lbl = tk.Label(inner, text=icon, font=("Segoe UI", 14),
                                  bg=PANEL, fg=SUBTEXT)
        self.icon_lbl.pack(side="left")

        self.text_lbl = tk.Label(inner, text=text, font=FONT_NAV,
                                  bg=PANEL, fg=TEXT, anchor="w")
        self.text_lbl.pack(side="left", padx=(10, 0), fill="x", expand=True)

        for widget in (self, inner, self.icon_lbl, self.text_lbl):
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
            widget.bind("<Button-1>", lambda e: self.command())

    def _on_enter(self, _e=None):
        if not self.active:
            self._paint(PANEL_ALT)

    def _on_leave(self, _e=None):
        if not self.active:
            self._paint(PANEL)

    def _paint(self, bg):
        self.configure(bg=bg)
        for w in self.winfo_children():
            w.configure(bg=bg)
            for c in w.winfo_children():
                c.configure(bg=bg)

    def set_active(self, active):
        self.active = active
        if active:
            self._paint(ACCENT_DIM)
            self.bar.configure(bg=ACCENT)
            self.icon_lbl.configure(fg=ACCENT)
            self.text_lbl.configure(fg=TEXT)
        else:
            self._paint(PANEL)
            self.bar.configure(bg=PANEL)
            self.icon_lbl.configure(fg=SUBTEXT)


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Recognition")
        self.root.geometry("1040x680")
        self.root.configure(bg=BG)
        self.root.minsize(860, 600)

        self.db = FaceDatabase()
        self.consistency_tracker = ConsistencyTracker()
        self.cap = None
        self.mode = None  # 'enroll' or 'attendance'
        self.enroll_name = None
        self.enroll_frames = []
        self._enroll_tick = 0
        self.marked_today = self._load_marked_today()

        self._build_ui()
        self._ensure_attendance_file()
        self._refresh_people_count()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        root_frame = tk.Frame(self.root, bg=BG)
        root_frame.pack(fill="both", expand=True)

        # ---- sidebar ----
        sidebar = tk.Frame(root_frame, bg=PANEL, width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand = tk.Frame(sidebar, bg=PANEL)
        brand.pack(fill="x", pady=(28, 24), padx=20)
        tk.Label(brand, text="◆", font=("Segoe UI", 18), bg=PANEL, fg=ACCENT).pack(side="left")
        tk.Label(brand, text=" FaceRec", font=("Segoe UI Semibold", 16),
                 bg=PANEL, fg=TEXT).pack(side="left")

        nav_frame = tk.Frame(sidebar, bg=PANEL)
        nav_frame.pack(fill="x", pady=(0, 10))

        self.nav_enroll = NavButton(nav_frame, "＋", "Enroll Face", self.start_enroll)
        self.nav_enroll.pack(fill="x")
        self.nav_attendance = NavButton(nav_frame, "▶", "Start Attendance", self.start_attendance)
        self.nav_attendance.pack(fill="x")
        self.nav_log = NavButton(nav_frame, "▤", "Attendance Log", self.view_log)
        self.nav_log.pack(fill="x")
        self.nav_stop = NavButton(nav_frame, "■", "Stop Camera", self.stop_camera)
        self.nav_stop.pack(fill="x")

        # ---- sidebar footer stats ----
        footer = tk.Frame(sidebar, bg=PANEL)
        footer.pack(side="bottom", fill="x", padx=20, pady=22)
        tk.Frame(footer, bg=BORDER, height=1).pack(fill="x", pady=(0, 14))
        tk.Label(footer, text="ENROLLED PEOPLE", font=("Segoe UI", 8, "bold"),
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w")
        self.people_count_var = tk.StringVar(value="0")
        tk.Label(footer, textvariable=self.people_count_var, font=("Segoe UI Semibold", 20),
                 bg=PANEL, fg=TEXT).pack(anchor="w")

        # ---- main content ----
        main = tk.Frame(root_frame, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=BG)
        header.pack(fill="x", padx=28, pady=(26, 14))
        tk.Label(header, text="Live Camera", font=FONT_TITLE, bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(header, text="Enroll faces, then start attendance to recognize them.",
                 font=FONT_SUB, bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(2, 0))

        # ---- video card ----
        video_card = tk.Frame(main, bg=PANEL, highlightbackground=BORDER,
                               highlightthickness=1)
        video_card.pack(fill="both", expand=True, padx=28, pady=(0, 14))

        self.video_label = tk.Label(video_card, bg=VIDEO_BG, text="Camera is off",
                                     font=("Segoe UI", 12), fg=SUBTEXT)
        self.video_label.pack(fill="both", expand=True, padx=14, pady=14)

        # ---- status bar ----
        status_bar = tk.Frame(main, bg=PANEL, highlightbackground=BORDER,
                               highlightthickness=1)
        status_bar.pack(fill="x", padx=28, pady=(0, 26))

        status_inner = tk.Frame(status_bar, bg=PANEL)
        status_inner.pack(fill="x", padx=16, pady=12)

        self.status_dot = tk.Label(status_inner, text="●", font=("Segoe UI", 10),
                                    bg=PANEL, fg=SUBTEXT)
        self.status_dot.pack(side="left")

        self.status_var = tk.StringVar(value="Idle. Choose an action from the sidebar.")
        tk.Label(status_inner, textvariable=self.status_var, font=FONT_STATUS,
                 bg=PANEL, fg=TEXT).pack(side="left", padx=(8, 0))

    def _set_status(self, message, kind="idle"):
        colors = {"idle": SUBTEXT, "info": ACCENT, "success": SUCCESS,
                  "warning": WARNING, "error": DANGER}
        self.status_var.set(message)
        self.status_dot.configure(fg=colors.get(kind, SUBTEXT))

    def _set_active_nav(self, name):
        for nav, key in ((self.nav_enroll, "enroll"), (self.nav_attendance, "attendance")):
            nav.set_active(key == name)

    def _refresh_people_count(self):
        self.people_count_var.set(str(len(self.db.all_names())))

    # ---------------------------------------------------- camera plumbing --
    def _open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self._loop()

    def stop_camera(self):
        self.mode = None
        self._set_active_nav(None)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="", text="Camera is off", bg=VIDEO_BG)
        self._set_status("Camera stopped.", "idle")

    def _loop(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            if self.mode == "enroll":
                self._handle_enroll_frame(frame)
            elif self.mode == "attendance":
                self._handle_attendance_frame(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")

        self.root.after(15, self._loop)

    # ------------------------------------------------------- enroll flow --
    def start_enroll(self):
        name = simpledialog.askstring("Enroll", "Enter the person's name:")
        if not name:
            return
        self.enroll_name = name.strip()
        self.enroll_frames = []
        self._enroll_tick = 0
        self.mode = "enroll"
        self._set_active_nav("enroll")
        self._set_status(f"Enrolling '{self.enroll_name}'... hold still, capturing 10 samples.", "info")
        self._open_camera()

    def _handle_enroll_frame(self, frame):
        self._enroll_tick += 1
        target_samples = 10

        cv2.putText(frame, f"Enrolling: {self.enroll_name} ({len(self.enroll_frames)}/{target_samples})",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 92, 124), 2)

        if self._enroll_tick % 10 == 0 and len(self.enroll_frames) < target_samples:
            self.enroll_frames.append(frame.copy())

        if len(self.enroll_frames) >= target_samples:
            ok = self.db.enroll(self.enroll_name, self.enroll_frames)
            self.mode = None
            self._enroll_tick = 0
            if ok:
                self._set_status(f"Enrolled '{self.enroll_name}' successfully.", "success")
                self._refresh_people_count()
            else:
                self._set_status("Enrollment failed — no face detected. Try again with better lighting.", "error")
            self.stop_camera()

    # --------------------------------------------------- attendance flow --
    def start_attendance(self):
        self.mode = "attendance"
        self._set_active_nav("attendance")
        self.consistency_tracker.reset()
        self._set_status("Attendance mode: recognizing faces...", "info")
        self._open_camera()

    def _handle_attendance_frame(self, frame):
        results = self.db.recognize(frame)
        for r in results:
            top, right, bottom, left = r["box"]
            name = r["name"]
            confirmed = False

            if name != "Unknown":
                self.consistency_tracker.update(name)
                confirmed = self.consistency_tracker.is_confirmed(name)

            if name == "Unknown":
                color = (122, 92, 255)  # BGR for the accent purple, roughly
                label = "Unknown"
            elif confirmed:
                color = (151, 220, 61)  # success green (BGR)
                label = f"{name} (confirmed)"
                if name not in self.marked_today:
                    self._mark_attendance(name)
            else:
                color = (72, 182, 255)  # warning amber (BGR)
                label = f"{name} (checking...)"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------------------------------------------------- attendance log --
    def _ensure_attendance_file(self):
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Name", "Date", "Time"])

    def _load_marked_today(self):
        marked = set()
        today = datetime.date.today().isoformat()
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "r") as f:
                for row in csv.DictReader(f):
                    if row.get("Date") == today:
                        marked.add(row["Name"])
        return marked

    def _mark_attendance(self, name):
        now = datetime.datetime.now()
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, now.date().isoformat(), now.strftime("%H:%M:%S")])
        self.marked_today.add(name)
        self._set_status(f"Attendance marked for {name} at {now.strftime('%H:%M:%S')}.", "success")

    def view_log(self):
        win = tk.Toplevel(self.root)
        win.title("Attendance Log")
        win.geometry("460x460")
        win.configure(bg=BG)

        header = tk.Frame(win, bg=BG)
        header.pack(fill="x", padx=20, pady=(20, 10))
        tk.Label(header, text="Attendance Log", font=("Segoe UI Semibold", 15),
                 bg=BG, fg=TEXT).pack(anchor="w")

        table_card = tk.Frame(win, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        table_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        style = ttk.Style(win)
        style.theme_use("default")
        style.configure("Log.Treeview",
                         background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                         rowheight=26, borderwidth=0, font=FONT_STATUS)
        style.configure("Log.Treeview.Heading",
                         background=PANEL_ALT, foreground=SUBTEXT,
                         font=("Segoe UI Semibold", 9), borderwidth=0)
        style.map("Log.Treeview", background=[("selected", ACCENT_DIM)],
                  foreground=[("selected", TEXT)])

        tree = ttk.Treeview(table_card, columns=("Name", "Date", "Time"),
                             show="headings", style="Log.Treeview")
        for col, w in (("Name", 160), ("Date", 140), ("Time", 120)):
            tree.heading(col, text=col.upper())
            tree.column(col, width=w, anchor="w")
        tree.pack(fill="both", expand=True, padx=12, pady=12)

        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "r") as f:
                for row in csv.DictReader(f):
                    tree.insert("", "end", values=(row["Name"], row["Date"], row["Time"]))


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()

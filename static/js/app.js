const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const videoWrap = document.getElementById("videoWrap");
const videoOff = document.getElementById("videoOff");

const btnEnroll = document.getElementById("btnEnroll");
const btnAttendance = document.getElementById("btnAttendance");
const btnLog = document.getElementById("btnLog");
const btnStop = document.getElementById("btnStop");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const peopleCount = document.getElementById("peopleCount");

const enrollModal = document.getElementById("enrollModal");
const enrollNameInput = document.getElementById("enrollNameInput");
const enrollConfirmBtn = document.getElementById("enrollConfirmBtn");
const enrollCancelBtn = document.getElementById("enrollCancelBtn");

const logModal = document.getElementById("logModal");
const logCloseBtn = document.getElementById("logCloseBtn");
const logTableBody = document.getElementById("logTableBody");
const logEmpty = document.getElementById("logEmpty");

let stream = null;
let mode = null; // 'enroll' | 'attendance' | null
let captureTimer = null;
const CAPTURE_INTERVAL_MS = 700; // how often a frame is sent to the backend

// ---------------------------------------------------------------- helpers --
function setStatus(message, kind = "idle") {
  statusText.textContent = message;
  statusDot.className = "status-dot " + kind;
}

function setActiveNav(name) {
  btnEnroll.classList.toggle("active", name === "enroll");
  btnAttendance.classList.toggle("active", name === "attendance");
}

function refreshPeopleCount() {
  fetch("/api/people_count")
    .then((r) => r.json())
    .then((d) => { peopleCount.textContent = d.count; })
    .catch(() => {});
}

function captureFrameDataURL() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.75);
}

function resizeOverlay() {
  overlay.width = videoWrap.clientWidth;
  overlay.height = videoWrap.clientHeight;
}
window.addEventListener("resize", resizeOverlay);

// ------------------------------------------------------------------ camera --
async function openCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    video.classList.remove("hidden");
    videoOff.style.display = "none";
    resizeOverlay();
  } catch (err) {
    setStatus("Could not access the camera: " + err.message, "error");
    throw err;
  }
}

function stopCamera() {
  mode = null;
  setActiveNav(null);
  clearTimeout(captureTimer);
  captureTimer = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.classList.add("hidden");
  videoOff.style.display = "block";
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  setStatus("Camera stopped.", "idle");
}

btnStop.addEventListener("click", stopCamera);

// ------------------------------------------------------------- enroll flow --
btnEnroll.addEventListener("click", () => {
  enrollNameInput.value = "";
  enrollModal.classList.add("open");
  enrollNameInput.focus();
});

enrollCancelBtn.addEventListener("click", () => {
  enrollModal.classList.remove("open");
});

enrollConfirmBtn.addEventListener("click", async () => {
  const name = enrollNameInput.value.trim();
  if (!name) {
    enrollNameInput.focus();
    return;
  }
  enrollModal.classList.remove("open");

  const res = await fetch("/api/enroll/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await res.json();
  if (data.error) {
    setStatus(data.error, "error");
    return;
  }

  mode = "enroll";
  setActiveNav("enroll");
  setStatus(`Enrolling '${name}'... hold still, capturing ${data.target} samples.`, "info");

  try {
    await openCamera();
  } catch {
    mode = null;
    return;
  }
  enrollLoop(name);
});

function enrollLoop(name) {
  if (mode !== "enroll") return;
  const imageData = captureFrameDataURL();

  fetch("/api/enroll/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        setStatus(data.error, "error");
        mode = null;
        return;
      }
      if (!data.done) {
        setStatus(`Enrolling '${name}'... (${data.count}/${data.target})`, "info");
        captureTimer = setTimeout(() => enrollLoop(name), CAPTURE_INTERVAL_MS);
      } else {
        if (data.success) {
          setStatus(`Enrolled '${data.name}' successfully.`, "success");
          refreshPeopleCount();
        } else {
          setStatus("Enrollment failed — no face detected. Try again with better lighting.", "error");
        }
        stopCamera();
      }
    })
    .catch(() => {
      setStatus("Lost connection while enrolling. Try again.", "error");
      stopCamera();
    });
}

// -------------------------------------------------------- attendance flow --
btnAttendance.addEventListener("click", async () => {
  mode = "attendance";
  setActiveNav("attendance");
  setStatus("Attendance mode: recognizing faces...", "info");

  await fetch("/api/attendance/reset_tracker", { method: "POST" });

  try {
    await openCamera();
  } catch {
    mode = null;
    return;
  }
  attendanceLoop();
});

function attendanceLoop() {
  if (mode !== "attendance") return;
  const imageData = captureFrameDataURL();

  fetch("/api/attendance/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData }),
  })
    .then((r) => r.json())
    .then((data) => {
      drawFaces(data);
      if (data.newly_marked && data.newly_marked.length) {
        const m = data.newly_marked[0];
        setStatus(`Attendance marked for ${m.name} at ${m.time}.`, "success");
      }
      captureTimer = setTimeout(attendanceLoop, CAPTURE_INTERVAL_MS);
    })
    .catch(() => {
      captureTimer = setTimeout(attendanceLoop, CAPTURE_INTERVAL_MS);
    });
}

function drawFaces(data) {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!data.faces || !data.faces.length || !data.width) return;

  // scale factor between the source frame (data.width/height) and the
  // displayed, object-fit:contain video element
  const scale = Math.min(overlay.width / data.width, overlay.height / data.height);
  const offsetX = (overlay.width - data.width * scale) / 2;
  const offsetY = (overlay.height - data.height * scale) / 2;

  data.faces.forEach((f) => {
    const { top, right, bottom, left } = f.box;
    const x = left * scale + offsetX;
    const y = top * scale + offsetY;
    const w = (right - left) * scale;
    const h = (bottom - top) * scale;

    let color = "#7c5cff", label = "Unknown";
    if (f.status === "confirmed") {
      color = "#3ddc97";
      label = `${f.name} (confirmed)`;
    } else if (f.status === "checking") {
      color = "#ffb648";
      label = `${f.name} (checking...)`;
    }

    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(x, y, w, h);

    overlayCtx.font = "600 13px Segoe UI, sans-serif";
    const textWidth = overlayCtx.measureText(label).width;
    overlayCtx.fillStyle = "rgba(5,6,10,0.75)";
    overlayCtx.fillRect(x, y - 22, textWidth + 10, 20);
    overlayCtx.fillStyle = color;
    overlayCtx.fillText(label, x + 5, y - 7);
  });
}

// -------------------------------------------------------------------- log --
btnLog.addEventListener("click", () => {
  fetch("/api/attendance/log")
    .then((r) => r.json())
    .then((data) => {
      logTableBody.innerHTML = "";
      if (!data.rows || !data.rows.length) {
        logEmpty.style.display = "block";
      } else {
        logEmpty.style.display = "none";
        data.rows.forEach((row) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${row.Name}</td><td>${row.Date}</td><td>${row.Time}</td>`;
          logTableBody.appendChild(tr);
        });
      }
      logModal.classList.add("open");
    });
});

logCloseBtn.addEventListener("click", () => logModal.classList.remove("open"));

// -------------------------------------------------------------------- init --
refreshPeopleCount();

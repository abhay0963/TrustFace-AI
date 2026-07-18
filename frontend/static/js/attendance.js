const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const captureCanvas = document.getElementById("capture-canvas");
const camStatusText = document.getElementById("cam-status-text");
const camDot = document.getElementById("cam-dot");
const btnToggle = document.getElementById("btn-toggle");
const lastRunText = document.getElementById("last-run-text");

const CAPTURE_INTERVAL_MS = 1500;
let running = true;
let busy = false;
let intervalHandle = null;

const decisionMeta = {
    auto_accept: { cls: "badge-accept", label: "AUTO ACCEPT", color: "#2dd4bf" },
    retry: { cls: "badge-retry", label: "RETRY — ask for another frame", color: "#f5a623" },
    reject: { cls: "badge-reject", label: "REJECT", color: "#ef4d5f" },
    unknown: { cls: "badge-unknown", label: "UNKNOWN FACE", color: "#ef4d5f" },
};

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        camStatusText.textContent = "Camera live";
        video.addEventListener("loadedmetadata", () => {
            overlay.width = video.videoWidth;
            overlay.height = video.videoHeight;
        });
    } catch (err) {
        camStatusText.textContent = "Camera access denied";
        camDot.classList.remove("live");
    }
}

function drawBoxes(results) {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    for (const r of results) {
        const [x1, y1, x2, y2] = r.bbox;
        const meta = decisionMeta[r.decision] || decisionMeta.unknown;
        ctx.strokeStyle = meta.color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.font = "600 13px IBM Plex Mono, monospace";
        const label = `${r.name} · ${r.trust_score.toFixed(0)}`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = meta.color;
        ctx.fillRect(x1, y1 - 22, textWidth + 12, 22);
        ctx.fillStyle = "#04211d";
        ctx.fillText(label, x1 + 6, y1 - 6);
    }
}

async function captureAndRecognize() {
    if (!running || busy || video.readyState < 2) return;
    busy = true;

    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.85);

    try {
        const res = await fetch("/api/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl }),
        });
        const data = await res.json();
        lastRunText.textContent = "Last analyzed: " + new Date().toLocaleTimeString();

        if (data.results && data.results.length > 0) {
            drawBoxes(data.results);
            renderResultPanel(data.results[0]);
        } else {
            overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
        }
    } catch (err) {
        console.error(err);
    } finally {
        busy = false;
    }
}

function renderResultPanel(r) {
    document.getElementById("result-empty").style.display = "none";
    document.getElementById("result-content").style.display = "block";

    document.getElementById("result-name").textContent = r.name;
    document.getElementById("result-id").textContent = r.external_id ? `ID: ${r.external_id}` : "Not in registry";
    document.getElementById("result-similarity").textContent = r.similarity.toFixed(3);

    const meta = decisionMeta[r.decision] || decisionMeta.unknown;
    const sourceLabel = r.decision_source === "ml" ? "ML MODEL" : "RULE ENGINE";
    let decisionHtml = `<span class="badge ${meta.cls}">${meta.label}</span> <span class="tag-mono">(${sourceLabel})</span>`;
    if (r.ml_decision && r.decision_source === "rule") {
        // ML exists but isn't promoted/available for this frame - show it as a preview of what it would say.
        const mlMeta = decisionMeta[r.ml_decision] || decisionMeta.unknown;
        decisionHtml += ` <span class="badge ${mlMeta.cls}" title="ML model prediction (not yet authoritative)">ML preview: ${r.ml_decision.toUpperCase()} (${(r.ml_confidence * 100).toFixed(0)}%)</span>`;
    }
    document.getElementById("result-decision").innerHTML = decisionHtml;

    // Gauge: circumference for r=64 is 2*PI*64 ≈ 402.1
    const circumference = 402;
    const offset = circumference - (r.trust_score / 100) * circumference;
    const arc = document.getElementById("gauge-arc");
    arc.style.stroke = meta.color;
    arc.setAttribute("stroke-dashoffset", offset);
    document.getElementById("gauge-num").textContent = r.trust_score.toFixed(0);

    const subEl = document.getElementById("subscores");
    subEl.innerHTML = "";
    const labels = { similarity: "Similarity", blur: "Sharpness", brightness: "Brightness", pose: "Pose", face_size: "Face Size", spoof: "Liveness" };
    for (const key of Object.keys(labels)) {
        const val = r.sub_scores[key];
        const weight = r.weights[key];
        subEl.innerHTML += `
            <div class="subscore-row">
                <div class="subscore-name">${labels[key]}</div>
                <div class="subscore-bar-track"><div class="subscore-bar-fill" style="width:${val}%"></div></div>
                <div class="subscore-val">${val.toFixed(0)}</div>
                <div class="tag-mono" style="width:38px; text-align:right;">×${weight}</div>
            </div>`;
    }
}

btnToggle.addEventListener("click", () => {
    running = !running;
    btnToggle.textContent = running ? "Pause Recognition" : "Resume Recognition";
    camDot.classList.toggle("live", running);
});

startCamera();
intervalHandle = setInterval(captureAndRecognize, CAPTURE_INTERVAL_MS);

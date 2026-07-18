const video = document.getElementById("video");
const captureCanvas = document.getElementById("capture-canvas");
const previewImg = document.getElementById("preview-img");
const btnCapture = document.getElementById("btn-capture");
const btnRetake = document.getElementById("btn-retake");
const btnSubmit = document.getElementById("btn-submit");
const flash = document.getElementById("flash");
const camStatusText = document.getElementById("cam-status-text");
const camDot = document.getElementById("cam-dot");

let capturedDataUrl = null;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        camStatusText.textContent = "Camera live";
    } catch (err) {
        camStatusText.textContent = "Camera access denied";
        camDot.classList.remove("live");
        showFlash("error", "Could not access webcam: " + err.message);
    }
}

function showFlash(type, message) {
    flash.className = `flash show flash-${type}`;
    flash.textContent = message;
}

function hideFlash() {
    flash.className = "flash";
}

btnCapture.addEventListener("click", () => {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    capturedDataUrl = captureCanvas.toDataURL("image/jpeg", 0.92);

    previewImg.src = capturedDataUrl;
    previewImg.style.display = "block";
    video.style.display = "none";
    btnCapture.style.display = "none";
    btnRetake.style.display = "inline-flex";
    checkSubmitEnabled();
});

btnRetake.addEventListener("click", () => {
    capturedDataUrl = null;
    previewImg.style.display = "none";
    video.style.display = "block";
    btnCapture.style.display = "inline-flex";
    btnRetake.style.display = "none";
    checkSubmitEnabled();
});

function checkSubmitEnabled() {
    const name = document.getElementById("input-name").value.trim();
    const id = document.getElementById("input-id").value.trim();
    btnSubmit.disabled = !(capturedDataUrl && name && id);
}

document.getElementById("input-name").addEventListener("input", checkSubmitEnabled);
document.getElementById("input-id").addEventListener("input", checkSubmitEnabled);

btnSubmit.addEventListener("click", async () => {
    hideFlash();
    btnSubmit.disabled = true;
    btnSubmit.textContent = "Encrypting & registering…";

    const payload = {
        name: document.getElementById("input-name").value.trim(),
        external_id: document.getElementById("input-id").value.trim(),
        image: capturedDataUrl,
    };

    try {
        const res = await fetch("/api/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.success) {
            showFlash("success", `Registered "${data.name}" (${data.external_id}) — embedding encrypted and stored.`);
            document.getElementById("input-name").value = "";
            document.getElementById("input-id").value = "";
            btnRetake.click();
        } else {
            showFlash("error", data.error || "Registration failed.");
        }
    } catch (err) {
        showFlash("error", "Network error: " + err.message);
    } finally {
        btnSubmit.textContent = "Encrypt & Register";
        checkSubmitEnabled();
    }
});

startCamera();

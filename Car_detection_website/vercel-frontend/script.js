// ---------- DOM Elements ----------
const video = document.querySelector("#live");
const buttonPlay = document.querySelector("#play");
const buttonEnd = document.querySelector("#end");
const canvas = document.querySelector("#can");
const context = canvas.getContext("2d");

// Metric dashboard elements
const elBackendFps = document.getElementById("m-backend-fps");
const elClientFps = document.getElementById("m-client-fps");
const elWsLatency = document.getElementById("m-ws-latency");
const elInference = document.getElementById("m-inference");
const elDropped = document.getElementById("m-dropped");
const elCpu = document.getElementById("m-cpu");
const elGpu = document.getElementById("m-gpu");

// ---------- Config ----------
// Backend address WITHOUT https://  e.g. "my-app.onrender.com"
// Leave empty ("") when the backend serves this page itself.
const BACKEND_HOST = "";

const FRAME_SIZE = 320;            // model input size (square)
const JPEG_QUALITY = 0.5;
const CONNECT_TIMEOUT_MS = 10000;
const WATCHDOG_MS = 1500;          // unlock sending if a reply never arrives

// Offscreen canvas: sends clean frames (no boxes drawn on them)
const sendCanvas = document.createElement("canvas");
sendCanvas.width = FRAME_SIZE;
sendCanvas.height = FRAME_SIZE;
const sendCtx = sendCanvas.getContext("2d");

// ---------- State ----------
let stream = null;
let ws = null;

let startAttempt = 0;   // bumped on every start AND every stop
let isStarting = false;
let hasConnected = false;
let isProcessing = false;

let frameSendTime = 0;
let totalCameraFrames = 0;
let processedFrames = 0;
let latestDetections = [];
let cameraFrameCounter = 0;
let lastFpsTime = performance.now();

let renderAnimId = null;
let frameCallbackId = null;
let sendTimeoutId = null;
let watchdogTimer = null;
let connectionTimeoutId = null;

// ---------- Helpers ----------

// Draw the video onto a ctx as a centre-cropped square (no stretching).
// Used for both the preview and the frames sent to the model, so box
// coordinates from the backend line up with what the user sees.
function drawSquare(ctx, size) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return false;
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);
    return true;
}

function formatMetric(val) {
    if (val === null || val === undefined || val === "") return "-";
    const num = Number(val);
    return Number.isFinite(num) ? num.toFixed(1) : "-";
}

function setText(el, value) {
    if (el) el.innerText = value;
}

function resetMetricsUI(statusMessage = "") {
    [elBackendFps, elClientFps, elWsLatency, elInference, elDropped, elCpu, elGpu]
        .forEach((el) => setText(el, "-"));

    if (statusMessage) {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "#FF5555";
        context.font = "bold 16px Arial";
        context.textBaseline = "alphabetic";
        // Truncate + maxWidth so long messages never run off the canvas
        const msg = statusMessage.length > 40 ? statusMessage.slice(0, 37) + "..." : statusMessage;
        context.fillText(msg, 10, 30, canvas.width - 20);
    }
}

// Complete stop and cleanup. Bumping startAttempt invalidates every
// pending async step (camera prompt, toBlob callback, timers).
function stopAll(statusMessage = "") {
    startAttempt++;
    isStarting = false;

    clearTimeout(connectionTimeoutId);
    connectionTimeoutId = null;
    clearTimeout(watchdogTimer);
    watchdogTimer = null;
    clearTimeout(sendTimeoutId);
    sendTimeoutId = null;

    if (renderAnimId) {
        cancelAnimationFrame(renderAnimId);
        renderAnimId = null;
    }
    if (frameCallbackId && "cancelVideoFrameCallback" in video) {
        video.cancelVideoFrameCallback(frameCallbackId);
    }
    frameCallbackId = null;

    if (stream) {
        stream.getTracks().forEach((track) => {
            track.onended = null;
            track.stop();
        });
        stream = null;
    }
    video.srcObject = null;

    if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
        ws = null;
    }

    isProcessing = false;
    hasConnected = false;
    latestDetections = [];
    totalCameraFrames = 0;
    processedFrames = 0;
    cameraFrameCounter = 0;

    context.clearRect(0, 0, canvas.width, canvas.height);
    resetMetricsUI(statusMessage);
}

// ---------- Render loop (visible canvas: preview + boxes) ----------
function renderLoop() {
    if (stream && video.readyState >= 2 && drawSquare(context, canvas.width)) {
        latestDetections.forEach((item) => {
            if (!Array.isArray(item.bbox) || item.bbox.length < 4) return;
            const [xmin, ymin, xmax, ymax] = item.bbox;
            const width = xmax - xmin;
            const height = ymax - ymin;
            const text = `${item.class || "obj"}: ${Math.round((item.confidence || 0) * 100)}%`;

            context.strokeStyle = "#0000FF";
            context.lineWidth = 3;
            context.strokeRect(xmin, ymin, width, height);

            context.font = "bold 14px Arial";
            context.textBaseline = "top";
            const bannerWidth = context.measureText(text).width + 10;
            const bannerHeight = 22;
            const bannerY = ymin - bannerHeight >= 0 ? ymin - bannerHeight : ymin;

            context.fillStyle = "#0000FF";
            context.fillRect(xmin, bannerY, bannerWidth, bannerHeight);
            context.fillStyle = "#FFFFFF";
            context.fillText(text, xmin + 5, bannerY + 3);
        });
    }
    renderAnimId = requestAnimationFrame(renderLoop);
}

// ---------- Send loop (offscreen canvas) ----------
function sendFramesLoop() {
    if (!stream || !ws || ws.readyState !== WebSocket.OPEN) return;

    const activeAttempt = startAttempt;
    totalCameraFrames++;

    // Camera FPS
    cameraFrameCounter++;
    const now = performance.now();
    const delta = now - lastFpsTime;
    if (delta >= 1000) {
        setText(elClientFps, Math.round((cameraFrameCounter * 1000) / delta).toString());
        cameraFrameCounter = 0;
        lastFpsTime = now;
    }

    if (!isProcessing && video.readyState >= 2 && drawSquare(sendCtx, FRAME_SIZE)) {
        sendCanvas.toBlob(
            (blob) => {
                if (blob && ws && ws.readyState === WebSocket.OPEN &&
                    !isProcessing && activeAttempt === startAttempt) {
                    isProcessing = true;
                    frameSendTime = performance.now();
                    ws.send(blob);

                    clearTimeout(watchdogTimer);
                    watchdogTimer = setTimeout(() => {
                        if (activeAttempt === startAttempt) {
                            isProcessing = false;
                            latestDetections = [];
                        }
                    }, WATCHDOG_MS);
                }
            },
            "image/jpeg",
            JPEG_QUALITY
        );
    }

    if ("requestVideoFrameCallback" in video) {
        frameCallbackId = video.requestVideoFrameCallback(sendFramesLoop);
    } else {
        sendTimeoutId = setTimeout(sendFramesLoop, 33);
    }
}

// ---------- Start ----------
buttonPlay.addEventListener("click", async () => {
    if (isStarting || stream ||
        (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN))) {
        return;
    }

    stopAll();
    isStarting = true;
    const currentAttempt = startAttempt; // stopAll() already bumped it

    // Step 1: camera
    if (!navigator.mediaDevices?.getUserMedia) {
        stopAll("Camera needs HTTPS");
        return;
    }

    let tempStream;
    try {
        tempStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: "environment" },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
    } catch (err) {
        if (currentAttempt !== startAttempt) return; // superseded by Stop/Start
        stopAll("Camera Error: " + err.message);
        return;
    }

    // Stop (or a newer Start) happened while the permission prompt was open
    if (currentAttempt !== startAttempt) {
        tempStream.getTracks().forEach((track) => track.stop());
        return;
    }

    stream = tempStream;
    video.srcObject = stream;
    stream.getVideoTracks()[0].onended = () => {
        if (currentAttempt === startAttempt) stopAll("Camera disconnected");
    };

    renderLoop(); // show live preview immediately

    // Step 2: WebSocket
    try {
        const wsHost = BACKEND_HOST || window.location.host;
        const protocol = BACKEND_HOST
            ? "wss:"
            : (window.location.protocol === "https:" ? "wss:" : "ws:");
        ws = new WebSocket(`${protocol}//${wsHost}/ws/predict`);
    } catch (err) {
        stopAll("WebSocket Error: " + err.message);
        return;
    }

    connectionTimeoutId = setTimeout(() => {
        if (!hasConnected && currentAttempt === startAttempt) {
            stopAll("Could not connect (Timeout)");
        }
    }, CONNECT_TIMEOUT_MS);

    ws.onopen = () => {
        clearTimeout(connectionTimeoutId);
        connectionTimeoutId = null;
        hasConnected = true;
        isStarting = false;
        isProcessing = false;
        lastFpsTime = performance.now();
        sendFramesLoop();
    };

    ws.onmessage = (event) => {
        clearTimeout(watchdogTimer);
        try {
            const response = JSON.parse(event.data);
            latestDetections = response.detections || [];
            const metrics = response.metrics || {};

            processedFrames++;

            setText(elWsLatency, Math.round(performance.now() - frameSendTime).toString());
            setText(elInference, formatMetric(metrics.inference_ms));
            setText(elBackendFps, formatMetric(metrics.backend_fps));
            setText(elCpu, formatMetric(metrics.cpu_usage));
            setText(elGpu, formatMetric(metrics.gpu_usage));
            setText(elDropped, Math.max(0, totalCameraFrames - processedFrames).toString());
        } catch (err) {
            console.error("Frame handling error:", err);
        } finally {
            isProcessing = false;
        }
    };

    ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
        isProcessing = false;
    };

    ws.onclose = () => {
        stopAll(hasConnected ? "Connection lost" : "Could not connect");
    };
});

// ---------- Stop ----------
buttonEnd.addEventListener("click", () => stopAll());

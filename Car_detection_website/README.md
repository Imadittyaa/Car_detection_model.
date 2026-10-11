# Real-Time Vehicle Detection

A web app that detects vehicles live from your camera. The browser captures camera frames, sends them over a WebSocket to a FastAPI server, which runs a YOLO model (ONNX) and sends back the detections. The browser draws the boxes on top of the live video.

## How it works

```
Browser (camera) ──JPEG frame──▶ FastAPI /ws/predict ──▶ YOLO (best.onnx)
       ▲                                                        │
       └────────── JSON: boxes + performance metrics ◀──────────┘
```

- The browser center-crops the camera image to a 320×320 square and sends **one frame at a time**, waiting for each reply before sending the next.
- The server decodes the JPEG, runs the model in a worker thread, and replies with detections and metrics.
- Because the preview and the frames sent to the server use the same crop, box coordinates line up with what you see.

## Features

- Live camera preview with bounding boxes, class names and confidence
- Live metrics panel: backend FPS, camera FPS, round-trip latency, model inference time, skipped frames, server CPU and GPU usage
- Clean start/stop handling, connection timeout, and clear on-screen messages ("Connection lost", "Camera disconnected", and so on)
- Works on desktop and phone browsers

## Project structure

```
project/
├── main.py                 FastAPI app (serves the page and the WebSocket)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── backend/
│   ├── __init__.py
│   ├── predictor.py        Loads the model and runs detection
│   └── preprocessor.py     Decodes frames, builds the JSON reply and metrics
├── frontend/               Page served by the backend
│   ├── index.html
│   └── script.js
├── vercel-frontend/        Optional: same page for hosting on Vercel (see below)
│   ├── index.html
│   └── script.js
└── model/
    └── best.onnx           Your trained YOLO model (add this yourself)
```

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put your model at `model/best.onnx`, then start the server from the project root:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and press **Start**.

### Using it from a phone

Browsers only allow camera access on **HTTPS** (or `localhost`), so plain `http://<laptop-ip>:8000` will not work on a phone. Pick one:

- **HTTPS tunnel** (ngrok, cloudflared, VS Code dev tunnels): easiest, but adds network delay.
- **Chrome flag** on the phone: open `chrome://flags/#unsafe-treat-insecure-origin-as-secure`, add `http://<laptop-ip>:8000`, enable it and relaunch Chrome. Both devices must be on the same Wi-Fi or hotspot.
- **Local HTTPS certificate** with `mkcert`, passed to uvicorn with `--ssl-keyfile` and `--ssl-certfile`.

On the same Wi-Fi, the network round trip is typically much lower than through a tunnel, so you get a higher frame rate.

## Docker

```bash
docker build -t vehicle-detect .
docker run -p 7860:7860 vehicle-detect
```

Open `http://localhost:7860`. The container listens on `$PORT` (default `7860`), so it also runs on hosts such as Render, Railway and Hugging Face Spaces (Docker SDK). For Hugging Face Spaces, add this block at the very top of this file:

```
---
title: Vehicle Detection
sdk: docker
app_port: 7860
---
```

Large model files may need Git LFS (GitHub rejects files over 100 MB).

## Hosting the page on Vercel (optional)

Vercel cannot run the WebSocket backend, so host the backend elsewhere (Docker host above) and put only the page on Vercel:

1. Deploy the backend and note its address, for example `my-app.onrender.com`.
2. In `vercel-frontend/script.js`, set `BACKEND_HOST` to that address, without `https://`.
3. Import the repo into Vercel with **Root Directory** set to `vercel-frontend` and no build command.

Free backend plans may sleep when idle. Open the backend address in a browser first to wake it, then press Start (the page gives up connecting after 10 seconds).

## WebSocket protocol

Endpoint: `/ws/predict`

**Client → server:** one binary message per frame, a 320×320 JPEG.

**Server → client:** one JSON message per frame.

```json
{
  "detections": [
    { "class": "car", "confidence": 0.87, "bbox": [x_min, y_min, x_max, y_max] }
  ],
  "metrics": {
    "inference_ms": 39.7,
    "decode_ms": 2.1,
    "backend_fps": 12.4,
    "cpu_usage": 45.0,
    "ram_usage": 61.2,
    "gpu_usage": null
  }
}
```

`bbox` is in pixels of the 320×320 frame. `gpu_usage` is `null` when no NVIDIA GPU is readable, and the page shows `-`. If a single frame fails on the server, the reply is an empty detection list so the stream keeps going.

## Metrics panel

| Label | Meaning |
|---|---|
| Backend FPS | Frames per second the server completed |
| Camera FPS | Frames per second the camera delivers |
| WS Latency (RTT) | Time from sending a frame to receiving its reply (includes model time) |
| YOLO Inference | Model-only time on the server |
| Skipped Frames | Camera frames not sent, because the previous frame was still being processed (expected to be high) |
| Server CPU / GPU | Usage on the machine running the backend, not on your phone |

## Configuration

| Setting | Where | Default |
|---|---|---|
| Confidence threshold | `backend/predictor.py` (`conf`) | `0.25` |
| Frame size sent to the model | `FRAME_SIZE` in `script.js` and `backend/preprocessor.py` (keep them equal) | `320` |
| JPEG quality | `JPEG_QUALITY` in `script.js` | `0.5` |
| Backend address (Vercel version) | `BACKEND_HOST` in `vercel-frontend/script.js` | same host as the page |

For the best speed, export the ONNX model at the same size you send (for example `imgsz=320`). A model exported at a larger size upscales every frame and does more work.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No supported WebSocket library detected" | Install `uvicorn[standard]` (it is in `requirements.txt`) |
| `libGL.so.1` error | Install `libgl1` and `libglib2.0-0` (already in the Dockerfile) |
| "Camera needs HTTPS" | Use `localhost`, an HTTPS tunnel, or the Chrome flag above |
| `/static/script.js` returns 404 | `script.js` must be inside `frontend/`, next to `index.html` |
| "Could not connect (Timeout)" | The server is not running, is asleep, or the address or port is wrong |
| No boxes appear | Check the server console, try a lower `conf`, and confirm the model's class names |
| Boxes look offset | Confirm `FRAME_SIZE` matches in `script.js` and `preprocessor.py` |

## Known limitations

- One frame is in flight at a time, so frame rate is limited by network round trip plus model time.
- If a reply takes longer than 1.5 seconds, the page unlocks and sends the next frame, and the late reply can be matched to the wrong frame. Echoing a frame id from the server would fix this.
- FPS and CPU counters are shared by all connected clients, so metrics are only meaningful with one viewer.
- GPU usage needs an NVIDIA GPU and `nvidia-ml-py`, and only helps if the model actually runs on the GPU (`onnxruntime-gpu`).

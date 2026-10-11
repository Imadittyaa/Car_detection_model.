from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.predictor import Predictor
from backend.preprocessor import Preprocessor

# Paths are built from this file's location, so the server works
# no matter which folder uvicorn is started from.
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "vercel-frontend"
MODEL_PATH = BASE_DIR / "model" / "best.onnx"

app = FastAPI(
    title="Car Detection API",
    description="Real-time frame detection API.",
)

predictor = Predictor(str(MODEL_PATH))
preprocessor = Preprocessor(predictor=predictor)

# /static/script.js  ->  frontend/script.js
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def get_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws/predict")
async def predict_frame(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            image_bytes = await websocket.receive_bytes()

            try:
                # Run the blocking model call in a worker thread so the
                # event loop stays free for other requests and keep-alives.
                result = await run_in_threadpool(preprocessor.process_frame, image_bytes)
            except Exception as e:
                # A single bad frame must not kill the connection.
                # Reply with an empty result so the client unlocks and continues.
                print(f"Frame processing error: {e}")
                result = {"detections": [], "metrics": {}}

            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("Frontend disconnected from WebSocket.")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass  # already closed

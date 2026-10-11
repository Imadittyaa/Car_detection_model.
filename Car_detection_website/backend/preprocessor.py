import threading
import time

import cv2
import numpy as np
import psutil

# Optional real GPU utilization via NVML (pip install nvidia-ml-py).
# If it is missing or there is no NVIDIA GPU, gpu_usage is sent as None
# and the frontend shows "-" instead of a fake 0.
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _GPU_HANDLE = None

FRAME_SIZE = 320          # must match the frontend canvas / send size
IDLE_RESET_SECONDS = 2.0  # restart the FPS window after a pause this long


class Preprocessor:
    def __init__(self, predictor):
        self.predictor = predictor

        # FPS tracking (shared by all connections)
        self._fps_lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.fps = 0.0

        # CPU usage cache (refreshed at most once per second)
        self.last_cpu_check = time.time()
        self.cached_cpu = 0.0
        psutil.cpu_percent(interval=None)  # prime the counter

    # ---------- system stats ----------
    def get_gpu_usage(self):
        if _GPU_HANDLE is None:
            return None
        try:
            return float(pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu)
        except Exception:
            return None

    def get_system_stats(self):
        now = time.time()
        if now - self.last_cpu_check >= 1.0:
            self.cached_cpu = round(psutil.cpu_percent(interval=None), 1)
            self.last_cpu_check = now
        ram_usage = psutil.virtual_memory().percent
        return self.cached_cpu, ram_usage, self.get_gpu_usage()

    # ---------- FPS ----------
    def _update_fps(self):
        with self._fps_lock:
            now = time.time()

            # After an idle gap (stream stopped, first frame of a new session)
            # start a fresh window so the first reading is not near zero.
            if now - self.last_frame_time > IDLE_RESET_SECONDS:
                self.frame_count = 0
                self.start_time = now
                self.fps = 0.0
            self.last_frame_time = now

            self.frame_count += 1
            elapsed = now - self.start_time
            if elapsed >= 1.0:
                self.fps = round(self.frame_count / elapsed, 1)
                self.frame_count = 0
                self.start_time = now
            return self.fps

    # ---------- main entry ----------
    def process_frame(self, image_bytes: bytes) -> dict:
        t0 = time.perf_counter()

        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"detections": [], "metrics": {}}

        decode_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Keep box coordinates in the frontend's 320x320 space.
        # The frontend already sends 320x320, so this normally does nothing.
        if frame.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        # Slim predictor returns plain dicts plus the model-only time in ms
        detections, model_ms = self.predictor.predict_frame(frame)
        inference_ms = round(float(model_ms), 2)

        fps = self._update_fps()

        cpu_usage, ram_usage, gpu_usage = self.get_system_stats()

        return {
            "detections": detections,
            "metrics": {
                "inference_ms": inference_ms,
                "decode_ms": decode_ms,
                "backend_fps": fps,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "gpu_usage": gpu_usage,  # None -> JSON null -> shown as "-"
            },
        }

import threading

import numpy as np
from ultralytics import YOLO


class Predictor:
    def __init__(self, model_path='model/best.onnx', conf=0.5):
        self.conf = conf
        self.model = YOLO(model_path, task="detect")

        # The YOLO object keeps internal state, so only one thread may run it at a time.
        self._lock = threading.Lock()

        # Warm-up: the first call builds the predictor and is slow.
        # Doing it here keeps the first real frame fast.
        self.predict_frame(np.zeros((320, 320, 3), dtype=np.uint8))

    def predict_frame(self, frame):
        """frame: BGR numpy array (OpenCV order). Returns an Ultralytics Results object."""
        with self._lock:
            results = self.model(frame, conf=self.conf, verbose=False)
        return results[0]

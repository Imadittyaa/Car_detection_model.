"""
Slim predictor: runs a YOLO ONNX model with onnxruntime only.
No PyTorch / Ultralytics needed, so RAM and image size are much smaller.

predict_frame(frame) -> (detections, model_ms)
    frame       : BGR numpy array (what cv2.imdecode returns)
    detections  : list of {"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
                  with coordinates in pixels of the frame you passed in
    model_ms    : time spent inside the ONNX model only
"""
import ast
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

IOU_THRESHOLD = 0.7   # same default as Ultralytics
MAX_DET = 300


class Predictor:
    def __init__(self, model_path='model/best.onnx', conf=0.25):
        self.conf = conf

        opts = ort.SessionOptions()
        # Few threads = less CPU contention on small cloud plans
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_fp16 = "float16" in inp.type
        shape = inp.shape  # normally [1, 3, H, W]
        self.in_h = shape[2] if isinstance(shape[2], int) else 320
        self.in_w = shape[3] if isinstance(shape[3], int) else 320

        self.names = self._load_names()

        out = self.session.get_outputs()[0]
        # Printed once at startup so it shows up in the host's logs
        print(f"[Predictor] input={inp.shape} output={out.shape} classes={self.names}")

        # Warm-up: the first run is slower
        self.predict_frame(np.zeros((320, 320, 3), dtype=np.uint8))

    # ---------- helpers ----------
    def _load_names(self):
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return {}
        try:
            names = ast.literal_eval(raw)
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            return {i: str(v) for i, v in enumerate(names)}
        except Exception:
            return {}

    def _letterbox(self, frame):
        """Resize keeping aspect ratio and pad to the model's input size."""
        h, w = frame.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = frame if (new_w, new_h) == (w, h) else cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        left = (self.in_w - new_w) // 2
        top = (self.in_h - new_h) // 2
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, scale, left, top

    def _to_blob(self, img):
        blob = img[..., ::-1].transpose(2, 0, 1)            # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None]                                    # add batch dim
        return blob.astype(np.float16) if self.input_fp16 else blob

    def _postprocess(self, out, scale, left, top, w, h):
        pred = np.squeeze(out, axis=0).astype(np.float32)

        # Standard Ultralytics export: (4+nc, N) -> (N, 4+nc)
        end2end = False
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        elif pred.shape[1] == 6:
            end2end = True  # NMS already done: [x1, y1, x2, y2, conf, class]

        if end2end:
            confs = pred[:, 4]
            keep = confs >= self.conf
            xyxy = pred[keep, :4].copy()
            confs = confs[keep]
            cls_ids = pred[keep, 5].astype(int)
        else:
            scores = pred[:, 4:]
            cls_ids = scores.argmax(axis=1)
            confs = scores[np.arange(len(scores)), cls_ids]
            keep = confs >= self.conf
            pred, confs, cls_ids = pred[keep], confs[keep], cls_ids[keep]
            if len(pred) == 0:
                return []
            cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

            # Class-aware NMS: shift each class far apart so classes never suppress each other
            shifted = xyxy + cls_ids[:, None] * 4096.0
            nms_boxes = np.stack(
                [shifted[:, 0], shifted[:, 1],
                 shifted[:, 2] - shifted[:, 0], shifted[:, 3] - shifted[:, 1]], axis=1
            ).tolist()
            idx = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), self.conf, IOU_THRESHOLD)
            idx = np.array(idx, dtype=int).reshape(-1)[:MAX_DET]
            xyxy, confs, cls_ids = xyxy[idx], confs[idx], cls_ids[idx]

        if len(xyxy) == 0:
            return []

        # Back to the coordinates of the original frame
        xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - left) / scale, 0, w)
        xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - top) / scale, 0, h)

        order = np.argsort(-confs)
        detections = []
        for i in order:
            cid = int(cls_ids[i])
            detections.append({
                "class": self.names.get(cid, str(cid)),
                "confidence": round(float(confs[i]), 2),
                "bbox": [int(v) for v in xyxy[i]],
            })
        return detections

    # ---------- main entry ----------
    def predict_frame(self, frame):
        h, w = frame.shape[:2]
        img, scale, left, top = self._letterbox(frame)
        blob = self._to_blob(img)

        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]
        model_ms = (time.perf_counter() - t0) * 1000

        return self._postprocess(out, scale, left, top, w, h), model_ms

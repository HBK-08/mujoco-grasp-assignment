"""Serve the fine-tuned YOLOv8 detector through a small local HTTP API.

The policy sends a JPEG and the public candidate identifiers named by the
instruction.  The service returns YOLO boxes and scores.  The policy performs
all RGB-D geometry locally from the evaluator's current observation.

Service root exposes ``GET /healthz`` (``{"ok": true}``) and ``POST /infer``.

Run:
    pip install ultralytics fastapi uvicorn pydantic pillow numpy
    python infer_server.py            # listens on 127.0.0.1:8765
Environment:
    YOLO_WEIGHTS   path to the trained checkpoint (default ./best.pt next to this file)
    YOLO_CONF      detection confidence threshold (default 0.25)
    YOLO_DEVICE    Ultralytics device selector (examples: cpu, 0, 1; default auto)
    YOLO_PORT      listen port (default 8765)
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "best.pt"

# Class ids MUST match the checkpoint's class names and yolo/objects.yaml.
CLASSES = [
    "red_cube", "green_cylinder", "blue_box",        # 0..2  primitives (task1)
    "banana", "apple", "orange",                      # 3..5  YCB fruit
    "mustard_bottle", "potted_meat_can",              # 6..7  YCB packaged food
    "scissors", "marker",                             # 8..9  YCB tools
    "square_tray", "round_tray",                      # 10..11 containers
]

CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}

app = FastAPI(title="GraspBench YOLO detector")


class InferRequest(BaseModel):
    image_jpeg_b64: str
    candidate_ids: list[str]


_model = None
_model_meta: dict[str, Any] = {}


def load_model() -> None:
    global _model, _model_meta
    if _model is not None:
        return
    import ultralytics
    from ultralytics import YOLO

    weights = os.getenv("YOLO_WEIGHTS", str(DEFAULT_WEIGHTS))
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {weights!r} "
            "(restore yolo/best.pt or set YOLO_WEIGHTS)"
        )
    _model = YOLO(weights)
    checkpoint_names = {int(key): str(value) for key, value in _model.names.items()}
    expected_names = dict(enumerate(CLASSES))
    if checkpoint_names != expected_names:
        raise RuntimeError(
            "YOLO checkpoint classes do not match the assignment catalogue: "
            f"expected={expected_names}, actual={checkpoint_names}"
        )
    _model_meta = {
        "weights": str(Path(weights).resolve()),
        "classes": CLASSES,
        "imgsz": 256,
        "conf": float(os.getenv("YOLO_CONF", "0.25")),
        "device": os.getenv("YOLO_DEVICE", "auto"),
        "ultralytics": ultralytics.__version__,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": _model is not None, **_model_meta}


@app.post("/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        raw = base64.b64decode(req.image_jpeg_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image_jpeg_b64: {exc}") from exc
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    candidate_ids = req.candidate_ids
    unknown = sorted(set(candidate_ids) - set(CLASSES))
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown candidate_ids: {unknown}")

    conf_threshold = float(os.getenv("YOLO_CONF", "0.25"))
    predict_kwargs: dict[str, Any] = {
        "imgsz": 256,
        "conf": conf_threshold,
        "verbose": False,
    }
    device = os.getenv("YOLO_DEVICE", "").strip()
    if device:
        predict_kwargs["device"] = device
    results = _model.predict(img, **predict_kwargs)
    res = results[0]

    if res.boxes is None or len(res.boxes) == 0:
        return {"detections": []}

    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy().astype(float)
    xyxy = res.boxes.xyxy.cpu().numpy()

    allowed_ids = {CLASS_TO_ID[name] for name in candidate_ids}
    detections = [
        {
            "target_id": CLASSES[int(class_id)],
            "score": float(score),
            "box_xyxy": [float(value) for value in box],
        }
        for class_id, score, box in zip(cls, conf, xyxy, strict=True)
        if int(class_id) in allowed_ids
    ]
    return {"detections": detections}


if __name__ == "__main__":
    import uvicorn

    load_model()
    print(f"[yolo] model ready: {_model_meta}", flush=True)
    port = int(os.getenv("YOLO_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

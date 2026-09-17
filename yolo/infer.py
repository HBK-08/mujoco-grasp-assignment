"""Minimal standalone inference for the fine-tuned YOLOv8n detector.

Usage:
    python infer.py [image ...]        # default: sample.jpg
    python infer.py a.jpg b.jpg --save

The checkpoint ``best.pt`` is self-contained: it embeds the model architecture,
the fine-tuned weights, and the 12 class names. Only ``ultralytics`` is required.

    pip install ultralytics
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "best.pt"
IMGSZ = 256          # must match the training resolution (see yolo/train.py --imgsz)
CONF = 0.25


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fine-tuned YOLOv8n detector.")
    ap.add_argument("images", nargs="*", default=["sample.jpg"])
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument(
        "--device",
        default=os.getenv("YOLO_DEVICE") or None,
        help="Ultralytics device selector (cpu, 0, 1, ...; default auto)",
    )
    ap.add_argument("--save", action="store_true", help="write annotated images next to the input")
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    print(f"[infer] weights={WEIGHTS.name} classes={len(model.names)}")

    for img in args.images:
        path = Path(img)
        if not path.exists():
            print(f"[infer] SKIP (not found): {img}")
            continue

        predict_kwargs = {"imgsz": IMGSZ, "conf": args.conf, "verbose": False}
        if args.device:
            predict_kwargs["device"] = args.device
        results = model.predict(str(path), **predict_kwargs)
        res = results[0]
        print(f"\n=== {path} ===")

        if res.boxes is None or len(res.boxes) == 0:
            print("  (no detections)")
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for cls_id, conf, box in zip(classes, confs, boxes):
            name = model.names[int(cls_id)]
            x1, y1, x2, y2 = (round(float(v), 1) for v in box)
            print(f"  {name:<16} conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")

        if args.save:
            out = path.with_name(f"{path.stem}_annotated{path.suffix}")
            res.save(filename=str(out))
            print(f"  -> saved {out}")


if __name__ == "__main__":
    main()

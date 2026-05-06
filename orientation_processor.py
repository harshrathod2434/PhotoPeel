#!/usr/bin/env python3
"""Orientation inference helper.

This is designed to be run as a subprocess from the main app server, so the
server can use a different Python version than the ONNX runtime environment.

Usage:
  python orientation_processor.py <model_path> <image_path>

Outputs a single JSON object on stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def _preprocess(pil: Image.Image) -> np.ndarray:
    """Match the training/inference preprocessing used in the notebook.

    - Resize to 384x384 (bilinear)
    - Scale to [0,1]
    - Normalize by ImageNet mean/std
    - Convert HWC -> CHW, add batch dim
    """
    pil = pil.resize((384, 384), Image.BILINEAR)
    arr = np.asarray(pil).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def main() -> int:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: orientation_processor.py <model_path> <image_path>"}))
        return 1

    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])

    try:
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name

        img = Image.open(image_path).convert("RGB")
        inp = _preprocess(img)
        outputs = session.run(None, {input_name: inp})
        logits = np.asarray(outputs[0]).squeeze()
        if logits.ndim != 1:
            logits = logits.reshape(-1)

        probs = _softmax(logits)
        pred = int(np.argmax(probs))

        labels = ["0deg", "90deg", "180deg", "270deg"]
        correction_turns = {0: 0, 1: 1, 2: 2, 3: 3}.get(pred, 0)

        print(
            json.dumps(
                {
                    "label": labels[pred] if 0 <= pred < len(labels) else "unknown",
                    "confidence": float(probs[pred]) if 0 <= pred < len(probs) else 0.0,
                    "turns": int(correction_turns),
                    "error": None,
                }
            )
        )
        return 0

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

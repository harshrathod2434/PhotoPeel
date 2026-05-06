#!/usr/bin/env python3
"""
Face detection + identity clustering for scanned photo exports.
Run as a subprocess:
  python3.11 face_processor.py <export_dir> <out_dir> [threshold] [margin]
Prints a single JSON object to stdout.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _detect_in_image(app, image):
    faces = app.get(image)
    result = []
    for face in faces:
        if face.embedding is None:
            continue
        result.append({
            "bbox":      [int(x) for x in face.bbox],
            "det_score": float(face.det_score),
            "embedding": face.embedding.tolist(),
        })
    return result


def _crop_face(image, bbox, margin):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    H, W = image.shape[:2]
    return image[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)]


def _cosine(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _cluster(all_faces, threshold):
    """Greedy centroid clustering on ArcFace embeddings."""
    clusters = []
    for face in all_faces:
        emb = np.asarray(face["embedding"], dtype=np.float32)
        best_sim, best_i = -1.0, -1
        for i, c in enumerate(clusters):
            sim = _cosine(emb, c["centroid"])
            if sim > best_sim:
                best_sim, best_i = sim, i
        if best_sim >= threshold:
            n = len(clusters[best_i]["faces"]) + 1
            clusters[best_i]["centroid"] = (
                clusters[best_i]["centroid"] * (n - 1) / n + emb / n
            )
            clusters[best_i]["faces"].append(face)
        else:
            clusters.append({"centroid": emb.copy(), "faces": [face]})
    clusters.sort(key=lambda c: -len(c["faces"]))
    return clusters


def process_export(export_dir, out_dir, threshold=0.35, margin=0.35):
    export_dir = Path(export_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in export_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_files:
        return {"persons": [], "total_faces": 0}

    app = _load_app()

    all_faces = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for face in _detect_in_image(app, img):
            face["source_file"] = img_path.name
            face["source_path"] = str(img_path)
            all_faces.append(face)

    if not all_faces:
        result = {"persons": [], "total_faces": 0}
        (out_dir / "groups.json").write_text(json.dumps(result))
        return result

    clusters = _cluster(all_faces, threshold)

    persons = []
    for p_idx, cluster in enumerate(clusters):
        person_id = f"person_{p_idx + 1:03d}"
        person_dir = out_dir / person_id
        person_dir.mkdir(exist_ok=True)
        faces_out = []
        for f_idx, face in enumerate(cluster["faces"]):
            src = cv2.imread(face["source_path"])
            if src is None:
                continue
            crop = _crop_face(src, face["bbox"], margin)
            if crop.size == 0:
                continue
            fname = f"face_{f_idx + 1:03d}.jpg"
            cv2.imwrite(str(person_dir / fname), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            faces_out.append({
                "file":   f"{person_id}/{fname}",
                "source": face["source_file"],
                "score":  round(face["det_score"], 3),
            })
        if faces_out:
            persons.append({"id": person_id, "count": len(faces_out), "faces": faces_out})

    result = {"persons": persons, "total_faces": len(all_faces)}
    (out_dir / "groups.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: face_processor.py <export_dir> <out_dir> [threshold] [margin]"}))
        sys.exit(1)
    _export_dir = sys.argv[1]
    _out_dir    = sys.argv[2]
    _threshold  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    _margin     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
    print(json.dumps(process_export(_export_dir, _out_dir, _threshold, _margin)))

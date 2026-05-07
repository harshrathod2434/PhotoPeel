import base64
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DATA_ROOT = ROOT / "appdata"
UPLOAD_ROOT = DATA_ROOT / "uploads"
EXPORT_ROOT = DATA_ROOT / "exports"
FACE_ROOT = DATA_ROOT / "faces"
SESSION_EXPORT_ROOT = DATA_ROOT / "session_exports"

FACE_PYTHON = "/opt/homebrew/bin/python3.11"
FACE_SCRIPT = ROOT / "face_processor.py"
ORIENTATION_ROOT = (ROOT.parent / "OBJECT_ORIENTATION_DETECTOR").resolve()
ORIENTATION_MODEL = ORIENTATION_ROOT / "orientation_model_v2_0.9882.onnx"
ORIENTATION_PYTHON = ORIENTATION_ROOT / "venv" / "bin" / "python"
ORIENTATION_SCRIPT = ROOT / "orientation_processor.py"
ORIENTATION_SESSION = None
ORIENTATION_INPUT_NAME = None
ORIENTATION_ERROR = None

for folder in (UPLOAD_ROOT, EXPORT_ROOT, FACE_ROOT, SESSION_EXPORT_ROOT):
    folder.mkdir(parents=True, exist_ok=True)


def load_orientation_model():
    global ORIENTATION_SESSION, ORIENTATION_INPUT_NAME, ORIENTATION_ERROR
    if ORIENTATION_SESSION is not None or ORIENTATION_ERROR is not None:
        return ORIENTATION_SESSION
    try:
        if not ORIENTATION_MODEL.exists():
            raise FileNotFoundError(f"Orientation model not found: {ORIENTATION_MODEL}")

        import onnxruntime as ort
        ORIENTATION_SESSION = ort.InferenceSession(str(ORIENTATION_MODEL))
        ORIENTATION_INPUT_NAME = ORIENTATION_SESSION.get_inputs()[0].name
        return ORIENTATION_SESSION
    except Exception as exc:
        ORIENTATION_ERROR = str(exc)
        return None


def orientation_subprocess_available():
    return ORIENTATION_MODEL.exists() and ORIENTATION_PYTHON.exists() and ORIENTATION_SCRIPT.exists()


def predict_orientation_subprocess(image):
    import tempfile

    if not orientation_subprocess_available():
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "turns": 0,
            "error": "Orientation subprocess not available",
        }

    tmp_path = None
    try:
        # Write the crop to a temporary JPEG file for the subprocess.
        fd, tmp_path = tempfile.mkstemp(prefix="orientation_", suffix=".jpg", dir=str(DATA_ROOT))
        os.close(fd)
        cv2.imwrite(tmp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

        proc = subprocess.run(
            [str(ORIENTATION_PYTHON), str(ORIENTATION_SCRIPT), str(ORIENTATION_MODEL), tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            return {
                "label": "unavailable",
                "confidence": 0.0,
                "turns": 0,
                "error": msg[-1] if msg else "orientation_processor failed",
            }
        last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        data = json.loads(last_line) if last_line else {}
        if data.get("error"):
            return {
                "label": "unavailable",
                "confidence": 0.0,
                "turns": 0,
                "error": data.get("error"),
            }
        return {
            "label": data.get("label", "unknown"),
            "confidence": float(data.get("confidence", 0.0)),
            "turns": int(data.get("turns", 0)),
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "turns": 0,
            "error": "Orientation subprocess timed out",
        }
    except Exception as exc:
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "turns": 0,
            "error": str(exc),
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _sanitize_folder_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "Person"
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in (" ", "_", "-", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip().strip(".")
    return out or "Person"


def _ensure_dir(path: Path) -> Path:
    path = Path(os.path.expanduser(str(path))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def decode_image(data_url):
    if "," in data_url:
        _, data = data_url.split(",", 1)
    else:
        data = data_url
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def encode_image(image, ext=".jpg", quality=92):
    params = []
    mime = "image/jpeg"
    if ext.lower() in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif ext.lower() == ".png":
        mime = "image/png"
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError("Unable to encode image")
    data = base64.b64encode(buf).decode("ascii")
    return f"data:{mime};base64,{data}"


def order_points(points):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    summed = pts.sum(axis=1)
    rect[0] = pts[np.argmin(summed)]   # top-left
    rect[2] = pts[np.argmax(summed)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # top-right
    rect[3] = pts[np.argmax(diff)]     # bottom-left
    return rect


def _find_gap_segments(profile, gap_brightness_ratio=0.75, min_gap_width=8, min_segment_width=80):
    """
    Find the 'gap' positions in a 1-D brightness profile (brighter = scanner background).
    Returns list of (start, end) inclusive index pairs for photo segments between gaps.
    """
    threshold = profile.max() * gap_brightness_ratio
    is_gap = profile >= threshold

    gap_ranges = []
    in_gap = False
    g_start = 0
    for i, v in enumerate(is_gap):
        if v and not in_gap:
            g_start = i
            in_gap = True
        elif not v and in_gap:
            if i - g_start >= min_gap_width:
                gap_ranges.append((g_start, i - 1))
            in_gap = False
    if in_gap and len(is_gap) - g_start >= min_gap_width:
        gap_ranges.append((g_start, len(is_gap) - 1))

    n = len(profile)
    boundaries = [0]
    for gs, ge in gap_ranges:
        boundaries.append(gs)
        boundaries.append(ge + 1)
    boundaries.append(n)

    segments = []
    for i in range(0, len(boundaries) - 1, 2):
        s, e = boundaries[i], boundaries[i + 1] - 1
        if e - s + 1 >= min_segment_width:
            segments.append((s, e))
    return segments


def detect_photos(image, threshold=240, min_area_ratio=0.012, padding=0):
    """
    Detect individual photo regions in a scanned image, including slanted/skewed photos.

    threshold      : Sensitivity slider value (80–252).
                     Higher = more sensitive = looser cutoff = picks up more.
    min_area_ratio : Minimum photo area as fraction of total image area (0–1).
    padding        : Extra pixels to expand each detected box outward.

    Each returned box has:
      - 'points' : 4 corner points of a ROTATED rectangle (handles skewed photos).
      - 'bounds' : Axis-aligned bounding box (x, y, w, h) for layout/overlap checks.
      - 'angle'  : Rotation angle of the photo in degrees.
    """
    height, width = image.shape[:2]
    image_area = width * height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def clamp_pt(x, y):
        return (
            float(max(0, min(width - 1, x))),
            float(max(0, min(height - 1, y))),
        )

    def clamp_bounds(x, y, w, h):
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(int(w), width - x)
        h = min(int(h), height - y)
        return x, y, w, h

    def bounds_iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx);  y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw);  y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / (aw * ah + bw * bh - inter + 1e-6)

    def center_inside(inner, outer):
        x, y, w, h = inner
        ox, oy, ow, oh = outer
        cx, cy = x + w / 2, y + h / 2
        return ox <= cx <= ox + ow and oy <= cy <= oy + oh

    min_dim = 100
    min_area = image_area * max(0.004, min_area_ratio)
    max_area = image_area * 0.92

    candidates = []

    def add_candidate_rotated(rot_pts, cnt_area, bbox_area, angle, method, base_score=0.7):
        """
        Add a candidate from a rotated-rectangle (minAreaRect).
        rot_pts: 4 corner points from cv2.boxPoints (already ordered).
        """
        # Axis-aligned bounding box for IoU / containment checks
        xs = rot_pts[:, 0];  ys = rot_pts[:, 1]
        bx = int(xs.min());  by = int(ys.min())
        bw = int(xs.max() - xs.min());  bh = int(ys.max() - ys.min())
        bx, by, bw, bh = clamp_bounds(bx, by, bw, bh)

        if bw < min_dim or bh < min_dim:
            return
        if bbox_area < min_area or bbox_area > max_area:
            return
        # Rotated-rect short vs long side
        sides = sorted([
            np.linalg.norm(rot_pts[1] - rot_pts[0]),
            np.linalg.norm(rot_pts[2] - rot_pts[1]),
        ])
        short, long = sides
        if short < min_dim:
            return
        aspect = long / max(short, 1)
        if aspect > 6.0:
            return

        fill = cnt_area / (bbox_area + 1e-6)
        if fill < 0.08:
            return

        score = fill * base_score + min(bbox_area / image_area, 0.40)
        candidates.append({
            "bounds": (bx, by, bw, bh),
            "points": rot_pts.tolist(),
            "angle":  float(angle),
            "score":  score,
            "area":   float(bbox_area),
            "method": method,
        })

    def add_candidate_rect(x, y, w, h, score, method):
        """Add a candidate from an axis-aligned rectangle (projection method)."""
        x, y, w, h = clamp_bounds(x, y, w, h)
        if w < min_dim or h < min_dim:
            return
        area = w * h
        if area < min_area or area > max_area:
            return
        aspect = w / max(h, 1)
        if aspect < 0.15 or aspect > 6.0:
            return
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
        candidates.append({
            "bounds": (x, y, w, h),
            "points": pts.tolist(),
            "angle":  0.0,
            "score":  score,
            "area":   float(area),
            "method": method,
        })

    def process_mask(mask, method, base_score=0.7):
        """
        Extract candidates from a binary mask using minAreaRect so that
        slanted/skewed photos are captured as rotated rectangles.
        """
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        m = cv2.morphologyEx(m,    cv2.MORPH_OPEN,  kernel_open,  iterations=2)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cnt_area = cv2.contourArea(contour)
            if cnt_area < min_area * 0.5:
                continue

            # ── Rotated minimum-area rectangle ──────────────────────────────
            rot_rect = cv2.minAreaRect(contour)
            center, (rw, rh), angle = rot_rect
            # Ensure rw >= rh (long side first)
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            bbox_area = rw * rh
            rot_pts = cv2.boxPoints(rot_rect).astype("float32")
            # Order: top-left, top-right, bottom-right, bottom-left
            rot_pts = order_points(rot_pts)
            add_candidate_rotated(rot_pts, cnt_area, bbox_area, angle, method, base_score)

    # ─── Method 1: Grayscale threshold (white-background scans) ─────────────────
    mask_gray = (gray < threshold).astype(np.uint8) * 255
    process_mask(mask_gray, "gray", base_score=0.85)

    # ─── Method 2: LAB background subtraction (non-white backgrounds) ────────────
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
    margin = max(10, int(min(height, width) * 0.04))
    border_pixels = np.concatenate([
        lab[:margin,  :,  :].reshape(-1, 3),
        lab[-margin:, :,  :].reshape(-1, 3),
        lab[:,  :margin,  :].reshape(-1, 3),
        lab[:, -margin:,  :].reshape(-1, 3),
    ])
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)

    lab_t = max(6, 14 + (252 - threshold) * 0.22)
    mask_lab = (distance > lab_t).astype(np.uint8) * 255
    process_mask(mask_lab, "lab", base_score=0.7)

    # ─── Method 3: Projection-profile gap detection ───────────────────────────────
    # Fires only when the first two methods find nothing or just one big blob.
    primary_found = len([c for c in candidates if c["area"] < max_area * 0.85])
    if primary_found <= 1:
        gap_ratio = 0.72 + (threshold - 80) / (252 - 80) * 0.10
        row_prof = gray.mean(axis=1).astype(float)
        col_prof = gray.mean(axis=0).astype(float)
        row_segs = _find_gap_segments(
            row_prof, gap_brightness_ratio=gap_ratio,
            min_gap_width=max(6, int(height * 0.01)),
            min_segment_width=max(80, int(height * 0.08)),
        )
        col_segs = _find_gap_segments(
            col_prof, gap_brightness_ratio=gap_ratio,
            min_gap_width=max(6, int(width * 0.01)),
            min_segment_width=max(80, int(width * 0.08)),
        )
        if len(row_segs) > 1 or len(col_segs) > 1:
            if not col_segs:
                col_segs = [(0, width - 1)]
            if not row_segs:
                row_segs = [(0, height - 1)]
            for rs, re in row_segs:
                for cs, ce in col_segs:
                    x, y = cs, rs
                    w, h = ce - cs + 1, re - rs + 1
                    area = w * h
                    score = 0.75 + min(area / image_area, 0.20)
                    add_candidate_rect(x, y, w, h, score, "projection")

    # ─── Deduplicate overlapping candidates ──────────────────────────────────────
    candidates.sort(key=lambda c: c["score"], reverse=True)
    keep = []
    for c in candidates:
        if all(bounds_iou(c["bounds"], k["bounds"]) < 0.45 for k in keep):
            keep.append(c)

    # ─── Remove sub-boxes whose centre sits inside a larger box ──────────────────
    final = []
    for c in sorted(keep, key=lambda c: c["area"], reverse=True):
        if any(
            center_inside(c["bounds"], f["bounds"]) and c["area"] < f["area"] * 0.75
            for f in final
        ):
            continue
        final.append(c)

    # ─── Build output ─────────────────────────────────────────────────────────────
    boxes = []
    for c in final:
        pts = np.array(c["points"], dtype="float32")

        # Apply padding by expanding the rotated rectangle outward from its centre
        if padding:
            center_pt = pts.mean(axis=0)
            offsets = pts - center_pt
            norms = np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-6
            # Move each corner outward by `padding` pixels
            pts = pts + offsets / norms * padding

        # Clamp all points to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        bx, by, bw, bh = c["bounds"]
        boxes.append({
            "id":     str(uuid.uuid4()),
            "points": pts.round(1).tolist(),
            "area":   float(bw * bh),
            "bounds": [int(bx), int(by), int(bw), int(bh)],
            "angle":  c["angle"],
            "status": c["method"],
        })

    boxes.sort(key=lambda b: (b["bounds"][1], b["bounds"][0]))
    return boxes


def crop_perspective(image, points):
    """Perspective-warp the 4-corner polygon (handles rotated/skewed photos)."""
    rect = order_points(points)
    tl, tr, br, bl = rect
    width_a  = np.linalg.norm(br - bl)
    width_b  = np.linalg.norm(tr - tl)
    max_width = max(1, int(max(width_a, width_b)))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(1, int(max(height_a, height_b)))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height), borderValue=(255, 255, 255))


def rotate_quarter_turns(image, turns):
    turns = int(turns) % 4
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    if turns == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def predict_orientation(image):
    session = load_orientation_model()
    if session is None:
        # Common case on machines where the server runs an unsupported Python
        # (e.g., 3.14) and onnxruntime can't be imported. Use a subprocess that
        # runs inside OBJECT_ORIENTATION_DETECTOR/venv instead.
        return predict_orientation_subprocess(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((384, 384), Image.BILINEAR)
    arr = np.asarray(pil).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    outputs = session.run(None, {ORIENTATION_INPUT_NAME: arr})
    logits = np.asarray(outputs[0]).squeeze()
    if logits.ndim != 1:
        logits = logits.reshape(-1)
    logits = logits - np.max(logits)
    probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-12)
    pred = int(np.argmax(probs))
    labels = ["0deg", "90deg", "180deg", "270deg"]
    correction_turns = {0: 0, 1: 1, 2: 2, 3: 3}[pred]
    return {
        "label":      labels[pred],
        "confidence": float(probs[pred]),
        "turns":      correction_turns,
        "error":      None,
    }


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            path = "/index.html"
        target = WEB_ROOT / unquote(path.lstrip("/"))
        if not target.exists() or not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/exports/"):
            return self.serve_file(EXPORT_ROOT / unquote(parsed.path.removeprefix("/exports/")))
        if parsed.path.startswith("/faces/"):
            return self.serve_file(FACE_ROOT / unquote(parsed.path.removeprefix("/faces/")))
        path = parsed.path
        if path == "/":
            path = "/index.html"
        return self.serve_file(WEB_ROOT / unquote(path.lstrip("/")))

    def do_POST(self):
        try:
            if self.path == "/api/detect":
                return self.detect()
            if self.path == "/api/crop":
                return self.crop()
            if self.path == "/api/faces":
                return self.group_faces()
            if self.path == "/api/faces_session":
                return self.group_faces_session()
            if self.path == "/api/save_all":
                return self.save_all_crops()
            if self.path == "/api/save_faces":
                return self.save_face_folders()
            if self.path == "/api/pick_folder":
                return self.pick_folder()
            json_response(self, {"error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, 500)

    def serve_file(self, path):
        if not path.exists() or not path.is_file():
            json_response(self, {"error": "Not found"}, 404)
            return
        mime = "text/plain"
        if path.suffix == ".html":
            mime = "text/html"
        elif path.suffix == ".css":
            mime = "text/css"
        elif path.suffix == ".js":
            mime = "text/javascript"
        elif path.suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif path.suffix == ".png":
            mime = "image/png"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def detect(self):
        payload = read_json(self)
        image = decode_image(payload["image"])
        image_id = f"{int(time.time())}_{uuid.uuid4().hex}"
        cv2.imwrite(str(UPLOAD_ROOT / f"{image_id}.jpg"), image)
        threshold      = int(payload.get("threshold", 240))
        min_area_ratio = float(payload.get("minAreaRatio", 0.012))
        padding        = int(payload.get("padding", 0))
        boxes = detect_photos(image, threshold, min_area_ratio, padding)
        json_response(
            self,
            {
                "imageId": image_id,
                "width":   int(image.shape[1]),
                "height":  int(image.shape[0]),
                "boxes":   boxes,
                "modelStatus": {
                    "orientation":  "available" if (load_orientation_model() or orientation_subprocess_available()) else "unavailable",
                    "faceGrouping": "planned",
                    "upscale":      "planned",
                },
            },
        )

    def crop(self):
        payload = read_json(self)
        image_id = payload.get("imageId") or f"{int(time.time())}_{uuid.uuid4().hex}"
        path = UPLOAD_ROOT / f"{image_id}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            image = decode_image(payload["image"])
            cv2.imwrite(str(path), image)

        export_dir = EXPORT_ROOT / image_id
        export_dir.mkdir(parents=True, exist_ok=True)

        results = []
        auto_orientation = bool(payload.get("autoOrientation", False))
        for index, box in enumerate(payload.get("boxes", []), start=1):
            # crop_perspective handles both straight and skewed polygons
            crop = crop_perspective(image, box["points"])
            orientation = None
            if auto_orientation:
                orientation = predict_orientation(crop)
                crop = rotate_quarter_turns(crop, orientation["turns"])
            crop = rotate_quarter_turns(crop, box.get("rotation", 0))
            filename = f"photo_{index:03}.jpg"
            out_path = export_dir / filename
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            results.append({
                "id":          box.get("id", str(uuid.uuid4())),
                "filename":    filename,
                "url":         f"/exports/{image_id}/{filename}",
                "preview":     encode_image(crop),
                "width":       int(crop.shape[1]),
                "height":      int(crop.shape[0]),
                "orientation": orientation,
            })

        json_response(self, {
            "imageId":    image_id,
            "results":    results,
            "exportPath": str(export_dir),
        })

    def group_faces(self):
        payload = read_json(self)
        export_id = payload.get("exportId", "")
        threshold = float(payload.get("threshold", 0.35))
        margin    = float(payload.get("margin", 0.35))
        if not export_id:
            return json_response(self, {"error": "exportId required"}, 400)
        export_dir = EXPORT_ROOT / export_id
        if not export_dir.is_dir():
            return json_response(self, {"error": f"Export not found: {export_id}"}, 404)
        face_out = FACE_ROOT / export_id
        try:
            proc = subprocess.run(
                [FACE_PYTHON, str(FACE_SCRIPT), str(export_dir), str(face_out),
                 str(threshold), str(margin)],
                capture_output=True, text=True, timeout=300,
            )
            if proc.stderr:
                print(f"[face_processor] {proc.stderr.strip()}", file=sys.stderr)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                return json_response(self, {"error": err[-1] if err else "face_processor failed"}, 500)
            # InsightFace prints loader messages to stdout; JSON is always the last line
            last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            data = json.loads(last_line)
        except subprocess.TimeoutExpired:
            return json_response(self, {"error": "Face processing timed out"}, 500)
        except Exception as exc:
            return json_response(self, {"error": str(exc)}, 500)

        # attach public URLs
        for person in data.get("persons", []):
            for face in person.get("faces", []):
                face["url"] = f"/faces/{export_id}/{face['file']}"
        json_response(self, data)

    def group_faces_session(self):
        payload = read_json(self)
        export_ids = payload.get("exportIds") or []
        threshold = float(payload.get("threshold", 0.35))
        margin    = float(payload.get("margin", 0.35))

        if not isinstance(export_ids, list) or not export_ids:
            return json_response(self, {"error": "exportIds must be a non-empty list"}, 400)

        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex}"
        session_export_dir = SESSION_EXPORT_ROOT / session_id
        session_export_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for export_id in export_ids:
            if not isinstance(export_id, str) or not export_id.strip():
                continue
            export_dir = EXPORT_ROOT / export_id
            if not export_dir.is_dir():
                continue
            for file in export_dir.iterdir():
                if file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                dest = session_export_dir / f"{export_id}__{file.name}"
                if dest.exists():
                    continue
                try:
                    os.link(file, dest)
                except Exception:
                    shutil.copy2(file, dest)
                copied += 1

        if copied == 0:
            return json_response(self, {"error": "No exported images found for this session"}, 400)

        face_out = FACE_ROOT / session_id
        try:
            proc = subprocess.run(
                [FACE_PYTHON, str(FACE_SCRIPT), str(session_export_dir), str(face_out),
                 str(threshold), str(margin)],
                capture_output=True, text=True, timeout=600,
            )
            if proc.stderr:
                print(f"[face_processor] {proc.stderr.strip()}", file=sys.stderr)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                return json_response(self, {"error": err[-1] if err else "face_processor failed"}, 500)
            last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            data = json.loads(last_line)
        except subprocess.TimeoutExpired:
            return json_response(self, {"error": "Face processing timed out"}, 500)
        except Exception as exc:
            return json_response(self, {"error": str(exc)}, 500)

        for person in data.get("persons", []):
            for face in person.get("faces", []):
                face["url"] = f"/faces/{session_id}/{face['file']}"

        data["sessionId"] = session_id
        data["total_images"] = copied
        json_response(self, data)

    def save_all_crops(self):
        payload = read_json(self)
        export_ids = payload.get("exportIds") or []
        dest_dir = (payload.get("destDir") or "").strip()
        if not isinstance(export_ids, list) or not export_ids:
            return json_response(self, {"error": "exportIds must be a non-empty list"}, 400)
        if not dest_dir:
            return json_response(self, {"error": "destDir required"}, 400)

        parent = _ensure_dir(Path(dest_dir))
        out_dir = parent / f"photo_studio_all_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for export_id in export_ids:
            if not isinstance(export_id, str) or not export_id.strip():
                continue
            export_dir = EXPORT_ROOT / export_id
            if not export_dir.is_dir():
                continue
            for file in export_dir.iterdir():
                if file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                dest = out_dir / f"{export_id}__{file.name}"
                if dest.exists():
                    continue
                shutil.copy2(file, dest)
                saved += 1

        json_response(self, {"outDir": str(out_dir), "saved": saved})

    def save_face_folders(self):
        payload = read_json(self)
        session_id = (payload.get("sessionId") or "").strip()
        dest_dir = (payload.get("destDir") or "").strip()
        top_k = int(payload.get("topK", 10))
        min_photos = int(payload.get("minPhotos", 2))
        names = payload.get("names") or {}

        if not session_id:
            return json_response(self, {"error": "sessionId required"}, 400)
        if not dest_dir:
            return json_response(self, {"error": "destDir required"}, 400)
        if top_k < 1:
            top_k = 1
        if min_photos < 1:
            min_photos = 1

        group_file = FACE_ROOT / session_id / "groups.json"
        if not group_file.exists():
            return json_response(self, {"error": f"groups.json not found for sessionId: {session_id}"}, 404)

        session_export_dir = SESSION_EXPORT_ROOT / session_id
        if not session_export_dir.is_dir():
            return json_response(self, {"error": f"session export dir not found: {session_id}"}, 404)

        parent = _ensure_dir(Path(dest_dir))
        out_dir = parent / f"photo_studio_faces_{session_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        data = json.loads(group_file.read_text())
        persons = data.get("persons") or []
        persons = [p for p in persons if int(p.get("count", 0)) >= min_photos]
        persons.sort(key=lambda p: -int(p.get("count", 0)))
        persons = persons[:top_k]

        saved = 0
        folders = 0

        for p in persons:
            pid = p.get("id") or "person"
            raw_name = None
            if isinstance(names, dict):
                raw_name = names.get(pid)
            folder_name = _sanitize_folder_name(raw_name or pid)
            person_dir = out_dir / folder_name
            person_dir.mkdir(parents=True, exist_ok=True)
            folders += 1

            seen_sources = set()
            for face in (p.get("faces") or []):
                src_name = face.get("source")
                if not src_name or src_name in seen_sources:
                    continue
                seen_sources.add(src_name)
                src_path = session_export_dir / src_name
                if not src_path.exists():
                    continue
                dest = person_dir / src_name
                if dest.exists():
                    continue
                shutil.copy2(src_path, dest)
                saved += 1

        json_response(self, {"outDir": str(out_dir), "saved": saved, "folders": folders})

    def pick_folder(self):
        # macOS-only: open a native Finder folder picker.
        if sys.platform != "darwin":
            return json_response(self, {"error": "Folder picker is only supported on macOS"}, 400)
        try:
            script = 'POSIX path of (choose folder with prompt "Choose destination folder")'
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                if "User canceled" in msg or "cancel" in msg.lower():
                    return json_response(self, {"error": "Cancelled"}, 400)
                return json_response(self, {"error": msg or "Folder picker failed"}, 500)
            path = (proc.stdout or "").strip()
            if path.endswith("/"):
                path = path[:-1]
            return json_response(self, {"path": path})
        except subprocess.TimeoutExpired:
            return json_response(self, {"error": "Folder picker timed out"}, 500)


def main():
    port = int(os.environ.get("PORT", "5173"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Photo Studio  →  http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

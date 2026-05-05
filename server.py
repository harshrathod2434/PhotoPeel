import base64
import json
import os
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
ORIENTATION_ROOT = Path("/Users/kriparathod/Desktop/6th sem/CV/OBJECT_ORIENTATION_DETECTOR")
ORIENTATION_SITE_PACKAGES = (
    ORIENTATION_ROOT / "venv" / "lib" / "python3.9" / "site-packages"
)
ORIENTATION_MODEL = ORIENTATION_ROOT / "orientation_model_v2_0.9882.onnx"
ORIENTATION_SESSION = None
ORIENTATION_INPUT_NAME = None
ORIENTATION_ERROR = None

for folder in (UPLOAD_ROOT, EXPORT_ROOT):
    folder.mkdir(parents=True, exist_ok=True)


def load_orientation_model():
    global ORIENTATION_SESSION, ORIENTATION_INPUT_NAME, ORIENTATION_ERROR
    if ORIENTATION_SESSION is not None or ORIENTATION_ERROR is not None:
        return ORIENTATION_SESSION
    try:
        if str(ORIENTATION_SITE_PACKAGES) not in sys.path:
            sys.path.append(str(ORIENTATION_SITE_PACKAGES))
        import onnxruntime as ort
        ORIENTATION_SESSION = ort.InferenceSession(str(ORIENTATION_MODEL))
        ORIENTATION_INPUT_NAME = ORIENTATION_SESSION.get_inputs()[0].name
        return ORIENTATION_SESSION
    except Exception as exc:
        ORIENTATION_ERROR = str(exc)
        return None


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


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
        return {"label": "unavailable", "confidence": 0.0, "turns": 0, "error": ORIENTATION_ERROR}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((384, 384), Image.BILINEAR)
    arr = np.asarray(pil).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    outputs = session.run(None, {ORIENTATION_INPUT_NAME: arr})
    logits = outputs[0][0]
    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred = int(np.argmax(probs))
    labels = ["0deg", "90deg", "180deg", "270deg"]
    correction_turns = {0: 0, 1: 3, 2: 2, 3: 1}[pred]
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
                    "orientation":  "available" if load_orientation_model() else "unavailable",
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


def main():
    port = int(os.environ.get("PORT", "5173"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Photo Studio  →  http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

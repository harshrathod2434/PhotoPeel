"""Process scanned images to detect and crop multiple photos.

This module provides functionality to detect and crop multiple photos
from scanned images. It handles white borders and supports multi-threading
for faster processing.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process scanned images to detect and crop multiple photos."
    )
    parser.add_argument(
        "--input-folder",
        default=os.getenv("INPUT_FOLDER", "raw"),
        help="Input folder containing scanned images (default: raw)",
    )
    parser.add_argument(
        "--output-folder",
        default=os.getenv("OUTPUT_FOLDER", "output_images"),
        help="Output folder for cropped images (default: output_images)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("THREADS", "1")),
        help="Number of processing threads (default: 1)",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=int(os.getenv("THRESHOLD_VALUE", "240")),
        help="Threshold value for image processing (default: 240)",
    )
    parser.add_argument(
        "--threshold-max",
        type=int,
        default=int(os.getenv("THRESHOLD_MAX", "255")),
        help="Maximum threshold value (default: 255)",
    )
    parser.add_argument(
        "--min-contour-width",
        type=int,
        default=int(os.getenv("MIN_CONTOUR_WIDTH", "50")),
        help="Minimum contour width to process (default: 50)",
    )
    parser.add_argument(
        "--min-contour-height",
        type=int,
        default=int(os.getenv("MIN_CONTOUR_HEIGHT", "50")),
        help="Minimum contour height to process (default: 50)",
    )
    parser.add_argument(
        "--allowed-extensions",
        default=os.getenv("ALLOWED_EXTENSIONS", ".png,.jpg,.jpeg"),
        help="Comma-separated list of allowed file extensions",
    )
    parser.add_argument(
        "--morph-kernel-size",
        type=int,
        default=int(os.getenv("MORPH_KERNEL_SIZE", "20")),
        help="Morphological closing kernel size in pixels (default: 20).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=int(os.getenv("PADDING", "10")),
        help="Extra pixels added around each detected photo crop (default: 10).",
    )

    args = parser.parse_args()
    args.allowed_extensions = tuple(args.allowed_extensions.split(","))
    return args


def _build_foreground_mask(
    image: "cv2.Mat", threshold_value: int, threshold_max: int
) -> "cv2.Mat":
    """Return a binary mask: photo pixels white, scanner background black.

    Color scans: scanner bed is grey (near-zero saturation). Real photos are
    colorful, so thresholding HSV saturation isolates them without overlap.

    B&W scans: saturation is zero everywhere; fall back to brightness threshold
    (dark content on bright scanner background).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())

    if mean_saturation > 8:
        sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
        _, mask = cv2.threshold(sat, 15, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.GaussianBlur(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0
        )
        _, thresh = cv2.threshold(gray, threshold_value, threshold_max, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(thresh)

    return mask


def _projection_segments(
    proj: np.ndarray, smooth_px: int = 25, valley_ratio: float = 0.12
) -> List[Tuple[int, int]]:
    """Return (start, end) index pairs of segments above the valley threshold.

    Smooths the projection first so small dips from photo content don't create
    false splits. A valley is any region below valley_ratio * max value.
    """
    if proj.max() == 0:
        return []
    kernel = np.ones(smooth_px, dtype=float) / smooth_px
    smoothed = np.convolve(proj, kernel, mode="same")
    cutoff = smoothed.max() * valley_ratio
    above = smoothed > cutoff

    segments: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(above)))
    return segments


def _split_bbox(
    mask: "cv2.Mat",
    x: int,
    y: int,
    w: int,
    h: int,
    min_w: int,
    min_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Try to split a bounding box using row/column saturation projections.

    When two touching photos are merged into one contour, a valley appears in
    the projection of the saturation mask at the junction. We find that valley
    and emit one sub-bbox per segment.

    Tries a horizontal split first (photos side by side), then vertical (stacked).
    Returns the original single bbox if no valid split is found.
    """
    sub = mask[y : y + h, x : x + w].astype(float)

    # --- horizontal split: valley in column-wise projection ---
    h_segs = _projection_segments(sub.mean(axis=0))
    if len(h_segs) > 1:
        crops = [
            (x + s, y, e - s, h)
            for s, e in h_segs
            if e - s >= min_w
        ]
        if len(crops) > 1:
            return crops

    # --- vertical split: valley in row-wise projection ---
    v_segs = _projection_segments(sub.mean(axis=1))
    if len(v_segs) > 1:
        crops = [
            (x, y + s, w, e - s)
            for s, e in v_segs
            if e - s >= min_h
        ]
        if len(crops) > 1:
            return crops

    return [(x, y, w, h)]


def remove_white_borders(
    image_path: str,
    output_folder: str,
    threshold_value: int,
    threshold_max: int,
    min_contour_width: int,
    min_contour_height: int,
    morph_kernel_size: int = 20,
    padding: int = 10,
) -> None:
    """Detect and crop individual photos from a scanned image.

    Pipeline:
      1. Foreground mask  — saturation-based for color scans; brightness for B&W
      2. Morphological closing — fills small internal gaps within each photo
      3. Contour detection — finds candidate photo regions
      4. Projection split — if two touching photos merged into one contour,
                            find the valley in the saturation profile and split
      5. Bounding-box crop + pad — saves each photo with a small margin
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return

    img_h, img_w = image.shape[:2]
    mask = _build_foreground_mask(image, threshold_value, threshold_max)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print(f"Warning: No contours found in {image_path}")
        return

    contours = sorted(
        contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])
    )

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_contour_width or h < min_contour_height:
            continue

        # Attempt to split merged adjacent photos via projection valleys
        sub_bboxes = _split_bbox(mask, x, y, w, h, min_contour_width, min_contour_height)

        for cx, cy, cw, ch in sub_bboxes:
            x1 = max(0, cx - padding)
            y1 = max(0, cy - padding)
            x2 = min(img_w, cx + cw + padding)
            y2 = min(img_h, cy + ch + padding)

            cropped_image = image[y1:y2, x1:x2]
            output_path = os.path.join(
                output_folder, f"{base_filename}_{count}.jpg"
            )
            cv2.imwrite(output_path, cropped_image)
            print(f"Saved: {output_path}")
            count += 1


def process_images(
    output_folder: str,
    input_folder: str,
    allowed_extensions: Tuple[str],
) -> List[str]:
    os.makedirs(output_folder, exist_ok=True)
    return [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(allowed_extensions)
    ]


def main() -> None:
    args = parse_args()
    image_files = process_images(
        output_folder=args.output_folder,
        input_folder=args.input_folder,
        allowed_extensions=args.allowed_extensions,
    )
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        executor.map(
            lambda x: remove_white_borders(
                x,
                args.output_folder,
                args.threshold_value,
                args.threshold_max,
                args.min_contour_width,
                args.min_contour_height,
                args.morph_kernel_size,
                args.padding,
            ),
            image_files,
        )


if __name__ == "__main__":
    main()

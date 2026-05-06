# PhotoPeel

PhotoPeel is a local, browser-based tool for cropping scanned photo sheets. It detects multiple photos per scan, fixes skew, and can auto-rotate and group faces for batch export.

## Features
- Detect and crop multiple photos from a single scanned image (including skewed photos).
- Auto orientation using an ONNX model (optional).
- Face grouping across crops using InsightFace.
- Export crops and grouped face folders.

## Requirements
- Python 3.11+ recommended (InsightFace runs best on 3.11).
- Packages: opencv-python, numpy, pillow, onnxruntime, insightface.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install opencv-python numpy pillow onnxruntime insightface
```

### Optional: Orientation model
Auto orientation expects the model folder at:
```
../OBJECT_ORIENTATION_DETECTOR/orientation_model_v2_0.9882.onnx
```
If you keep the model elsewhere, update the paths in server.py.

## Run
From this repo root:
```bash
PORT=5173 python server.py
```
Then open:
```
http://localhost:5173
```

## Usage
1. Click "Select scan images" and choose one or more scans.
2. Adjust detection sliders and click "Detect Photos" or "Detect All Scans".
3. Review crops, rotate if needed, then click "Generate Results".
4. Use "Group Faces" to cluster identities, then "Save Face Folders".

## Data output
Exports and intermediate files are stored under appdata/ (ignored by git).

## Troubleshooting
- Face grouping error about ml_dtypes: `python -m pip install -U ml_dtypes`.
- Auto orientation unavailable: confirm the ONNX model path or disable Auto orientation in the UI.

# Drone Tracker Setup - COMPLETE ✓

**Date:** December 16, 2025  
**Status:** Ready for video processing

## What Was Done

### 1. Environment Setup (Miniconda)
- ✓ Miniconda3 installed at `C:\Users\chipn\Miniconda3`
- ✓ Conda environment `drone39` created with Python 3.9
- ✓ All dependencies installed:
  - `ultralytics==8.1.0` (YOLOv8 framework)
  - `torch` (CPU version, 2.8.0)
  - `torchvision`, `torchaudio` (CPU versions)
  - `opencv-python` (video I/O)
  - `numpy`, `scipy`, `scikit-learn`, `scikit-image`
  - `filterpy` (Kalman Filter)
  - `tqdm` (progress bars)
  - `Pillow` (image processing)

### 2. Project Structure
```
e:\kshatra\drone-tracker/
├── src/
│   ├── __init__.py          ✓
│   ├── main.py              ✓ (entry point with unsafe load patch)
│   ├── tracker.py           ✓ (Track, Tracker classes)
│   ├── kalman_filter.py     ✓ (KF creation, bbox conversions)
│   └── utils.py             ✓ (IoU, NMS utilities)
├── models/
│   └── (yolov8n.pt will be here after first run)
├── videos/                  ← Place your test videos here
├── outputs/                 ← Results will be saved here
├── requirements.txt         ✓ (pinned dependencies)
├── README.md                ✓ (detailed guide)
├── track.py                 ✓ (legacy single-file version)
└── SETUP_COMPLETE.md        ← This file
```

### 3. Core Implementation

#### **main.py** (Entry Point)
- Loads YOLOv8n model with unsafe-load patch (allows weights_only=False)
- Processes videos from `videos/` folder
- Uses Kalman Filter tracker for object association
- Saves outputs to `outputs/`:
  - `tracked_<video>.mp4` with bounding boxes and track IDs
  - `tracked_<video>.csv` with frame-by-frame tracking data

#### **tracker.py** (Tracking Logic)
- `Track` class: Individual object track with Kalman Filter
- `Tracker` class: Multi-object tracker using Hungarian algorithm
- IoU-based data association
- Configurable parameters: max_age (30 frames), min_hits (1), iou_threshold (0.3)

#### **kalman_filter.py** (Motion Prediction)
- 7D state vector: [cx, cy, s, r, vx, vy, vs] (center, scale, aspect, velocities)
- 4D measurement: [cx, cy, s, r]
- Handles frame-to-frame prediction and Bayesian update

#### **utils.py** (Helper Functions)
- IoU (Intersection over Union) calculation
- NMS (Non-Maximum Suppression) for duplicate detection removal

### 4. Model & Weights
- YOLOv8n (Nano) is **6.3M parameters**, optimized for speed
- Auto-downloads on first run to current directory
- Unsafe load enabled (patch applied in main.py for torch 2.8+ compatibility)

## How to Use (Quick Reference)

### Step 1: Activate Environment
```powershell
# First time only (initialize conda):
& "$env:UserProfile\Miniconda3\Scripts\conda.exe" init powershell

# Then restart PowerShell and activate:
conda activate drone39
cd e:\kshatra\drone-tracker
```

### Step 2: Place Your Videos
Copy your 3 test videos into `videos/` folder:
```
e:\kshatra\drone-tracker\videos\
├── video1.mp4
├── video2.mp4
└── video3.mp4
```
Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

### Step 3: Run the Tracker
```powershell
python -m src.main --input-dir videos --output-dir outputs --weights yolov8n.pt --conf 0.35
```

### Step 4: Check Results
Results will be in `e:\kshatra\drone-tracker\outputs/`:
```
outputs/
├── tracked_video1.mp4      (annotated video)
├── tracked_video1.csv      (tracking data)
├── tracked_video2.mp4
├── tracked_video2.csv
└── ...
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | `videos` | Folder containing input videos |
| `--output-dir` | `outputs` | Folder to save results |
| `--weights` | `yolov8n.pt` | YOLO model weights file |
| `--conf` | `0.35` | Detection confidence (0.0-1.0) |

**Examples:**
```powershell
# Higher confidence, fewer detections:
python -m src.main --conf 0.5

# Lower confidence, more detections:
python -m src.main --conf 0.2

# Use larger model (if available):
python -m src.main --weights yolov8m.pt
```

## Performance Expectations

| Hardware | Expected FPS |
|----------|-------------|
| CPU (modern, 4+ cores) | 5-15 FPS |
| GPU (GTX 1080+) | 30-60 FPS |
| Jetson Nano (baseline) | 2-5 FPS |
| Jetson Nano (TensorRT) | 15-40 FPS |

*Note: FPS depends on video resolution and number of objects.*

## CSV Output Format

Each `tracked_<video>.csv` contains:
```
frame,id,x1,y1,x2,y2
1,1,150,200,350,400
1,2,450,150,600,350
2,1,152,202,352,402
2,2,452,152,602,352
...
```

**Columns:**
- `frame`: Frame number (1-indexed)
- `id`: Unique track ID (assigned at first detection, persistent across frames)
- `x1, y1`: Top-left corner of bounding box
- `x2, y2`: Bottom-right corner of bounding box

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
**Solution:** Make sure you're in `e:\kshatra\drone-tracker` when running and use:
```powershell
python -m src.main
```
(NOT `python src/main.py`)

### "No video files found in videos"
**Solution:** Ensure videos are in the correct folder with supported extensions:
```powershell
# Check what's there:
dir e:\kshatra\drone-tracker\videos
# Should show your .mp4, .avi, .mov, .mkv files
```

### Script runs but no output or very slow
**Solution:** Check if GPU is available:
```powershell
conda activate drone39
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
- If `False`: Using CPU (slower but still works)
- If `True`: GPU is available; script should be faster

### "Weights only load failed" error
**Solution:** This is already handled by the unsafe-load patch in `main.py`. If it still occurs, the patch may need updating.

## Next Steps

1. **Place your 3 videos** in `e:\kshatra\drone-tracker\videos/`
2. **Activate environment** (if not already):
   ```powershell
   conda activate drone39
   ```
3. **Run the tracker**:
   ```powershell
   cd e:\kshatra\drone-tracker
   python -m src.main
   ```
4. **Check outputs** in `e:\kshatra\drone-tracker\outputs/`

## Files Generated

- `tracked_<video>.mp4` — Video with bounding boxes and track IDs drawn
- `tracked_<video>.csv` — Detailed tracking data (frame, track_id, bbox coords)

## Advanced Options

### Use CPU-Only (if GPU causes issues)
Environment already uses CPU PyTorch; no changes needed.

### Use a Different Model Size
Download the model first, then specify it:
```powershell
# Download (run once):
conda activate drone39
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Then use:
python -m src.main --weights yolov8m.pt
```

Model sizes: `yolov8n.pt` (nano, fastest), `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` (large, slowest, most accurate).

### Jetson Nano Deployment
For faster inference on Jetson Nano, convert to TensorRT:
```bash
# On Jetson (after setup):
conda activate drone39
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0, half=True, imgsz=480)
"
# Creates yolov8n.engine (smaller, 2-5x faster)

# Then run:
python -m src.main --weights yolov8n.engine
```

## Environment Details

```
Conda Environment: drone39
Python: 3.9.25
PyTorch: 2.8.0 (CPU)
YOLOv8: 8.1.0
OpenCV: 4.12.0.88
Filterpy: 1.4.5
Operating System: Windows 11
```

To verify, run:
```powershell
conda activate drone39
python -c "
import torch, ultralytics, cv2, filterpy
print(f'PyTorch: {torch.__version__}')
print(f'YOLOv8: {ultralytics.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'FilterPy: {filterpy.__version__}')
"
```

## Support & Documentation

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **FilterPy (Kalman Filter):** https://filterpy.readthedocs.io/
- **OpenCV:** https://docs.opencv.org/

---

**Setup completed successfully on 2025-12-16**  
Ready to process your drone videos! 🚁

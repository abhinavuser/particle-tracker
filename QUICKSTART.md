# 🚁 QUICK START GUIDE - Drone Tracker

## ⏱️ 60-Second Setup

Your drone tracker is **ready to use**. Here's what to do:

### Step 1: Add Your Videos (30 seconds)
Copy your 3 test videos to this folder:
```
e:\kshatra\drone-tracker\videos\
```

**Supported formats:** `.mp4` `.avi` `.mov` `.mkv`

Example:
```
videos/
├── drone_flight_1.mp4
├── drone_flight_2.mp4
└── drone_flight_3.mp4
```

### Step 2: Run the Tracker (30 seconds)

Open **PowerShell** and run:

```powershell
conda activate drone39
cd e:\kshatra\drone-tracker
python -m src.main
```

**That's it!** The script will:
1. Load YOLOv8n model (6.3M parameters)
2. Process each video from `videos/`
3. Track drone movement frame-by-frame using Kalman Filter
4. Save results to `outputs/`

### Step 3: Check Results

Results appear in:
```
e:\kshatra\drone-tracker\outputs\
├── tracked_drone_flight_1.mp4      ← Video with bounding boxes
├── tracked_drone_flight_1.csv      ← Tracking data (frame, ID, bbox)
├── tracked_drone_flight_2.mp4
├── tracked_drone_flight_2.csv
└── tracked_drone_flight_3.mp4
    tracked_drone_flight_3.csv
```

**MP4 files:** Watch these in any video player. Green boxes = detected drones, numbers = track IDs.

**CSV files:** Frame-by-frame data:
```
frame,id,x1,y1,x2,y2
1,1,150,200,350,400
2,1,152,202,352,402
3,2,450,150,600,350
...
```

---

## 🎛️ Optional: Customize Detection

Adjust detection sensitivity:

```powershell
# Higher confidence (fewer false positives):
python -m src.main --conf 0.5

# Lower confidence (catch more drones):
python -m src.main --conf 0.2

# Default:
python -m src.main --conf 0.35
```

---

## 📊 Performance Expectations

| Hardware | Speed |
|----------|-------|
| **CPU** (modern, 4+ cores) | 5-15 FPS |
| **GPU** (GTX 1080+) | 30-60 FPS |

*FPS = frames per second. Faster = smoother video processing.*

---

## ✅ Checklist Before Running

- [ ] Videos placed in `e:\kshatra\drone-tracker\videos/`
- [ ] Videos are readable (try playing them first)
- [ ] Conda installed (run `conda --version` to check)
- [ ] No other heavy processes running (frees up memory)

---

## 🐛 Troubleshooting in 30 Seconds

| Problem | Solution |
|---------|----------|
| "No video files found" | Check folder path: `e:\kshatra\drone-tracker\videos\` |
| "ModuleNotFoundError: src" | Make sure you're in `e:\kshatra\drone-tracker` folder |
| Very slow (1-2 FPS) | Normal on CPU. Try GPU if available, or reduce video resolution. |
| Script freezes on start | Model is downloading (~6MB). Takes ~30sec first run. |

---

## 📁 Project Structure (What You Need to Know)

```
e:\kshatra\drone-tracker/
├── src/                   ← Don't touch; tracking code
├── videos/                ← PUT YOUR VIDEOS HERE
├── outputs/               ← RESULTS APPEAR HERE
├── models/                ← (auto-filled with yolov8n.pt)
├── README.md              ← Detailed guide
├── SETUP_COMPLETE.md      ← Setup details
└── requirements.txt       ← Python packages (already installed)
```

---

## 🚀 Running for the First Time

```powershell
# 1. Activate environment
conda activate drone39

# 2. Go to project folder
cd e:\kshatra\drone-tracker

# 3. Run the tracker
python -m src.main

# Expected output:
# Loading YOLO model from yolov8n.pt...
# ✓ Model loaded successfully
# Found 3 video(s) to process:
#   - drone_flight_1.mp4
#   - drone_flight_2.mp4
#   - drone_flight_3.mp4
# 
# Processing drone_flight_1.mp4...
#   Frame 50 (12.3 fps)
#   Frame 100 (13.1 fps)
#   ...
# ✓ Finished drone_flight_1.mp4
#   Output video: outputs/tracked_drone_flight_1.mp4
#   Tracking CSV: outputs/tracked_drone_flight_1.csv
#   Total time: 45.23s, Avg FPS: 12.5
```

---

## 💾 Output Files Explained

**tracked_<video>.mp4**
- Your original video with green bounding boxes drawn
- Each box = detected drone
- Number on box = track ID (consistent across frames)
- Open with any video player (VLC, Windows Media Player, etc.)

**tracked_<video>.csv**
- Frame-by-frame tracking data
- Open with Excel or text editor
- Columns:
  - `frame`: Frame number (1, 2, 3, ...)
  - `id`: Track ID (1, 2, 3, ...)
  - `x1, y1`: Top-left corner of bounding box
  - `x2, y2`: Bottom-right corner of bounding box

---

## 📖 More Help

- **Full guide:** See `README.md`
- **Setup details:** See `SETUP_COMPLETE.md`
- **Jetson Nano:** See "Jetson Nano Deployment" in `README.md`

---

**Ready?** Copy your videos to `videos/` and run:
```powershell
conda activate drone39
cd e:\kshatra\drone-tracker
python -m src.main
```

Good luck! 🚁✨

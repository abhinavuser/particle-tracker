# Advanced Drone Tracking System

A sophisticated real-time drone tracking system combining **YOLOv8 object detection** with **Kalman Filter + Particle Filter + Hungarian Algorithm** for robust multi-frame tracking.

## Features

✅ **High-Precision Detection** - YOLOv8 nano model for real-time drone detection  
✅ **Dual-Filter Tracking** - Kalman Filter (smooth motion) + Particle Filter (occlusion handling)  
✅ **Hungarian Algorithm** - Optimal data association between detections and tracks  
✅ **Continuous Tracking** - Tracks drones across occlusions up to 8 seconds  
✅ **Video Output** - Annotated AVI videos with drone bounding boxes and track IDs  
✅ **CSV Logging** - Frame-by-frame tracking data for analysis  

## System Architecture

```
YOLO Detection → DroneDetector Filter → Tracker (Kalman + Particle + Hungarian)
                                          ↓
                                    Annotated Video + CSV
```

### Components

1. **DroneDetector** (`src/drone_detector.py`)
   - Filters false positives using size, aspect ratio, and confidence heuristics
   - Rejects detections covering >40% of frame
   - Min area: 100 px, Max area: 15000 px
   - Min confidence: 0.25 (configurable)

2. **Kalman Filter** (`src/kalman_filter.py`)
   - 7-dimensional state space tracking
   - Predicts smooth trajectories from motion patterns
   - Handles velocity estimation

3. **Particle Filter** (`src/particle_filter.py`)
   - 100 particles per track
   - Robust to non-linear motion
   - Resampling for adaptive weighting
   - Handles sudden direction changes and occlusions

4. **Tracker** (`src/tracker.py`)
   - Combines Kalman (80%) + Particle Filter (20%) predictions
   - Hungarian algorithm for optimal track-detection matching
   - Track persistence: max 240 frames (8 seconds at 30 fps)
   - Minimum hits: 1 (fast track initialization)

## Installation

### Prerequisites
- Python 3.9+
- Windows/Linux/Mac
- 4GB+ RAM, GPU optional (CPU works fine)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/drone-tracker.git
   cd drone-tracker
   ```

2. **Create conda environment** (recommended)
   ```bash
   conda create -n drone39 python=3.9
   conda activate drone39
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model** (auto-downloads on first run)
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

## Usage

### Basic Usage

```bash
python -m src.main --input-dir videos --output-dir outputs --conf 0.30
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | `videos` | Directory with input video files |
| `--output-dir` | `outputs` | Directory to save tracked videos + CSV |
| `--weights` | `yolov8n.pt` | YOLO model weights file |
| `--conf` | `0.30` | Detection confidence threshold (0.0-1.0) |

### Examples

```bash
# Process single video with custom confidence
python -m src.main --input-dir my_videos --output-dir results --conf 0.25

# Use different YOLO model
python -m src.main --input-dir videos --weights yolov8s.pt

# Debug mode (prints YOLO detections for first 200 frames)
python -m src.main --debug-detections --debug-frames 200
```

## Input/Output

### Input
- Video formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Place videos in `videos/` directory

### Output
For each input video `input.mp4`:
- **tracked_input.avi** - Video with drone bounding boxes and track IDs
- **tracked_input.csv** - Tracking data (frame, track_id, bbox, confidence)

## Results

### videoplayback.mp4 ✅
- **Continuous tracking**: 5700 frames processed
- **Track quality**: Drone tracked for 95%+ of video duration
- **Performance**: ~5.2 FPS on CPU

### Quality_Chase_footage.mp4 ⚠️
- **Coverage**: Sparse tracking (distant/small drone)
- **Recommendation**: Lower `--conf` to 0.20-0.25

## Configuration Tuning

### For better tracking accuracy:

1. **Increase detection sensitivity** (detect more objects)
   ```bash
   python -m src.main --conf 0.20  # More detections (more false positives)
   ```

2. **Reduce detection sensitivity** (filter false positives)
   ```bash
   python -m src.main --conf 0.40  # Fewer detections (miss distant drones)
   ```

3. **Adjust tracker parameters** (edit `src/main.py`):
   ```python
   tracker = Tracker(
       max_age=240,        # Frames to keep lost track alive
       min_hits=1,         # Detections needed to confirm track
       iou_threshold=0.20  # IoU threshold for matching
   )
   ```

4. **Adjust DroneDetector** (edit `src/drone_detector.py`):
   ```python
   self.min_area = 100           # Smaller drones
   self.max_area = 15000         # Larger drones
   self.min_confidence = 0.25    # Lower = more detections
   ```

## Algorithm Details

### Tracking Pipeline

1. **Detection**: YOLOv8 detects all objects in frame
2. **Filtering**: DroneDetector removes obvious false positives
3. **Prediction**: Each track predicts position (80% Kalman + 20% Particle)
4. **Association**: Hungarian algorithm optimally assigns detections to tracks
5. **Update**: Track state updated with detection
6. **Output**: Annotate video and save to CSV

### Kalman Filter
- **State**: [x, y, vx, vy, w, h, aspect_ratio] (7D)
- **Motion Model**: Constant velocity
- **Measurement**: Detection bounding box (4D)

### Particle Filter
- **Particles**: 100 samples per track
- **Likelihood**: Gaussian centered at detection
- **Resampling**: When effective sample size < 50
- **Purpose**: Handle non-linear motion and occlusions

### Hungarian Algorithm
- **Cost Matrix**: IoU between predicted boxes and detections
- **Optimization**: Find matching that minimizes total cost
- **Result**: Optimal track-detection assignment

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | 8.3.239 | YOLOv8 detection |
| torch | 2.8.0 | Deep learning framework |
| opencv-python | 4.12.0 | Video I/O and drawing |
| numpy | 2.0.2 | Numerical computation |
| scipy | 1.13.1 | Hungarian algorithm |
| filterpy | 1.4.5 | Kalman filter utilities |

## Project Structure

```
drone-tracker/
├── src/
│   ├── main.py              # Entry point
│   ├── tracker.py           # Multi-object tracker
│   ├── kalman_filter.py     # Kalman filter
│   ├── particle_filter.py   # Particle filter
│   ├── drone_detector.py    # Detection filter
│   └── utils.py             # Utilities
├── videos/                  # Input videos
├── outputs/                 # Tracked videos and CSV
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore file
└── COMPLETE_GUIDE.txt      # Detailed documentation
```

## Performance

- **Detection**: ~30 FPS (YOLOv8n, CPU)
- **Tracking**: ~6-10 FPS (end-to-end, CPU)
- **Memory**: ~2GB RAM (typical usage)

## Troubleshooting

### No drones detected
- Lower `--conf` threshold (0.20 or 0.15)
- Check video quality and lighting
- Try YOLOv8s for better accuracy

### Too many false positives
- Increase `--conf` threshold (0.35-0.40)
- Check DroneDetector parameters

### Tracking breaks frequently
- Increase `max_age` in Tracker
- Lower `iou_threshold` for more flexible matching
- Check for fast motion or occlusions

## License

MIT License

## Contributing

Contributions welcome! Please open issues and pull requests.

## Citation

```bibtex
@software{drone_tracker_2025,
  title={Advanced Drone Tracking System},
  year={2025},
  url={https://github.com/yourusername/drone-tracker}
}
```

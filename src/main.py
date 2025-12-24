"""
Main tracker script for processing videos with smart drone detection.
"""
import os
import sys
import cv2
import time
import argparse

# Allow unsafe YOLO load
import torch
import ultralytics.nn.tasks

# Monkey-patch torch_safe_load to allow weights_only=False
from ultralytics.nn import tasks as ultralytics_tasks
original_torch_safe_load = ultralytics_tasks.torch_safe_load

def patched_torch_safe_load(file):
    """Patched version that loads with weights_only=False."""
    return torch.load(file, map_location="cpu", weights_only=False), file

ultralytics_tasks.torch_safe_load = patched_torch_safe_load

from ultralytics import YOLO
from .tracker import Tracker
from .drone_detector import DroneDetector


def run_on_video(input_path, output_path, model, tracker, drone_detector, conf=0.35,
                 debug_detections=False, debug_frames=200):
    """
    Process a single video file with smart drone detection.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        model: YOLO model
        tracker: Tracker instance
        drone_detector: DroneDetector instance
        conf: Detection confidence threshold
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'ERROR: Could not open {input_path}')
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use MJPEG codec (compatible and efficient)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # Change file extension to .avi for MJPEG
    avi_path = output_path.replace('.mp4', '.avi')
    out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
    output_path = avi_path
    
    if not out.isOpened():
        print(f'ERROR: Could not create output video writer')
        cap.release()
        return
    
    frame_idx = 0
    start = time.time()
    csv_lines = ["frame,id,x1,y1,x2,y2,confidence"]
    
    print(f'Processing {os.path.basename(input_path)}...')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Run YOLO detection
        results = model(frame, conf=conf, verbose=False)
        boxes = []
        confidences = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, cls_id = box[0], box[1], box[2], box[3], box[4], box[5]
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                confidences.append(float(score))

        # Debug: print raw YOLO detections for first N frames if enabled
        if debug_detections and frame_idx <= debug_frames:
            if boxes:
                print(f"DEBUG: Frame {frame_idx} - {len(boxes)} raw detection(s)")
                for i, (b, c) in enumerate(zip(boxes, confidences)):
                    bx = [round(v, 1) for v in b]
                    print(f"  DET {i}: bbox={bx} conf={c:.3f}")
            else:
                print(f"DEBUG: Frame {frame_idx} - no detections")
        
        # SMART DRONE FILTERING - choose the single primary detection per frame
        # We pick the largest/confident drone detection and track that alone to keep a focused track
        primary_bbox, primary_conf = drone_detector.get_largest_drone(boxes, confidences, frame.shape)
        if primary_bbox is not None:
            drone_boxes = [primary_bbox]
            drone_confs = [primary_conf]
        else:
            drone_boxes = []
            drone_confs = []

        # Update tracker with only the primary drone bbox
        tracks = tracker.update(drone_boxes)
        
        # Draw results on frame
        for tid, bbox in tracks:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Ensure valid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            
            if x1 < x2 and y1 < y2:
                # Draw bounding box (green for tracked drone)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(frame, f'Drone {tid}', (x1, max(20, y1 - 8)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Log to CSV (use detection confidence if available)
                conf_val = drone_confs[0] if (len(drone_confs) > 0) else 1.0
                csv_lines.append(f"{frame_idx},{tid},{x1},{y1},{x2},{y2},{conf_val}")
        
        # Write frame to output video (robust)
        write_ok = True
        try:
            res = out.write(frame)
            # Some OpenCV builds return None from write(); treat None as success
            if res is None:
                res = True
            write_ok = bool(res)
        except Exception as e:
            write_ok = False
            write_err = e

        if not write_ok:
            print(f"  WARNING: Failed to write frame {frame_idx} - attempting to reopen writer")
            try:
                out.release()
            except Exception:
                pass
            # Small pause before reopening
            time.sleep(0.1)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # Ensure we still write to the same output_path (AVI expected)
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not out.isOpened():
                    print("  ERROR: Could not reopen VideoWriter (out.isOpened() is False)")
                else:
                    try:
                        res2 = out.write(frame)
                        if res2 is None:
                            res2 = True
                        if res2:
                            print(f"  INFO: Rewrote frame {frame_idx} after reopening writer")
                        else:
                            print(f"  ERROR: Re-write failed for frame {frame_idx}")
                    except Exception as e2:
                        print(f"  ERROR: Exception on re-write: {e2}")
            except Exception as e3:
                print(f"  ERROR: Failed to reopen VideoWriter: {e3}")
        
        if frame_idx % 30 == 0:
            elapsed = time.time() - start
            fps_actual = frame_idx / elapsed
            print(f"  Frame {frame_idx} ({fps_actual:.2f} fps) - {len(tracks)} drone(s) tracked")
    
    cap.release()
    out.release()
    
    # Save CSV file
    csv_path = os.path.splitext(output_path)[0] + '.csv'
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    
    elapsed = time.time() - start
    print(f'✓ Finished {os.path.basename(input_path)}')
    print(f'  Output video: {output_path}')
    print(f'  Tracking CSV: {csv_path}')
    print(f'  Total time: {elapsed:.2f}s, Avg FPS: {frame_idx/elapsed:.2f}')
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Advanced Drone Tracking System')
    parser.add_argument('--input-dir', type=str, default='videos',
                       help='Directory with input videos')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                       help='YOLO weights file')
    parser.add_argument('--conf', type=float, default=0.30,
                       help='Detection confidence threshold (0.0-1.0, lower = more sensitive)')
    parser.add_argument('--debug-detections', action='store_true',
                       help='Print raw YOLO detections for the first --debug-frames frames')
    parser.add_argument('--debug-frames', type=int, default=200,
                       help='Number of frames to print raw detections for when --debug-detections is set')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f'ERROR: Input directory "{args.input_dir}" not found')
        print(f'Please create it and place your videos inside.')
        return
    
    # Load YOLO model
    print(f'Loading YOLO model from {args.weights}...')
    try:
        model = YOLO(args.weights)
        print('✓ Model loaded successfully')
    except Exception as e:
        print(f'ERROR loading model: {e}')
        return
    
    # Initialize tracker for drone tracking - TRACKING IS THE MAIN FOCUS
    # min_hits=1: accept a single confirmed detection to start the track
    # max_age=240: keep track alive longer (8 seconds at 30fps) for robust tracking through occlusions
    # iou_threshold=0.20: slightly stricter matching to reduce false associations
    tracker = Tracker(max_age=240, min_hits=1, iou_threshold=0.20)
    
    # Initialize drone detector
    drone_detector = DroneDetector()
    
    # Find video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    files = [f for f in os.listdir(args.input_dir) 
            if f.lower().endswith(video_extensions)]
    
    if not files:
        print(f'ERROR: No video files found in {args.input_dir}')
        return
    
    print(f'\nFound {len(files)} video(s) to process:')
    for f in files:
        print(f'  - {f}')
    print()
    
    # Process each video
    for f in sorted(files):
        inp = os.path.join(args.input_dir, f)
        outp = os.path.join(args.output_dir, 'tracked_' + f)
        
        # Reset tracker for new video
        tracker.reset()

        run_on_video(inp, outp, model, tracker, drone_detector,
                 conf=args.conf,
                 debug_detections=args.debug_detections,
                 debug_frames=args.debug_frames)
    
    print(f'✓ All videos processed. Results saved to {args.output_dir}/')


if __name__ == '__main__':
    main()

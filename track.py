import os
import cv2
import argparse
import time
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


class Track:
    def __init__(self, bbox, track_id):
        # bbox = [x1, y1, x2, y2]
        self.id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)
        # state: [cx, cy, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.
        # state transition
        self.kf.F = np.array([[1,0,0,0,dt,0,0],
                              [0,1,0,0,0,dt,0],
                              [0,0,1,0,0,0,dt],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.P *= 10.
        self.kf.R *= 1.
        self.kf.Q *= 0.01
        self.kf.x[:4,0] = np.array([cx, cy, s, r])
        self.last_bbox = bbox

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        cx, cy, s, r = self.kf.x[:4,0]
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = cx - w/2.
        y1 = cy - h/2.
        x2 = cx + w/2.
        y2 = cy + h/2.
        self.last_bbox = [x1, y1, x2, y2]
        return self.last_bbox

    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)
        z = np.array([cx, cy, s, r])
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.last_bbox = bbox


class Tracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2]
        # predict existing
        preds = [t.predict() for t in self.tracks]
        if len(preds) == 0:
            unmatched_dets = list(range(len(detections)))
            matches = []
            unmatched_trks = []
        else:
            # compute cost matrix
            cost = np.zeros((len(preds), len(detections)), dtype=np.float32)
            for i, p in enumerate(preds):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - iou(p, d)
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = []
            unmatched_trks = list(range(len(preds)))
            unmatched_dets = list(range(len(detections)))
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] > 1.0 - self.iou_threshold:
                    continue
                matches.append((r, c))
                unmatched_trks.remove(r)
                unmatched_dets.remove(c)

        # update matched
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        # create new tracks for unmatched detections
        for idx in unmatched_dets:
            self.tracks.append(Track(detections[idx], self._next_id))
            self._next_id += 1

        # age and delete old tracks
        to_remove = []
        for i, t in enumerate(self.tracks):
            if t.time_since_update > self.max_age:
                to_remove.append(i)
        for idx in sorted(to_remove, reverse=True):
            del self.tracks[idx]

        # return active tracks (that have at least min_hits)
        output = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                output.append((t.id, t.last_bbox))
        return output


def run_on_video(input_path, output_path, model, tracker, conf=0.35):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'ERROR opening {input_path}')
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    frame_idx = 0
    start = time.time()
    csv_lines = ["frame,id,x1,y1,x2,y2,conf"]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = model(frame, conf=conf, verbose=False)
        boxes = []
        scores = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes.data.cpu().numpy():
                # box: [x1,y1,x2,y2,conf,cls]
                x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], box[4]
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
        tracks = tracker.update(boxes)
        # draw
        for tid, bbox in tracks:
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{tid}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            csv_lines.append(f"{frame_idx},{tid},{x1},{y1},{x2},{y2},")
        out.write(frame)
        if frame_idx % 50 == 0:
            elapsed = time.time() - start
            print(f"Processed frame {frame_idx} ({frame_idx/elapsed:.2f} fps)")
    cap.release()
    out.release()
    # save CSV
    csv_path = os.path.splitext(output_path)[0] + '.csv'
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f'Finished {input_path}. Output saved to {output_path}, tracks to {csv_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='videos', help='Directory with input videos')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='YOLO weights file')
    parser.add_argument('--conf', type=float, default=0.35)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.input_dir):
        print(f'Create a folder named "{args.input_dir}" and put your videos (mp4/avi) inside and re-run this script.')
        return

    model = YOLO(args.weights)
    # safe globals for newer torch if needed
    try:
        import torch, ultralytics.nn.tasks as tasks
        torch.serialization.add_safe_globals([tasks.DetectionModel])
    except Exception:
        pass

    tracker = Tracker(max_age=40, min_hits=1, iou_threshold=0.3)
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not files:
        print('No video files found in', args.input_dir)
        return

    for f in files:
        inp = os.path.join(args.input_dir, f)
        outp = os.path.join(args.output_dir, 'tracked_' + f)
        run_on_video(inp, outp, model, tracker, conf=args.conf)


if __name__ == '__main__':
    main()

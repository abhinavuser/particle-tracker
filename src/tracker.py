"""
Advanced tracker using Kalman Filter + Particle Filter + Hungarian algorithm.

This tracker is optimized for drone tracking:
- Combines Kalman Filter (smooth motion) + Particle Filter (robustness)
- Uses Hungarian algorithm for robust data association
- Tracks continuous drone motion even with missed detections
- Maintains single dominant track (primary drone)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import create_kalman_filter, init_kalman_state, bbox_from_kalman
from .particle_filter import ParticleFilter
from .utils import iou


class Track:
    """
    Single track object with Kalman + Particle Filter.
    
    Combines:
    - Kalman Filter: predicts motion based on velocity model
    - Particle Filter: robust to noise and missed detections
    - Track age and hits: ensures track maturity before output
    """
    
    def __init__(self, bbox, track_id):
        """
        Initialize a track.
        
        Args:
            bbox: [x1, y1, x2, y2] initial bounding box
            track_id: unique identifier
        """
        self.id = track_id
        self.time_since_update = 0
        self.hits = 1  # Number of detections matched to this track
        self.age = 0   # Total age in frames
        self.last_bbox = bbox
        
        # Initialize Kalman Filter
        self.kf = create_kalman_filter()
        self.kf.x[:7, 0] = init_kalman_state(bbox)
        
        # Initialize Particle Filter for robustness
        # State: [center_x, center_y, vx, vy]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        initial_state = np.array([cx, cy, 0.0, 0.0])
        self.pf = ParticleFilter(
            initial_state,
            num_particles=50,
            process_noise=5.0,
            measurement_noise=15.0
        )
        
        self.bbox_w = bbox[2] - bbox[0]  # Track width
        self.bbox_h = bbox[3] - bbox[1]  # Track height
    
    def predict(self):
        """
        Predict next position using both Kalman and Particle Filter.
        
        Kalman provides smooth velocity estimates.
        Particle Filter provides robustness.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Predict with Kalman
        self.kf.predict()
        
        # Predict with Particle Filter
        pf_state = self.pf.predict(dt=1.0)
        
        # Get predictions
        kalman_bbox = bbox_from_kalman(self.kf.x[:, 0])
        pf_pos = pf_state[:2]
        
        # TRACKING IS THE MAIN FOCUS: Combine predictions 80% Kalman, 20% Particle Filter
        # (Favor Kalman more for SMOOTHER, more stable tracking)
        combined_x1 = 0.8 * kalman_bbox[0] + 0.2 * (pf_pos[0] - self.bbox_w / 2)
        combined_y1 = 0.8 * kalman_bbox[1] + 0.2 * (pf_pos[1] - self.bbox_h / 2)
        combined_x2 = 0.8 * kalman_bbox[2] + 0.2 * (pf_pos[0] + self.bbox_w / 2)
        combined_y2 = 0.8 * kalman_bbox[3] + 0.2 * (pf_pos[1] + self.bbox_h / 2)
        
        self.last_bbox = [combined_x1, combined_y1, combined_x2, combined_y2]
        
        # Track state
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        
        return self.last_bbox
    
    def update(self, bbox):
        """
        Update track with new detection.
        
        Args:
            bbox: [x1, y1, x2, y2] detection bounding box
        """
        # Extract detection center and size
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)
        z = np.array([cx, cy, s, r])
        
        # Update Kalman Filter
        self.kf.update(z)
        
        # Update Particle Filter with detection center
        measurement = np.array([cx, cy])
        self.pf.update(measurement)
        
        # Update track state
        self.time_since_update = 0
        self.hits += 1
        self.last_bbox = bbox
        
        # Update bbox dimensions
        self.bbox_w = bbox[2] - bbox[0]
        self.bbox_h = bbox[3] - bbox[1]


class Tracker:
    """
    Multi-object tracker using Kalman + Particle Filters + Hungarian algorithm.
    
    Optimized for drone tracking:
    - Robust to missed detections (Particle Filters)
    - Smooth motion prediction (Kalman Filters)
    - Intelligent matching (Hungarian algorithm with IoU cost)
    - Single primary drone priority
    """
    
    def __init__(self, max_age=60, min_hits=2, iou_threshold=0.25):
        """
        Initialize tracker.
        
        Args:
            max_age: Max frames to keep track without updates (default 60 = ~2 sec at 30fps)
            min_hits: Min detections before track becomes active (default 2 = confirmed)
            iou_threshold: IoU threshold for matching (default 0.25)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._next_id = 1
        
        # Track primary drone (largest/most confident)
        self.primary_track_id = None
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Uses Hungarian algorithm for optimal matching.
        
        Args:
            detections: List of [x1, y1, x2, y2] bounding boxes
        
        Returns:
            List of (track_id, bbox) for active tracks
        """
        # Predict all tracks
        preds = [t.predict() for t in self.tracks]
        
        # Perform matching
        if len(preds) == 0:
            unmatched_dets = list(range(len(detections)))
            matches = []
            unmatched_trks = []
        else:
            # Compute cost matrix using IoU
            cost = np.zeros((len(preds), len(detections)), dtype=np.float32)
            for i, p in enumerate(preds):
                for j, d in enumerate(detections):
                    iou_val = iou(p, d)
                    # Cost = 1 - IoU (lower cost = better match)
                    cost[i, j] = 1.0 - iou_val
            
            # Sanitize cost matrix
            cost = np.nan_to_num(cost, nan=1e10, posinf=1e10, neginf=0)
            cost = np.clip(cost, 0, 1e10)
            
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = []
            unmatched_trks = list(range(len(preds)))
            unmatched_dets = list(range(len(detections)))
            
            # Keep matches with IoU > threshold
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.iou_threshold):
                    matches.append((r, c))
                    if r in unmatched_trks:
                        unmatched_trks.remove(r)
                    if c in unmatched_dets:
                        unmatched_dets.remove(c)
        
        # Update matched tracks
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for idx in unmatched_dets:
            new_track = Track(detections[idx], self._next_id)
            self.tracks.append(new_track)
            self._next_id += 1
        
        # Age unmatched tracks (predicted but not detected)
        # They continue to predict their position
        
        # Remove dead tracks
        to_remove = []
        for i, t in enumerate(self.tracks):
            if t.time_since_update > self.max_age:
                to_remove.append(i)
        
        for idx in sorted(to_remove, reverse=True):
            if self.primary_track_id == self.tracks[idx].id:
                self.primary_track_id = None
            del self.tracks[idx]
        
        # Output active tracks
        output = []
        for t in self.tracks:
            # Show track only if it has enough hits (confirmed track)
            # This prevents showing tracks before they're confident
            if t.hits >= self.min_hits:
                output.append((t.id, t.last_bbox))
        
        # Update primary track (largest by area)
        if output:
            largest_idx = 0
            largest_area = 0
            for idx, (tid, bbox) in enumerate(output):
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > largest_area:
                    largest_area = area
                    largest_idx = idx
            self.primary_track_id = output[largest_idx][0]
        
        return output
    
    def get_primary_track(self):
        """Get the primary (largest) drone track."""
        for t in self.tracks:
            if t.id == self.primary_track_id and t.hits >= self.min_hits:
                return (t.id, t.last_bbox)
        return None
    
    def reset(self):
        """Reset all tracks (for new video)."""
        self.tracks = []
        self.primary_track_id = None
        self._next_id = 1

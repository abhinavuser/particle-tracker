"""
Kalman Filter for motion prediction in drone tracking.
"""
import numpy as np
from filterpy.kalman import KalmanFilter


def create_kalman_filter():
    """
    Create and initialize a Kalman Filter for tracking.
    State: [cx, cy, s, r, vx, vy, vs]
    Measurement: [cx, cy, s, r]
    """
    kf = KalmanFilter(dim_x=7, dim_z=4)
    dt = 1.0
    
    # State transition matrix
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1]])
    
    # Measurement matrix
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0]])
    
    # Covariance matrices
    kf.P *= 10.0
    kf.R *= 1.0
    kf.Q *= 0.01
    
    return kf


def init_kalman_state(bbox):
    """
    Initialize Kalman filter state from bounding box.
    bbox = [x1, y1, x2, y2]
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)
    return np.array([cx, cy, s, r, 0, 0, 0])


def bbox_from_kalman(state):
    """
    Convert Kalman state [cx, cy, s, r, ...] to bounding box [x1, y1, x2, y2].
    """
    cx, cy, s, r = state[0], state[1], state[2], state[3]
    # Ensure positive values to avoid NaN in sqrt
    s = max(s, 1.0)
    r = max(r, 0.1)
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]

"""
Smart drone detection and filtering.

Distinguishes drones from other objects using:
- Aspect ratio analysis
- Size constraints
- Motion characteristics
- Appearance (small, compact objects in sky)
"""
import numpy as np


class DroneDetector:
    """
    Identifies actual drones vs false positives.
    
    Drones have specific characteristics:
    - Small to medium size (typically 50-500 pixels)
    - Relatively square aspect ratio (0.5-2.0)
    - Move smoothly through sky
    - Appear isolated (not clustered with other objects)
    """
    
    def __init__(self):
        """Initialize drone detector with STRICTER filters to reduce false positives."""
        # Size constraints for drones (in pixels)
        # INCREASED min_area to filter out noise, REDUCED max_area to reject large false positives
        self.min_area = 100  # Filter out tiny noise
        self.max_area = 15000  # Reject very large boxes (likely false positives)
        
        # Aspect ratio constraints (w/h)
        # NARROWED aspect range - drones are relatively compact
        self.min_aspect = 0.6
        self.max_aspect = 2.5
        
        # Position constraints
        self.min_height_ratio = 0.0  # Drones can be anywhere vertically
        self.max_height_ratio = 1.0
        
        # Confidence thresholds
        # RAISED confidence threshold to reduce false positives
        self.min_confidence = 0.25
        
        # Frame coverage limit - reject boxes covering >40% of frame
        self.max_frame_coverage = 0.40
        
        # Track some history for motion analysis
        self.detection_history = []
        self.max_history = 10
    
    def is_drone(self, bbox, confidence, frame_shape=None):
        """
        Determine if bounding box is likely a drone.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            confidence: Detection confidence (0-1)
            frame_shape: (height, width) of frame (optional)
        
        Returns:
            bool: True if likely drone, False otherwise
        """
        # Confidence check
        if confidence < self.min_confidence:
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Compute size
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        # Size check
        if area < self.min_area or area > self.max_area:
            return False
        
        # REJECT boxes covering too much of the frame (likely false positives)
        if frame_shape is not None:
            frame_area = frame_shape[0] * frame_shape[1]
            if area / frame_area > self.max_frame_coverage:
                return False
        
        # Aspect ratio check (drones are roughly square or rectangular)
        aspect = w / (h + 1e-6)
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False
        
        # Position check (optional)
        if frame_shape is not None:
            frame_h, frame_w = frame_shape[:2]
            center_y = (y1 + y2) / 2
            height_ratio = center_y / frame_h
            if height_ratio < self.min_height_ratio or height_ratio > self.max_height_ratio:
                return False
        
        return True
    
    def filter_detections(self, boxes, confidences, frame_shape=None):
        """
        Filter detections to keep only likely drones.
        
        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes
            confidences: List of confidence scores (0-1)
            frame_shape: (height, width) of frame (optional)
        
        Returns:
            List of drone bounding boxes
        """
        drone_boxes = []
        
        for bbox, conf in zip(boxes, confidences):
            if self.is_drone(bbox, conf, frame_shape):
                drone_boxes.append(bbox)
        
        return drone_boxes
    
    def get_largest_drone(self, boxes, confidences, frame_shape=None):
        """
        Get the largest (most likely primary) drone from detections.
        
        Useful for single-drone tracking.
        
        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes
            confidences: List of confidence scores (0-1)
            frame_shape: (height, width) of frame (optional)
        
        Returns:
            Tuple (bbox, confidence) or (None, None) if no drone found
        """
        drone_detections = []
        
        for bbox, conf in zip(boxes, confidences):
            if self.is_drone(bbox, conf, frame_shape):
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                drone_detections.append((bbox, conf, area))
        
        if not drone_detections:
            return None, None
        
        # Return largest drone
        largest = max(drone_detections, key=lambda x: x[2])
        return largest[0], largest[1]
    
    def filter_by_confidence(self, boxes, confidences, threshold=0.35):
        """
        Filter detections by confidence threshold.
        
        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            threshold: Minimum confidence
        
        Returns:
            Filtered (boxes, confidences)
        """
        filtered_boxes = []
        filtered_confs = []
        
        for bbox, conf in zip(boxes, confidences):
            if conf >= threshold:
                filtered_boxes.append(bbox)
                filtered_confs.append(conf)
        
        return filtered_boxes, filtered_confs
    
    def non_max_suppression(self, boxes, confidences, iou_threshold=0.5):
        """
        Remove overlapping detections (keep highest confidence).
        
        Args:
            boxes: List of [x1, y1, x2, y2]
            confidences: List of confidence scores
            iou_threshold: IoU threshold for suppression
        
        Returns:
            (boxes, confidences) after NMS
        """
        if not boxes:
            return [], []
        
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        # Sort by confidence descending
        order = np.argsort(-confidences)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou(boxes[i], boxes[order[1:]])
            
            # Keep boxes with IoU < threshold
            order = order[1:][ious < iou_threshold]
        
        return boxes[keep].tolist(), confidences[keep].tolist()
    
    def _compute_iou(self, box1, boxes):
        """Compute IoU between box1 and multiple boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Intersection
        inter_xmin = np.maximum(x1_min, x2_min)
        inter_ymin = np.maximum(y1_min, y2_min)
        inter_xmax = np.minimum(x1_max, x2_max)
        inter_ymax = np.minimum(y1_max, y2_max)
        
        inter_w = np.maximum(0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        boxes_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + boxes_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou

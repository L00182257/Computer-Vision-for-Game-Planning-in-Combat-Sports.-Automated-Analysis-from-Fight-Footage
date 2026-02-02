import numpy as np

class FighterFilter:
    """
    Filters detections to retain only the two primary fighters.
    """

    def __init__(self, min_area_ratio=0.05):
        """
        Args:
            min_area_ratio (float): Minimum bbox area relative to frame area
        """
        self.min_area_ratio = min_area_ratio

    def filter(self, boxes_xyxy, track_ids, frame_shape):
        """
        Args:
            boxes_xyxy (np.ndarray): Nx4 bounding boxes
            track_ids (np.ndarray): N track IDs
            frame_shape (tuple): (H, W, C)

        Returns:
            filtered_boxes, filtered_ids
        """
        if len(boxes_xyxy) <= 2:
            return boxes_xyxy, track_ids

        h, w = frame_shape[:2]
        frame_area = h * w

        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * \
                (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        valid = areas > (self.min_area_ratio * frame_area)

        boxes_xyxy = boxes_xyxy[valid]
        track_ids = track_ids[valid]

        if len(boxes_xyxy) <= 2:
            return boxes_xyxy, track_ids

        # Keep the two largest boxes
        idx = np.argsort(areas[valid])[::-1][:2]

        return boxes_xyxy[idx], track_ids[idx]
import numpy as np


class FighterFilter:
    """
    Keeps only the two most central, largest detections.
    Designed specifically for fight footage.
    """

    def __init__(self, min_area_ratio=0.02, centre_weight=1.5):
        self.min_area_ratio = min_area_ratio
        self.centre_weight = centre_weight

    def filter(self, boxes, track_ids, frame_shape):

        if boxes is None or len(boxes) == 0:
            return np.array([]), np.array([])

        h, w = frame_shape[:2]
        frame_area = h * w
        cx, cy = w / 2, h / 2

        candidates = []

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area_ratio * frame_area:
                continue

            # centre distance
            bx = (x1 + x2) / 2
            by = (y1 + y2) / 2
            centre_dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

            # score = area minus weighted centre distance
            score = area - self.centre_weight * centre_dist

            candidates.append((score, box, tid))

        if len(candidates) == 0:
            return np.array([]), np.array([])

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Keep top 2
        top = candidates[:2]

        filtered_boxes = [item[1] for item in top]
        filtered_ids = [item[2] for item in top]

        return np.array(filtered_boxes), np.array(filtered_ids)
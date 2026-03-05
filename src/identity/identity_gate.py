import numpy as np


class IdentityGate:
    """
    Basic identity gate.
    Opens when at least two people are sufficiently separated
    for a number of consecutive frames.
    """

    def __init__(self, required_frames=15, min_sep_ratio=0.15):
        self.required_frames = required_frames
        self.min_sep_ratio = min_sep_ratio
        self.counter = 0
        self.locked = False

    def update(self, boxes, frame_shape):

        if self.locked:
            return True, "LOCKED"

        if boxes is None or len(boxes) < 2:
            self.counter = 0
            return False, "Need at least 2 people"

        # Take two largest boxes
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = np.argsort(areas)[-2:]
        b1, b2 = boxes[idx]

        # Compute centre distance
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)

        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = np.sqrt(dx * dx + dy * dy)

        frame_h, frame_w = frame_shape[:2]
        frame_diag = np.sqrt(frame_w ** 2 + frame_h ** 2)

        norm_sep = dist / frame_diag

        if norm_sep < self.min_sep_ratio:
            self.counter = 0
            return False, f"Too close ({norm_sep:.2f})"

        self.counter += 1

        if self.counter >= self.required_frames:
            self.locked = True
            return True, "LOCKED"

        return False, f"Stable frames: {self.counter}/{self.required_frames}"
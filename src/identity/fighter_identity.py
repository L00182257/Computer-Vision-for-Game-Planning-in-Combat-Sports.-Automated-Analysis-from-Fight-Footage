import cv2
import numpy as np


class FighterIdentity:

    def __init__(self, fighter_a_hsv, fighter_b_hsv, sat_threshold=40):

        self.sat_threshold = sat_threshold

        self.fighter_state = {
            "A": {
                "track_id": None,
                "hue": float(fighter_a_hsv[0])
            },
            "B": {
                "track_id": None,
                "hue": float(fighter_b_hsv[0])
            }
        }

    # ----------------------------
    # Hue wrap distance
    # ----------------------------
    def hue_distance(self, h1, h2):
        diff = abs(h1 - h2)
        return min(diff, 180 - diff)

    # ----------------------------
    # Extract shorts hue
    # ----------------------------
    def extract_shorts_hue(self, frame, box):

        x1, y1, x2, y2 = map(int, box)

        h = y2 - y1
        sample_y1 = y1 + int(0.55 * h)
        sample_y2 = y1 + int(0.85 * h)

        crop = frame[sample_y1:sample_y2, x1:x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        pixels = pixels[pixels[:, 1] > self.sat_threshold]

        if len(pixels) < 50:
            return None

        return float(np.mean(pixels[:, 0]))

    # ----------------------------
    # Core assignment logic
    # ----------------------------
    def assign(self, frame, boxes, track_ids):

        identities = []

        visible_tracks = []

        for box, tid in zip(boxes, track_ids):
            hue = self.extract_shorts_hue(frame, box)
            if hue is not None:
                visible_tracks.append((tid, hue))

        # If exactly 2 visible fighters → force pairwise assignment
        if len(visible_tracks) == 2:

            (tid1, hue1), (tid2, hue2) = visible_tracks

            d1A = self.hue_distance(hue1, self.fighter_state["A"]["hue"])
            d1B = self.hue_distance(hue1, self.fighter_state["B"]["hue"])

            d2A = self.hue_distance(hue2, self.fighter_state["A"]["hue"])
            d2B = self.hue_distance(hue2, self.fighter_state["B"]["hue"])

            cost_AB = d1A + d2B
            cost_BA = d1B + d2A

            if cost_AB < cost_BA:
                self.fighter_state["A"]["track_id"] = tid1
                self.fighter_state["B"]["track_id"] = tid2
            else:
                self.fighter_state["A"]["track_id"] = tid2
                self.fighter_state["B"]["track_id"] = tid1

        # Single visible fighter case
        elif len(visible_tracks) == 1:

            tid, hue = visible_tracks[0]

            # If already assigned, skip
            if tid not in (
                self.fighter_state["A"]["track_id"],
                self.fighter_state["B"]["track_id"]
            ):

                dA = self.hue_distance(hue, self.fighter_state["A"]["hue"])
                dB = self.hue_distance(hue, self.fighter_state["B"]["hue"])

                # Assign to whichever slot is free
                if self.fighter_state["A"]["track_id"] is None:
                    self.fighter_state["A"]["track_id"] = tid
                elif self.fighter_state["B"]["track_id"] is None:
                    self.fighter_state["B"]["track_id"] = tid
                else:
                    # Replace closer one
                    if dA < dB:
                        self.fighter_state["A"]["track_id"] = tid
                    else:
                        self.fighter_state["B"]["track_id"] = tid

        # Build identity list
        for tid in track_ids:

            if tid == self.fighter_state["A"]["track_id"]:
                identities.append("A")
            elif tid == self.fighter_state["B"]["track_id"]:
                identities.append("B")
            else:
                identities.append(None)

        return identities
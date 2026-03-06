import numpy as np


class DefenceDetector:

    def __init__(self):

        # adjusted threshold based on your debug data
        self.duck_threshold = 1.65

        # frames required for confirmation
        self.required_frames = 3

        self.counter = {"A": 0, "B": 0}


    def detect(self, keypoints, fighter_id):

        if keypoints is None or len(keypoints) < 15:
            return None

        nose = keypoints[0]

        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]

        l_hip = keypoints[11]
        r_hip = keypoints[12]

        l_knee = keypoints[13]
        r_knee = keypoints[14]

        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        knee_y = (l_knee[1] + r_knee[1]) / 2
        head_y = nose[1]

        # basic pose sanity check
        if not (head_y < shoulder_y < hip_y < knee_y):
            return None

        torso_height = hip_y - shoulder_y
        if torso_height < 10:
            return None

        head_to_knee = knee_y - head_y

        ratio = head_to_knee / torso_height

        print("DuckRatio:", round(ratio, 2))

        if ratio < self.duck_threshold:

            self.counter[fighter_id] += 1

            if self.counter[fighter_id] >= self.required_frames:
                return "Duck"

        else:
            self.counter[fighter_id] = 0

        return None
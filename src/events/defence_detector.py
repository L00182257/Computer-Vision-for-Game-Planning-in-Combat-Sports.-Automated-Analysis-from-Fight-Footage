import numpy as np
import math


class DefenceDetector:

    def __init__(self):

        # Duck thresholds
        self.head_drop_threshold = 1.8
        self.torso_angle_threshold = 30

        # Frames required to confirm duck
        self.required_frames = 4

        # EMA smoothing strength
        self.smoothing_alpha = 0.6

        # Pose smoothing buffers
        self.smoothed_pose = {
            "A": None,
            "B": None
        }

        # Previous head positions for velocity check
        self.prev_head_y = {
            "A": None,
            "B": None
        }

        self.counter = {
            "A": 0,
            "B": 0
        }


    def _smooth_pose(self, keypoints, fighter_id):

        keypoints = np.array(keypoints[:, :2], dtype=np.float32)

        if self.smoothed_pose[fighter_id] is None:
            self.smoothed_pose[fighter_id] = keypoints
            return keypoints

        prev = self.smoothed_pose[fighter_id]

        smoothed = (
            self.smoothing_alpha * keypoints +
            (1 - self.smoothing_alpha) * prev
        )

        self.smoothed_pose[fighter_id] = smoothed

        return smoothed


    def detect(self, keypoints, fighter_id):

        if keypoints is None or len(keypoints) < 17:
            return None


        keypoints = self._smooth_pose(keypoints, fighter_id)


        nose = keypoints[0]

        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]

        l_hip = keypoints[11]
        r_hip = keypoints[12]

        l_knee = keypoints[13]
        r_knee = keypoints[14]


        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2
        knee_mid = (l_knee + r_knee) / 2


        torso_height = hip_mid[1] - shoulder_mid[1]

        if torso_height < 15:
            return None


        head_to_knee = knee_mid[1] - nose[1]
        ratio = head_to_knee / torso_height


        torso_vec = hip_mid - shoulder_mid

        angle = abs(math.degrees(
            math.atan2(torso_vec[0], torso_vec[1])
        ))


        # Head velocity check
        velocity = 0

        if self.prev_head_y[fighter_id] is not None:
            velocity = nose[1] - self.prev_head_y[fighter_id]

        self.prev_head_y[fighter_id] = nose[1]


        print(
            "DuckRatio:", round(ratio, 2),
            "TorsoAngle:", round(angle, 1),
            "HeadVel:", round(velocity, 2)
        )


        duck_detected = False


        # Geometry condition
        if ratio < self.head_drop_threshold or angle > self.torso_angle_threshold:

            # Must also be moving downward
            if velocity > 0.5:
                duck_detected = True


        if duck_detected:

            self.counter[fighter_id] += 1

            if self.counter[fighter_id] >= self.required_frames:
                return "Duck"

        else:

            self.counter[fighter_id] = 0


        return None
import numpy as np
import math


class DefenceDetector:

    def __init__(self):

        # --- thresholds ---
        self.duck_ratio = 1.6
        self.slip_threshold = 0.18
        self.pull_threshold = 0.15

        # frames required to confirm
        self.required_frames = 4

        # cooldown to prevent spam
        self.cooldown_frames = 12

        # fighter state
        self.counters = {"A":0, "B":0}
        self.cooldowns = {"A":0, "B":0}

        # previous head positions
        self.prev_head = {"A":None, "B":None}


    def detect(self, keypoints, fighter_id):

        if keypoints is None or len(keypoints) < 17:
            return None


        # -------------------------------------------------
        # Keypoints
        # -------------------------------------------------

        nose = keypoints[0]

        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]

        l_hip = keypoints[11]
        r_hip = keypoints[12]

        l_knee = keypoints[13]
        r_knee = keypoints[14]

        l_wrist = keypoints[9]
        r_wrist = keypoints[10]


        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2
        knee_mid = (l_knee + r_knee) / 2


        torso_height = hip_mid[1] - shoulder_mid[1]

        if torso_height < 10:
            return None


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        head_to_knee = knee_mid[1] - nose[1]
        ratio = head_to_knee / torso_height

        torso_vec = hip_mid - shoulder_mid

        torso_angle = abs(math.degrees(
            math.atan2(torso_vec[0], torso_vec[1])
        ))

        head_offset = (nose[0] - shoulder_mid[0]) / torso_height

        # head movement
        velocity = 0

        if self.prev_head[fighter_id] is not None:
            velocity = nose[1] - self.prev_head[fighter_id]

        self.prev_head[fighter_id] = nose[1]


        print(
            "ratio:", round(ratio,2),
            "angle:", round(torso_angle,1),
            "offset:", round(head_offset,2),
            "vel:", round(velocity,2)
        )


        # -------------------------------------------------
        # Cooldown
        # -------------------------------------------------

        if self.cooldowns[fighter_id] > 0:
            self.cooldowns[fighter_id] -= 1
            return None


        event = None


        # -------------------------------------------------
        # Duck
        # -------------------------------------------------

        if ratio < self.duck_ratio and torso_angle > 30 and velocity > 2:
            event = "Duck"


        # -------------------------------------------------
        # Slip Left / Right
        # -------------------------------------------------

        elif head_offset < -self.slip_threshold:
            event = "Slip Left"

        elif head_offset > self.slip_threshold:
            event = "Slip Right"


        # -------------------------------------------------
        # Pull
        # -------------------------------------------------

        elif velocity < -self.pull_threshold * torso_height:
            event = "Pull"


        # -------------------------------------------------
        # High Block
        # -------------------------------------------------

        elif (
            l_wrist[1] < nose[1] and
            r_wrist[1] < nose[1]
        ):
            event = "Block High"


        # -------------------------------------------------
        # Low Block
        # -------------------------------------------------

        elif (
            l_wrist[1] > shoulder_mid[1] and
            r_wrist[1] > shoulder_mid[1]
        ):
            event = "Block Low"


        # -------------------------------------------------
        # Temporal confirmation
        # -------------------------------------------------

        if event:

            self.counters[fighter_id] += 1

            if self.counters[fighter_id] >= self.required_frames:

                self.counters[fighter_id] = 0
                self.cooldowns[fighter_id] = self.cooldown_frames

                return event

        else:

            self.counters[fighter_id] = 0


        return None
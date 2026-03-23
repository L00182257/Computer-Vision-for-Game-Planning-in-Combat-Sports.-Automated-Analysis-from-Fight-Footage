import numpy as np
import math


class DefenceDetector:

    def __init__(self):

        # --- thresholds ---
        self.duck_ratio = 1.6  # reverted
        self.slip_offset = 0.3  # increased for less sensitivity
        self.pull_threshold = 0.25  # increased for less sensitivity

        # frames required to confirm
        self.required_frames = 5  # keep higher

        # cooldown to prevent spam
        self.cooldown_frames = 15  # keep higher

        # fighter state
        self.counters = {"A": 0, "B": 0}
        self.cooldowns = {"A": 0, "B": 0}

        # previous head positions and angles
        self.prev_head = {"A": None, "B": None}
        self.prev_torso_angle = {"A": None, "B": None}


    def _is_pose_valid(self, keypoints):
        """Check if pose keypoints are reasonably valid (not all zeros, within frame bounds)."""
        if keypoints is None or len(keypoints) < 17:
            return False
        # Check if key points are not at origin
        key_points = [keypoints[0], keypoints[5], keypoints[6], keypoints[11], keypoints[12]]  # nose, shoulders, hips
        for kp in key_points:
            if np.allclose(kp, [0, 0], atol=1):
                return False
        return True


    def detect(self, keypoints, fighter_id):

        if not self._is_pose_valid(keypoints):
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
        if torso_height < 20:  # stricter min height
            return None

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        head_to_knee = knee_mid[1] - nose[1]
        ratio = head_to_knee / torso_height

        torso_vec = hip_mid - shoulder_mid
        torso_angle = abs(math.degrees(math.atan2(torso_vec[0], torso_vec[1])))

        head_offset = (nose[0] - shoulder_mid[0]) / torso_height

        # head movement velocity
        velocity_y = 0
        if self.prev_head[fighter_id] is not None:
            velocity_y = nose[1] - self.prev_head[fighter_id]

        self.prev_head[fighter_id] = nose[1]

        # -------------------------------------------------
        # Cooldown
        # -------------------------------------------------

        if self.cooldowns[fighter_id] > 0:
            self.cooldowns[fighter_id] -= 1
            return None

        event = None

        # -------------------------------------------------
        # Duck (reverted to original logic)
        # -------------------------------------------------
        if ratio < 1.6 and torso_angle > 30 and velocity_y > 2:
            event = "Duck"

        # -------------------------------------------------
        # Slip Left / Right (less sensitive)
        # -------------------------------------------------
        elif (head_offset < -self.slip_offset and  # increased threshold
              abs(velocity_y) < 2 and  # stricter vertical movement
              torso_angle < 25):  # more upright
            event = "Slip Left"

        elif (head_offset > self.slip_offset and  # increased threshold
              abs(velocity_y) < 2 and
              torso_angle < 25):
            event = "Slip Right"

        # -------------------------------------------------
        # Pull (less sensitive)
        # -------------------------------------------------
        elif velocity_y < -self.pull_threshold * torso_height:  # stricter velocity
            event = "Pull"

        # -------------------------------------------------
        # Shoulder Roll Left / Right
        # -------------------------------------------------
        elif (abs(head_offset) > 0.22 and
              torso_angle > 25 and
              (l_wrist[1] < shoulder_mid[1] or r_wrist[1] < shoulder_mid[1])):
            if head_offset < 0:
                event = "Shoulder Roll Left"
            else:
                event = "Shoulder Roll Right"

        # -------------------------------------------------
        # Blocking commented out
        # -------------------------------------------------
        # elif (l_wrist[1] < nose[1] - self.block_high_threshold * torso_height and
        #       r_wrist[1] < nose[1] - self.block_high_threshold * torso_height and
        #       abs(head_offset) < 0.1):
        #     event = "Block High"

        # elif (l_wrist[1] < shoulder_mid[1] and l_wrist[0] < shoulder_mid[0] - 0.1 * torso_height):
        #     event = "Block Left"

        # elif (r_wrist[1] < shoulder_mid[1] and r_wrist[0] > shoulder_mid[0] + 0.1 * torso_height):
        #     event = "Block Right"

        # elif (l_wrist[1] > shoulder_mid[1] + self.block_low_threshold * torso_height and
        #       r_wrist[1] > shoulder_mid[1] + self.block_low_threshold * torso_height):
        #     event = "Block Low"

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
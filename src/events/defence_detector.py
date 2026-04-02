import numpy as np
import math


class DefenceDetector:

    def __init__(self):

        # --- thresholds ---
        self.duck_ratio = 1.8  
        #self.slip_offset = 0.3  
        self.pull_threshold = 0.25  

        # frames required to confirm
        self.required_frames = 4 

        # cooldown to prevent spam
        self.cooldown_frames = 28  

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


    def detect_block(self, defender_pose, attacker_pose):
        """Detect block when defender and attacker gloves (wrists) come into contact.

        Conditions:
        - Both poses valid.
        - Defender wrist and attacker wrist are within a tight threshold.
        - Defender wrist is near defender's head/shoulder area (indicating a defensive block position).
        - This is intentionally conservative (err on side of caution).
        """
        if not self._is_pose_valid(defender_pose) or not self._is_pose_valid(attacker_pose):
            return None

        d_nose = defender_pose[0]
        d_l_shoulder = defender_pose[5]
        d_r_shoulder = defender_pose[6]
        d_l_wrist = defender_pose[9]
        d_r_wrist = defender_pose[10]

        a_l_wrist = attacker_pose[9]
        a_r_wrist = attacker_pose[10]

        shoulder_mid = (d_l_shoulder + d_r_shoulder) / 2
        torso_height = np.linalg.norm(shoulder_mid - ((defender_pose[11] + defender_pose[12]) / 2))
        if torso_height <= 0:
            return None

        # requirement: two gloves are close enough (contact)
        contact_threshold = max(50.0, torso_height * 0.25)

        def _dist(p, q):
            return np.linalg.norm(np.array(p) - np.array(q))

        contacts = [
            _dist(d_l_wrist, a_l_wrist),
            _dist(d_l_wrist, a_r_wrist),
            _dist(d_r_wrist, a_l_wrist),
            _dist(d_r_wrist, a_r_wrist),
        ]
        matched = [d for d in contacts if d <= contact_threshold]

        if not matched:
            return None

        # defender wrist should be placed near own head/upper-chest area
        face_region_threshold = max(20.0, torso_height * 0.25)
        if (_dist(d_l_wrist, d_nose) < face_region_threshold or
            _dist(d_r_wrist, d_nose) < face_region_threshold or
            _dist(d_l_wrist, shoulder_mid) < face_region_threshold or
            _dist(d_r_wrist, shoulder_mid) < face_region_threshold):
            return "Block"

        return None


    def detect(self, keypoints, fighter_id):
        """Detect single-fighter defensive actions, excluding block (handled by detect_block).

        Actions covered:
        - Duck: lower height quickly (head-to-knee ratio drops) with downward movement.
        - Slip Left/Right: lateral head offset with stable vertical position and neutral torso.
        - Pull: retracting body quickly backwards (negative vertical velocity) as counter-defence.

        This method is intended as a conservative fallback when block is not present.
        """

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
        # Duck
        # -------------------------------------------------
        if ratio < 1.8 and torso_angle > 30 and velocity_y > 2:
            event = "Duck"

        # -------------------------------------------------
        # Slip Left / Right
        # -------------------------------------------------
        #elif (head_offset < -self.slip_offset and  # increased threshold
         #     abs(velocity_y) < 2 and  # stricter vertical movement
         #     torso_angle < 25):  # more upright
         #   event = "Slip Left"

        # -------------------------------------------------
        # Pull (less sensitive)
        # -------------------------------------------------
        elif velocity_y < -self.pull_threshold * torso_height:  # stricter velocity
            event = "Pull"

        # -------------------------------------------------
        # Simplified defense list: remove bobbing/weaving and complex rolls
        # -------------------------------------------------
        # (Shoulder roll and complex moves removed for stability and coach-focus)

        # -------------------------------------------------
        # Blocking handled separately in detect_block for more direct glove contact detection, so removed these more heuristic block detections
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
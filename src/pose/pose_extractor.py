import numpy as np
from ultralytics import YOLO


class PoseExtractor:

    def __init__(self):

        # Pose model
        self.model = YOLO("yolov8n-pose.pt")

        # Last known pose for each track
        self.pose_memory = {}


    def extract(self, frame, boxes, track_ids):

        poses_for_tracks = {}

        for box, tid in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            h, w = frame.shape[:2]

            # Clamp bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                poses_for_tracks[tid] = None
                continue

            results = self.model(
                roi,
                verbose=False,
                device=0
            )[0]

            if results.keypoints is None:

                # fallback to last pose
                if tid in self.pose_memory:
                    poses_for_tracks[tid] = self.pose_memory[tid]
                else:
                    poses_for_tracks[tid] = None

                continue


            kpts = results.keypoints.xy.cpu().numpy()

            if len(kpts) == 0:

                if tid in self.pose_memory:
                    poses_for_tracks[tid] = self.pose_memory[tid]
                else:
                    poses_for_tracks[tid] = None

                continue


            pose = kpts[0]

            # Convert ROI coordinates → frame coordinates
            pose[:, 0] += x1
            pose[:, 1] += y1

            poses_for_tracks[tid] = pose

            self.pose_memory[tid] = pose

        return poses_for_tracks
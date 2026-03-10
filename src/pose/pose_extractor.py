import numpy as np
from ultralytics import YOLO


class PoseExtractor:

    def __init__(self):

        # YOLO pose model
        self.model = YOLO("yolov8n-pose.pt")

        # Pose memory per track
        self.pose_memory = {}


    def _iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        if boxAArea == 0 or boxBArea == 0:
            return 0

        return interArea / float(boxAArea + boxBArea - interArea)


    def extract(self, frame, boxes, track_ids):

        """
        Returns poses aligned to each track_id
        """

        poses_for_tracks = {tid: None for tid in track_ids}

        results = self.model(frame, verbose=False)[0]

        if results.keypoints is None:
            return poses_for_tracks


        pose_boxes = results.boxes.xyxy.cpu().numpy()
        pose_kpts = results.keypoints.xy.cpu().numpy()


        for track_box, tid in zip(boxes, track_ids):

            best_iou = 0
            best_pose = None

            for pbox, pkpt in zip(pose_boxes, pose_kpts):

                iou = self._iou(track_box, pbox)

                if iou > best_iou:
                    best_iou = iou
                    best_pose = pkpt

            if best_iou > 0.2:
                poses_for_tracks[tid] = best_pose
                self.pose_memory[tid] = best_pose

            else:
                # fallback to last known pose if occluded
                if tid in self.pose_memory:
                    poses_for_tracks[tid] = self.pose_memory[tid]

        return poses_for_tracks
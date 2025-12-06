#Video loading and frame editor
import cv2

def iterate_frames(video_path, stride=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield frame_idx, timestamp, frame

        frame_idx += 1

    cap.release()
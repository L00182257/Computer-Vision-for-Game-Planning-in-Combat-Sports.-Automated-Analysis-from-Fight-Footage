#Video loading and frame editor
import cv2

def iterate_frames(video_path, stride=1):
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % stride == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield idx, timestamp, frame
        
        idx += 1

    cap.release()
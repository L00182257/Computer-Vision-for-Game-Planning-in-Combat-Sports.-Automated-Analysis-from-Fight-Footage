from ultralytics import YOLO
import cv2
import numpy as np

from src.detection.filtering.fighter_filtering import FighterFilter 

VIDEO_PATH = "data/raw/Max H VS Justin G.mp4"

def main():
    model = YOLO("yolov8n.pt")
    filterer = FighterFilter()

    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            device=0,
            persist=True,
            classes=[0],
            conf=0.4,
            verbose=False
        )[0]

        if results.boxes is None:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()

        if results.boxes.id is None:
          continue  # skip frame if no tracking IDs yet

        track_ids = results.boxes.id.cpu().numpy().astype(int)


        boxes, track_ids = filterer.filter(boxes, track_ids, frame.shape)

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"FIGHTER", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Filtered Fighters", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
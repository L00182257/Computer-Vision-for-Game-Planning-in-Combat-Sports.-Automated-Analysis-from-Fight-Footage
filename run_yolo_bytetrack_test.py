from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/raw/Max H VS Justin G.mp4"

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            device=0,
            persist=True,
            classes=[0],   # person only
            conf=0.4,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow("YOLO + ByteTrack", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
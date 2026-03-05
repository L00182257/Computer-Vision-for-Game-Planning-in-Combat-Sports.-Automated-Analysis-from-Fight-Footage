import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk

from src.detection.filtering.fighter_filtering import FighterFilter
from src.identity.fighter_identity import FighterIdentity


VIDEO_PATH = "data/raw/(Test) Max H vs Justin G.mp4"


# ---------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------

COLOUR_PALETTE = {
    "Red":    (0, 200, 200),
    "Blue":   (110, 200, 200),
    "Green":  (60, 200, 200),
    "Yellow": (30, 200, 200),
    "Orange": (15, 200, 200),
    "Purple": (140, 200, 200),
    "Pink":   (165, 200, 200),
    "White":  (0, 0, 255),
    "Black":  (0, 0, 30),
    "Grey":   (0, 0, 120),
    "Brown":  (20, 150, 150),
}


# ---------------------------------------------------------
# GUI COLOUR SELECTION
# ---------------------------------------------------------

def select_fighter_colours():

    result = {}

    def submit():
        if combo_a.get() == combo_b.get():
            error_label.config(text="Fighters cannot share the same colour.")
            return

        result["A"] = COLOUR_PALETTE[combo_a.get()]
        result["B"] = COLOUR_PALETTE[combo_b.get()]
        root.destroy()

    root = tk.Tk()
    root.title("Select Fighter Shorts Colours")
    root.geometry("400x250")

    ttk.Label(root, text="Fighter A Shorts Colour").pack(pady=5)
    combo_a = ttk.Combobox(root, values=list(COLOUR_PALETTE.keys()))
    combo_a.current(0)
    combo_a.pack()

    ttk.Label(root, text="Fighter B Shorts Colour").pack(pady=5)
    combo_b = ttk.Combobox(root, values=list(COLOUR_PALETTE.keys()))
    combo_b.current(1)
    combo_b.pack()

    error_label = ttk.Label(root, text="", foreground="red")
    error_label.pack(pady=5)

    ttk.Button(root, text="Confirm", command=submit).pack(pady=15)

    root.mainloop()

    return result["A"], result["B"]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\nRunning YOLO + ByteTrack + Shorts Identity Test\n")

    fighter_a_hsv, fighter_b_hsv = select_fighter_colours()

    model = YOLO("yolov8n.pt")
    filterer = FighterFilter(min_area_ratio=0.02)
    identity_assigner = FighterIdentity(fighter_a_hsv, fighter_b_hsv)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            device=0,
            persist=True,
            classes=[0],   # person class
            conf=0.4,
            verbose=False
        )[0]

        if results.boxes is None or results.boxes.id is None:
            cv2.imshow("Fighters", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            continue

        raw_boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        boxes, track_ids = filterer.filter(
            raw_boxes,
            track_ids,
            frame.shape
        )

        identities = identity_assigner.assign(frame, boxes, track_ids)

        for box, identity in zip(boxes, identities):

            x1, y1, x2, y2 = map(int, box)

            if identity == "A":
                color = (0, 0, 255)
                label = "Fighter A"
            elif identity == "B":
                color = (255, 0, 0)
                label = "Fighter B"
            else:
                color = (0, 255, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Fighters", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
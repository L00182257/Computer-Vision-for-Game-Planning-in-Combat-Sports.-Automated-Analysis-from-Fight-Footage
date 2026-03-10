import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

from ultralytics import YOLO

from src.detection.filtering.fighter_filtering import FighterFilter
from src.identity.fighter_identity import FighterIdentity
from src.pose.pose_extractor import PoseExtractor
from src.events.defence_detector import DefenceDetector


COLOUR_PALETTE = {
    "Red": (0,200,200),
    "Blue": (110,200,200),
    "Green": (60,200,200),
    "Yellow": (30,200,200),
    "Orange": (15,200,200),
    "Purple": (140,200,200),
    "Pink": (165,200,200),
    "White": (0,0,255),
    "Black": (0,0,30),
    "Grey": (0,0,120),
}


class CombatVisionGUI:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Combat Vision System")
        self.root.geometry("1300x780")

        self.video_path = None
        self.cap = None

        self.running = False
        self.paused = True
        self.processing_enabled = False

        self.playback_speed = 1.0

        self.detector = YOLO("yolov8n.pt")

        self.filterer = FighterFilter()
        self.pose_extractor = PoseExtractor()
        self.defence_detector = DefenceDetector()

        self.identity_assigner = None

        self._build_layout()

        self.root.mainloop()

    # --------------------------------------------------

    def _build_layout(self):

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        controls = ttk.Frame(main, width=320)
        controls.pack(side="left", fill="y", padx=10, pady=10)

        video_frame = ttk.Frame(main)
        video_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        ttk.Button(controls, text="Load Video", command=self._load_video).pack(pady=5)

        ttk.Button(controls, text="Play", command=self._play).pack(pady=5)
        ttk.Button(controls, text="Pause", command=self._pause).pack(pady=5)
        ttk.Button(controls, text="Stop", command=self._stop).pack(pady=5)

        ttk.Label(controls, text="Playback Speed").pack(pady=4)

        self.speed_slider = tk.Scale(
            controls,
            from_=0.25,
            to=2.0,
            resolution=0.25,
            orient="horizontal",
            command=self._set_speed
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack()

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Fighter A Shorts").pack()
        self.combo_a = ttk.Combobox(
            controls,
            values=list(COLOUR_PALETTE.keys()),
            state="disabled"
        )
        self.combo_a.pack()

        ttk.Label(controls, text="Fighter B Shorts").pack()
        self.combo_b = ttk.Combobox(
            controls,
            values=list(COLOUR_PALETTE.keys()),
            state="disabled"
        )
        self.combo_b.pack()

        ttk.Button(
            controls,
            text="Enable Identity",
            command=self._enable_processing
        ).pack(pady=6)

    # --------------------------------------------------

    def _load_video(self):

        path = filedialog.askopenfilename(
            filetypes=[("Video Files","*.mp4 *.avi *.mov *.mkv")]
        )

        if not path:
            return

        self._stop()

        self.video_path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            print("Failed to open video.")
            return

        ret, frame = self.cap.read()

        if ret:
            self._update_canvas(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.combo_a.config(state="readonly")
        self.combo_b.config(state="readonly")

        self.running = True
        self.paused = True

        threading.Thread(
            target=self._video_loop,
            daemon=True
        ).start()

    # --------------------------------------------------

    def _play(self):
        if self.running:
            self.paused = False

    def _pause(self):
        self.paused = True

    def _stop(self):

        self.running = False
        self.paused = True

        if self.cap:
            self.cap.release()
            self.cap = None

    def _set_speed(self, value):
        self.playback_speed = float(value)

    # --------------------------------------------------

    def _enable_processing(self):

        if not self.paused:
            print("Pause video before enabling identity.")
            return

        fighter_a = COLOUR_PALETTE[self.combo_a.get()]
        fighter_b = COLOUR_PALETTE[self.combo_b.get()]

        self.identity_assigner = FighterIdentity(
            fighter_a,
            fighter_b
        )

        self.processing_enabled = True

        print("Identity enabled.")

    # --------------------------------------------------

    def _video_loop(self):

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        while self.running and self.cap:

            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()

            if not ret:
                break

            if self.processing_enabled:

                results = self.detector.track(
                    source=frame,
                    persist=True,
                    conf=0.4,
                    device=0,
                    verbose=False
                )[0]

                if results.boxes is not None and results.boxes.id is not None:

                    boxes = results.boxes.xyxy.cpu().numpy()
                    track_ids = results.boxes.id.cpu().numpy().astype(int)

                    boxes, track_ids = self.filterer.filter(
                        boxes,
                        track_ids,
                        frame.shape
                    )

                    identities = self.identity_assigner.assign(
                        frame,
                        boxes,
                        track_ids
                    )

                    poses = self.pose_extractor.extract(
                        frame,
                        boxes,
                        track_ids
                    )

                    for box, tid, identity in zip(boxes, track_ids, identities):

                        x1,y1,x2,y2 = map(int,box)

                        color = (0,255,255)
                        label = "Unknown"

                        if identity == "A":
                            color = (0,0,255)
                            label = "Fighter A"

                        elif identity == "B":
                            color = (255,0,0)
                            label = "Fighter B"

                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                        cv2.putText(
                            frame,
                            label,
                            (x1,max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                        pose = poses.get(tid)

                        if pose is not None and identity is not None:

                            for kp in pose:
                                cv2.circle(
                                    frame,
                                    (int(kp[0]),int(kp[1])),
                                    3,
                                    (0,255,0),
                                    -1
                                )

                            event = self.defence_detector.detect(
                                pose,
                                identity
                            )

                            if event == "Duck":

                                cv2.rectangle(
                                    frame,
                                    (x1,y1),
                                    (x2,y2),
                                    (0,255,255),
                                    4
                                )

                                cv2.putText(
                                    frame,
                                    "DUCK",
                                    (x1,y1-40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (0,255,255),
                                    3
                                )

            self._update_canvas(frame)

            time.sleep(1.0/(fps*self.playback_speed))

        self.running = False

    # --------------------------------------------------

    def _update_canvas(self,frame):

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w>0 and h>0:
            img = img.resize((w,h))

        self.photo = ImageTk.PhotoImage(img)

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.photo
        )
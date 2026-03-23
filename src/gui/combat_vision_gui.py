import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import time
import numpy as np
import torch

from ultralytics import YOLO

from src.detection.filtering.fighter_filtering import FighterFilter
from src.identity.fighter_identity import FighterIdentity
from src.pose.pose_extractor import PoseExtractor
from src.events.defence_detector import DefenceDetector
from src.events.event_logger import EventLogger
from src.pattern.prefixspan_miner import PrefixSpanMiner
from src.analytics.coach_insights import CoachInsights


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
        self.root.title("Combat Vision Analytics - Coach Dashboard")
        self.root.geometry("1600x900")

        self.cap = None
        self.total_frames = 0
        self.frame_idx = 0

        self.playing = False
        self.processing_enabled = False

        self.playback_speed = 1.0

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use a CUDA-enabled GPU and drivers.")

        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 0:
            raise RuntimeError("CUDA device count reports 0; cannot run on GPU")

        self.device = "cuda:1" if self.gpu_count > 1 else "cuda:0"
        print(f"Using device {self.device} (GPU count: {self.gpu_count})")

        self.detector = YOLO("yolov8n.pt")
        self.detector.to(self.device)

        self.filterer = FighterFilter()
        self.pose_extractor = PoseExtractor(device=self.device)
        self.defence_detector = DefenceDetector()
        self.event_logger = EventLogger()

        self.pattern_miner = PrefixSpanMiner()
        self.coach_insights = CoachInsights()

        self.identity_assigner = None

        self.show_pose = True
        self.show_boxes = True

        self._build_layout()

        self.root.after(10, self._update_loop)

        self.root.mainloop()

    # ------------------------------------------------
    # GUI LAYOUT - TABBED INTERFACE
    # ------------------------------------------------

    def _build_layout(self):

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Video Playback & Real-time Analysis
        self.playback_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.playback_tab, text="Video Playback")

        # Tab 2: Event Log & Insights
        self.insights_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.insights_tab, text="Insights & Patterns")

        # Tab 3: Export & Reports
        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.export_tab, text="Export & Reports")

        self._build_playback_tab()
        self._build_insights_tab()
        self._build_export_tab()

    def _build_playback_tab(self):

        main = ttk.Frame(self.playback_tab)
        main.pack(fill="both", expand=True)

        # Left panel: Controls
        controls = ttk.Frame(main, width=350)
        controls.pack(side="left", fill="y", padx=10, pady=10)

        # Right panel: Video canvas
        video_frame = ttk.Frame(main)
        video_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Controls
        ttk.Button(controls, text="Load Video", command=self._load_video).pack(pady=5)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Button(controls, text="Play", command=self._play).pack(pady=3)
        ttk.Button(controls, text="Pause", command=self._pause).pack(pady=3)
        ttk.Button(controls, text="Stop", command=self._stop).pack(pady=3)
        ttk.Button(controls, text="Step Frame", command=self._step_frame).pack(pady=3)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Playback Speed").pack()
        self.speed = tk.Scale(
            controls,
            from_=0.25,
            to=2.0,
            resolution=0.25,
            orient="horizontal",
            command=self._set_speed
        )
        self.speed.set(1.0)
        self.speed.pack()

        ttk.Separator(controls).pack(fill="x", pady=10)

        self.timeline = tk.Scale(
            controls,
            from_=0,
            to=100,
            orient="horizontal",
            command=self._seek_frame
        )
        self.timeline.pack(fill="x")

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Checkbutton(
            controls,
            text="Enable Processing",
            command=self._toggle_processing
        ).pack()

        ttk.Checkbutton(
            controls,
            text="Show Pose",
            command=lambda: setattr(self, "show_pose", not self.show_pose)
        ).pack()

        ttk.Checkbutton(
            controls,
            text="Show Boxes",
            command=lambda: setattr(self, "show_boxes", not self.show_boxes)
        ).pack()

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
            command=self._enable_identity
        ).pack(pady=6)

        ttk.Separator(controls).pack(fill="x", pady=10)

        self.info_label = ttk.Label(controls, text="Frame: 0 | FPS: 0")
        self.info_label.pack()

        # Real-time event feed
        ttk.Label(controls, text="Recent Events:").pack(pady=5)
        self.event_feed = scrolledtext.ScrolledText(controls, width=40, height=10)
        self.event_feed.pack()

    def _build_insights_tab(self):

        main = ttk.Frame(self.insights_tab)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: Event summary
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")

        ttk.Button(left, text="Refresh Insights", command=self._refresh_insights).pack(pady=5)

        self.insights_text = scrolledtext.ScrolledText(left, width=50, height=20)
        self.insights_text.pack()

        # Right: Pattern visualization
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        ttk.Button(right, text="Mine Patterns", command=self._mine_patterns).pack(pady=5)

        self.pattern_canvas = tk.Canvas(right, bg="white")
        self.pattern_canvas.pack(fill="both", expand=True)

    def _build_export_tab(self):

        main = ttk.Frame(self.export_tab)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Button(main, text="Export Events CSV", command=self._export_events).pack(pady=10)
        ttk.Button(main, text="Export Insights JSON", command=self._export_insights).pack(pady=10)
        ttk.Button(main, text="Export Annotated Video", command=self._export_video).pack(pady=10)

        self.export_status = ttk.Label(main, text="")
        self.export_status.pack(pady=20)

    # ------------------------------------------------
    # VIDEO LOAD
    # ------------------------------------------------

    def _load_video(self):

        path = filedialog.askopenfilename(
            filetypes=[("Video Files","*.mp4 *.avi *.mov *.mkv")]
        )

        if not path:
            return

        self.cap = cv2.VideoCapture(path)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.timeline.config(to=self.total_frames)

        self.combo_a.config(state="readonly")
        self.combo_b.config(state="readonly")

        self.frame_idx = 0

        self._read_frame()

    # ------------------------------------------------
    # PLAYBACK CONTROLS
    # ------------------------------------------------

    def _play(self):
        self.playing = True

    def _pause(self):
        self.playing = False

    def _stop(self):

        self.playing = False

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.frame_idx = 0

    def _step_frame(self):

        self.playing = False
        self._read_frame()

    def _set_speed(self, val):
        self.playback_speed = float(val)

    # ------------------------------------------------
    # SEEK
    # ------------------------------------------------

    def _seek_frame(self, val):

        if self.cap is None:
            return

        frame = int(val)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        self.frame_idx = frame

        self._read_frame()

    # ------------------------------------------------
    # PROCESSING TOGGLE
    # ------------------------------------------------

    def _toggle_processing(self):

        self.processing_enabled = not self.processing_enabled

    # ------------------------------------------------
    # IDENTITY
    # ------------------------------------------------

    def _enable_identity(self):

        fighter_a = COLOUR_PALETTE[self.combo_a.get()]
        fighter_b = COLOUR_PALETTE[self.combo_b.get()]

        self.identity_assigner = FighterIdentity(
            fighter_a,
            fighter_b
        )

        print("Identity enabled.")

    # ------------------------------------------------
    # FRAME LOOP
    # ------------------------------------------------

    def _update_loop(self):

        if self.playing and self.cap:

            start = time.time()

            self._read_frame()

            delay = max(1, int(1000 / (30 * self.playback_speed)))

            elapsed = time.time() - start

            fps = 1.0 / max(elapsed, 0.001)

            self.info_label.config(
                text=f"Frame: {self.frame_idx} | FPS: {fps:.1f}"
            )

        self.root.after(10, self._update_loop)

    # ------------------------------------------------
    # READ FRAME
    # ------------------------------------------------

    def _read_frame(self):

        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        self.frame_idx += 1
        self.event_logger.update_frame(self.frame_idx)

        self.timeline.set(self.frame_idx)

        if self.processing_enabled:

            frame = self._process_frame(frame)

        self._display(frame)

    # ------------------------------------------------
    # PROCESS FRAME
    # ------------------------------------------------

    def _process_frame(self, frame):

        results = self.detector.track(
            source=frame,
            persist=True,
            conf=0.4,
            verbose=False
        )[0]

        if results.boxes is None or results.boxes.id is None:
            return frame

        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        boxes, track_ids = self.filterer.filter(
            boxes,
            track_ids,
            frame.shape
        )

        if self.identity_assigner is not None:
            identities = self.identity_assigner.assign(
                frame,
                boxes,
                track_ids
            )
        else:
            identities = [None] * len(track_ids)

        poses = self.pose_extractor.extract(
            frame,
            boxes,
            track_ids
        )

        active_events = set()

        for box, tid, identity in zip(boxes, track_ids, identities):

            x1,y1,x2,y2 = map(int,box)

            if self.show_boxes:

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            pose = poses.get(tid)

            if pose is not None and identity:

                if self.show_pose:

                    for kp in pose:
                        cv2.circle(
                            frame,
                            (int(kp[0]),int(kp[1])),
                            3,
                            (0,255,255),
                            -1
                        )

                event = self.defence_detector.detect(pose, identity)

                if event:

                    active_events.add((identity,event))

                    self.event_logger.update(identity,event)

                    cv2.putText(
                        frame,
                        event,
                        (x1,y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,255),
                        2
                    )

                    # Update real-time event feed
                    self.event_feed.insert(tk.END, f"Frame {self.frame_idx}: {identity} - {event}\n")
                    self.event_feed.see(tk.END)

        self.event_logger.finalize_inactive(active_events)

        return frame

    # ------------------------------------------------
    # DISPLAY
    # ------------------------------------------------

    def _display(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    # ------------------------------------------------
    # INSIGHTS & PATTERNS
    # ------------------------------------------------

    def _refresh_insights(self):

        insights = self.coach_insights.summarize_events(self.event_logger)

        text = f"Total events: {insights.get('total_events', 0)}\n"
        text += f"Dominant action: {insights.get('dominant_defensive_action', 'N/A')}\n\n"

        for fighter in ["A","B"]:
            actions = insights.get("per_fighter", {}).get(fighter, {})
            text += f"Fighter {fighter}:\n"
            if actions:
                for event, count in actions.items():
                    text += f"  {event}: {count}\n"
            else:
                text += "  None\n"
            text += "\n"

        self.insights_text.delete(1.0, tk.END)
        self.insights_text.insert(tk.END, text)

    def _mine_patterns(self):

        sequences = self.event_logger.get_sequences()

        if not sequences or all(len(s) == 0 for s in sequences):
            messagebox.showinfo("Pattern Mining", "No events recorded yet.")
            return

        patterns = self.pattern_miner.mine(sequences, min_support=2, max_pattern_len=5)

        # Simple visualization: list top patterns
        self.pattern_canvas.delete("all")
        y = 20
        for p in patterns[:10]:
            self.pattern_canvas.create_text(10, y, text=f"{p['pattern']} (support {p['support']})", anchor="nw")
            y += 20

    # ------------------------------------------------
    # EXPORT
    # ------------------------------------------------

    def _export_events(self):

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )

        if not path:
            return

        self.event_logger.export_csv(path)
        self.export_status.config(text=f"Events exported to {path}")

    def _export_insights(self):

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )

        if not path:
            return

        insights = self.coach_insights.summarize_events(self.event_logger)
        self.coach_insights.export_json(insights, path)
        self.export_status.config(text=f"Insights exported to {path}")

    def _export_video(self):

        # Placeholder: implement annotated video export
        if self.cap is None:
            messagebox.showerror("Export Video", "No video loaded.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Files", "*.mp4")]
        )

        if not path:
            return

        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.event_logger = EventLogger()  # Reset logger for export

        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.export_status.config(text="Exporting video...")

        for frame_num in range(total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_idx = frame_num + 1
            self.event_logger.update_frame(self.frame_idx)

            # Process frame for annotations
            if self.processing_enabled:
                frame = self._process_frame_for_export(frame)

            out.write(frame)

            # Update progress
            if frame_num % 100 == 0:
                self.export_status.config(text=f"Exporting... {frame_num}/{total_frames}")

        out.release()
        self.export_status.config(text=f"Video exported to {path}")

        messagebox.showinfo("Export Video", f"Annotated video saved to {path}")

    def _process_frame_for_export(self, frame):

        results = self.detector.track(
            source=frame,
            persist=True,
            conf=0.4,
            verbose=False
        )[0]

        if results.boxes is None or results.boxes.id is None:
            return frame

        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        boxes, track_ids = self.filterer.filter(
            boxes,
            track_ids,
            frame.shape
        )

        if self.identity_assigner is not None:
            identities = self.identity_assigner.assign(
                frame,
                boxes,
                track_ids
            )
        else:
            identities = [None] * len(track_ids)

        poses = self.pose_extractor.extract(
            frame,
            boxes,
            track_ids
        )

        active_events = set()

        for box, tid, identity in zip(boxes, track_ids, identities):

            x1, y1, x2, y2 = map(int, box)

            if self.show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            pose = poses.get(tid)

            if pose is not None and identity:
                if self.show_pose:
                    for kp in pose:
                        cv2.circle(
                            frame,
                            (int(kp[0]), int(kp[1])),
                            3,
                            (0, 255, 255),
                            -1
                        )

                event = self.defence_detector.detect(pose, identity)

                if event:
                    active_events.add((identity, event))
                    self.event_logger.update(identity, event)

                    cv2.putText(
                        frame,
                        event,
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )

        self.event_logger.finalize_inactive(active_events)

        return frame
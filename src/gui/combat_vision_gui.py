import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import json

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
        self.processing_enabled = True  

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

        self.selected_fighter_hsv = {
            "A": COLOUR_PALETTE["Red"],
            "B": COLOUR_PALETTE["Blue"]
        }

        self.selected_fighter_bgr = {
            "A": self._hsv_to_bgr(self.selected_fighter_hsv["A"]),
            "B": self._hsv_to_bgr(self.selected_fighter_hsv["B"])
        }

        self.short_picker_target = None
        self.current_frame_orig = None

        self.show_pose = True
        self.show_boxes = True

        # Round configuration
        self.fps = 30.0
        self.rounds_count = tk.IntVar(value=3)
        self.round_duration_sec = tk.IntVar(value=180)
        self.show_round_overlay = tk.BooleanVar(value=True)
        self.selected_round_actions = tk.Variable(value="Block Duck Pull")

        # Chart canvas reference
        self.chart_canvas_tk = None
        self.chart_figure = None

        self._build_layout()

        self.root.after(10, self._update_loop)

        self.root.mainloop()

    # ------------------------------------------------
    # GUI LAYOUT - TABBED INTERFACE
    # ------------------------------------------------

    def _hsv_to_bgr(self, hsv):
        h, s, v = hsv
        hsv_pixel = np.uint8([[[h, s, v]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])

    def _set_fighter_color(self, fighter, hsv):
        self.selected_fighter_hsv[fighter] = hsv
        self.selected_fighter_bgr[fighter] = self._hsv_to_bgr(hsv)

        # Keep combo in sync for user feedback
        if fighter == "A":
            self.combo_a.set(self._nearest_color_name(hsv))
        else:
            self.combo_b.set(self._nearest_color_name(hsv))

        if self.identity_assigner is not None:
            self.identity_assigner = FighterIdentity(
                self.selected_fighter_hsv["A"],
                self.selected_fighter_hsv["B"]
            )

    def _nearest_color_name(self, hsv):
        # Find the closest COLOUR_PALETTE key by hue distance (circular) + saturation/value
        h, s, v = hsv
        best_name = None
        best_dist = float('inf')
        for name, (ph, ps, pv) in COLOUR_PALETTE.items():
            dh = min(abs(ph - h), 180 - abs(ph - h))
            ds = abs(ps - s)
            dv = abs(pv - v)
            dist = dh + ds * 0.2 + dv * 0.2
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def _start_short_picker(self, fighter):
        self.short_picker_target = fighter
        messagebox.showinfo("Shorts Color Picker", f"Click on the video area to pick shorts color for Fighter {fighter}.")

    def _on_canvas_click(self, event):
        if self.short_picker_target is None or self.current_frame_orig is None:
            return

        frame_h, frame_w = self.current_frame_orig.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w == 0 or canvas_h == 0:
            return

        fx = int(event.x * frame_w / canvas_w)
        fy = int(event.y * frame_h / canvas_h)

        if fx < 0 or fy < 0 or fx >= frame_w or fy >= frame_h:
            return

        bgr = self.current_frame_orig[fy, fx]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        self._set_fighter_color(self.short_picker_target, (int(hsv[0]), int(hsv[1]), int(hsv[2])))

        messagebox.showinfo("Shorts Color Picker", f"Fighter {self.short_picker_target} color set by dropper.")
        self.short_picker_target = None

    def _compute_round_metrics(self, events, fps, round_duration_sec, max_rounds=None):
        """Compute per-round action counts for both fighters."""
        if fps <= 0 or round_duration_sec <= 0:
            return {}, []

        frames_per_round = fps * round_duration_sec
        round_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for event in events:
            start_frame = event.get("start_frame", 0)
            fighter = event.get("fighter")
            event_name = event.get("event")

            round_idx = int(start_frame / frames_per_round)
            if max_rounds and round_idx >= max_rounds:
                continue

            round_num = round_idx + 1
            round_stats[fighter][round_num][event_name] += 1

        # Build a clean dict with all rounds from 1 to max for both fighters
        result = {"A": {}, "B": {}}
        if events:
            max_round = int(max(e.get("start_frame", 0) for e in events) / frames_per_round) + 1
            if max_rounds:
                max_round = min(max_round, max_rounds)
            for fighter in ["A", "B"]:
                for r in range(1, max_round + 1):
                    result[fighter][r] = dict(round_stats[fighter].get(r, {}))

        return result, list(range(1, len(result.get("A", {})) + 1))

    def _get_round_from_frame(self, frame_idx):
        """Convert frame index to round number."""
        if self.fps <= 0 or self.round_duration_sec.get() <= 0:
            return 1
        frames_per_round = self.fps * self.round_duration_sec.get()
        return int(frame_idx / frames_per_round) + 1

    def _plot_round_metrics(self, round_stats, rounds):
        """Create and display matplotlib figure with round-based metrics."""
        if not rounds or not round_stats:
            return

        actions = self.selected_round_actions.get().split()

        try:
            if self.chart_figure:
                plt.close(self.chart_figure)
            if self.chart_canvas_tk:
                self.chart_canvas_tk.get_tk_widget().destroy()

            self.chart_figure, axes = plt.subplots(1, len(actions), figsize=(15, 4))
            if len(actions) == 1:
                axes = [axes]

            for idx, action in enumerate(actions):
                ax = axes[idx]

                a_counts = [round_stats.get("A", {}).get(r, {}).get(action, 0) for r in rounds]
                b_counts = [round_stats.get("B", {}).get(r, {}).get(action, 0) for r in rounds]

                x = np.arange(len(rounds))
                width = 0.35

                ax.bar(x - width / 2, a_counts, width, label="Fighter A", color=self.selected_fighter_bgr["A"], alpha=0.8)
                ax.bar(x + width / 2, b_counts, width, label="Fighter B", color=self.selected_fighter_bgr["B"], alpha=0.8)

                ax.set_xlabel("Round")
                ax.set_ylabel(f"{action} Count")
                ax.set_title(f"{action} per Round")
                ax.set_xticks(x)
                ax.set_xticklabels(rounds)
                ax.legend()
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()

            # Embed into Tkinter
            self.chart_canvas_tk = FigureCanvasTkAgg(self.chart_figure, master=self.chart_frame)
            self.chart_canvas_tk.draw()
            self.chart_canvas_tk.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            print(f"Error plotting chart: {e}")

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
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Controls
        ttk.Button(controls, text="Load Video", command=self._load_video).pack(pady=5)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Button(controls, text="Play", command=self._play).pack(pady=3)
        ttk.Button(controls, text="Pause", command=self._pause).pack(pady=3)
        ttk.Button(controls, text="Stop", command=self._stop).pack(pady=3)
        ttk.Button(controls, text="Step Frame", command=self._step_frame).pack(pady=3)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Playback Speed (max 10x)").pack()
        self.speed = tk.Scale(
            controls,
            from_=0.25,
            to=10.0,
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

        frame_jump_frame = ttk.Frame(controls)
        frame_jump_frame.pack(pady=5)
        ttk.Label(frame_jump_frame, text="Go to frame:").pack(side="left")
        self.frame_entry = ttk.Entry(frame_jump_frame, width=8)
        self.frame_entry.pack(side="left", padx=4)
        ttk.Button(frame_jump_frame, text="Go", command=self._jump_to_frame).pack(side="left")

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Processing: Enabled (always on)").pack(pady=2)
        ttk.Label(controls, text="Pose Overlay: Enabled").pack(pady=2)
        ttk.Label(controls, text="Bounding Boxes: Enabled").pack(pady=2)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Round Configuration", font=("Arial", 10, "bold")).pack()
        
        ttk.Label(controls, text="Number of Rounds:").pack()
        rounds_spin = ttk.Spinbox(controls, from_=1, to=12, textvariable=self.rounds_count, width=8)
        rounds_spin.pack()

        ttk.Label(controls, text="Round Duration (sec):").pack()
        duration_spin = ttk.Spinbox(controls, from_=30, to=600, textvariable=self.round_duration_sec, width=8)
        duration_spin.pack()

        ttk.Label(controls, text="FPS (auto-detected):").pack()
        self.fps_label = ttk.Label(controls, text=f"{self.fps:.1f}")
        self.fps_label.pack()

        ttk.Checkbutton(controls, text="Show Round Overlay", variable=self.show_round_overlay).pack()

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Fighter A Shorts").pack()
        self.combo_a = ttk.Combobox(
            controls,
            values=list(COLOUR_PALETTE.keys()),
            state="disabled"
        )
        self.combo_a.pack()

        ttk.Button(
            controls,
            text="Pick A Shorts from Frame",
            command=lambda: self._start_short_picker("A")
        ).pack(pady=2)

        ttk.Label(controls, text="Fighter B Shorts").pack()
        self.combo_b = ttk.Combobox(
            controls,
            values=list(COLOUR_PALETTE.keys()),
            state="disabled"
        )
        self.combo_b.pack()

        ttk.Button(
            controls,
            text="Pick B Shorts from Frame",
            command=lambda: self._start_short_picker("B")
        ).pack(pady=2)

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

        # Right: Chart visualization
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(right, text="Round-Based Metrics", font=("Arial", 10, "bold")).pack(pady=5)
        
        ttk.Label(right, text="Select actions:").pack()
        action_frame = ttk.Frame(right)
        action_frame.pack(pady=5)
        actions = ["Block", "Duck", "Pull"]
        for action in actions:
            ttk.Checkbutton(action_frame, text=action, onvalue=action, offvalue="", 
                          variable=tk.StringVar()).pack(anchor="w")

        self.chart_frame = ttk.Frame(right)
        self.chart_frame.pack(fill="both", expand=True)

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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

        self.fps_label.config(text=f"{self.fps:.1f}")

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

    def _jump_to_frame(self):

        if self.cap is None:
            return

        try:
            frame = int(self.frame_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Frame", "Please enter a valid frame number.")
            return

        if frame < 0 or frame >= self.total_frames:
            messagebox.showerror("Invalid Frame", "Frame number out of range.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.frame_idx = frame
        self.timeline.set(frame)
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

        if self.combo_a.get() not in COLOUR_PALETTE or self.combo_b.get() not in COLOUR_PALETTE:
            messagebox.showerror("Identity", "Please select both Fighter A and Fighter B shorts colors.")
            return

        self._set_fighter_color("A", COLOUR_PALETTE[self.combo_a.get()])
        self._set_fighter_color("B", COLOUR_PALETTE[self.combo_b.get()])

        self.identity_assigner = FighterIdentity(
            self.selected_fighter_hsv["A"],
            self.selected_fighter_hsv["B"]
        )

        print("Identity enabled.")

    # ------------------------------------------------
    # FRAME LOOP
    # ------------------------------------------------

    def _update_loop(self):

        if self.playing and self.cap:

            start = time.time()

            # High-speed mode: read multiple frames per update for smoother fast playback
            frames_to_advance = max(1, int(self.playback_speed))
            for _ in range(frames_to_advance):
                self._read_frame()
                if not self.playing or self.cap is None:
                    break

            delay = max(1, int(1000 / (30 * self.playback_speed)))

            elapsed = time.time() - start

            fps = 1.0 / max(elapsed, 0.001)

            self.info_label.config(
                text=f"Frame: {self.frame_idx} | FPS: {fps:.1f} | Speed: {self.playback_speed:.2f}x"
            )

            # Avoid unnecessary CPU spin when running at slow speed
            time.sleep(delay / 1000.0)

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

        self.current_frame_orig = frame.copy()

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

        pose_by_identity = {i: p for (b, t, i), p in zip(zip(boxes, track_ids, identities), [poses.get(t) for t in track_ids]) if i in ("A", "B")}

        # Draw round overlay
        if self.show_round_overlay.get():
            round_num = self._get_round_from_frame(self.frame_idx)
            round_text = f"Round {round_num}"
            
            # Get action counts for current round
            round_stats, rounds = self._compute_round_metrics(
                self.event_logger.events,
                self.fps,
                self.round_duration_sec.get(),
                max_rounds=self.rounds_count.get()
            )
            
            if round_num in rounds:
                a_actions = round_stats.get("A", {}).get(round_num, {})
                b_actions = round_stats.get("B", {}).get(round_num, {})
                round_text += f"\nA: " + ", ".join([f"{a}={c}" for a, c in a_actions.items()]) if a_actions else "A: -"
                round_text += f"\nB: " + ", ".join([f"{a}={c}" for a, c in b_actions.items()]) if b_actions else "B: -"
            
            # Draw semi-transparent overlay background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            # Draw text
            y_offset = 30
            for line in round_text.split('\n'):
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

        for box, tid, identity in zip(boxes, track_ids, identities):

            x1,y1,x2,y2 = map(int,box)

            draw_color = self.selected_fighter_bgr.get(identity, (0,255,0))

            if self.show_boxes:

                cv2.rectangle(frame,(x1,y1),(x2,y2),draw_color,2)

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

                opponent_id = "B" if identity == "A" else "A"
                opponent_pose = pose_by_identity.get(opponent_id)

                block_event = self.defence_detector.detect_block(pose, opponent_pose)
                event = block_event or self.defence_detector.detect(pose, identity)

                if event:

                    active_events.add((identity,event))

                    self.event_logger.update(identity,event)

                    cv2.putText(
                        frame,
                        event,
                        (x1,y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        draw_color,
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

        # Compute round-based metrics
        round_stats, rounds = self._compute_round_metrics(
            self.event_logger.events,
            self.fps,
            self.round_duration_sec.get(),
            max_rounds=self.rounds_count.get()
        )

        if rounds:
            text += f"\n--- Round-by-Round Summary ---\n"
            for round_num in rounds:
                text += f"\nRound {round_num}:\n"
                for fighter in ["A", "B"]:
                    actions = round_stats.get(fighter, {}).get(round_num, {})
                    text += f"  {fighter}: "
                    if actions:
                        text += ", ".join([f"{a}={c}" for a, c in actions.items()])
                    else:
                        text += "No actions"
                    text += "\n"

        self.insights_text.delete(1.0, tk.END)
        self.insights_text.insert(tk.END, text)

        # Plot chart
        if round_stats:
            self._plot_round_metrics(round_stats, rounds)

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
        
        # Include round-based metrics
        round_stats, rounds = self._compute_round_metrics(
            self.event_logger.events,
            self.fps,
            self.round_duration_sec.get(),
            max_rounds=self.rounds_count.get()
        )
        
        insights["round_data"] = {
            "rounds": rounds,
            "stats": round_stats,
            "fps": self.fps,
            "round_duration_sec": self.round_duration_sec.get()
        }
        
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

        pose_by_identity = {}
        for box, tid, identity in zip(boxes, track_ids, identities):
            if identity in ("A", "B") and poses.get(tid) is not None:
                pose_by_identity[identity] = poses.get(tid)

        # Draw round overlay
        if self.show_round_overlay.get():
            round_num = self._get_round_from_frame(self.frame_idx)
            round_text = f"Round {round_num}"
            
            # Get action counts for current round
            round_stats, rounds = self._compute_round_metrics(
                self.event_logger.events,
                self.fps,
                self.round_duration_sec.get(),
                max_rounds=self.rounds_count.get()
            )
            
            if round_num in rounds:
                a_actions = round_stats.get("A", {}).get(round_num, {})
                b_actions = round_stats.get("B", {}).get(round_num, {})
                round_text += f"\nA: " + ", ".join([f"{a}={c}" for a, c in a_actions.items()]) if a_actions else "A: -"
                round_text += f"\nB: " + ", ".join([f"{a}={c}" for a, c in b_actions.items()]) if b_actions else "B: -"
            
            # Draw semi-transparent overlay background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            # Draw text
            y_offset = 30
            for line in round_text.split('\n'):
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

        for box, tid, identity in zip(boxes, track_ids, identities):

            x1, y1, x2, y2 = map(int, box)

            draw_color = self.selected_fighter_bgr.get(identity, (0, 255, 0))

            if self.show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

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

                opponent_id = "B" if identity == "A" else "A"
                opponent_pose = pose_by_identity.get(opponent_id)

                block_event = self.defence_detector.detect_block(pose, opponent_pose)
                event = block_event or self.defence_detector.detect(pose, identity)

                if event:
                    active_events.add((identity, event))
                    self.event_logger.update(identity, event)

                    cv2.putText(
                        frame,
                        event,
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        draw_color,
                        2
                    )

        self.event_logger.finalize_inactive(active_events)

        return frame
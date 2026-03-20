import csv
from collections import defaultdict


class EventLogger:

    def __init__(self):

        # active events being tracked
        self.active_events = {}

        # finished events
        self.events = []

        # frame index
        self.frame_idx = 0


    def update_frame(self):

        self.frame_idx += 1


    def update(self, fighter_id, event_name):

        key = (fighter_id, event_name)

        # start new event
        if key not in self.active_events:

            self.active_events[key] = {
                "fighter": fighter_id,
                "event": event_name,
                "start_frame": self.frame_idx,
                "last_frame": self.frame_idx
            }

        else:

            self.active_events[key]["last_frame"] = self.frame_idx


    def finalize_inactive(self, active_events_current_frame):

        finished = []

        for key in list(self.active_events.keys()):

            fighter, event = key

            if (fighter, event) not in active_events_current_frame:

                data = self.active_events[key]

                start = data["start_frame"]
                end = data["last_frame"]

                duration = end - start + 1

                data["end_frame"] = end
                data["duration_frames"] = duration

                self.events.append(data)

                finished.append(key)


        for key in finished:
            del self.active_events[key]


    def export_csv(self, path):

        with open(path, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fighter",
                    "event",
                    "start_frame",
                    "end_frame",
                    "duration_frames"
                ]
            )

            writer.writeheader()

            for e in self.events:
                writer.writerow(e)


    def summary(self):

        counts = defaultdict(int)

        for e in self.events:
            counts[(e["fighter"], e["event"])] += 1

        return counts
import json
from collections import defaultdict


class CoachInsights:

    def __init__(self):
        pass

    def summarize_events(self, event_logger):
        counts = event_logger.summary()

        total = sum(counts.values())

        fighter_summary = {
            "A": defaultdict(int),
            "B": defaultdict(int)
        }

        for (fighter, event), c in counts.items():
            fighter_summary[fighter][event] = c

        insights = {
            "total_events": total,
            "per_fighter": {
                "A": dict(fighter_summary["A"]),
                "B": dict(fighter_summary["B"])
            }
        }

        if total > 0:
            insights["dominant_defensive_action"] = max(counts.items(), key=lambda x: x[1])[0]

        return insights

    def export_json(self, insights, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2)

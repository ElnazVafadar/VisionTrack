from collections import defaultdict, deque
from math import hypot
import csv
import json
from pathlib import Path

import cv2
import numpy as np


class MotionAnalyzer:
    def __init__(self, trail_length=30):
        self.positions = {}
        self.trails = defaultdict(lambda: deque(maxlen=trail_length))
        self.previous_side = {}
        self.crossings = defaultdict(int)
        self.unique_ids = set()

    def update(self, track_id, center, frame_index, fps, line_y):
        x, y = center
        self.unique_ids.add(track_id)
        self.trails[track_id].append((int(x), int(y)))

        speed = 0.0
        direction = "stationary"

        previous = self.positions.get(track_id)
        if previous is not None:
            old_x, old_y, old_frame = previous
            frame_delta = max(frame_index - old_frame, 1)
            distance = hypot(x - old_x, y - old_y)
            speed = distance * fps / frame_delta
            dx, dy = x - old_x, y - old_y

            if abs(dx) + abs(dy) > 1:
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"

        self.positions[track_id] = (x, y, frame_index)

        current_side = 1 if y >= line_y else -1
        old_side = self.previous_side.get(track_id)
        if old_side is not None and current_side != old_side:
            self.crossings["down" if current_side == 1 else "up"] += 1
        self.previous_side[track_id] = current_side

        return speed, direction

    def trail(self, track_id):
        return list(self.trails[track_id])


class CsvLogger:
    fields = [
        "frame", "timestamp_seconds", "track_id", "class_id",
        "class_name", "confidence", "center_x", "center_y",
        "speed_pixels_per_second", "direction", "bbox_width",
        "bbox_height", "mask_area_pixels", "mask_area_ratio"
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def save_summary(path, frame_count, fps, analyzer, class_counts):
    data = {
        "processed_frames": frame_count,
        "source_fps": fps,
        "unique_tracked_objects": len(analyzer.unique_ids),
        "line_crossings": dict(analyzer.crossings),
        "class_counts": dict(class_counts)
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

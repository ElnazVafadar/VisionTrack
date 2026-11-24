from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

from .analytics import MotionAnalyzer, CsvLogger, save_summary
from .visualization import draw_overlay


def open_source(source):
    value = int(source) if str(source).isdigit() else str(source)
    capture = cv2.VideoCapture(value)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def run(settings, source):
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(settings.model)
    capture = open_source(source)

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * settings.line_position)

    video_writer = None
    if settings.save_video:
        output_path = settings.output_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, source_fps, (width, height)
        )

    logger = CsvLogger(settings.output_dir / "detections.csv")
    analyzer = MotionAnalyzer(settings.trail_length)
    class_counts = Counter()
    frame_index = 0
    processing_start = perf_counter()
    inference_fps = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            start = perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker=settings.tracker,
                conf=settings.confidence,
                iou=settings.iou,
                device=settings.device or None,
                verbose=False
            )[0]
            elapsed = max(perf_counter() - start, 1e-6)
            inference_fps = 1.0 / elapsed

            if result.boxes is not None:
                masks = result.masks.data.cpu().numpy() if result.masks is not None else None

                for index, box in enumerate(result.boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    class_name = result.names.get(cls_id, str(cls_id))
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    speed = 0.0
                    direction = "untracked"

                    if track_id >= 0:
                        speed, direction = analyzer.update(
                            track_id, (center_x, center_y), frame_index,
                            source_fps, line_y
                        )

                    mask_area = 0
                    if masks is not None and index < len(masks):
                        resized_mask = cv2.resize(
                            masks[index].astype("uint8"), (width, height)
                        )
                        mask_area = int(np.count_nonzero(resized_mask))

                    bbox_width = max(x2 - x1, 0)
                    bbox_height = max(y2 - y1, 0)
                    class_counts[class_name] += 1

                    logger.write({
                        "frame": frame_index,
                        "timestamp_seconds": round(frame_index / source_fps, 3),
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "center_x": round(float(center_x), 2),
                        "center_y": round(float(center_y), 2),
                        "speed_pixels_per_second": round(float(speed), 2),
                        "direction": direction,
                        "bbox_width": round(float(bbox_width), 2),
                        "bbox_height": round(float(bbox_height), 2),
                        "mask_area_pixels": mask_area,
                        "mask_area_ratio": round(mask_area / (width * height), 6)
                    })

            output = draw_overlay(
                frame, result, analyzer, frame_index, source_fps,
                line_y, inference_fps
            )

            if video_writer is not None:
                video_writer.write(output)

            if settings.show:
                cv2.imshow("VisionTrack", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        capture.release()
        logger.close()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    save_summary(
        settings.output_dir / "summary.json",
        frame_index,
        source_fps,
        analyzer,
        class_counts
    )
    print(f"Processed frames: {frame_index}")
    print(f"Unique tracked objects: {len(analyzer.unique_ids)}")
    print(f"Results saved to: {settings.output_dir.resolve()}")

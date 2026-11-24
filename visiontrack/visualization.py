import cv2
import numpy as np


def draw_overlay(frame, results, analyzer, frame_index, fps, line_y, inference_fps):
    output = frame.copy()
    class_names = results.names
    boxes = results.boxes

    if results.masks is not None and len(results.masks.data):
        masks = results.masks.data.cpu().numpy()
        overlay = output.copy()
        for mask in masks:
            resized = cv2.resize(mask.astype("uint8"), (output.shape[1], output.shape[0]))
            color = np.random.default_rng(int(mask.sum()) % 100000).integers(50, 255, 3).tolist()
            overlay[resized > 0] = (
                0.55 * overlay[resized > 0] + 0.45 * np.array(color)
            ).astype("uint8")
        output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0)

    height, width = output.shape[:2]
    cv2.line(output, (0, line_y), (width, line_y), (0, 220, 255), 2)
    cv2.putText(output, "COUNTING LINE", (12, max(line_y - 10, 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    if boxes is not None:
        for index, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            track_value = box.id[0].item() if box.id is not None else -1
            track_id = int(track_value)
            label = class_names.get(cls_id, str(cls_id))

            if track_id >= 0:
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                speed, direction = analyzer.update(
                    track_id, center, frame_index, fps, line_y
                )
                points = analyzer.trail(track_id)
                for start, end in zip(points[:-1], points[1:]):
                    cv2.line(output, start, end, (255, 180, 0), 2)

                text = f"ID {track_id} {label} {confidence:.2f} {speed:.1f}px/s {direction}"
            else:
                text = f"{label} {confidence:.2f}"

            cv2.rectangle(output, (x1, y1), (x2, y2), (70, 220, 70), 2)
            cv2.putText(output, text, (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 220, 70), 2)

    cv2.putText(output, f"Frame: {frame_index}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(output, f"Source FPS: {fps:.1f} | Processing FPS: {inference_fps:.1f}",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(output, f"Tracked objects: {len(analyzer.unique_ids)}",
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return output

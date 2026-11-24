import argparse

from visiontrack.config import Settings
from visiontrack.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Object detection, tracking, segmentation and motion analysis"
    )
    parser.add_argument("--source", required=True, help="Video path or webcam index")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--line-position", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.line_position < 1:
        raise ValueError("--line-position must be between 0 and 1")

    settings = Settings(
        model=args.model,
        tracker=args.tracker,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        trail_length=args.trail_length,
        line_position=args.line_position,
        output_dir=__import__("pathlib").Path(args.output),
        save_video=not args.no_video,
        show=not args.no_display
    )
    run(settings, args.source)


if __name__ == "__main__":
    main()

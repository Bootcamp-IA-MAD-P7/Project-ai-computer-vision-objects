"""Run the AeroVision AI detector on an image, a folder of images or a video."""

import argparse
import sys
from collections import Counter
from pathlib import Path

DEFAULT_WEIGHTS = Path("models/best.pt")
DEFAULT_OUTPUT = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect airplanes, birds, drones and helicopters with YOLO11n."
    )
    parser.add_argument("--source", required=True, help="Image, folder or video to analyse.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the trained weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence to keep a detection.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for inference.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Folder where results are saved.")
    parser.add_argument("--device", default=None, help="Device: 0 for GPU, cpu for CPU. Automatic by default.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not Path(args.source).exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1
    if not Path(args.weights).exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 1

    # Imported here so that argument errors are reported without loading PyTorch.
    from ultralytics import YOLO

    model = YOLO(args.weights)
    counts = Counter()

    # stream=True processes one frame at a time and keeps memory usage low on long videos.
    for result in model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        stream=True,
        verbose=False,
        project=args.output,
        name="predictions",
        exist_ok=True,
    ):
        for class_id in result.boxes.cls.tolist():
            counts[result.names[int(class_id)]] += 1

    print("Detections per class (summed over all frames):")
    for name, total in sorted(counts.items()):
        print(f"  {name}: {total}")
    print(f"Results saved in: {Path(args.output) / 'predictions'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the AeroVision AI detector on an image, a folder of images or a video."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_OUTPUT, DEFAULT_WEIGHTS
from src.detector import AeroVisionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect airplanes, birds, drones and helicopters with YOLO11n."
    )
    parser.add_argument("--source", required=True, help="Image, folder or video to analyse.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the trained weights.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Minimum confidence to keep a detection.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image size used for inference.")
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

    detector = AeroVisionDetector(
        weights=Path(args.weights),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    counts: Counter = Counter()
    for result in detector.predict_source(args.source, args.output):
        counts.update(detector.count_classes(result))

    print("Detections per class (summed over all frames):")
    for name, total in sorted(counts.items()):
        print(f"  {name}: {total}")
    print(f"Results saved in: {Path(args.output) / 'predictions'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

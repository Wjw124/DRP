"""Compute per-video flow patches and tracks after IPV frame extraction."""

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from ruamel.yaml import YAML
from wmlib.motion.models import MotionModels
from wmlib.motion.extraction import AtomicActionExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N eligible videos (0: all)",
    )
    args = parser.parse_args()
    model_config = YAML(typ="safe").load(args.models)
    extractor = AtomicActionExtractor(MotionModels(model_config, segmentation=False))
    patch_dir, track_dir = args.output / "patches", args.output / "tracks"
    patch_dir.mkdir(parents=True, exist_ok=True)
    track_dir.mkdir(parents=True, exist_ok=True)
    rows = [line.split() for line in args.list.read_text().splitlines()]
    rows = [row for row in rows if int(row[1]) >= args.min_frames]
    if args.limit:
        rows = rows[: args.limit]
    for video_id, count, _ in rows:
        patch_path, track_path = (
            patch_dir / f"patches_{video_id}.pt",
            track_dir / f"pred_tracks_{video_id}.pt",
        )
        if patch_path.exists() and track_path.exists():
            continue
        # Video-ID seeding gives the same sampling when resuming a partial run.
        seed = (args.seed + int(video_id)) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        frames = []
        for frame in range(1, int(count) + 1):
            with Image.open(args.frames / video_id / f"{frame:06d}.jpg") as image:
                frames.append(np.asarray(image.convert("RGB")))
        patches, tracks = extractor.ssv2(np.stack(frames))
        torch.save(patches, patch_path)
        torch.save(tracks, track_path)
        print(f"{video_id}: {count} frames, {tracks.shape[2]} tracks", flush=True)


if __name__ == "__main__":
    main()

"""SSV2 RGB clips and matching precomputed atomic-action tensors."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SomethingV2(Dataset):
    def __init__(self, root_path, list_file, patches_dir, tracks_dir, segment_len=25):
        self.root = Path(root_path)
        self.patches_dir = Path(patches_dir)
        self.tracks_dir = Path(tracks_dir)
        self.segment_len = segment_len
        self.records = []
        for line in Path(list_file).read_text().splitlines():
            video_id, frame_count, label = line.split()
            if int(frame_count) >= segment_len:
                self.records.append((video_id, int(frame_count), int(label)))
        if not self.records:
            raise ValueError(
                f"No videos with at least {segment_len} frames in {list_file}"
            )

    @property
    def total_steps(self):
        return sum(count for _, count, _ in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        video_id, count, _ = self.records[index]
        # Resolve all three modalities by video ID, independent of list order.
        patches = torch.load(
            self.patches_dir / f"patches_{video_id}.pt",
            map_location="cpu",
            weights_only=True,
        ).squeeze(0)
        tracks = torch.load(
            self.tracks_dir / f"pred_tracks_{video_id}.pt",
            map_location="cpu",
            weights_only=True,
        ).squeeze(0)
        if patches.shape[0] != count or tracks.shape[0] != count:
            raise ValueError(
                f"{video_id}: RGB list has {count} frames, patches/tracks have {patches.shape[0]}/{tracks.shape[0]}"
            )
        start = np.random.randint(0, count - self.segment_len + 1)
        frames = []
        for frame in range(start + 1, start + self.segment_len + 1):
            with Image.open(self.root / video_id / f"{frame:06d}.jpg") as image:
                frames.append(np.asarray(image.convert("RGB")))
        return (
            np.stack(frames),
            patches[start : start + self.segment_len],
            tracks[start : start + self.segment_len],
        )

"""Expose offline video clips through the training replay interface."""

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from ..core.replay import seed_worker


class ClipStream(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            image, patches, tracks = self.dataset[np.random.randint(len(self.dataset))]
            first = np.zeros(len(image), dtype=bool)
            first[0] = True
            yield dict(
                image=image.transpose(0, 3, 1, 2),
                patches=patches,
                tracks=tracks,
                is_first=first,
                action=np.zeros((len(image), 1), dtype=np.float32),
            )


def collate_clips(batch):
    return {
        k: torch.stack([torch.as_tensor(item[k]) for item in batch]) for k in batch[0]
    }


class DummyReplay:
    def __init__(self, dataset):
        self.video_dataset = dataset

    def dataset(self, batch, length, pin_memory=True, num_workers=8):
        if length != self.video_dataset.segment_len:
            raise ValueError(
                "Dataset segment length must equal the training batch length"
            )
        return DataLoader(
            ClipStream(self.video_dataset),
            batch_size=batch,
            pin_memory=pin_memory,
            drop_last=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_clips,
        )

    @property
    def stats(self):
        return dict(
            total_steps=0,
            total_episodes=0,
            loaded_steps=self.video_dataset.total_steps,
            loaded_episodes=len(self.video_dataset),
        )

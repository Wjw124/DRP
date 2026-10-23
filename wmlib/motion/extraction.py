"""Extract tracked local flow patches for SSV2 and completed RL episodes."""

import numpy as np
import torch


def crop_patches(flows, tracks, patch_size=16):
    """Crop [B,T,H,W,2] at [B,T,K,2]; zero-pad image boundaries."""
    batch, time, height, width, _ = flows.shape
    count = tracks.shape[2]
    patches = flows.new_zeros(batch, time, count, patch_size, patch_size, 2)
    half = patch_size // 2
    for b in range(batch):
        for t in range(time):
            for k in range(count):
                x, y = tracks[b, t, k]
                x, y = min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1)
                x0, x1 = max(x - half, 0), min(x + half, width)
                y0, y1 = max(y - half, 0), min(y + half, height)
                px, py = half - (x - x0), half - (y - y0)
                patches[b, t, k, py : py + y1 - y0, px : px + x1 - x0] = flows[
                    b, t, y0:y1, x0:x1
                ]
    return patches


def queries_from_mask(mask, count, frame, device):
    indices = np.flatnonzero(mask)
    picked = np.random.choice(indices, count, replace=False)
    xy = np.stack([picked % mask.shape[1], picked // mask.shape[1]], axis=-1)
    xy = torch.from_numpy(xy).float().to(device)[None]
    return torch.cat([xy.new_full((1, count, 1), frame), xy], dim=-1)


class AtomicActionExtractor:
    def __init__(self, models):
        self.models = models

    def _video(self, frames):
        video = torch.as_tensor(
            np.asarray(frames), dtype=torch.float32, device=self.models.device
        )
        return video.permute(0, 3, 1, 2).unsqueeze(0)

    def _flow(self, video):
        flow, _ = self.models.flow(video[0, :-1], video[0, 1:])
        flow = flow.permute(0, 2, 3, 1)
        # Align T observations with T motion entries; the final flow is zero.
        return torch.cat([flow, torch.zeros_like(flow[:1])], dim=0)[None]

    @torch.no_grad()
    def ssv2(self, frames):
        """Track 16 points from a last-frame temporal-difference mask, then retrack."""
        video = self._video(frames)
        _, time, _, height, width = video.shape
        flow = self._flow(video)
        attribution = (video - video.mean(dim=1, keepdim=True)).abs().amax(dim=2)
        threshold = torch.quantile(attribution[0, -1].float().flatten(), 0.80)
        mask = (attribution[0, -1] >= threshold).cpu().numpy()
        count = 16
        queries = queries_from_mask(mask, count, time - 1, video.device)
        with torch.autocast(
            device_type=video.device.type, enabled=video.device.type == "cuda"
        ):
            tracks, visibility = self.models.tracker(
                video, queries=queries, backward_tracking=True
            )
        variance = tracks.var(dim=1).sum(dim=-1)
        selected = torch.where(variance[0] > 10.0)[0]
        if len(selected) == 0:
            selected = torch.topk(variance[0], k=10).indices
        queries = queries[:, selected]
        if queries.shape[1] < count:
            queries = queries.repeat(1, count // queries.shape[1] + 1, 1)[:, :count]
            noise_scale = 0.05 * height * (1 - torch.sigmoid(variance.mean()))
            queries[..., 1:] += torch.randn_like(queries[..., 1:]) * noise_scale
        with torch.autocast(
            device_type=video.device.type, enabled=video.device.type == "cuda"
        ):
            tracks, visibility = self.models.tracker(
                video, queries=queries, backward_tracking=True
            )
        return crop_patches(flow, tracks).cpu(), tracks.cpu()

    @torch.no_grad()
    def episode(self, frames):
        """Segment each frame, track 32 last-frame queries, retain the 16 most mobile."""
        video = self._video(frames)
        _, time, _, height, width = video.shape
        masks, box = [], None
        for frame in np.asarray(frames):
            mask, new_box = self.models.segment(
                frame.astype(np.uint8), last_known_box=box
            )
            if new_box is not None:
                box = new_box
            masks.append(mask)
        masks = torch.stack(masks)
        flow = self._flow(video) * masks[None, ..., None]
        mask = masks[-1].cpu().numpy()
        if np.count_nonzero(mask) < 32:
            mask = np.ones((height, width), dtype=bool)
        queries = queries_from_mask(mask, 32, time - 1, video.device)
        from .. import ENABLE_FP16

        with torch.autocast(
            device_type=video.device.type,
            enabled=ENABLE_FP16 and video.device.type == "cuda",
        ):
            tracks, visibility = self.models.tracker(
                video, queries=queries, backward_tracking=True
            )
        indices = tracks.var(dim=1).sum(dim=-1)[0].topk(16).indices
        tracks = tracks[:, :, indices]
        patches = crop_patches(flow, tracks)
        return {
            "patches": patches[0].cpu().numpy().astype(np.float32),
            "tracks": tracks[0].cpu().numpy().astype(np.float32),
        }

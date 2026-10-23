# Motion data

## SSV2

Input RGB frames have shape `[T,64,64,3]`, dtype `uint8`, and values in `[0,255]`.

1. Estimate forward flow between neighboring frames using SEA-RAFT at twice the image resolution with four refinement iterations. Resize the flow back and rescale displacement to the 64 × 64 coordinate system. Append a zero flow for the final observation.
2. Compute absolute differences from the temporal mean image. On the last frame, take the maximum across color channels and threshold at the 80th percentile.
3. Sample 16 points in this mask and run CoTracker with last-frame queries and backward tracking.
4. Keep trajectories whose summed x/y temporal variance exceeds 10. If no trajectory qualifies, use the ten largest variances. Repeat selected queries to reach 16 and perturb repeated points with the variance-dependent noise scale.
5. Track the selected queries again and crop 16 × 16 two-channel flow patches at their coordinates. Coordinates are converted to integers and clamped to the image; crops extending beyond the boundary are padded with zeros.

The cache contains:

```text
motion_train/
├── patches/patches_<video_id>.pt      # float32 [1,T,16,16,16,2]
└── tracks/pred_tracks_<video_id>.pt   # float32 [1,T,16,2], x/y in pixels
```

RGB and both motion tensors are looked up by the same video ID and sliced at the same temporal offset. Train and validation splits should use separate cache directories. Pretraining clips have length 25; the learned temporal embeddings support up to 50 frames. The model's patch projection retains the tensor layout used by its checkpoints.

To process validation videos, run the same extraction command with `--list data/ssv2/val_video_folder.txt --output data/ssv2/motion_val`. A `--limit` extraction is intended for a matching subset list when testing the training entry point; a full training list requires all of its eligible videos to be cached.

## RL replay

Episode processing is called by `Replay.add_episode`. GroundingDINO and SAM 2 segment each frame. The detector uses `robot`, followed by the configurable fallback prompt in `configs/motion.yaml`; when detection is missing, the last detected box is reused. With no detected or previous box, the mask is empty.

Flow is masked by the foreground of the source frame. The last frame supplies 32 tracking queries. If fewer than 32 foreground pixels are available, queries are sampled from the full image. The 16 trajectories with greatest temporal variance are retained for patch extraction.

Stored episode fields include:

| Field | Shape |
| --- | --- |
| `image` | `[T,64,64,3]` |
| `patches` | `[T,16,16,16,2]` |
| `tracks` | `[T,16,2]` |
| `action` | `[T,A]` |
| `reward`, `is_first`, `is_last`, `is_terminal` | `[T]` |

`T` includes the reset observation, whose action is zero. An episode with `T` observations has `T-1` agent actions. The final flow is zero. Cached episodes already containing both motion fields are not processed again.

Online policy inference samples 16 foreground points in the reset observation, updates them on consecutive frame pairs, and keeps at most ten flows and track positions. The reset observation has zero motion. Training and evaluation drivers each carry their own tracking history.

# Local Motion Matters — DRP (CVPR 2026)

PyTorch implementation of **Local Motion Matters: A Deconstruct–Recompose Paradigm for Reinforcement Learning Pre-training from Videos**.

**Jinwen Wang, Youfang Lin, Xiaobo Hu, Shuo Wang, Kai Lv** · Beijing Jiaotong University

DRP extracts tracked local optical-flow patches (atomic actions), encodes their spatial and temporal relationships with a Dual-Attention Encoder, and aggregates them into motion tokens for world-model pretraining and downstream reinforcement learning.

## Setup

```bash
git clone https://github.com/Wjw124/DRP.git
cd DRP
```

Follow [installation](docs/installation.md) for the training environment, perception models, and simulator assets. Run all commands below from this repository's root. Training and motion extraction use CUDA; the online policy uses one environment per process.

## 1. Prepare Something-Something V2

Follow the **Datasets → Something-Something-V2** section of the [official IPV / ContextWM repository](https://github.com/thuml/ContextWM#datasets) to download SSV2, extract **64 × 64 RGB frames**, and generate `train_video_folder.txt` / `val_video_folder.txt` using its `extract_frames.py` and `process_somethingv2.py` scripts. Set the paths in those scripts for your machine.

The expected layout is:

```text
data/ssv2/
├── frames/
│   ├── 78687/000001.jpg
│   ├── 78687/000002.jpg
│   └── ...
├── train_video_folder.txt
└── val_video_folder.txt
```

Each list entry contains `video_id frame_count class_id`. Frames are one-indexed, six-digit JPEG files.

Set the model paths in [`configs/motion.yaml`](configs/motion.yaml), then extract tracks and flow patches:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_ssv2.py \
  --frames data/ssv2/frames \
  --list data/ssv2/train_video_folder.txt \
  --output data/ssv2/motion_train \
  --models configs/motion.yaml
```

This stage uses SEA-RAFT and CoTracker; Grounded SAM is used during RL fine-tuning. Videos shorter than 25 frames are excluded. Completed video pairs are skipped when restarting the command. Use `--limit 1` for an initial extraction check and a separate output directory for validation videos. [Data format and extraction details](docs/data.md).

## 2. Pretrain

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py \
  --frames data/ssv2/frames \
  --list data/ssv2/train_video_folder.txt \
  --motion-data data/ssv2/motion_train \
  --logdir runs/pretrain/seed0 \
  --seed 0 --steps 600000 --save_all_models True
```

[`configs/pretrain.yaml`](configs/pretrain.yaml) sets a batch size of 16 and clip length of 25. It jointly trains the motion encoder and world model. Use dotted flags such as `--dataset.batch 8` to override a configuration field. To use another split, pass its list and its own matching motion cache.

The output directory contains `variables.pt` and the four transferable module checkpoints:

```text
encoder_variables.pt
decoder_variables.pt
rssm_variables.pt
flow_mae_variables.pt
```

`--load-model /path/to/variables.pt` initializes model weights. Optimizer state and the training step are initialized for a new run.

## 3. Fine-tune

Point `--load_logdir` to the directory containing the four pretrained module checkpoints, and configure all perception models in `configs/motion.yaml`.

**DMControl Remastered — Walker Run**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --configs dmc_remastered contextualized \
  --task dmcr_walker_run --seed 1 \
  --logdir runs/finetune/walker_run \
  --load_logdir runs/pretrain/seed0 \
  --loss_scales.reward 1.0 --loss_scales.aux_reward 1.0
```

**DMControl Remastered — Hopper Stand**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --configs dmc_remastered dmcr_hopper contextualized \
  --task dmcr_hopper_stand --seed 0 \
  --logdir runs/finetune/hopper_stand \
  --load_logdir runs/pretrain/seed0 \
  --loss_scales.reward 1.0 --loss_scales.aux_reward 1.0
```

The `dmcr_hopper` preset fixes the camera while varying the other visual factors.

**Meta-World — Drawer Open**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --configs metaworld contextualized \
  --task metaworld_drawer_open --seed 0 \
  --logdir runs/finetune/drawer_open \
  --load_logdir runs/pretrain/seed0 \
  --loss_scales.reward 1.0 --loss_scales.aux_reward 1.0 \
  --encoder_ctx.ctx_aug erasing
```

For Meta-World, the first 100 replay-only updates keep the pretrained encoder, pretrained RSSM, and motion encoder frozen, then enable their optimizer. DMCR updates these modules from the start.

Completed episodes are processed automatically before entering replay: foreground segmentation, optical-flow estimation, point tracking, and local flow cropping. The resulting `patches` and `tracks` are stored with RGB, actions, rewards, and episode flags in `.npz` files. Policy inference updates tracks from the latest two observations and encodes a window of up to ten observations.

Each run saves its configuration, source snapshot, JSONL metrics, TensorBoard events, replay episodes, and `variables.pt`. `steps`, `eval_every`, and `time_limit` count agent actions; logged steps are multiplied by `action_repeat`. Evaluation runs every `eval_every` collected actions. Use a distinct log directory for each experiment.

## Code map

| Path | Purpose |
| --- | --- |
| `scripts/preprocess_ssv2.py` | Offline SSV2 motion extraction |
| `scripts/pretrain.py` | Video pretraining |
| `scripts/finetune.py` | RL collection, replay, training, and evaluation |
| `wmlib/motion/` | Frozen perception models and patch extraction |
| `wmlib/nets/motion/` | Dual attention, masked reconstruction, motion aggregation |
| `wmlib/agents/` | World models and actor–critic training |
| `wmlib/datasets/` | Aligned RGB / patch / track clip sampling |
| `wmlib/core/replay.py` | Episode storage and replay sampling |
| `configs/` | Stage and model settings |

The `flow_mae` checkpoint name refers to the motion encoder and masked reconstruction model; `cls_token` is the learned motion aggregation token.

## Validation

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

See [validation notes](docs/validation.md) for the tested paths and runtime. For a short training check, reduce the batch size and step count with command-line overrides. Full benchmark runs require the original training budgets.

## Citation and acknowledgments

```bibtex
@inproceedings{wang2026local,
  title={Local Motion Matters: A Deconstruct--Recompose Paradigm for Reinforcement Learning Pre-training from Videos},
  author={Wang, Jinwen and Lin, Youfang and Hu, Xiaobo and Wang, Shuo and Lv, Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

This code builds on [IPV / ContextWM](https://github.com/thuml/ContextWM), [APV](https://github.com/younggyoseo/apv), and [DreamerV2](https://github.com/danijar/dreamerv2). Perception uses [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [CoTracker](https://github.com/facebookresearch/co-tracker), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [SAM 2](https://github.com/facebookresearch/sam2). Environment code includes [DMControl Remastered](https://github.com/QData/dmc_remastered).

The inherited MIT license is retained in [LICENSE](LICENSE). The bundled DMControl Remastered environment code retains its [Apache 2.0 license](wmlib/envs/dmcr/LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution. External datasets, model weights, and dependencies retain their respective terms; they are obtained separately.

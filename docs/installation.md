# Installation

Use Linux, an NVIDIA GPU, and a CUDA-enabled PyTorch installation. `rsync` is used to save experiment source snapshots; `ffmpeg` supports video logging. Install the system OpenGL/EGL and compiler libraries required by your MuJoCo installation.

## Reference training environment

The runtime used for the release smoke checks has Python 3.8.13, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, NumPy 1.23.0, dm_control 0.0.318037100, and mujoco-py 2.0.2.8. The legacy simulator versions are retained for the task definitions used in these experiments.

```bash
conda create -n drp python=3.8.13 pip -y
conda activate drp
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The package constraints resolve for Linux / Python 3.8. The smoke checks use the existing training environment; a complete installation into an empty environment has not yet been validated.

## Perception repositories

```bash
mkdir -p third_party checkpoints
git clone https://github.com/princeton-vl/SEA-RAFT.git third_party/SEA-RAFT
git clone https://github.com/facebookresearch/co-tracker.git third_party/co-tracker
python -m pip install --no-deps -e third_party/co-tracker
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
python -m pip install --no-build-isolation --no-deps -e third_party/GroundingDINO

git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 98fcb164bf880f70799c324c283c758c4d20bf82
export PYTHONPATH="$PWD/third_party/sam2${PYTHONPATH:+:$PYTHONPATH}"
```

The SAM 2 checkout contains the SAM 2.1 image predictor. The pinned image predictor was also checked with the reference training stack. Importing its source package is the route used here; its optional CUDA postprocessing extension is not needed for the default image-predictor settings. This pinned revision's [installation metadata](https://github.com/facebookresearch/sam2/blob/98fcb164bf880f70799c324c283c758c4d20bf82/setup.py) declares newer Python/NumPy minimums than the legacy runtime above. For a modern environment, follow the [official SAM 2 instructions](https://github.com/facebookresearch/sam2#installation) and recheck the legacy simulator dependencies.

GroundingDINO builds a CUDA extension. Set `CUDA_HOME` to a matching CUDA toolkit when needed and follow its [official installation instructions](https://github.com/IDEA-Research/GroundingDINO#install) if compilation fails. Its text encoder uses `bert-base-uncased`. On an offline machine, download the model/tokenizer first and set `text_encoder_type` in its Swin-B config to the local BERT directory.

Download these weights from the respective official repositories:

| Model | Checkpoint | Official source |
| --- | --- | --- |
| SEA-RAFT medium | `Tartan-C-T-TSKH-spring540x960-M.pth` | [SEA-RAFT pretrained models](https://github.com/princeton-vl/SEA-RAFT#pretrained-models) |
| CoTracker3 offline | `scaled_offline.pth` | [CoTracker checkpoints](https://github.com/facebookresearch/co-tracker#installation-instructions) |
| GroundingDINO Swin-B | `groundingdino_swinb_cogcoor.pth` | [GroundingDINO checkpoints](https://github.com/IDEA-Research/GroundingDINO#checkpoints) |
| SAM 2.1 Hiera large | `sam2.1_hiera_large.pt` | [SAM 2.1 checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints) |

Place them under `checkpoints/`, or edit [`configs/motion.yaml`](../configs/motion.yaml). SSV2 extraction needs only SEA-RAFT and CoTracker. Fine-tuning additionally uses GroundingDINO and SAM 2.1. SEA-RAFT may fetch torchvision's ResNet-34 initialization weights on its first construction.

## DMControl Remastered

Download the [DMCR assets](https://github.com/QData/dmc_remastered/tree/main/dmc_remastered/assets) into `wmlib/envs/dmcr/assets`, preserving their subdirectory structure. For example:

```bash
git clone --depth 1 https://github.com/QData/dmc_remastered.git third_party/dmc_remastered
cp -r third_party/dmc_remastered/dmc_remastered/assets wmlib/envs/dmcr/assets
export MUJOCO_GL=egl
```

The model XMLs, floor textures, and sky textures are required even for a short environment check. Assets are excluded from this repository's source snapshots.

## Meta-World

Use the version specified by [IPV / ContextWM](https://github.com/thuml/ContextWM#dependencies):

```bash
python -m pip install --no-deps 'git+https://github.com/rlworkgroup/metaworld.git@a0009ed9a208ff9864a5c1368c04c273bb20dd06'
```

Install [MuJoCo 2.0](https://www.roboti.us/download.html) and configure `mujoco-py` according to its installation instructions. Add the MuJoCo binary directory to `LD_LIBRARY_PATH`; for a standard installation:

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Also include the NVIDIA library directory requested by `mujoco-py` on your system. This adapter uses the legacy Meta-World API; current Meta-World releases have different environment interfaces.

## Compute and storage

`CUDA_VISIBLE_DEVICES` selects the GPU. The supplied batch sizes are 16 for SSV2/DMCR and 50 for Meta-World. Reduce `--dataset.batch` for a small-memory smoke check. Full-episode tracking and foreground segmentation add computation at episode boundaries; the processed replay cache avoids repeating that work when sampling training batches.

Keep frames, motion caches, replay, and checkpoints on a disk with sufficient free space. A float32 SSV2 motion cache uses about 32 KiB per frame for the 16 patches, plus tracks. Run directories also contain module checkpoints and TensorBoard videos.

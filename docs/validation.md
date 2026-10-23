# Validation

The release candidate was exercised in the existing Linux / CUDA training environment (Python 3.8.13, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121). These are integration checks, not new benchmark results.

| Check | Result |
| --- | --- |
| Existing full pretrained world-model checkpoint | Strict state-dict loading passed |
| Encoder, decoder, pretrained RSSM, and motion encoder transfer | All four module checkpoints loaded strictly |
| Existing complete Walker Run fine-tuning checkpoint | All 626 state-dict entries matched; floating-point tensors were finite |
| Float32 training | Pretraining and fine-tuning forward/backward/optimizer steps and video reports passed |
| FP16 pretraining entry point | 16 iterations with real RGB and motion caches; finite gradients and changed motion-encoder parameters after scaler adaptation; all five checkpoint files saved |
| SSV2 perception | One 54-frame video processed by SEA-RAFT and CoTracker into 16 tracks and corresponding patches |
| RL perception | 12 DMCR observations processed by GroundingDINO, SAM 2.1, SEA-RAFT, and CoTracker; finite caches and zero final flow |
| Online policy | 12 finite actions, including the transition to the ten-frame history window |
| Fine-tuning entry point | Random prefill, cached replay, module transfer, evaluation, FP16 training with finite gradients in both world-model optimizers, and full checkpoint saving completed on a short Walker Run episode |
| Meta-World entry point | Drawer Open with context erasing completed reset/render/step, prefill, replay processing, strict module transfer, evaluation, short training, and checkpoint saving |
| SAM 2.1 pinned source | Image predictor loaded the large checkpoint and returned finite mask scores in the reference environment |
| Python dependencies | Constraints resolved for Linux / Python 3.8 |
| Data/model tests | Five tests cover video-ID alignment, boundary crops, motion-encoder causality and updates, replay persistence/counts, and random-prefill driver integration |

Run the lightweight checks from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Mixed precision can skip initial optimizer steps while GradScaler reduces its loss scale. In the short pretraining check, finite gradients and updates were observed from iteration 12 onward. The float32 checks separately verified optimizer updates without scaling.

A full 600k-step pretraining run, downstream benchmark sweeps, and installation into an empty environment have not been repeated during code cleanup. The provided run commands retain the experiment settings; smoke-check budgets are intentionally shorter.

# Validation

The release was exercised in the Linux / CUDA training environment (Ubuntu 20.04, Python 3.8.13, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121). These are integration checks, not new benchmark results.

| Check | Result |
| --- | --- |
| Existing full pretrained world-model checkpoint | Strict state-dict loading passed |
| Encoder, decoder, pretrained RSSM, and motion encoder transfer | All four module checkpoints loaded strictly |
| Existing complete Walker Run fine-tuning checkpoint | All 626 state-dict entries matched; floating-point tensors were finite |
| Float32 training | Pretraining and fine-tuning forward/backward/optimizer steps and video reports passed |
| FP16 pretraining entry point | 16 iterations with real RGB and motion caches; finite gradients and changed motion-encoder parameters after scaler adaptation; all five checkpoint files saved |
| SSV2 perception | One 54-frame video processed by SEA-RAFT and CoTracker into 16 tracks and corresponding patches |
| RL perception | 12 DMCR observations processed by GroundingDINO, SAM 2.1, SEA-RAFT, and CoTracker; finite caches and zero final flow |
| Full-length replay | A 500-action Walker episode (501 observations) completed segmentation, flow, tracking, and replay persistence; all motion values were finite and final flow was zero |
| Default training sizes | SSV2: 16 iterations at batch 16, length 25, eight data workers, with video logging and saving. DMCR batch 16 × 50 and Meta-World preset batch 50 × 50 both completed updates on the cached Walker episode with finite world-model gradients |
| Optimizer grouping and warmup | Pretrained encoder, RSSM, and flow MAE parameters belong to the same optimizer; a shortened warmup test kept all three unchanged during pre-updates and changed all three on the next update |
| Online policy | 12 finite actions, including the transition to the ten-frame history window |
| Fine-tuning entry point | Random prefill, cached replay, module transfer, evaluation, FP16 training with finite gradients in both world-model optimizers, and full checkpoint saving completed on a short Walker Run episode |
| Meta-World entry point | Drawer Open with context erasing completed reset/render/step, prefill, replay processing, strict module transfer, evaluation, short training, and checkpoint saving |
| SAM 2.1 pinned source | Image predictor loaded the large checkpoint and returned finite mask scores in the reference environment |
| Python dependencies | Constraints resolved for Linux / Python 3.8 |
| Fresh Python environment | Installed the requirements and compiled GroundingDINO in an isolated Python 3.8.13 environment; all seven tests passed, Hopper Stand / Drawer Open reset-render-step passed, and the four perception models processed real frames |
| Fresh-environment training entry points | Pretraining completed weight loading, float32 updates, video logging, and saving. Hopper Stand and Drawer Open with context erasing completed prefill, replay processing, module transfer, evaluation, FP16 updates, and saving. All three saved models had finite floating-point tensors and changed motion-encoder parameters |
| Data/model/logging tests | Seven tests cover video-ID alignment, boundary crops, motion-encoder causality and updates, replay persistence/counts, random-prefill driver integration, GIF encoding, and first-frame logging if encoding fails |
| FFmpeg versions | GIF encoding passed with FFmpeg 4.2.7 on Linux and FFmpeg 8.1 on macOS; the encoder no longer requires the removed `fifo` filter |

Run the lightweight checks from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Mixed precision can skip initial optimizer steps while GradScaler reduces its loss scale. In the short pretraining check, finite gradients and updates were observed from iteration 12 onward. The float32 checks separately verified optimizer updates without scaling.

On a 24 GiB RTX 4090, the 501-observation replay extraction peaked at about 19.2 GiB of PyTorch allocated memory. The two fine-tuning batch checks peaked at about 8.8 GiB and 20.0 GiB respectively, with the perception models resident. Memory use varies with video length, batch size, and backend workspace allocations.

The fresh Python environment reused existing system drivers, the CUDA toolkit, MuJoCo binaries, assets, data, and model checkpoints. GroundingDINO's distribution-name warning for headless OpenCV is explained in [installation](installation.md).

A full 600k-step pretraining run and downstream benchmark sweeps have not been repeated during code cleanup. The provided run commands retain the experiment settings; smoke-check budgets are intentionally shorter.

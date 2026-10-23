"""Pretrain the motion encoder and world model from cached SSV2 clips."""

import argparse
import collections
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ruamel.yaml import YAML
import wmlib
from wmlib import utils
from wmlib.agents.pretrain import APV_Pretrain
from wmlib.datasets import SomethingV2, DummyReplay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/pretrain.yaml",
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--motion-data", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument(
        "--load-model",
        type=Path,
        help="Optional full variables.pt to initialize model weights",
    )
    args, overrides = parser.parse_known_args()
    config = utils.Flags(utils.Config(YAML(typ="safe").load(args.config))).parse(
        overrides
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    args.logdir.mkdir(parents=True, exist_ok=True)
    config.save(args.logdir / "config.yaml")
    (args.logdir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.set_num_threads(1)
    wmlib.ENABLE_FP16 = config.precision == 16
    replay = DummyReplay(
        SomethingV2(
            args.frames,
            args.list,
            args.motion_data / "patches",
            args.motion_data / "tracks",
            segment_len=config.dataset.length,
        )
    )
    iterator = iter(replay.dataset(**config.dataset))
    # Offline pretraining needs only image/action shapes, not an RL simulator.
    obs_space = {"image": SimpleNamespace(shape=(*config.render_size, 3))}
    act_space = {"action": SimpleNamespace(shape=(1,))}
    step = utils.Counter(0)
    agent = APV_Pretrain(config, obs_space, act_space, step).to("cuda")
    if args.load_model:
        agent.load_state_dict(
            torch.load(args.load_model, map_location="cpu", weights_only=True),
            strict=True,
        )
    agent.init_optimizers()
    logger = utils.Logger(
        step,
        [
            utils.TerminalOutput(),
            utils.JSONLOutput(args.logdir),
            utils.TensorBoardOutputPytorch(args.logdir),
        ],
    )
    metrics = collections.defaultdict(list)

    def next_batch():
        dtype = torch.float16 if wmlib.ENABLE_FP16 else torch.float32
        return {k: v.to("cuda", dtype=dtype) for k, v in next(iterator).items()}

    state = None
    for _ in range(int(config.steps)):
        state, values = agent.train(next_batch(), state)
        step.increment()
        for key, value in values.items():
            metrics[key].append(value)
        if int(step) % int(config.log_every) == 0:
            logger.add({key: np.mean(value) for key, value in metrics.items()})
            logger.write(fps=True)
            metrics.clear()
        if config.video_every and int(step) % int(config.video_every) == 0:
            logger.add(agent.report(next_batch(), recon=True), prefix="train")
            logger.write()
        if int(step) % int(config.eval_every) == 0:
            agent.save_model(args.logdir)
        if config.save_all_models and int(step) % 50000 == 0:
            agent.save_model(args.logdir, "_s" + str(int(step)))
    agent.save_model(args.logdir)


if __name__ == "__main__":
    main()

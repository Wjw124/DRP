import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from wmlib.datasets import SomethingV2
from wmlib.motion.extraction import crop_patches, AtomicActionExtractor
from wmlib.nets.motion.pretrain import PatchMAE as PretrainMAE
from wmlib.nets.motion.finetune import PatchMAE as FinetuneMAE
from wmlib.core.replay import Replay
from wmlib.core.driver import Driver
from wmlib.agents.random_agent import RandomAgent
from types import SimpleNamespace


class DataAndMotionTest(unittest.TestCase):
    def test_rgb_patches_tracks_use_same_video_id(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "patches").mkdir()
            (root / "tracks").mkdir()
            # The list order is deliberately different from numeric video-ID order.
            (root / "list.txt").write_text("20 3 0\n10 3 1\n")
            for identifier in [10, 20]:
                (root / str(identifier)).mkdir()
                for t in range(1, 4):
                    Image.fromarray(np.full((8, 8, 3), identifier + t, np.uint8)).save(
                        root / str(identifier) / f"{t:06d}.jpg"
                    )
                values = torch.tensor(
                    [identifier + t for t in range(1, 4)], dtype=torch.float32
                )
                torch.save(
                    values.view(1, 3, 1, 1, 1, 1).expand(1, 3, 16, 16, 16, 2).clone(),
                    root / "patches" / f"patches_{identifier}.pt",
                )
                torch.save(
                    values.view(1, 3, 1, 1).expand(1, 3, 16, 2).clone(),
                    root / "tracks" / f"pred_tracks_{identifier}.pt",
                )
            dataset = SomethingV2(
                root,
                root / "list.txt",
                root / "patches",
                root / "tracks",
                segment_len=2,
            )
            for i in range(2):
                image, patches, tracks = dataset[i]
                np.testing.assert_array_equal(image[:, 0, 0, 0], patches[:, 0, 0, 0, 0])
                np.testing.assert_array_equal(image[:, 0, 0, 0], tracks[:, 0, 0])

    def test_patch_crop_padding_and_coordinates(self):
        flow = torch.arange(8 * 8 * 2).reshape(1, 1, 8, 8, 2).float()
        tracks = torch.tensor([[[[4, 4], [0, 0], [7, 7], [-5, -5]]]])
        patches = crop_patches(flow, tracks, 4)
        torch.testing.assert_close(patches[0, 0, 0], flow[0, 0, 2:6, 2:6])
        torch.testing.assert_close(patches[0, 0, 1, 2:, 2:], flow[0, 0, :2, :2])
        self.assertEqual(patches[0, 0, 1, :2].abs().sum(), 0)
        torch.testing.assert_close(patches[0, 0, 2, :3, :3], flow[0, 0, 5:8, 5:8])
        torch.testing.assert_close(patches[0, 0, 3], patches[0, 0, 1])

    def test_encoder_causality_gradient_and_stage_checkpoint(self):
        torch.manual_seed(5)
        kw = dict(
            embed_dim=48,
            num_heads=8,
            enc_layers=2,
            dec_layers=2,
            num_patches=16,
            mask_ratio=0.5,
        )
        pre, fine = PretrainMAE(**kw).eval(), FinetuneMAE(**kw).eval()
        fine.load_state_dict(pre.state_dict(), strict=True)
        patches = torch.randn(1, 6, 16, 16, 16, 2)
        tracks = torch.rand(1, 6, 16, 2) * 64
        with torch.no_grad():
            token = pre.get_flow_token(patches, tracks=tracks)
            torch.testing.assert_close(
                token, fine.get_flow_token(patches, tracks=tracks)
            )
            changed = patches.clone()
            changed[:, 3:] += 20
            altered = pre.get_flow_token(changed, tracks=tracks)
            torch.testing.assert_close(token[:, :3], altered[:, :3])
        before = pre.patch_embed.conv.weight.detach().clone()
        optimizer = torch.optim.Adam(pre.parameters(), lr=1e-3)
        _, loss = pre(patches, tracks=tracks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, pre.patch_embed.conv.weight))

    def test_replay_persists_motion_and_counts_actions(self):
        class Processor:
            def episode(self, frames):
                return dict(
                    patches=np.ones((len(frames), 16, 16, 16, 2), np.float32),
                    tracks=np.ones((len(frames), 16, 2), np.float32),
                )

        episode = dict(
            image=np.zeros((6, 64, 64, 3), np.uint8),
            action=np.zeros((6, 4)),
            reward=np.zeros(6),
            is_first=np.array([True] + [False] * 5),
            is_last=np.array([False] * 5 + [True]),
            is_terminal=np.zeros(6, bool),
        )
        with tempfile.TemporaryDirectory() as root:
            replay = Replay(root, motion=Processor(), minlen=3, maxlen=3)
            replay.add_episode(episode)
            self.assertEqual(replay.stats["total_steps"], 5)
            loaded = Replay(root, minlen=3, maxlen=3)
            self.assertEqual(loaded.stats["total_steps"], 5)
            sample = next(iter(loaded.dataset(batch=1, length=3)))
            self.assertEqual(tuple(sample["patches"].shape), (1, 3, 16, 16, 16, 2))
            self.assertEqual(tuple(sample["tracks"].shape), (1, 3, 16, 2))

    def test_random_prefill_driver_signature(self):
        class Env:
            act_space = {
                "action": SimpleNamespace(
                    shape=(2,), low=np.full(2, -1.0), high=np.ones(2)
                )
            }

            def reset(self):
                self.t = 0
                return self.obs(True)

            def step(self, action):
                self.t += 1
                return self.obs(False)

            def obs(self, first):
                return dict(
                    image=np.zeros((64, 64, 3), np.uint8),
                    reward=np.float32(0),
                    is_first=first,
                    is_last=self.t == 3,
                    is_terminal=False,
                )

        env = Env()
        episodes = []
        driver = Driver([env], "cpu")
        driver.on_episode(episodes.append)
        driver(RandomAgent(env.act_space), episodes=1)
        self.assertEqual(len(episodes[0]["image"]), 4)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

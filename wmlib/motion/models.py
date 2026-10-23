"""Frozen perception models shared by preprocessing, replay, and the policy."""

from argparse import Namespace
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F


class MotionModels:
    def __init__(self, config, device="cuda", segmentation=False):
        self.device = torch.device(device)
        self.config = config
        # SEA-RAFT exposes its inference modules from its core directory.
        sys.path.insert(0, str(Path(config["sea_raft_root"]).expanduser() / "core"))
        from raft import RAFT
        from cotracker.predictor import CoTrackerPredictor

        args = Namespace(
            use_var=True,
            var_min=0,
            var_max=10,
            pretrain="resnet34",
            initial_dim=64,
            block_dims=[64, 128, 256],
            radius=4,
            dim=128,
            num_blocks=2,
            iters=4,
        )
        self.raft = RAFT(args).to(self.device).eval()
        self.raft.load_state_dict(
            torch.load(config["sea_raft_checkpoint"], map_location="cpu"), strict=True
        )
        self.raft.requires_grad_(False)
        self.tracker = (
            CoTrackerPredictor(checkpoint=config["cotracker_checkpoint"])
            .to(self.device)
            .eval()
        )
        self.tracker.requires_grad_(False)
        self.predictor = None
        if segmentation:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from groundingdino.util.inference import load_model

            sam = build_sam2(
                config["sam2_config"],
                config["sam2_checkpoint"],
                device=str(self.device),
            )
            sam.eval().requires_grad_(False)
            self.predictor = SAM2ImagePredictor(sam)
            self.detector = load_model(
                config["grounding_config"],
                config["grounding_checkpoint"],
                device=str(self.device),
            )
            self.detector.eval().requires_grad_(False)

    @torch.no_grad()
    def flow(self, image1, image2):
        """SEA-RAFT at 2x resolution; return flow in original-image pixels."""
        image1 = F.interpolate(
            image1, scale_factor=2, mode="bilinear", align_corners=False
        )
        image2 = F.interpolate(
            image2, scale_factor=2, mode="bilinear", align_corners=False
        )
        batch = self.config.get("flow_batch_size", 32)
        flows, infos = [], []
        for a, b in zip(image1.split(batch), image2.split(batch)):
            output = self.raft(a, b, iters=4, test_mode=True)
            flows.append(
                F.interpolate(
                    output["flow"][-1],
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False,
                )
                * 0.5
            )
            infos.append(
                F.interpolate(output["info"][-1], scale_factor=0.5, mode="area")
            )
        return torch.cat(flows), torch.cat(infos)

    @torch.no_grad()
    def segment(
        self,
        frame_np,
        text_prompt="robot",
        box_threshold=0.35,
        text_threshold=0.25,
        last_known_box=None,
    ):
        """Grounded SAM mask, using the previous box when detection is missing."""
        from PIL import Image
        import groundingdino.datasets.transforms as transforms
        from groundingdino.util.inference import predict

        transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(Image.fromarray(frame_np), None)
        prompts = [text_prompt, self.config["fallback_prompt"]]
        for prompt in prompts:
            boxes, scores, _ = predict(
                model=self.detector,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=str(self.device),
            )
            if len(boxes):
                break
        height, width = frame_np.shape[:2]
        new_box = boxes[np.argmax(scores)] if len(boxes) else None
        box = new_box if new_box is not None else last_known_box
        if box is None:
            return (
                torch.zeros(height, width, dtype=torch.bool, device=self.device),
                None,
            )
        self.predictor.set_image(frame_np)
        cx, cy, bw, bh = box
        xyxy = np.array(
            [
                int((cx - bw / 2) * width),
                int((cy - bh / 2) * height),
                int((cx + bw / 2) * width),
                int((cy + bh / 2) * height),
            ]
        )
        masks, scores, _ = self.predictor.predict(box=xyxy, multimask_output=True)
        return torch.from_numpy(masks[np.argmax(scores)]).to(self.device), new_box

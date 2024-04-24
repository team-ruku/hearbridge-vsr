import time

import numpy as np
import torch
from loguru import logger

from .data import *
from .detectors import *
from .model import ModelModule


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        logger.info("[Phase] 0. Initializing")

        self.device = cfg.device
        self.time_enabled = cfg.time

        logger.debug(f"[Config] Accel device: {self.device}")

        if self.time_enabled:
            start = time.time()

        self.landmark_detector = LandmarksDetectorMediaPipe()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform()

        logger.debug("[Init] Creating ModelModule")
        self.modelmodule = ModelModule(cfg)

        logger.debug("[Init] Loading model")
        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )

        logger.debug("[Init] Setting model to evaluation mode")
        self.modelmodule.to(self.device).eval()

        if self.time_enabled:
            end = time.time()
            logger.debug(f"[Time] Init time: {end - start}")

    @logger.catch
    @torch.inference_mode()
    def forward(self, video):
        if self.time_enabled:
            start = time.time()

        video = video.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        landmarks = self.landmark_detector.obsolete(
            video.astype(np.uint8)
            .astype(np.uint8)
            .astype(np.uint8)
            .astype(np.uint8)
            .astype(np.uint8)
            .astype(np.uint8)
            .astype(np.uint8)
        )
        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))

        transcript = self.modelmodule(self.video_transform(video))

        if self.time_enabled:
            end = time.time()
            logger.debug(f"[Time] Inference time: {end - start}")

        return transcript


__all__ = [InferencePipeline]

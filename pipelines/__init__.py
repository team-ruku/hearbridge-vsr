import os

import cv2
import torch
import torchvision
from loguru import logger

from .data import *
from .detectors import *
from .model import ModelModule


class InferencePipeline(torch.nn.Module):
    def __init__(self, hydra_cfg, model_cfg):
        super(InferencePipeline, self).__init__()
        logger.info("[Phase 0] Initializing")

        self.legacy_enabled = hydra_cfg.enable_legacy
        self.save_roi = hydra_cfg.save_mouth_roi

        logger.debug("creating LandmarkDetector, VideoProcess")

        if self.legacy_enabled:
            logger.debug("legacy option enabled, loading mediapipe")
            self.landmarks_detector = LandmarksDetectorMediaPipe()
        else:
            logger.debug("no legacy option, loading retinaface")
            self.landmarks_detector = LandmarksDetectorRetinaFace()

        self.video_process = VideoProcess(convert_gray=True)
        if self.save_roi:
            self.colorized_video = VideoProcess(convert_gray=False)

        logger.debug("transforming video")
        self.video_transform = VideoTransform(speed_rate=1)

        logger.debug("creating model module")
        self.modelmodule = ModelModule(model_cfg)

    @logger.catch
    def forward(self, filename):
        logger.info("Starting Inference")
        filename = os.path.abspath(filename)
        assert os.path.isfile(filename), f"filename: {filename} does not exist."

        video = self.load_video(filename)

        if self.save_roi:
            logger.info("Mouth ROI capture enabled, saving ROI crop result")
            fps = cv2.VideoCapture(filename).get(cv2.CAP_PROP_FPS)
            self.__save_to_video("demos/roi.mp4", video, fps)

        logger.info("[Phase 2] Getting transcript")
        with torch.no_grad():
            transcript = self.modelmodule(video)

        return transcript

    @logger.catch
    def load_video(self, filename):
        logger.info("[Phase 1-1] Preprocess Video")
        logger.debug(f"reading video using torchvision, filename: {filename}")
        video = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()
        landmarks = self.landmarks_detector(video)
        video = torch.tensor(self.video_process(video, landmarks))
        if self.save_roi:
            self.__save_to_video(
                "demos/roi.mp4",
                torch.tensor(self.colorized_video(video, landmarks)),
                cv2.VideoCapture(filename).get(cv2.CAP_PROP_FPS),
            )
        return self.video_transform(video)

    def __save_to_video(self, filename, vid, frames_per_second):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torchvision.io.write_video(filename, vid, frames_per_second)


__all__ = [InferencePipeline]

import os
import time

import cv2
import torch
import torchvision
from loguru import logger

from .data import *
from .detectors import *
from .model import ModelModule


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        logger.info("[Phase 0] Initializing")

        self.device = cfg.device
        self.time_enabled = cfg.time
        self.detector = cfg.detector

        logger.debug(f"Accel device: {self.device}")
        logger.debug(f"Face Detector: {self.detector}")

        if self.time_enabled:
            start = time.time()

        if self.detector == "retinaface":
            logger.debug("loading retinaface")
            self.landmarks_detector = LandmarksDetectorRetinaFace()
        else:
            logger.debug("loading mediapipe")
            self.landmarks_detector = LandmarksDetectorMediaPipe()

        self.save_roi = cfg.save_mouth_roi

        logger.debug("creating VideoProcess")
        self.video_process = VideoProcess(convert_gray=False)
        if self.save_roi:
            self.colorized_video = VideoProcess(convert_gray=False)

        logger.debug("creating VideoTransform")
        self.video_transform = VideoTransform()

        logger.debug("creating ModelModule")
        self.modelmodule = ModelModule(cfg)

        logger.debug("loading model")
        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )

        logger.debug("setting model to evaluation mode")
        self.modelmodule.to(self.device).eval()

        if self.time_enabled:
            end = time.time()
            logger.debug(f"Init time: {end - start}")

    @logger.catch
    def forward(self, filename):
        logger.info("Starting Inference")
        filename = os.path.abspath(filename)
        assert os.path.isfile(filename), f"filename: {filename} does not exist."

        video = self.load_video(filename)

        if self.time_enabled:
            start = time.time()

        logger.info("[Phase 2] Getting transcript")
        with torch.no_grad():
            transcript = self.modelmodule(video)

        if self.time_enabled:
            end = time.time()
            logger.debug(f"Inference time: {end - start}")

        return transcript

    @logger.catch
    def load_video(self, filename):
        logger.info("[Phase 1-1] Preprocess Video")
        logger.debug(f"reading video, filename: {filename}")

        if self.time_enabled:
            start = time.time()

        video = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()
        landmarks = self.landmarks_detector(video)

        if self.save_roi:
            logger.info("Mouth ROI capture enabled, saving ROI crop result")
            fps = cv2.VideoCapture(filename).get(cv2.CAP_PROP_FPS)
            self.__save_to_video(
                f"{filename.replace('.mp4','')}_roi.mp4",
                torch.tensor(self.colorized_video(video, landmarks)),
                fps,
            )

        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))

        if self.time_enabled:
            end = time.time()
            logger.debug(f"Preprocess time: {end - start}")

        return self.video_transform(video)

    def __save_to_video(self, filename, vid, fps):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.debug(f"writing on {filename}, {fps} fps")
        torchvision.io.write_video(filename, vid, fps)


__all__ = [InferencePipeline]

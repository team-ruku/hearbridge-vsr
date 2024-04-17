import os

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

        logger.debug("creating LandmarkDetector, VideoProcess")

        if cfg.enable_legacy:
            logger.debug("legacy option enabled, loading mediapipe")
            self.landmarks_detector = LandmarksDetectorMediaPipe()
        else:
            logger.debug("no legacy option, loading retinaface")
            self.landmarks_detector = LandmarksDetectorRetinaFace()

        self.save_roi = cfg.save_mouth_roi

        self.video_process = VideoProcess(convert_gray=False)
        if self.save_roi:
            self.colorized_video = VideoProcess(convert_gray=False)

        logger.debug("transforming video")
        self.video_transform = VideoTransform()

        logger.debug("creating model module")
        self.modelmodule = ModelModule(cfg)

        logger.debug("loading model file")
        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )

        logger.debug("setting model to evaluation mode")
        self.modelmodule.eval()

    @logger.catch
    def forward(self, filename):
        logger.info("Starting Inference")
        filename = os.path.abspath(filename)
        assert os.path.isfile(filename), f"filename: {filename} does not exist."

        video = self.load_video(filename)

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

        if self.save_roi:
            logger.info("Mouth ROI capture enabled, saving ROI crop result")
            fps = cv2.VideoCapture(filename).get(cv2.CAP_PROP_FPS)
            self.__save_to_video(
                f"{filename.replace('.mp4','')}_roi.mp4",
                torch.tensor(self.colorized_video(video, landmarks)),
                fps,
            )

        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))
        return self.video_transform(video)

    def __save_to_video(self, filename, vid, frames_per_second):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torchvision.io.write_video(filename, vid, frames_per_second)


__all__ = [InferencePipeline]

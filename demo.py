import os
import sys

import hydra
import torch
import torchvision
from loguru import logger

from preprocessing import ModelModule, VideoTransform, VideoProcess


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        logger.info("[Phase 0] Initializing")

        logger.debug("creating LandmarkDetector, VideoProcess")

        if cfg.enable_legacy:
            logger.debug("legacy option enabled, loading mediapipe")
            from preprocessing.detector import LandmarksDetectorMediaPipe

            self.landmarks_detector = LandmarksDetectorMediaPipe()
        else:
            logger.debug("no legacy option, loading retinaface")
            from preprocessing.detector import LandmarksDetectorRetinaFace

            self.landmarks_detector = LandmarksDetectorRetinaFace()

        self.video_process = VideoProcess(convert_gray=False)

        logger.debug("transforming video")
        self.video_transform = VideoTransform(subset="test")

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

        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        return video


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    logger.debug(f"CFG: {cfg}")
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.filename)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()

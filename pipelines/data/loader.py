import torchvision
from pipelines.detectors import *


class DataLoader:
    def __init__(self, filename) -> None:
        self.original_video = torchvision.io.read_video(filename, pts_unit="sec")[
            0
        ].numpy()

        self.landmark_detector = LandmarksDetectorMediaPipe()
        self.video_process = VideoProcess(convert_gray=False)

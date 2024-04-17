import torch
import torchvision

from .transforms import VideoTransform


class DataModule:
    def __init__(
        self, speed_rate=1, transform=True, detector="retinaface", convert_gray=True
    ):
        self.transform = transform
        if detector == "mediapipe":
            from pipelines.detectors import VideoProcess

            self.video_process = VideoProcess(convert_gray=convert_gray)
        if detector == "retinaface":
            from pipelines.detectors import VideoProcess

            self.video_process = VideoProcess(convert_gray=convert_gray)
        self.video_transform = VideoTransform(speed_rate=speed_rate)

    def load_data(self, filename, landmarks=None):
        video = self.load_video(filename)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return self.video_transform(video)

    def load_video(self, filename):
        return torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()

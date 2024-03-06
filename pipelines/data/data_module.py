import torch
import torchvision
from .transforms import VideoTransform


class AVSRDataLoader:
    def __init__(
        self,
        speed_rate=1,
        transform=True,
        convert_gray=True,
    ):
        self.transform = transform
        from pipelines.detectors.mediapipe.video_process import VideoProcess

        self.video_process = VideoProcess(convert_gray=convert_gray)
        self.video_transform = VideoTransform(speed_rate=speed_rate)

    def load_data(self, data_filename, landmarks=None):
        video = self.load_video(data_filename)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return self.video_transform(video) if self.transform else video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

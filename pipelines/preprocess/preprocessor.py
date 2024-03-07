import numpy as np
import torch
import torchvision

from pipelines.detector import LandmarksDetector, VideoProcess


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class Preprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess()
        self.video_transform = torch.nn.Sequential(
            FunctionalModule(
                lambda n: [
                    (
                        lambda x: torchvision.transforms.functional.resize(
                            x, 44, antialias=True
                        )
                    )(i)
                    for i in n
                ]
            ),
            FunctionalModule(lambda x: torch.stack(x)),
            torchvision.transforms.Normalize(0.0, 255.0),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def forward(self, audio, video):
        video = video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video).permute(0, 3, 1, 2).float()
        video = self.video_transform(video)
        audio = audio.mean(axis=-1, keepdim=True)
        return audio, video

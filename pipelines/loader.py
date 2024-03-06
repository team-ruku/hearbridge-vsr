import torch
import torchvision


class Loader:
    def __init__(
        self,
        speed_rate=1,
        transform=True,
        convert_gray=True,
    ):
        self.transform = transform
        from pipelines.detector.video_process import VideoProcess

        self.video_process = VideoProcess(convert_gray=convert_gray)
        self.video_transform = Transform(speed_rate=speed_rate)

    def load_data(self, data_filename, landmarks=None):
        video = self.load_video(data_filename)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return self.video_transform(video) if self.transform else video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class Transform:
    def __init__(self, speed_rate):
        self.video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x.unsqueeze(-1)),
            FunctionalModule(
                lambda x: (
                    x
                    if speed_rate == 1
                    else torch.index_select(
                        x,
                        dim=0,
                        index=torch.linspace(
                            0,
                            x.shape[0] - 1,
                            int(x.shape[0] / speed_rate),
                            dtype=torch.int64,
                        ),
                    )
                )
            ),
            FunctionalModule(lambda x: x.permute(3, 0, 1, 2)),
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, sample):
        return self.video_pipeline(sample)

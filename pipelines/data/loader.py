import torch
import torchaudio

from loguru import logger

from pipelines.detectors import *

from .cacher import ContextCacher
from .transforms import VideoTransform


class DataLoader:
    def __init__(
        self, filename, format, buffer_size=32, segment_length=8, context_length=4
    ) -> None:
        self.landmark_detector = LandmarksDetectorMediaPipe()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform()

        self.buffer_size = buffer_size
        self.segment_length = segment_length
        self.context_length = context_length

        self.cacher = ContextCacher(self.buffer_size, self.context_length)

        # self.streamer = torchaudio.io.StreamReader(
        #    src="0",
        #    format=format,
        #   option={"framerate": "30", "pixel_format": "rgb24"},
        # )

    def stream(self, queue):
        self.streamer.add_basic_video_stream(
            frames_per_chunk=self.segment_length,
            buffer_chunk_size=500,
            width=600,
            height=340,
        )

        for chunk in self.streamer.stream(timeout=-1, backoff=1.0):
            queue.put(chunk)

    @classmethod
    def preprocess(cls, video):

        landmarks = cls.landmark_detector.obsolete(video)
        video = torch.tensor(cls.video_process(video, landmarks)).permute((0, 3, 1, 2))
        return cls.video_transform(video)

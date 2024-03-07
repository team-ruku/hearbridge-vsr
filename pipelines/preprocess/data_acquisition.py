import torch
import torchaudio


def stream(q, format, option, src, segment_length, sample_rate):
    print("Building StreamReader...")
    streamer = torchaudio.io.StreamReader(src=src, format=format, option=option)
    streamer.add_basic_video_stream(
        frames_per_chunk=segment_length, buffer_chunk_size=500, width=600, height=340
    )
    streamer.add_basic_audio_stream(
        frames_per_chunk=segment_length * 640, sample_rate=sample_rate
    )

    print(streamer.get_src_stream_info(0))
    print(streamer.get_src_stream_info(1))
    print("Streaming...")
    print()
    for chunk_v, chunk_a in streamer.stream(timeout=-1, backoff=1.0):
        q.put([chunk_v, chunk_a])


class ContextCacher:
    def __init__(self, segment_length: int, context_length: int, rate_ratio: int):
        self.segment_length = segment_length
        self.context_length = context_length

        self.context_length_v = context_length
        self.context_length_a = context_length * rate_ratio
        self.context_v = torch.zeros([self.context_length_v, 3, 340, 600])
        self.context_a = torch.zeros([self.context_length_a, 1])

    def __call__(self, chunk_v, chunk_a):
        if chunk_v.size(0) < self.segment_length:
            chunk_v = torch.nn.functional.pad(
                chunk_v, (0, 0, 0, 0, 0, 0, 0, self.segment_length - chunk_v.size(0))
            )
        if chunk_a.size(0) < self.segment_length * 640:
            chunk_a = torch.nn.functional.pad(
                chunk_a, (0, 0, 0, self.segment_length * 640 - chunk_a.size(0))
            )

        if self.context_length == 0:
            return chunk_v.float(), chunk_a.float()
        else:
            chunk_with_context_v = torch.cat((self.context_v, chunk_v))
            chunk_with_context_a = torch.cat((self.context_a, chunk_a))
            self.context_v = chunk_v[-self.context_length_v :]
            self.context_a = chunk_a[-self.context_length_a :]
            return chunk_with_context_v.float(), chunk_with_context_a.float()

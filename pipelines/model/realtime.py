import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torchaudio

from pipelines.data import ContextCacher
from pipelines.preprocess import Preprocessing
from pipelines.token import SentencePieceTokenProcessor

ctx = mp.get_context("spawn")


class RealtimeAVSR(torch.nn.Module):
    def __init__(
        self,
        model_path,
        spm_model_path,
        buffer_size,
        segment_length,
        context_length,
        sample_rate,
        frame_rate,
        device="cuda:0",
    ) -> None:
        super(RealtimeAVSR, self).__init__()
        self.device = device

        self.model = torch.jit.load(model_path)
        self.model.to(device=self.device).eval()

        self.sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)
        self.token_processor = SentencePieceTokenProcessor(self.sp_model)
        self.preprocessor = Preprocessing()

        self.buffer_size = buffer_size
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

        self.cacher = ContextCacher(
            self.buffer_size, context_length, self.sample_rate // self.frame_rate
        )
        self.decoder = torchaudio.models.RNNTBeamSearch(
            self.model.model, self.sp_model.get_piece_size()
        )

        self.state = None
        self.hypotheses = None

    def stream(self, format):
        print("Building StreamReader...")
        streamer = torchaudio.io.StreamReader(
            src="0:1",
            format=format,
            option={"framerate": self.frame_rate, "pixel_format": "rgb24"},
        )
        streamer.add_basic_video_stream(
            frames_per_chunk=self.segment_length,
            buffer_chunk_size=500,
            width=600,
            height=340,
        )
        streamer.add_basic_audio_stream(
            frames_per_chunk=self.segment_length * 640, sample_rate=self.sample_rate
        )

        print(streamer.get_src_stream_info(0))
        print(streamer.get_src_stream_info(1))
        print("Streaming...")
        print()
        for chunk_v, chunk_a in streamer.stream(timeout=-1, backoff=1.0):
            self.queue.put([chunk_v, chunk_a])

    def pipe(self, audio, video):
        audio, video = self.preprocessor(audio, video)
        feats = self.model(audio.unsqueeze(0), video.unsqueeze(0))
        length = torch.tensor([feats.size(1)], device=audio.device)
        self.hypotheses, self.state = self.decoder.infer(
            feats, length, 10, state=self.state, hypothesis=self.hypotheses
        )
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript

    def infer(self, data_filename):
        self.queue = ctx.Queue()
        self.process = ctx.Process(target=self.stream, args=(self, data_filename))

        num_video_frames = 0
        video_chunks = []
        audio_chunks = []

        @torch.inference_mode()
        def inner():
            while True:
                chunk_v, chunk_a = self.queue.get()
                num_video_frames += chunk_a.size(0) // 640

                video_chunks.append(chunk_v)
                audio_chunks.append(chunk_a)

                if num_video_frames < self.buffer_size:
                    continue

                video = torch.cat(video_chunks)
                audio = torch.cat(audio_chunks)

                video, audio = self.cacher(video, audio)
                self.state, self.hypotheses = None, None

                transcript = self.pipe(audio, video.float())
                print(transcript, end="", flush=True)

                num_video_frames = 0
                video_chunks = []
                audio_chunks = []

        self.process.start()
        inner()
        self.process.join()

import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torchaudio

from pipelines.token import SentencePieceTokenProcessor


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

        self.buffer_size = buffer_size
        self.segment_length = segment_length
        self.context_length = context_length
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.rate_ratio = self.sample_rate // self.frame_rate

        self.decoder = torchaudio.models.RNNTBeamSearch(
            self.model.model, self.sp_model.get_piece_size()
        )

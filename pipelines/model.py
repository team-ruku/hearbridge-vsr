import torch
from loguru import logger
from pytorch_lightning import LightningModule

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus

from .data import TextTransform


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)
        self.beam_search = self.get_beam_search_decoder()

    def get_beam_search_decoder(self, ctc_weight=0.1, beam_size=40):
        scorers = {
            "decoder": self.model.decoder,
            "ctc": CTCPrefixScorer(self.model.ctc, self.model.eos),
            "length_bonus": LengthBonus(len(self.token_list)),
            "lm": None,
        }

        weights = {
            "decoder": 1.0 - ctc_weight,
            "ctc": ctc_weight,
            "lm": 0.0,
            "length_bonus": 0.0,
        }

        return BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=len(self.token_list),
            weights=weights,
            scorers=scorers,
            sos=self.model.sos,
            eos=self.model.eos,
            token_list=self.token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        )

    @logger.catch
    def forward(self, sample):
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.cfg.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace(
            "<eos>", ""
        )

        print(predicted)

        return predicted

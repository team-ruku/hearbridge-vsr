import argparse
import json
import os
import pathlib

import torch
from loguru import logger

from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus


SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())),
    "models",
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())),
    "models",
    "spm",
    "unigram",
    "unigram5000_units.txt",
)


class ModelModule(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        logger.info("[Phase 0-1] Initializing model")

        with open(cfg["model"]["model_conf"], "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        labels_type = getattr(self.train_args, "labels_type", "char")
        if labels_type == "char":
            self.token_list = self.train_args.char_list
        elif labels_type == "unigram5000":
            self.token_list = (
                ["<blank>"]
                + [word.split()[0] for word in open(DICT_PATH).read().splitlines()]
                + ["<eos>"]
            )
        self.odim = len(self.token_list)

        self.model = E2E(self.odim, self.train_args)
        self.model.load_state_dict(
            torch.load(
                cfg["model"]["model_path"], map_location=lambda storage, loc: storage
            )
        )
        self.model.to().eval()

        self.beam_search = get_beam_search_decoder(
            self.model,
            self.token_list,
            cfg["model"]["rnnlm"],
            cfg["model"]["rnnlm_conf"],
            cfg["decode"]["penalty"],
            cfg["decode"]["ctc_weight"],
            cfg["decode"]["lm_weight"],
            cfg["decode"]["beam_size"],
        )
        self.beam_search.to().eval()

    @logger.catch
    @torch.inference_mode()
    def forward(self, data):
        if isinstance(data, tuple):
            enc_feats = self.model.encode(data[0], data[1])
        else:
            enc_feats = self.model.encode(data)
        nbest_hyps = self.beam_search(enc_feats)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        transcription = add_results_to_json(nbest_hyps, self.token_list)
        transcription = transcription.replace("▁", " ").strip()
        return transcription.replace("<eos>", "")


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty,
    )

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

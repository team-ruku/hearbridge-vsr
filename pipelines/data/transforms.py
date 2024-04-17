#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import pathlib

import sentencepiece
import torch
import torchvision

SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())),
    "models",
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())),
    "models",
    "spm",
    "unigram",
    "unigram5000_units.txt",
)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
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


class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):
        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")

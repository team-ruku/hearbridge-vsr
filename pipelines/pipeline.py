import os
import pickle
from configparser import ConfigParser

import sentencepiece as spm
import torch

from pipelines.data import AVSRDataLoader
from pipelines.detector import LandmarksDetector
from pipelines.model import AVSR, RealtimeAVSR
from pipelines.token import SentencePieceTokenProcessor


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, device_override=""):
        super(InferencePipeline, self).__init__()

        self.config = ConfigParser()
        self.config.read(f"./configs/{config_filename}.ini")

        self.modality = self.config.get("input", "modality")

        # data configuration
        self.input_v_fps = self.config.getfloat("input", "v_fps")
        self.model_v_fps = self.config.getfloat("model", "v_fps")

        # model configuration
        self.model_path = self.config.get("model", "model_path")
        self.model_conf = self.config.get("model", "model_conf")

        # language model configuration
        self.rnnlm = self.config.get("model", "rnnlm")
        self.rnnlm_conf = self.config.get("model", "rnnlm_conf")
        self.penalty = self.config.getfloat("decode", "penalty")
        self.ctc_weight = self.config.getfloat("decode", "ctc_weight")
        self.lm_weight = self.config.getfloat("decode", "lm_weight")
        self.beam_size = self.config.getint("decode", "beam_size")

        self.device = torch.device(self.__device_automatch(device_override))

        self.dataloader = AVSRDataLoader(
            self.modality, speed_rate=self.input_v_fps / self.model_v_fps
        )

        self.model = AVSR(
            self.modality,
            self.model_path,
            self.model_conf,
            self.rnnlm,
            self.rnnlm_conf,
            self.penalty,
            self.ctc_weight,
            self.lm_weight,
            self.beam_size,
            self.device,
        )

        self.landmarks_detector = LandmarksDetector()

    def __process_landmarks(self, data_filename, landmarks_filename):
        if isinstance(landmarks_filename, str):
            landmarks = pickle.load(open(landmarks_filename, "rb"))
        else:
            landmarks = self.landmarks_detector(data_filename)
        return landmarks

    def __device_automatch(self, override: str = "") -> str:
        if override:
            return override

        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."
        landmarks = self.__process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

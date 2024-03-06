import os
import torch
import pickle
from configparser import ConfigParser

from pipelines.detector import LandmarksDetector


from .model import VSR
from .loader import Loader


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, device_override=""):
        super(InferencePipeline, self).__init__()

        config = ConfigParser()
        config.read(config_filename)

        # data configuration
        input_v_fps = config.getfloat("input", "v_fps")
        model_v_fps = config.getfloat("model", "v_fps")

        # model configuration
        model_path = config.get("model", "model_path")
        model_conf = config.get("model", "model_conf")

        # language model configuration
        rnnlm = config.get("model", "rnnlm")
        rnnlm_conf = config.get("model", "rnnlm_conf")
        penalty = config.getfloat("decode", "penalty")
        ctc_weight = config.getfloat("decode", "ctc_weight")
        lm_weight = config.getfloat("decode", "lm_weight")
        beam_size = config.getint("decode", "beam_size")

        device = torch.device(self.__device_automatch(device_override))

        self.dataloader = Loader(speed_rate=input_v_fps / model_v_fps)
        self.model = VSR(
            model_path,
            model_conf,
            rnnlm,
            rnnlm_conf,
            penalty,
            ctc_weight,
            lm_weight,
            beam_size,
            device,
        )

        self.landmarks_detector = LandmarksDetector()

    def process_landmarks(self, data_filename, landmarks_filename):
        if isinstance(landmarks_filename, str):
            landmarks = pickle.load(open(landmarks_filename, "rb"))
        else:
            landmarks = self.landmarks_detector(data_filename)
        return landmarks

    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def __device_automatch(self, override: str = "") -> str:
        if override:
            return override

        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"

        return "cpu"

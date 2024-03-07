import os
import pickle
from configparser import ConfigParser

import torch

from pipelines.data import AVSRDataLoader
from pipelines.detector import LandmarksDetector
from pipelines.model import AVSR, RealtimeAVSR


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, device_override=""):
        super(InferencePipeline, self).__init__()

        self.config = ConfigParser()
        self.config.read(f"./configs/{config_filename}.ini")

        self.modality = self.config.get("input", "modality")
        self.device = torch.device(self.__device_automatch(device_override))

        if self.modality == "realtime":
            model_path = self.config.get("model", "model_path")
            spm_model_path = self.config.get("model", "spm_model_path")

            buffer_size = self.config.getint("decode", "buffer_size")
            segment_length = self.config.getint("decode", "segment_length")
            context_length = self.config.getint("decode", "context_length")
            sample_rate = self.config.getint("decode", "sample_rate")
            frame_rate = self.config.getint("decode", "frame_rate")

            self.model = RealtimeAVSR(
                model_path,
                spm_model_path,
                buffer_size,
                segment_length,
                context_length,
                sample_rate,
                frame_rate,
            )

        else:
            # data configuration
            input_v_fps = self.config.getfloat("input", "v_fps")
            model_v_fps = self.config.getfloat("model", "v_fps")

            # model configuration
            model_path = self.config.get("model", "model_path")
            model_conf = self.config.get("model", "model_conf")

            # language model configuration
            rnnlm = self.config.get("model", "rnnlm")
            rnnlm_conf = self.config.get("model", "rnnlm_conf")
            penalty = self.config.getfloat("decode", "penalty")
            ctc_weight = self.config.getfloat("decode", "ctc_weight")
            lm_weight = self.config.getfloat("decode", "lm_weight")
            beam_size = self.config.getint("decode", "beam_size")

            self.dataloader = AVSRDataLoader(
                self.modality, speed_rate=input_v_fps / model_v_fps
            )

            self.model = AVSR(
                self.modality,
                model_path,
                model_conf,
                rnnlm,
                rnnlm_conf,
                penalty,
                ctc_weight,
                lm_weight,
                beam_size,
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
        if self.modality == "realtime":
            self.model.infer(data_filename)
        else:
            assert os.path.isfile(
                data_filename
            ), f"data_filename: {data_filename} does not exist."
            landmarks = self.__process_landmarks(data_filename, landmarks_filename)
            data = self.dataloader.load_data(data_filename, landmarks)
            transcript = self.model.infer(data)
            return transcript

import threading
import time
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
import torch
from loguru import logger

from .data import DetectorModule, SinglePerson, VideoProcess, VideoTransform
from .model import ModelModule


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        self.device = cfg.device
        self.debug = cfg.debug

        logger.debug(f"[Config] Accel device: {self.device}")
        logger.debug(f"[Config] Face Number: {cfg.num_faces}")

        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform()
        self.datamodule = DetectorModule(cfg.num_faces)
        self.modelmodule = ModelModule(cfg)

        logger.debug("[Init] Loaded Modules")

        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )
        logger.debug("[Init] Loaded VSR Model")

        self.modelmodule.to(self.device).eval()
        logger.debug("[Init] Setting VSR Model to evaluation mode")

        self.inference_threads = []
        self.persons: Dict[str, SinglePerson] = {}

    def __load_video(self, video, landmarks):
        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))
        return self.video_transform(video)

    @logger.catch
    def infer(self, index, image, keypoints):
        logger.debug(f"[Task] Created for index {index}")

        transcript: str = self.modelmodule(
            self.__load_video(np.stack(image, axis=0), keypoints)
        )

        print(f"{index} - {transcript.lower()}")
        self.persons[index].inferred_string.append(transcript.lower())
        logger.debug(f"[Task] Index {index} task End")
        return transcript

    @logger.catch
    @torch.inference_mode()
    def forward(self):
        start_timestamp = time.time()
        last_timestamp = 0

        while self.datamodule.capture.isOpened():
            status, frame = self.datamodule.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            current_timestamp = int((time.time() - start_timestamp) * 1000)

            logger.debug(f"[Inference] Current Timestamp: {current_timestamp}")

            if last_timestamp == current_timestamp:
                continue

            self.datamodule.face_landmark.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped),
                current_timestamp,
            )

            last_timestamp = current_timestamp

            if self.datamodule.landmark_output is not None:
                image = cv2.cvtColor(self.datamodule.landmark_output, cv2.COLOR_RGB2BGR)

                for idx, detected_face in enumerate(
                    self.datamodule.landmark_result.face_landmarks
                ):
                    logger.debug(f"[Inference] IDX {idx}")
                    if idx not in self.persons:
                        logger.debug(f"[Inference] Person {idx} created")
                        self.persons[idx] = SinglePerson(idx)

                    self.persons[idx].current_mouth_status = (
                        self.datamodule.calculate_mouth_distance(
                            detected_face[13], detected_face[14]
                        )
                    )
                    keypoints = self.datamodule.calculate_keypoints(
                        detected_face, image
                    )

                    self.persons[idx].update_mouth_timestamp()
                    current_status = self.persons[idx].check_mouth_status()

                    if "OPENED" in current_status:
                        logger.debug("[Infernece] Mouth is opened")
                        self.persons[idx].add_chunk(image, keypoints)

                        if "SECOND" in current_status:
                            self.persons[idx].mouth_opened_timestamp = time.time()

                    if "CLOSED" in current_status:
                        logger.debug("[Infernece] Mouth is closed")
                        self.persons[idx].mouth_closed_timestamp = time.time()

                    if self.persons[idx].infer_status:
                        logger.debug("[Infernece] Inference Task Created")
                        t = threading.Thread(
                            target=self.infer,
                            args=(
                                self.persons[idx].index,
                                self.persons[idx].frame_chunk,
                                self.persons[idx].calculated_keypoints,
                            ),
                        )

                        t.start()
                        self.inference_threads.append(t)

                        self.persons[idx].reset()

                    self.persons[idx].update_mouth_status()


__all__ = [InferencePipeline]

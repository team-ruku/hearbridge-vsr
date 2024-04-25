import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from loguru import logger

from .data import DataModule, VideoProcess, VideoTransform
from .model import ModelModule


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        self.device = cfg.device
        self.debug = cfg.debug

        logger.debug(f"[Config] Accel device: {self.device}")

        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform()
        self.datamodule = DataModule()
        self.modelmodule = ModelModule(cfg)

        logger.debug(f"[Init] Loaded Modules")

        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )
        logger.debug(f"[Init] Loaded VSR Model")

        self.modelmodule.to(self.device).eval()
        logger.debug(f"[Init] Setting VSR Model to evaluation mode")

        self.inference_threads = []

    def __load_video(self, video, landmarks):
        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))
        return self.video_transform(video)

    def __update_mouth_status(self):
        if self.last_mouth_closed > self.last_mouth_opened:
            passed_time = time.time() - self.last_mouth_closed

            if passed_time > 2:
                self.infer_status = True

    def __reset_status(self):
        self.infer_status = False
        self.last_mouth_closed = 0
        self.last_mouth_opened = 0

    @logger.catch
    def infer(self, video, landmarks):
        logger.debug("[Task] Created")
        transcript = self.modelmodule(self.__load_video(video, landmarks))
        print(transcript)
        logger.debug("[Task] End")
        return transcript

    @logger.catch
    @torch.inference_mode()
    def forward(self):
        self.last_mouth_closed = 0
        self.last_mouth_opened = 0
        self.infer_status = False

        self.datamodule.reset_chunk()

        start_timestamp = time.time()
        last_timestamp = 0

        while self.datamodule.capture.isOpened():
            status, frame = self.datamodule.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            current_timestamp = int((time.time() - start_timestamp) * 1000)

            logger.debug(f"[Inference] Current Timestamp: {current_timestamp}")
            logger.debug(f"[Inference] Last Mouth Closed: {self.last_mouth_closed}")
            logger.debug(f"[Inference] Last Mouth Opened: {self.last_mouth_opened}")
            logger.debug(f"[Inference] Infer Status: {self.infer_status}")

            if last_timestamp == current_timestamp:
                continue

            self.datamodule.face_landmark.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped),
                current_timestamp,
            )

            last_timestamp = current_timestamp

            if self.datamodule.landmark_output is not None:
                image = cv2.cvtColor(self.datamodule.landmark_output, cv2.COLOR_RGB2BGR)

                for detected_faces in self.datamodule.landmark_result.face_landmarks:
                    self.datamodule.calculate_mouth_distance(
                        detected_faces[13], detected_faces[14]
                    )
                    keypoints = self.datamodule.calculate_keypoints(
                        detected_faces, image
                    )

                    self.__update_mouth_status()

                    if self.datamodule.mouth_status:
                        logger.debug("[Infernece] Mouth is opened")
                        self.datamodule.frame_chunk.append(image)
                        self.datamodule.calculated_keypoints.append(keypoints)

                        if not self.datamodule.prev_status:
                            self.last_mouth_opened = time.time()

                    if (
                        self.datamodule.prev_status != self.datamodule.mouth_status
                        and self.datamodule.prev_status
                    ):
                        logger.debug("[Infernece] Mouth is closed")
                        self.last_mouth_closed = time.time()

                    if self.infer_status:
                        logger.debug("[Infernece] Creating numpy stack")
                        numpy_arrayed_chunk = np.stack(
                            self.datamodule.frame_chunk, axis=0
                        )

                        logger.debug("[Infernece] Inference Task Created")
                        t = threading.Thread(
                            target=self.infer,
                            args=(
                                numpy_arrayed_chunk,
                                self.datamodule.calculated_keypoints,
                            ),
                        )

                        t.start()
                        self.inference_threads.append(t)

                        self.datamodule.reset_chunk()
                        self.__reset_status()

                    self.datamodule.prev_status = self.datamodule.mouth_status


__all__ = [InferencePipeline]

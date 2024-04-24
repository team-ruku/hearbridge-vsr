import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from loguru import logger

from .data import VideoProcess, VideoTransform, DataModule
from .model import ModelModule


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        self.device = cfg.device

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

    def __load_video(self, video, landmarks):
        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))
        return self.video_transform(video)

    @logger.catch
    @torch.inference_mode()
    def forward(self):
        start_time = time.time()

        self.datamodule.reset_chunk()
        self.ctx = torch.multiprocessing.get_context("spawn")
        self.queue = self.ctx.Queue()

        while self.datamodule.capture.isOpened():
            status, frame = self.datamodule.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            timestamp_ms = int((time.time() - start_time) * 1000)

            logger.debug(f"[Inference] Current Timestamp: {timestamp_ms}")

            self.datamodule.face_landmark.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped), timestamp_ms
            )

            if self.datamodule.landmark_output is not None:
                image = cv2.cvtColor(self.datamodule.landmark_output, cv2.COLOR_RGB2BGR)

                for detected_faces in self.datamodule.landmark_result.face_landmarks:
                    self.datamodule.calculate_mouth_distance(
                        detected_faces[13], detected_faces[14]
                    )
                    landmark = self.datamodule.calculate_keypoints(
                        detected_faces, image
                    )

                    if self.datamodule.mouth_status is True:
                        logger.debug("[Infernece] Mouth is opened")
                        self.datamodule.frame_chunk.append(image)
                        self.datamodule.calculated_keypoints.append(landmark)

                    if (
                        self.datamodule.prev_status != self.datamodule.mouth_status
                        and self.datamodule.prev_status == True
                    ):
                        logger.debug("[Infernece] Mouth is closed")
                        numpy_arrayed_chunk = np.stack(
                            self.datamodule.frame_chunk, axis=0
                        )

                        process = self.ctx.Process(
                            target=self.modelmodule,
                            args=(
                                self.__load_video(
                                    numpy_arrayed_chunk,
                                    self.datamodule.calculated_keypoints,
                                ),
                                self.queue,
                            ),
                        )
                        process.start()

                        self.datamodule.reset_chunk()

                    self.datamodule.prev_status = self.datamodule.mouth_status


__all__ = [InferencePipeline]

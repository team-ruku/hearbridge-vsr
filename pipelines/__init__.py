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
        start_timestamp = time.time()
        last_timestamp = 0
        last_mouth_closed = -1

        self.datamodule.reset_chunk()

        while self.datamodule.capture.isOpened():
            status, frame = self.datamodule.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            current_timestamp = int((time.time() - start_timestamp) * 1000)

            logger.debug(f"[Inference] Current Timestamp: {current_timestamp}")
            logger.debug(f"[Inference] Last Mouth Closed: {last_mouth_closed}")

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
                    landmark = self.datamodule.calculate_keypoints(
                        detected_faces, image
                    )

                    if self.datamodule.mouth_status is True or last_mouth_closed != -1:
                        logger.debug("[Infernece] Mouth is opened")
                        if last_mouth_closed != -1:
                            logger.debug("This frame is between with error corrections")
                            last_mouth_closed = -2

                        self.datamodule.frame_chunk.append(image)
                        self.datamodule.calculated_keypoints.append(landmark)

                    if (
                        self.datamodule.prev_status != self.datamodule.mouth_status
                        and self.datamodule.prev_status == True
                    ):
                        logger.debug("[Infernece] Mouth is closed")
                        last_mouth_closed = time.time()

                    if (time.time() - last_mouth_closed) > 2 and last_mouth_closed > 0:
                        logger.debug(
                            f"[Inference] Created task at {time.time()} - {last_mouth_closed}"
                        )
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
                        last_mouth_closed = -1

                    self.datamodule.prev_status = self.datamodule.mouth_status


__all__ = [InferencePipeline]

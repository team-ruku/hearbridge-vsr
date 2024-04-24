import time
import cv2
import numpy as np
import torch
from loguru import logger

from .data import *
from .detectors import *
from .model import ModelModule
import mediapipe as mp


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg):
        super(InferencePipeline, self).__init__()
        self.device = cfg.device

        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform()

        self.data_loader = DataLoader()

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(
                "models/visual/model.pth",
                map_location=lambda storage, loc: storage,
            )
        )
        self.modelmodule.to(self.device).eval()

    def __load_video(self, video, landmarks):
        video = torch.tensor(self.video_process(video, landmarks)).permute((0, 3, 1, 2))
        return self.video_transform(video)

    @logger.catch
    @torch.inference_mode()
    def forward(self, video):
        start_time = time.time()

        self.data_loader.__reset_chunk()

        while self.data_loader.capture.isOpened():
            status, frame = self.data_loader.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            timestamp_ms = int((time.time() - start_time) * 1000)

            self.data_loader.face_landmark.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped), timestamp_ms
            )

            if self.data_loader.landmark_output is not None:
                image = cv2.cvtColor(
                    self.data_loader.landmark_output, cv2.COLOR_RGB2BGR
                )

                for (
                    detected_landmark
                ) in self.data_loader.landmark_result.face_landmarks:
                    self.data_loader.__calculate_mouth_distance(
                        detected_landmark[13], detected_landmark[14]
                    )
                    landmark = self.data_loader.__calculate_keypoints(
                        detected_landmark, image
                    )

                    if self.data_loader.mouth_status is True:
                        self.data_loader.frame_chunk.append(image)
                        self.data_loader.calculated_keypoints.append(landmark)

                    if (
                        self.data_loader.prev_status != self.data_loader.mouth_status
                        and self.data_loader.prev_status == True
                    ):
                        # Inference
                        numpy_arrayed_chunk = np.stack(
                            self.data_loader.frame_chunk, axis=0
                        )
                        self.data_loader.__reset_chunk()

                        transcript = self.modelmodule(
                            self.__load_video(
                                numpy_arrayed_chunk,
                                self.data_loader.calculated_keypoints,
                            )
                        )

                        print(transcript)

                    self.data_loader.prev_status = self.data_loader.mouth_status


__all__ = [InferencePipeline]

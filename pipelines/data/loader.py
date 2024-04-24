import os
import pathlib

import cv2
import mediapipe as mp
import numpy as np

from loguru import logger


class DataModule:
    def __init__(self) -> None:
        self.face_landmark_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(
                    os.path.dirname(pathlib.Path(__file__).parent.parent.absolute()),
                    "models",
                    "mediapipe",
                    "face_landmarker.task",
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.__landmarker_callback,
        )

        self.face_landmark = mp.tasks.vision.FaceLandmarker.create_from_options(
            self.face_landmark_options
        )

        self.landmark_result = None
        self.landmark_output = None

        self.mouth_status = False  # 아가리가 벌려져있는지
        self.prev_status = False  # 아가리 이전 상태

        self.capture = cv2.VideoCapture(0)

    @logger.catch
    def __landmarker_callback(
        self,
        result: mp.tasks.vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        self.landmark_result = result
        self.landmark_output = output_image.numpy_view()

    def calculate_mouth_distance(self, a, b):
        distance = abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)

        if distance < 0.005:
            self.mouth_status = False
        else:
            self.mouth_status = True

        return distance

    def calculate_keypoints(self, landmark, image):
        lmx = [
            [  # 오른쪽 눈
                int(landmark[472].x * image.shape[1]),
                int(landmark[472].y * image.shape[0]),
            ],
            [  # 왼쪽 눈
                int(landmark[467].x * image.shape[1]),
                int(landmark[467].y * image.shape[0]),
            ],
            [  # 코 끝?
                int(landmark[0].x * image.shape[1]),
                int(landmark[0].y * image.shape[0]),
            ],
            [  # 입술
                int(landmark[13].x * image.shape[1]),
                int(landmark[13].y * image.shape[0]),
            ],
        ]

        return np.array(lmx)

    def reset_chunk(self):
        self.frame_chunk = []
        self.calculated_keypoints = []

import os
import pathlib

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from typing import List


class DetectorModule:
    def __init__(self, num_faces: int) -> None:
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
            num_faces=num_faces,
        )

        self.face_landmark = mp.tasks.vision.FaceLandmarker.create_from_options(
            self.face_landmark_options
        )

        self.landmark_result = None
        self.landmark_output = None

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

    @staticmethod
    def calculate_mouth_distance(a, b) -> bool:
        distance = abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)

        if distance < 0.005:
            return False
        else:
            return True

    @staticmethod
    def calculate_keypoints(landmark, image) -> np.ndarray:
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

    @staticmethod
    def show(image):
        cv2.imshow("HearBridge", image)

    @staticmethod
    def putText(image, keypoint, array: List):
        cv2.putText(
            image,
            " ".join(array),
            (
                int(keypoint[10].x * image.shape[1]),
                int(keypoint[10].y * image.shape[0]),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

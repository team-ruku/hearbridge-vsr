import os
import cv2
import pathlib

import mediapipe as mp
from loguru import logger

import time

from pipelines.detectors import *

# RIGHT_EYE, LEFT_EYE, NOSE_TIP, MOUTH_CENTER
# 472 467 0 13


class DataLoader:
    def __init__(self) -> None:
        self.face_detector_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(
                    os.path.dirname(pathlib.Path(__file__).parent.parent.absolute()),
                    "models",
                    "mediapipe",
                    "short_range.tflite",
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.__detector_callback,
            min_detection_confidence=0.5,
        )

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

        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            self.face_detector_options
        )
        self.face_landmark = mp.tasks.vision.FaceLandmarker.create_from_options(
            self.face_landmark_options
        )

        self.capture = cv2.VideoCapture(0)

    def __detector_callback(
        self,
        result: mp.tasks.vision.FaceDetectorResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        self.detector_result = result
        self.detector_output = output_image.numpy_view()

    def __landmarker_callback(
        self,
        result: mp.tasks.vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        self.landmark_result = result
        self.landmark_output = output_image.numpy_view()

    def __calculate_mouth_distance(self, a, b):
        return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)

    def __call__(self):
        start_time = time.time()

        while self.capture.isOpened():
            status, frame = self.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            timestamp_ms = int((time.time() - start_time) * 1000)

            self.face_landmark.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped), timestamp_ms
            )

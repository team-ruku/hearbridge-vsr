import os
import pathlib
import time

import cv2
import mediapipe as mp

from pipelines.detectors import *


class DataLoader:
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

        self.capture = cv2.VideoCapture(0)

    def __landmarker_callback(
        self,
        result: mp.tasks.vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        self.landmark_result = result
        self.landmark_output = output_image.numpy_view()

    def __calculate_mouth_distance(self, a, b):
        distance = abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)

        if distance < 0.003:
            self.mouth_status = True
        else:
            self.mouth_status = False

        return distance

    def __calculate_keypoints(self, landmark, image):
        return [
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

    def __call__(self):
        start_time = time.time()  # MediaPipe async 타임스탬프

        self.mouth_status = False  # 아가리가 벌려져있는지
        self.prev_status = False  # 아가리 이전 상태
        self.frame_chunk = []  # 아가리 벌려있을 동안의 프레임
        self.calculated_keypoints = []  # 아갈통 벌려있을 동안의 Landmarks

        while self.capture.isOpened():
            status, frame = self.capture.read()

            if not status:
                continue

            flipped = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            timestamp_ms = int((time.time() - start_time) * 1000)

            self.face_landmark.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped), timestamp_ms
            )

            if self.landmark_output is not None:
                image = cv2.cvtColor(self.landmark_output, cv2.COLOR_RGB2BGR)

                for detected_landmark in self.landmark_result.face_landmarks:
                    distance = self.__calculate_mouth_distance(
                        detected_landmark[13], detected_landmark[14]
                    )
                    landmark = self.__calculate_keypoints(detected_landmark, image)

                    if self.mouth_status:
                        self.frame_chunk.append()
                        self.calculated_keypoints.append(landmark)

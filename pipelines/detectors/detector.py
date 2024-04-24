import os
import pathlib

import mediapipe as mp
import numpy as np
from loguru import logger


class LandmarksDetectorMediaPipe:
    def __init__(self):
        self.face_detector = mp.tasks.vision.FaceDetector
        self.face_landmark = mp.tasks.vision.FaceLandmarker

        self.options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(
                    os.path.dirname(pathlib.Path(__file__).parent.parent.absolute()),
                    "models",
                    "mediapipe",
                    "short_range.tflite",
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

        self.landmark_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(
                    os.path.dirname(pathlib.Path(__file__).parent.parent.absolute()),
                    "models",
                    "mediapipe",
                    "face_landmarker.task",
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

        self.landmarker = self.face_landmark.create_from_options(self.landmark_options)

    @logger.catch
    def __call__(self, video_frames):
        with self.face_detector.create_from_options(self.options) as detector:
            landmarks = []
            for frame in video_frames:
                results = detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                )

                landmark_result = self.landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                )

                if not results.detections:
                    landmarks.append(None)
                    continue

                face_points = []
                for idx, detected_faces in enumerate(results.detections):
                    # logger.debug(
                    #    f"[Preprocess] Face Detected for Index {idx}: \n{pformat(detected_faces, indent=4)}"
                    # )
                    max_id, max_size = 0, 0
                    ih, iw, ic = frame.shape
                    bbox = detected_faces.bounding_box
                    bbox_size = (bbox.width - bbox.origin_x) + (
                        bbox.height - bbox.origin_y
                    )
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size

                    keypoints = detected_faces.keypoints
                    resultlandmarks = landmark_result.face_landmarks[0]
                    lmx = [
                        [
                            int(resultlandmarks[472].x * iw),
                            int(resultlandmarks[472].y * ih),
                        ],
                        [
                            int(resultlandmarks[467].x * iw),
                            int(resultlandmarks[467].y * ih),
                        ],
                        [
                            int(resultlandmarks[3].x * iw),
                            int(resultlandmarks[3].y * ih),
                        ],
                        [
                            int(resultlandmarks[13].x * iw),
                            int(resultlandmarks[13].y * ih),
                        ],
                    ]
                    face_points.append(lmx)
                landmarks.append(np.array(face_points[max_id]))
            return landmarks

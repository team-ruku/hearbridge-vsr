import os
import pathlib

import mediapipe as mp
import numpy as np

from loguru import logger


class LandmarksDetectorMediaPipeNew:
    def __init__(self):
        self.face_detector = mp.tasks.vision.FaceDetector

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

    @logger.catch
    def __call__(self, video_frames):
        logger.info("[Phase 1-2] Landmark Detection")

        with self.face_detector.create_from_options(self.options) as detector:
            landmarks = []
            for frame in video_frames:
                results = detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                )

                if not results.detections:
                    landmarks.append(None)
                    continue

                face_points = []
                for idx, detected_faces in enumerate(results.detections):
                    logger.debug(f"Face Detected for Index {idx}: {detected_faces}")
                    max_id, max_size = 0, 0
                    ih, iw, ic = frame.shape
                    bbox = detected_faces.bounding_box
                    bbox_size = (bbox.width - bbox.origin_x) + (
                        bbox.height - bbox.origin_y
                    )
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size

                    keypoints = detected_faces.keypoints
                    lmx = [
                        [
                            int(keypoints[0].x * iw),
                            int(keypoints[0].y * ih),
                        ],
                        [
                            int(keypoints[1].x * iw),
                            int(keypoints[1].y * ih),
                        ],
                        [
                            int(keypoints[2].x * iw),
                            int(keypoints[2].y * ih),
                        ],
                        [
                            int(keypoints[3].x * iw),
                            int(keypoints[3].y * ih),
                        ],
                    ]

                    face_points.append(lmx)
                landmarks.append(np.array(face_points[max_id]))
            return landmarks

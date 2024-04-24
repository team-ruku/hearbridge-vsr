import os
import pathlib
from pprint import pformat

import mediapipe as mp
import numpy as np
from loguru import logger


class LandmarksDetectorMediaPipe:
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
        logger.info("[Phase] 1-2. Landmark Detection")

        with self.face_detector.create_from_options(self.options) as detector:
            landmarks = {}
            max_id = 0
            for frame in video_frames:
                results = detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))
                )

                if not results.detections:
                    landmarks.append(None)
                    continue

                face_points = []
                for idx, detected_face in enumerate(results.detections):
                    ih, iw, ic = frame.shape

                    if idx > max_id:
                        max_id = idx

                    keypoints = detected_face.keypoints
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

                for idx, current_face_point in enumerate(face_points):
                    if not landmarks.keys():
                        landmarks[idx] = []

                    landmarks[idx].append(current_face_point)

            return max_id, landmarks

    @logger.catch
    def obsolete(self, video_frames):
        # logger.info("[Phase] 1-2. Landmark Detection")

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
                    logger.debug(pformat(face_points, indent=4))
                logger.debug(
                    f"[Preprocess] Max ID for this frame: {pformat(max_id, indent=4)}"
                )
                landmarks.append(np.array(face_points[max_id]))
            logger.debug(
                f"[Preprocess] Final Landmark: \n{pformat(landmarks, indent=4)}"
            )
            return landmarks

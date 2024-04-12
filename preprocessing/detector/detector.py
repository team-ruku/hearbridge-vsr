import numpy as np
from retinaface import RetinaFace

from loguru import logger


class LandmarksDetector:
    def __init__(self) -> None:
        self.detector = RetinaFace.detect_faces

    @logger.catch()
    def __call__(self, video_frames):
        logger.debug("[Phase 1-1] Landmark Detection")
        total_landmarks = []

        for frame in video_frames:
            detected_faces = self.detector(frame, threshold=0.8)  # 감지하자
            max_id, max_size = (
                0,
                0,
            )  # we should edit here later for multi-party conversations

            face_points = []
            new_array = [value for key, value in detected_faces.items()]

            for idx, result in enumerate(new_array):
                logger.debug(f"Landmark Detected for Index {idx}: {result}")
                current_bbox = result["facial_area"]
                current_landmarks = result["landmarks"]

                bbox_size = (current_bbox[2] - current_bbox[0]) + (
                    current_bbox[3] - current_bbox[1]
                )  # Bounding Box 사이즈 계산
                if bbox_size > max_size:  # 최대치만 쏙쏙 골라뺴기
                    max_id, max_size = idx, bbox_size

                face_points.append(  # 계산
                    [
                        current_landmarks["right_eye"],
                        current_landmarks["left_eye"],
                        current_landmarks["nose"],
                        [
                            (
                                current_landmarks["mouth_left"][0]
                                + current_landmarks["mouth_right"][0]
                            )
                            / 2,
                            (
                                current_landmarks["mouth_left"][1]
                                + current_landmarks["mouth_right"][1]
                            )
                            / 2,
                        ],
                    ]
                )
            total_landmarks.append(np.array(face_points[max_id]))
        return total_landmarks

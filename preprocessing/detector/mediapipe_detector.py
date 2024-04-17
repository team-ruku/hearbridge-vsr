#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import mediapipe as mp
import numpy as np

from loguru import logger


class LandmarksDetectorMediaPipe:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        self.full_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1
        )

    def __call__(self, video_frames):
        logger.info("[Phase 1-2] Landmark Detection")
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            assert any(
                l is not None for l in landmarks
            ), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                logger.debug(f"Face Detected for Index {idx}: {detected_faces}")
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                logger.debug(f"Bounding Box for {idx}: {bbox}")
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].y
                            * ih
                        ),
                    ],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks

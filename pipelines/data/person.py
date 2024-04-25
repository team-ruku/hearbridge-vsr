import time

from loguru import logger


class SinglePerson:
    def __init__(self, index) -> None:
        self.index = index

        self.frame_chunk = []
        self.calculated_keypoints = []

        self.current_mouth_status = False
        self.previous_mouth_status = False

        self.mouth_closed_timestamp = 0
        self.mouth_opened_timestamp = 0

        self.infer_status = False

    def update_mouth_timestamp(self) -> None:
        logger.debug(f"[Person {self.index}] UPDATE_MOUTH_TIMESTAMP")
        if self.mouth_closed_timestamp > self.mouth_opened_timestamp:
            passed_time = time.time() - self.mouth_closed_timestamp

            if passed_time > 2:
                self.infer_status = True

    def update_mouth_status(self) -> None:
        logger.debug(f"[Person {self.index}] UPDATE_MOUTH_STATUS")
        self.previous_mouth_status = self.current_mouth_status

    def check_mouth_status(self) -> str:
        logger.debug(f"[Person {self.index}] CHECK_MOUTH_STATUS")
        logger.debug(
            f"[Person {self.index}] {self.previous_mouth_status} {self.current_mouth_status}"
        )
        if self.previous_mouth_status != self.current_mouth_status:
            logger.debug(f"[Person {self.index}] PREV != CUR")
            if not self.previous_mouth_status:
                logger.debug(f"[Person {self.index}] ERROR CORRECTION")
                return "OPENED_SECOND"

            if self.previous_mouth_status:
                logger.debug(f"[Person {self.index}] CHECK POINT")
                return "CLOSED"

        if self.current_mouth_status:
            logger.debug(f"[Person {self.index}] OPENED")
            return "OPENED"

        return "IN_PROGRESS"

    def reset_status(self) -> None:
        logger.debug(f"[Person {self.index}] RESET_STATUS")
        self.infer_status = False
        self.mouth_closed_timestamp = 0
        self.mouth_opened_timestamp = 0

    def reset_chunk(self) -> None:
        logger.debug(f"[Person {self.index}] RESET_CHUNK")
        self.frame_chunk = []
        self.calculated_keypoints = []

    def reset(self) -> None:
        logger.debug(f"[Person {self.index}] RESET")
        self.reset_chunk()
        self.reset_status()

    def add_chunk(self, image, keypoints) -> None:
        logger.debug(f"[Person {self.index}] ADD_CHUNK")
        self.frame_chunk.append(image)
        self.calculated_keypoints.append(keypoints)

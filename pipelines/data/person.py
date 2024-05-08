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
        self.inferend_string = []

    def __log(self, message: str) -> None:
        logger.debug(f"[Person {self.index}] {message}")

    def update_mouth_timestamp(self) -> None:
        self.__log("UPDATE_MOUTH_TIMESTAMP")
        if self.mouth_closed_timestamp > self.mouth_opened_timestamp:
            passed_time = time.time() - self.mouth_closed_timestamp

            if passed_time > 2:
                self.infer_status = True

    def update_mouth_status(self) -> None:
        self.__log("UPDATE_MOUTH_STATUS")
        self.previous_mouth_status = self.current_mouth_status

    def check_mouth_status(self) -> str:
        self.__log("CHECK_MOUTH_STATUS")
        self.__log(f"{self.previous_mouth_status} {self.current_mouth_status}")
        if self.previous_mouth_status != self.current_mouth_status:
            self.__log("PREV != CUR")
            if not self.previous_mouth_status:
                self.__log("ERROR CORRECTION")
                return "OPENED_SECOND"

            if self.previous_mouth_status:
                self.__log("CHECK POINT")
                return "CLOSED"

        if self.current_mouth_status:
            self.__log("OPENED")
            return "OPENED"

        return "IN_PROGRESS"

    def reset_status(self) -> None:
        self.__log("RESET_STATUS")
        self.infer_status = False
        self.mouth_closed_timestamp = 0
        self.mouth_opened_timestamp = 0

    def reset_chunk(self) -> None:
        self.__log("RESET_CHUNK")
        self.frame_chunk = []
        self.calculated_keypoints = []

    def reset(self) -> None:
        self.__log("RESET")
        self.reset_chunk()
        self.reset_status()

    def add_chunk(self, image, keypoints) -> None:
        self.__log("ADD_CHUNK")
        self.frame_chunk.append(image)
        self.calculated_keypoints.append(keypoints)

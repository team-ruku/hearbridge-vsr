import time


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
        if self.mouth_closed_timestamp > self.mouth_opened_timestamp:
            passed_time = time.time() - self.mouth_closed_timestamp

            if passed_time > 2:
                self.infer_status = True

    def update_mouth_status(self) -> None:
        self.previous_mouth_status = self.current_mouth_status

    def reset_status(self) -> None:
        self.infer_status = False
        self.mouth_closed_timestamp = 0
        self.mouth_opened_timestamp = 0

    def reset_chunk(self) -> None:
        self.frame_chunk = []
        self.calculated_keypoints = []

    def reset(self) -> None:
        self.reset_chunk()
        self.reset_status()

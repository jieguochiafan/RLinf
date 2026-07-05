from dataclasses import dataclass


@dataclass
class DynamicBatchState:
    target_batch_size: int
    max_wait_time_s: float
    first_request_time: float | None = None

    def mark_first_request(self, now: float) -> None:
        if self.first_request_time is None:
            self.first_request_time = now

    def reset(self) -> None:
        self.first_request_time = None

    def should_flush(self, queue_size: int, now: float) -> bool:
        if queue_size <= 0:
            return False
        if queue_size >= self.target_batch_size:
            return True
        if self.first_request_time is None:
            return False
        return now - self.first_request_time >= self.max_wait_time_s

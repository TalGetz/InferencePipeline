import time


class StopWatch:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.dt = None

    def __enter__(self):
        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time_ns()
        self.dt = self.end_time - self.start_time

    def get_time_in_seconds(self):
        return self.dt / 1e6

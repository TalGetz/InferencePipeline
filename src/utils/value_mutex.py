import queue
import threading


class ValueMutex:
    def __init__(self):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.not_trying_to_get = threading.Event()
        self.not_trying_to_get.set()
        self.is_full = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        value = None
        while value is None:
            value = self.get()
        return value

    def put(self, value):
        # todo: rewrite with count and a single notify using a condition variable instead of "not_trying_to_get"
        self.not_trying_to_get.wait(timeout=3600)
        with self.lock:
            while self.queue.qsize() > 0:
                self.queue.get(timeout=3600)
            self.queue.put(value)
            self.is_full.set()

    def get(self):
        # todo: rewrite with count and a single notify using a condition variable instead of "not_trying_to_get"
        self.is_full.wait(timeout=3600)
        self.not_trying_to_get.clear()
        with self.lock:
            self.not_trying_to_get.set()
            if not self.is_full.is_set():
                return None
            self.is_full.clear()
            return self.queue.get(timeout=3600)

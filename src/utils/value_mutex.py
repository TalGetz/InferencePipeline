import multiprocessing


class ValueMutex:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.not_trying_to_get = multiprocessing.Event()
        self.not_trying_to_get.set()
        self.is_full = multiprocessing.Event()

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    def put(self, value):
        self.not_trying_to_get.wait()
        with self.lock:
            while self.queue.qsize() > 0:
                self.queue.get()
            self.queue.put(value)
            self.is_full.set()

    def get(self, *args, **kwargs):
        self.is_full.wait()
        self.not_trying_to_get.clear()
        with self.lock:
            self.not_trying_to_get.set()
            self.is_full.clear()
            return self.queue.get()

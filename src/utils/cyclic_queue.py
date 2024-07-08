from multiprocessing import Queue, Lock


class CyclicQueue:
    def __init__(self, capacity):
        self.queue = Queue(maxsize=capacity)
        self.capacity = capacity
        self.size = 0
        self.lock = Lock()

    def put(self, item):
        with self.lock:
            if self.size >= self.capacity:
                self.queue.get()  # Remove the oldest item to make space for the new item
            else:
                self.size += 1
            self.queue.put(item)

    def get(self):
        with self.lock:
            if self.size > 0:
                self.size -= 1
            return self.queue.get()

    def qsize(self):
        with self.lock:
            return self.queue.qsize()

    def empty(self):
        with self.lock:
            return self.queue.empty()

    def full(self):
        with self.lock:
            return self.qsize() == self.capacity

import abc
import multiprocessing
from multiprocessing import Queue


class TProcess(abc.ABC):
    def __init__(self, input_queue, output_queue_capacity):
        self.process = multiprocessing.Process(target=self.process_loop)
        self._input_queue: Queue = input_queue
        self._output_queue: Queue = Queue(output_queue_capacity)

    def init_in_process(self):
        pass

    def process_loop(self):
        self.init_in_process()
        while True:
            input = self._input_queue.get()
            output = self.infer(input)
            self._output_queue.put(output)

    def start(self):
        self.process.start()

    def join(self):
        self.process.join()

    @property
    def output_queue(self):
        return self._output_queue

    @abc.abstractmethod
    def infer(self, item):
        raise NotImplementedError()

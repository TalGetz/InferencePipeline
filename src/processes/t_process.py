import abc
import multiprocessing
from multiprocessing import Queue


class TProcess(abc.ABC):
    def __init__(self, input_queue, output_queue_capacity):
        self.process = multiprocessing.Process(target=self.process_loop)
        self._input_queue: Queue = input_queue
        self._output_queue: Queue = Queue(output_queue_capacity)
        self._is_initiated = False

    def _repeatable_init_in_process(self):
        if not self._is_initiated:
            self.init_in_process()
            self._is_initiated = True

    def init_in_process(self):
        pass

    def process_loop(self):
        self._repeatable_init_in_process()
        while True:
            input = self._input_queue.get()
            output = self.infer(input)
            self._output_queue.put(output)

    def start(self):
        self.process.start()
        return self

    def join(self):
        self.process.join()

    @property
    def output_queue(self):
        return self._output_queue

    def infer(self, item):
        self._repeatable_init_in_process()
        return self.overridable_infer(item)

    @abc.abstractmethod
    def overridable_infer(self, item):
        raise NotImplementedError()

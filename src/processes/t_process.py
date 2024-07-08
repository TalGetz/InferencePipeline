import abc
import multiprocessing

from src.utils.cyclic_queue import CyclicQueue


class TProcess(abc.ABC):
    def __init__(self, input_queue, output_queue_capacity):
        self.process = multiprocessing.Process(target=self.loop)
        self._input_queue: CyclicQueue = input_queue
        self._output_queue: CyclicQueue = CyclicQueue(output_queue_capacity)

    def loop(self):
        while True:
            input = self._input_queue.get()
            output = self.infer(input)
            self._output_queue.put(output)

    @property
    def output_queue(self):
        return self._output_queue

    @abc.abstractmethod
    def infer(self, item):
        raise NotImplementedError()

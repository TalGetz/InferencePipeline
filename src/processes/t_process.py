import abc
import multiprocessing
import sys
import time
from multiprocessing import Queue

from src.utils.value_mutex import ValueMutex


class TProcess(abc.ABC):
    def __init__(self, input_queue, kill_flag=None):
        self.process = multiprocessing.Process(target=self.process_loop, daemon=True)
        self._input_queue: Queue = input_queue
        self._output_queue: Queue = ValueMutex()
        self._is_initiated = False
        self.kill_flag: multiprocessing.Event = kill_flag

    def _repeatable_init_in_process(self):
        if not self._is_initiated:
            self.init_in_process()
            self._is_initiated = True

    def init_in_process(self):
        pass

    def process_loop(self):
        try:
            self._repeatable_init_in_process()
            while self.kill_flag is None or not self.kill_flag.is_set():
                input = self._input_queue.get(timeout=30)
                outputs = self.infer(input)
                for output in outputs:
                    self._output_queue.put(output)
        except Exception as e:
            print(e, file=sys.stderr)
            self.kill()
        except:
            self.kill()
        self.kill()
        print(f"shutting down: {self.__class__.__name__}")
        time.sleep(1)

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

    def kill(self):
        self.kill_flag.set()

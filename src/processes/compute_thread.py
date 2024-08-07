import abc
import sys
import time
from threading import Thread

from src.utils.value_mutex import ValueMutex


class ComputeThread(abc.ABC):
    def __init__(self, input_queue, kill_flag=None, output_queue=None):
        self.thread = Thread(target=self.process_loop, daemon=True)
        self._input_queue = input_queue
        self._output_queue = ValueMutex() if output_queue is None else output_queue
        self._is_initiated = False
        self.kill_flag = kill_flag

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
                input = next(self._input_queue)
                outputs = self.infer(input)
                for output in outputs:
                    self._output_queue.put(output)
        except (Exception, RuntimeError) as e:
            print(e, file=sys.stderr)
        finally:
            self.kill()

        print(f"shutting down: {self.__class__.__name__}")
        time.sleep(1)

    def start(self):
        self.thread.start()
        return self

    def join(self):
        self.thread.join()

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

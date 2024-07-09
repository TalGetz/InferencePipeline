from src.frame_readers.camera_reader import CameraReader
from src.processes.t_process import TProcess


class CameraReaderProcess(TProcess):
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        super().__init__(None, 1)

    def init_in_process(self):
        self._input_queue = CameraReader(self.camera_index)

    def infer(self, x):
        return x

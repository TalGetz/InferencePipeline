from src.frame_readers.camera_reader import CameraReader
from src.processes.compute_thread import ComputeThread


class CameraReaderProcess(ComputeThread):
    def __init__(self, camera_index=0, kill_flag=None):
        self.camera_index = camera_index
        super().__init__(None, kill_flag=kill_flag)

    def init_in_process(self):
        self._input_queue = CameraReader(self.camera_index)

    def overridable_infer(self, x):
        return [x]

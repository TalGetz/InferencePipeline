from src.utils.value_mutex import ValueMutex


class MultipleInstancesWrapper:

    def __init__(self, wrapped_class, n_instances, input_queue, *args, **kwargs):
        self.output_queue = ValueMutex()
        self.instances = [
            wrapped_class(input_queue=input_queue, output_queue=self.output_queue, *args, **kwargs) for
            _ in range(n_instances)]

    def start(self):
        for instance in self.instances:
            instance.start()
        return self

    def join(self):
        for instance in self.instances:
            instance.join()

    def __iter__(self):
        return self

    def __next__(self):
        return self.output_queue.get()

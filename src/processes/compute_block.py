import abc


class ComputeBlock(abc.ABC):
    @abc.abstractmethod
    def start(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def join(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError()
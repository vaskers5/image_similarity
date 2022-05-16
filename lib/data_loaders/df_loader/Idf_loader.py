from typing import Iterable


class IDfLoader:
    def __init__(self, seq_len: int):
        raise NotImplementedError

    def load(self, **kwargs) -> Iterable:
        raise NotImplementedError

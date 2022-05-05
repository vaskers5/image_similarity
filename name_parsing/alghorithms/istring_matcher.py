from typing import List
from abc import ABC


class IStringMatcher(ABC):
    def find_in_list(self, text: List[str], line: str) -> bool:
        raise NotImplementedError

    def get_index_in_list(self, some_dict: List[str], line: str) -> int:
        raise NotImplementedError


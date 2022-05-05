from typing import Optional, List, Tuple
from .istring_matcher import IStringMatcher

from copy import deepcopy
from tqdm.notebook import tqdm_notebook
from multiprocessing import Pool

import nltk
import re
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords

STEMMER = Mystem()
nltk.download("stopwords")
RUSSIAN_STOPWORDS = stopwords.words("russian")

EPSILON = 0.000000001


class AbstractMatcher(IStringMatcher):
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def find_in_list(self, some_dict: List[str], line: str) -> bool:
        return self._get_match(some_dict, line)[2] >= self.threshold

    def get_index_in_list(self, some_dict: List[str], line: str,
                          elem_split: bool = False) -> Optional[int]:
        res = self._get_match(some_dict, line, elem_split)
        return res[1] if res[2] >= self.threshold else None

    def get_indexes_in_list(self, some_dict: List[str], line: str) -> List[int]:
        data = deepcopy(some_dict)
        res = list(filter(lambda elem: elem[2] >= self.threshold,
                          self._get_result_list(data, line)))
        indexes = [elem[1] for elem in res]
        return indexes

    def get_item_in_list(self,
                         some_dict: List[str],
                         line: str,
                         elem_split: bool = False) -> Optional[str]:
        res = self._get_match(some_dict, line, elem_split)
        return res[0] if res[2] >= self.threshold else None

    def _get_result_list(self,
                         some_dict: List[str],
                         line: str) -> Tuple[str, int, float]:

        if not line:
            return "", "", -1.0

        with Pool() as pool:
            pool_data = [(line, idx, x) for idx, x in enumerate(some_dict)]
            result = list(tqdm_notebook(pool.starmap(self._compare, pool_data),
                                        leave=False,
                                        total=len(pool_data)))
        return result

    def _compare(self, line: str, idx: int, x: str, elem_split=False) -> Tuple[str, int, float]:
        line = self._clear_string(line)
        elem = self._clear_string(x)
        elem_list = elem.split(';')
        if len(elem_list) > 1 and elem_split:
            word_results = []
            for index, word in enumerate(elem_list):
                if word[0] == ' ':
                    elem_list[index] = word[1:]
                if word[-1] == ' ':
                    elem_list[index] = word[:-1]
                word_results += [self._compare_algorithm(elem_list[index], line)]
            return x, idx, max(word_results) - EPSILON
        else:
            return x, idx, self._compare_algorithm(elem, line)


    def _get_match(self,
                   some_dict: List[str],
                   line: str,
                   elem_split: bool = False) -> Tuple[str, int, float]:
        result_list = self._get_result_list(some_dict, line, elem_split)
        if result_list[0] == "":
            return result_list
        result = max(result_list, key=lambda elem: elem[2])
        return result

    @staticmethod
    def _clear_string(s: str) -> str:
        s = s.lower()
        tokens = STEMMER.lemmatize(s)
        if not tokens:
            return s
        tokens = [token for token in tokens if token not in RUSSIAN_STOPWORDS \
                  and token != " " \
                  and token.strip() not in punctuation]
        return "".join(tokens)

    @staticmethod
    def _compare_algorithm(s1: str, s2: str) -> float:
        pass

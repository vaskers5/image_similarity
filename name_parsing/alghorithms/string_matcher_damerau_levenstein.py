from .abstract_matcher import AbstractMatcher


class StringMatcherDamerauLevenstein(AbstractMatcher):
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    @staticmethod
    def _compare_algorithm(s1: str, s2: str) -> float:
        d = {}
        len_1, len_2 = len(s1), len(s2)

        for i in range(-1, len_1 + 1):
            d[(i, -1)] = i + 1
        for j in range(-1, len_2 + 1):
            d[(-1, j)] = j + 1

        for i in range(len_1):
            for j in range(len_2):
                if s1[i] == s2[j]:
                    cost = 0
                else:
                    cost = 1
                d[(i, j)] = min(
                    d[(i - 1, j)] + 1,  # deletion
                    d[(i, j - 1)] + 1,  # insertion
                    d[(i - 1, j - 1)] + cost,  # substitution
                )
                if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                    d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + 1)  # transposition

        return 1 - d[len_1 - 1, len_2 - 1] / max(len_1, len_2)


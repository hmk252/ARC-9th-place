import numpy as np
from src.data import Problem, Case, Matter


class Freq:
    def __init__(self):
        pass

    @classmethod
    def case(cls, c: Case, descending: bool = False) -> Case:
        new_case = c.copy()
        counter = dict()
        for m in c.matter_list:
            assert m.a is not None
            if m.a not in counter.keys():
                counter[m.a] = 1
            else:
                counter[m.a] += 1

        new_case.matter_list = []
        for m in c.matter_list:
            new_m = m.deepcopy()
            new_m.a = counter[new_m.a]
            new_case.matter_list.append(new_m)

        new_case.matter_list = list(sorted(new_case.matter_list, key=lambda x: -x.a))

        return new_case

    @classmethod
    def problem(cls, p: Problem) -> Problem:
        q: Problem = p.copy()
        q.train_x_list = [cls.case(c) for c in p.train_x_list]
        q.test_x_list = [cls.case(c) for c in p.test_x_list]
        return q

from typing import List, Union, Tuple, Any, Container, Callable, FrozenSet, Iterable
import os
import json
import hashlib
from copy import deepcopy

from utils import plot_task, is_grid, format_task

Integer = int

Grid = Tuple[Tuple[Integer]]


def height_of_grid(grid: Grid) -> int:
    return len(grid)


def width_of_grid(grid: Grid) -> int:
    return len(grid[0])


class Example:
    def __init__(self, input: Grid, output: Grid):
        assert is_grid(input)
        assert is_grid(output)
        self.input = input
        self.output = output
        self._hash = None

    def __getitem__(self, item):
        if item == "input":
            return self.input
        if item == "output":
            return self.output
        raise KeyError(f"Invalid key {item}")

    def plot(self, title=None, save_to_disk=False, output_folder=None):
        plot_task([self], title, save_to_disk, output_folder)

    def hash(self):
        if self._hash is not None:
            return self._hash
        input_hash = hashlib.md5(json.dumps(self.input).encode()).hexdigest()[:8]
        output_hash = hashlib.md5(json.dumps(self.output).encode()).hexdigest()[:8]
        hash = input_hash + output_hash
        self._hash = hash
        return hash


class Task:
    def __init__(self, id: str, data: Any):
        self.id = id
        assert isinstance(data, dict)
        assert "train" in data
        assert "test" in data
        self.train_examples = []
        self.test_input_grid = None
        self.test_solution_grid = None
        self._hash = None
        self.originalId = None
        self.originalTestIndex = None
        data = format_task(data)
        for train_data in data["train"]:
            example = Example(train_data["input"], train_data["output"])
            self.train_examples.append(example)
        self.test_input_grid = data["test"]["input"]
        if "output" in data["test"]:
            self.test_solution_grid = data["test"]["output"]

    def plot(self, title=None, save_to_disk=False, output_folder=None):
        if title == None:
            title = self.id
        dummy_grid = tuple([tuple([0, 0]), tuple([0, 0])])
        DISPLAY_SOLUTIONS = True
        dict = self.to_dict()
        if DISPLAY_SOLUTIONS == False or self.test_solution_grid == None:
            dict["test"][0]["output"] = dummy_grid
        flattened_dict = []
        for example in dict["train"]:
            flattened_dict.append(example)
        flattened_dict.append(dict["test"][0])
        plot_task(flattened_dict, title, save_to_disk, output_folder)

    def hash(self):
        if self._hash is not None:
            return self._hash
        input_hashes = [example.hash() for example in self.train_examples]
        input_hashes += [hashlib.md5(json.dumps(self.test_input_grid).encode()).hexdigest()[:8]]
        hash = hashlib.md5(json.dumps(input_hashes).encode()).hexdigest()[:8]
        self._hash = hash
        return hash

    def is_input_size_equal_output_size(self):
        assert len(self.train_examples) > 0
        for example in self.train_examples:
            if height_of_grid(example.input) != height_of_grid(example.output):
                return False
            if width_of_grid(example.input) != width_of_grid(example.output):
                return False
        return True

    def to_dict(self):
        task_dict = {"train": [], "test": []}
        for example in self.train_examples:
            task_dict["train"].append({"input": example.input, "output": example.output})
        if self.test_solution_grid is not None:
            task_dict["test"].append({"input": self.test_input_grid, "output": self.test_solution_grid})
        else:
            task_dict["test"].append({"input": self.test_input_grid})
        return json.loads(json.dumps(task_dict))


class Solution:
    def __init__(self, task_id, test_output_grid_candidates, train_score):
        # test_outputs contiene para cada test_input un array de 2 soluciones
        assert len(test_output_grid_candidates) == 2
        assert "_" not in task_id
        self.task_id = task_id
        self.test_output_grid_candidates = test_output_grid_candidates
        self.train_score = train_score

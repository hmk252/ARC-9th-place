from typing import List, Union, Tuple, Any, Container, Callable, FrozenSet, Iterable

from classes import Grid, Example, Task
from utils import RUNNING_IN_KAGGLE


def is_input_size_equal_output_size(task):
    return task.is_input_size_equal_output_size()


def num_test_inputs(task, tasks):
    res = 1
    for t in tasks:
        if t.id == task.id:
            continue
        if task.originalId != None:
            if t.originalId == task.originalId:
                res += 1
    return res


def total_train_size(task):
    total_size = 0
    for example in task.train_examples:
        total_size += len(example["input"]) * len(example["output"])
    return total_size


def unique_number(task):
    unique_number = 0
    for example in task.train_examples:
        for row in example["input"]:
            for value in row:
                unique_number += value
        for row in example["output"]:
            for value in row:
                unique_number += value
    if task.originalTestIndex != None:
        unique_number += task.originalTestIndex
    return unique_number


def get_task_ordering_score(task, tasks):
    input_same_size_output_score = 0 if is_input_size_equal_output_size(task) else 10000000.0
    num_test_inputs_score = num_test_inputs(task, tasks) * 1000000.0
    total_train_size_score = total_train_size(task) * 100.0
    unique_number_score = unique_number(task) * 0.01
    return input_same_size_output_score + num_test_inputs_score + total_train_size_score + unique_number_score


def reoder_tasks(tasks):
    task_ordering_score_and_id = []
    for task in tasks:
        task_ordering_score = get_task_ordering_score(task, tasks)
        task_ordering_score_and_id.append((task_ordering_score, task.id))
    task_ordering_score_and_id.sort(key=lambda x: x[0], reverse=True)
    # Reorder tasks according to the ordering score
    ordered_tasks_ids = [task_id for _, task_id in task_ordering_score_and_id]
    reordered_tasks = []
    for task_id in ordered_tasks_ids:
        for task in tasks:
            if task.id == task_id:
                reordered_tasks.append(task)
                break
    assert len(tasks) == len(reordered_tasks)
    return reordered_tasks

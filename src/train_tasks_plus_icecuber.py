from datetime import datetime
import resource
import os
from time import time
import numpy as np
from copy import deepcopy

from utils import (
    plot_grid,
    format_grid,
    is_grid,
    RUNNING_IN_KAGGLE,
    seed_everything,
    hash_text,
    timed_function,
    print_timer_logs,
    print_to_logfile,
)
from classes import Task, Grid, Example, height_of_grid, width_of_grid, Solution

from icecuber import solve_generated_task_icecuber, solve_generated_task_icecuber_parallel
import verifiers


def strip_prefix(string: str, prefix: str) -> str:
    return string[len(prefix) :]


def get_verifiers() -> dict:
    prefix = "verify_"
    return {strip_prefix(n, prefix): getattr(verifiers, n) for n in dir(verifiers) if n.startswith(prefix)}


verifiers_mapper = get_verifiers()
keys = sorted(verifiers_mapper.keys())
values = [verifiers_mapper[k] for k in keys]
all_functions = values

from concurrent.futures import ProcessPoolExecutor, as_completed


def generate_train_test(task, func):
    new_train_test = []
    for example in task.train_examples:
        input_grid = deepcopy(example["input"])
        predicted_grid = func(input_grid)
        if not is_grid(predicted_grid):
            return None
        new_train_test.append(predicted_grid)
    input_test_grid = deepcopy(task.test_input_grid)
    predicted_test_grid = func(input_test_grid)
    if not is_grid(predicted_test_grid):
        return None
    return new_train_test, predicted_test_grid


def parallel_generate_train_test(task_func):
    task, func = task_func
    try:
        return generate_train_test(task, func)
    except Exception as e:
        return None


def train_tasks_plus_icecuber_process_task(task, depth):
    CHUNK_SIZE = 32
    functions_chunks = [all_functions[i : i + CHUNK_SIZE] for i in range(0, len(all_functions), CHUNK_SIZE)]
    selected_solution = None
    for functions_chunk in functions_chunks:
        new_train_test = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(parallel_generate_train_test, (task, func)) for func in functions_chunk]

            # Wait for the futures to complete and process results
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    new_train_test.append(result)

        solution = solve_generated_task_icecuber_parallel(task, new_train_test, depth)
        if solution is not None:
            if selected_solution is None or solution.train_score > selected_solution.train_score:
                selected_solution = solution
    return selected_solution

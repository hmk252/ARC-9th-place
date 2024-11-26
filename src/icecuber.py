from typing import List, Union, Tuple, Any, Container, Callable, FrozenSet, Iterable
from subprocess import Popen, PIPE, STDOUT
from glob import glob
import os
import json
from classes import Task, Grid, Example, Solution, height_of_grid, width_of_grid
from copy import deepcopy
from utils import RUNNING_IN_KAGGLE, timed_function, mySystem


class Icecuber_solution:
    def __init__(self, task_id, train_outputs, test_output, score):
        self.task_id = task_id
        self.train_outputs = train_outputs
        self.test_output = test_output
        self.score = score


icecuber_dir = "/home/horacio/git/ARC-2024/2020_icecuber"
if RUNNING_IN_KAGGLE:
    icecuber_dir = "/kaggle/working/2020-icecuber"


def _icecuber_solve_task_parallel(tasks, depth):
    assert depth in [2, 3, 30, 4, 44]
    for task in tasks:
        assert "_" not in task.id
    current_dir = os.getcwd()
    # Change directory to icecuber's working directory and then back to the current one
    os.chdir(icecuber_dir)
    # Run icecuber's solution
    mySystem(f"rm -rf ./tmp", quiet=True)
    os.makedirs("./tmp", exist_ok=True)
    task_dir = f"./tmp/input_task"
    os.makedirs(task_dir, exist_ok=True)
    # Execute main.py
    for task in tasks:
        with open(f"{task_dir}/{task.id}.json", "w") as file:
            json.dump(task.to_dict(), file)
    mySystem(f"python mi_run.py {depth}")

    with open("./tmp/output/solutions.json", "r") as f:
        solutions = json.load(f)
    os.chdir(current_dir)
    icecuber_solutions = []
    for solution in solutions:
        icecuber_solutions.append(
            Icecuber_solution(solution["task_id"], solution["train_outputs"], solution["test_output"], solution["score"])
        )

    # If two or more solutions have the same task_id and test_output, keep only the one with highest score
    filtered_solutions = []
    for solution in icecuber_solutions:
        existing = False
        for i in range(len(filtered_solutions)):
            if filtered_solutions[i].task_id == solution.task_id and filtered_solutions[i].test_output == solution.test_output:
                existing = True
                if solution.score > filtered_solutions[i].score:
                    filtered_solutions[i] = solution
        if not existing:
            filtered_solutions.append(solution)
    icecuber_solutions = filtered_solutions

    # Check if all tasks have 2 solutions, if not, add a dummy solution
    for task in tasks:
        task_solutions = [solution for solution in icecuber_solutions if solution.task_id == task.id]
        if len(task_solutions) == 1:
            icecuber_solutions.append(task_solutions[0])

    # sort from highest score to lowest
    icecuber_solutions.sort(key=lambda x: x.score, reverse=True)

    return icecuber_solutions


# @timed_function
def solve_generated_task_icecuber_parallel(task, predicted_train_grids_and_predicted_test_grid, depth):
    index = 0
    new_test_tasks = []
    for predicted_train_grids, predicted_test_grid in predicted_train_grids_and_predicted_test_grid:
        index += 1
        assert len(predicted_train_grids) == len(task.train_examples)
        # Create a task with the predicted grids
        new_task = deepcopy(task)
        # Format index as a 6 digit number
        new_task.id = f"1{index:07d}"
        new_task.train_examples = [
            Example(predicted_train_grids[i], task.train_examples[i].output) for i in range(len(task.train_examples))
        ]
        new_task.test_input_grid = predicted_test_grid
        new_test_tasks.append(new_task)
        # new_task.plot()
    icecuber_solutions = _icecuber_solve_task_parallel(new_test_tasks, depth)
    # Pick top 2 solutions
    icecuber_solutions.sort(key=lambda x: x.score, reverse=True)
    assert len(icecuber_solutions) >= 2
    # Keep first 2
    icecuber_solutions = icecuber_solutions[:2]
    assert icecuber_solutions[0].score >= icecuber_solutions[1].score

    train_score = icecuber_solutions[0].score
    test_output_grid_candidates = [icecuber_solutions[0].test_output, icecuber_solutions[1].test_output]

    task_solution = Solution(task.id, test_output_grid_candidates, train_score)
    return task_solution


def solve_generated_task_icecuber(task, predicted_train_grids, predicted_test_grid, depth):
    return solve_generated_task_icecuber_parallel(task, [(predicted_train_grids, predicted_test_grid)], depth)


def icecuber_solve_tasks_parallel(tasks, depth):
    icecuber_solutions = _icecuber_solve_task_parallel(tasks, depth)
    solutions = {}
    for task in tasks:
        task_icecuber_solutions = [solution for solution in icecuber_solutions if solution.task_id == task.id]
        # Pick top 2 solutions
        task_icecuber_solutions.sort(key=lambda x: x.score, reverse=True)
        assert len(task_icecuber_solutions) >= 2
        # Keep first 2
        task_icecuber_solutions = task_icecuber_solutions[:2]
        assert task_icecuber_solutions[0].score >= task_icecuber_solutions[1].score

        train_score = task_icecuber_solutions[0].score
        test_output_grid_candidates = [task_icecuber_solutions[0].test_output, task_icecuber_solutions[1].test_output]
        task_solution = Solution(task.id, test_output_grid_candidates, train_score)
        solutions[task.id] = task_solution
    return solutions


def icecuber_solve_task(task, depth):
    solution = icecuber_solve_tasks_parallel([task], depth=depth)[task.id]
    assert solution is not None
    return solution

# From https://github.com/michaelhodel/re-arc

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from my_dsl import *
import os
import random
import numpy as np
import torch
import hashlib
from functools import wraps
from time import time
from subprocess import Popen, PIPE, STDOUT

RUNNING_IN_KAGGLE = os.path.exists("/kaggle")


def is_grid(grid: Any) -> bool:
    """
    returns True if and only if argument is a valid grid
    """
    if not isinstance(grid, tuple):
        return False
    if not (0 < len(grid) <= 30):
        return False
    if not all(isinstance(r, tuple) for r in grid):
        return False
    if not all(0 < len(r) <= 30 for r in grid):
        return False
    if not len(set(len(r) for r in grid)) == 1:
        return False
    if not all(all(isinstance(x, int) for x in r) for r in grid):
        return False
    if not all(all(0 <= x <= 9 for x in r) for r in grid):
        return False
    return True


def strip_prefix(string: str, prefix: str) -> str:
    """
    removes prefix
    """
    return string[len(prefix) :]


def format_grid(grid: List[List[int]]) -> Grid:
    """
    grid type casting
    """
    return tuple(tuple(row) for row in grid)


def format_example(example: dict) -> dict:
    """
    example data type
    """
    return {"input": format_grid(example["input"]), "output": format_grid(example["output"])}


def format_task(task: dict) -> dict:
    assert len(task["test"]) == 1
    task_test = task["test"][0]
    has_test_output = "output" in task_test
    assert "input" in task_test
    if has_test_output:
        return {
            "train": [format_example(example) for example in task["train"]],
            "test": format_example(task_test),
        }

    return {
        "train": [format_example(example) for example in task["train"]],
        "test": {"input": format_grid(task_test["input"])},
    }


alt_color_scheme_consts_name = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "pink",
    7: "orange",
    8: "purple",
    9: "brown",
}


def color_to_name(color: int) -> str:
    """
    returns color name
    """
    if color == -1:
        return "(multiple colors)"
    assert 0 <= color <= 9
    return alt_color_scheme_consts_name[color]


def name_to_color(color: str) -> int:
    """
    returns color number
    """
    assert color in alt_color_scheme_consts_name.values()
    return {v: k for k, v in alt_color_scheme_consts_name.items()}[color]


# black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9
colormap = ["#000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#A020F0", "#964B00"]


def print_grid(grid: List[List[int]]) -> None:
    """
    prints grid
    """
    row_copy = [row.copy() for row in grid]
    for row in row_copy:
        for i in range(len(row)):
            row[i] = "X" if row[i] == -1 else row[i]
    for row in row_copy:
        print(",".join(map(str, row)))


def plot_grid(grid: List[List[int]]) -> None:
    grid = format_grid(grid)
    cmap = ListedColormap(colormap)
    norm = Normalize(vmin=0, vmax=9)
    args = {"cmap": cmap, "norm": norm}
    # Add grid separation lines as gray lines
    for i in range(len(grid)):
        plt.axhline(i - 0.5, color="gray", linewidth=0.5)
    for i in range(len(grid[0])):
        plt.axvline(i - 0.5, color="gray", linewidth=0.5)
    plt.imshow(grid, **args)
    plt.axis("off")
    plt.show()
    plt.close()


def plot_task(task: List[dict], title: str = None, save_to_disk: Boolean = False, output_folder: str = None) -> None:
    """
    displays a task
    """
    cmap = ListedColormap(colormap)
    norm = Normalize(vmin=0, vmax=9)
    args = {"cmap": cmap, "norm": norm}
    height = 2
    width = len(task)
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    for column, example in enumerate(task):
        if width == 1:
            for i in range(len(example["input"])):
                axes[0].axhline(i - 0.5, color="gray", linewidth=0.5)
            for i in range(len(example["input"][0])):
                axes[0].axvline(i - 0.5, color="gray", linewidth=0.5)
            axes[0].imshow(example["input"], **args)
            for i in range(len(example["output"])):
                axes[1].axhline(i - 0.5, color="gray", linewidth=0.5)
            for i in range(len(example["output"][0])):
                axes[1].axvline(i - 0.5, color="gray", linewidth=0.5)
            axes[1].imshow(example["output"], **args)
            axes[0].axis("off")
            axes[1].axis("off")
            continue
        for i in range(len(example["input"])):
            axes[0, column].axhline(i - 0.5, color="gray", linewidth=0.5)
        for i in range(len(example["input"][0])):
            axes[0, column].axvline(i - 0.5, color="gray", linewidth=0.5)
        axes[0, column].imshow(example["input"], **args)
        for i in range(len(example["output"])):
            axes[1, column].axhline(i - 0.5, color="gray", linewidth=0.5)
        for i in range(len(example["output"][0])):
            axes[1, column].axvline(i - 0.5, color="gray", linewidth=0.5)
        axes[1, column].imshow(example["output"], **args)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if save_to_disk:
        assert title is not None
        # Warn if file already exists
        path = os.path.join(output_folder, f"{title}.png")
        if os.path.exists(path):
            print(f"Warning: file {path} already exists, overwriting")
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def seed_everything(seed):
    """
    seed everything
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def find_task_by_id(task_id, tasks):
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]


timer_logs = {}


def timed_function(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        global timer_logs
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print("> %s took %d seconds to finish" % (f.__name__, elapsed))
        if f.__name__ not in timer_logs:
            timer_logs[f.__name__] = 0
        timer_logs[f.__name__] += elapsed
        return result

    return wrapper


def print_timer_logs():
    global timer_logs
    print("%%%%%%%%%%%%%% Timer Logs %%%%%%%%%%%%%%")
    for k, v in timer_logs.items():
        print("%s - %d seconds TOTAL time" % (k, v))


# Delete logfile if it exists
if os.path.exists("logfile.txt"):
    os.remove("logfile.txt")


def print_to_logfile(text):
    if RUNNING_IN_KAGGLE:
        return
    # Check if logfile.txt exists
    if os.path.exists("logfile.txt"):
        # Open the file in append mode
        f = open("logfile.txt", "a")
    else:
        # Create a new file and write
        f = open("logfile.txt", "w")
    f.write(text + "\n")
    f.close()


def mySystem(cmd, quiet=True):
    if not quiet:
        print(cmd)
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    output = ""
    for line in iter(process.stdout.readline, b""):
        output += line.decode("utf-8")
        if not quiet:
            print(line.decode("utf-8"), end="")
    assert process.wait() == 0
    return output


def diagonal_flip(matrix, direction):
    rows = len(matrix)
    cols = len(matrix[0])
    flipped_matrix = [[0] * rows for _ in range(cols)]
    if direction == "none":
        return matrix
    if direction == "main":
        for i in range(rows):
            for j in range(cols):
                flipped_matrix[j][i] = matrix[i][j]
    elif direction == "anti":
        for i in range(rows):
            for j in range(cols):
                flipped_matrix[cols - j - 1][rows - i - 1] = matrix[i][j]
    else:
        raise ValueError("Direction must be 'main' or 'anti'")

    return flipped_matrix


def diagonal_flip_task(task, direction):
    from classes import Task

    task = Task(task.id, task.to_dict())
    if direction == "none":
        return task
    for example in task.train_examples:
        example.input = format_grid(diagonal_flip(example.input, direction))
        example.output = format_grid(diagonal_flip(example.output, direction))
    task.test_input_grid = format_grid(diagonal_flip(task.test_input_grid, direction))
    if task.test_solution_grid is not None:
        task.test_solution_grid = format_grid(diagonal_flip(task.test_solution_grid, direction))
    return task


def diagonal_flip_tasks(tasks, direction):
    return [diagonal_flip_task(task, direction) for task in tasks]

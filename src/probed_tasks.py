from typing import List, Union, Tuple, Any, Container, Callable, FrozenSet, Iterable
import os
import json
from copy import deepcopy

from utils import plot_task, is_grid, format_task, plot_grid, RUNNING_IN_KAGGLE
from classes import Solution


def solve_task_2(task):
    colors = []
    for example in task.train_examples:
        input_grid = example.input
        output_grid = example.output
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                colors.append(input_grid[i][j])
        for i in range(len(output_grid)):
            for j in range(len(output_grid[0])):
                colors.append(output_grid[i][j])
    unique_colors_in_examples = set(colors)
    test_input_grid = task.test_input_grid
    input_grid = test_input_grid
    input_grid = [list(row) for row in input_grid]
    output_grid = [list(row) for row in input_grid]

    colors_in_test = []
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            colors_in_test.append(input_grid[i][j])

    zero_color = max(set(colors_in_test), key=colors_in_test.count)
    assert len(set(colors_in_test)) == 6
    # Assert there are 3 colors in each column
    for col in range(len(input_grid[0])):
        column = [input_grid[row][col] for row in range(len(input_grid))]
        unique_colors_in_column = list(set(column))
        assert len(unique_colors_in_column) == 3 or len(unique_colors_in_column) == 1

    for col in range(len(input_grid[0])):
        column = [input_grid[row][col] for row in range(len(input_grid))]
        blue_color = None
        red_color = None
        for i in range(len(column)):
            if column[i] != zero_color:
                if blue_color is None:
                    blue_color = column[i]
                else:
                    if column[i] != blue_color:
                        if red_color is None:
                            red_color = column[i]
                        else:
                            assert column[i] == red_color

        dots_in_column = []
        for i in range(len(column)):
            if column[i] == blue_color:
                dots_in_column.append(i)
        # First append blue ones, then red ones
        for i in range(len(column)):
            if column[i] == red_color:
                dots_in_column.append(i)

        for _ in range(1000):
            new_blue_dots = []
            new_red_dots = []
            for i in dots_in_column:
                if column[i] == blue_color:
                    if i + 1 == len(column):
                        continue
                    if column[i + 1] == red_color:
                        continue
                    if i + 1 not in dots_in_column:
                        column[i + 1] = blue_color
                        new_blue_dots.append(i + 1)

                if column[i] == red_color:
                    if i - 1 == -1:
                        continue
                    if column[i - 1] == blue_color:
                        if i - 1 in new_blue_dots:
                            new_blue_dots.remove(i - 1)
                            column[i - 1] = zero_color
                    else:
                        if i - 1 not in dots_in_column:
                            column[i - 1] = red_color
                            new_red_dots.append(i - 1)
            dots_in_column = new_blue_dots + dots_in_column + new_red_dots
            print(column)
        for row in range(len(input_grid)):
            output_grid[row][col] = column[row]
    return output_grid


def solve_probed_tasks(tasks):
    assert len(tasks) == 2

    solutions = {}
    for task in tasks:
        task_solution = [[0]]
        try:
            res = solve_task_2(task)
            task_solution = res
        except Exception as e:
            pass
        task_solution = Solution(task.id, [task_solution, task_solution], 1337)
        solutions[task.id] = task_solution
    return solutions

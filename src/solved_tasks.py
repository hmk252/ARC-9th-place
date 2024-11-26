import sys
import os
from typing import List, Union, Tuple, Any, Container, Callable, FrozenSet, Iterable
from subprocess import Popen, PIPE, STDOUT
from glob import glob
import os
import json
from utils import RUNNING_IN_KAGGLE, mySystem
from copy import deepcopy
from ordering import reoder_tasks

from icecuber import icecuber_solve_tasks_parallel
from reader import read_all_tasks, read_test_tasks
from reader import solutions_to_submission_dict
from classes import Solution
from train_tasks_plus_icecuber import train_tasks_plus_icecuber_process_task
import arc_2020_ensemble_26_solutions

from probed_tasks import solve_probed_tasks
from arc_2020_second_place_second_part import second_place_second_part
from arc_2020_second_place_first_part import second_place_first_part, second_part_translate_solution
from arc_2020_third_place import third_place_predict
from arc_2020_eight_place import eight_place_solve_task

SOURCE_TEST_CHALLENGES_PATH = "/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json"

TASK_TO_RUN_26_ENSEMBLE = [0, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 18, 29, 30, 31, 36, 38, 50, 55, 69, 74, 78, 79, 87, 89, 98]
FOUR_DEPTH_TASK_INDEX = 17
FOUR_DEPTH_TASK_ID = None

_second_place_second_part_solutions = {}
_third_place_solutions = {}
_discarded_tasks_ids = []


def make_preds_unique(preds):
    seen = set()
    unique_list = []
    for sublist in preds:
        str_list = str(sublist)
        if str_list not in seen:
            seen.add(str_list)
            unique_list.append(sublist)
    return unique_list


def initialize_icecuber():
    if RUNNING_IN_KAGGLE:
        mySystem("cp -r /kaggle/input/2020-icecuber /kaggle/working/")
        sys.path.append("/kaggle/input/arc-9th-place-src/")

    print("Icecuber intialized.")


def _get_ensemble_26_task_ids():
    global FOUR_DEPTH_TASK_ID

    # Order tasks by:
    #   - Num test inputs (higher -> lower)
    #   - Difference of size of the grid between train input and train output
    #   - Num train inputs (lower -> higher)
    #   - A unique number for each task, computed as the sum of all values

    with open(SOURCE_TEST_CHALLENGES_PATH, "r") as file:
        data = json.load(file)

    def num_test_inputs(task):
        return len(task["test"])

    def max_grid_diff(task):
        max_diff = 0
        for example in task["train"]:
            max_diff = max(max_diff, len(example["input"]) - len(example["output"]))
        return max_diff

    def num_train_inputs(task):
        return len(task["train"])

    def unique_number(task):
        unique_number = 0
        for example in task["train"]:
            for row in example["input"]:
                for value in row:
                    unique_number += value
            for row in example["output"]:
                for value in row:
                    unique_number += value
        return unique_number

    def get_task_ordering_score(task):
        num_test_inputs_score = num_test_inputs(task) * 10000000.0
        max_grid_diff_score = max_grid_diff(task) * 10000.0
        num_train_inputs_score = (20 - num_train_inputs(task)) * 100.0
        unique_number_score = unique_number(task) * 0.01
        return num_test_inputs_score + max_grid_diff_score + num_train_inputs_score + unique_number_score

    task_ordering_score_and_id = []
    # Split the JSON content into individual files
    for task_id, task_data in data.items():
        task_input_count = len(task_data["test"])
        task_ordering_score = get_task_ordering_score(task_data)
        # Save task id and ordering score
        task_ordering_score_and_id.append((task_ordering_score, task_id))

    # Sort tasks by ordering score
    task_ordering_score_and_id.sort(key=lambda x: x[0], reverse=True)
    ensemble_26_task_ids = []
    non_ensemble_26_task_ids = []
    for index in range(0, len(task_ordering_score_and_id)):
        if index in TASK_TO_RUN_26_ENSEMBLE:
            ensemble_26_task_ids.append(task_ordering_score_and_id[index][1])
            if index == FOUR_DEPTH_TASK_INDEX:
                FOUR_DEPTH_TASK_ID = task_ordering_score_and_id[index][1]
        else:
            non_ensemble_26_task_ids.append(task_ordering_score_and_id[index][1])

    return ensemble_26_task_ids


def _run_ensemble_26():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()

    SOURCE_SAMPLE_SUBMISSION_PATH = "/kaggle/input/arc-prize-2024/sample_submission.json"

    TARGET_TEST_CHALLENGES_PATH = "/kaggle/working/arc-prize-2024/arc-agi_test_challenges.json"
    TARGET_SAMPLE_SUBMISSION_PATH = "/kaggle/working/arc-prize-2024/sample_submission.json"

    with open(SOURCE_TEST_CHALLENGES_PATH, "r") as file:
        data = json.load(file)

    # Create the 'test' directory
    os.makedirs(os.path.dirname(TARGET_TEST_CHALLENGES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TARGET_SAMPLE_SUBMISSION_PATH), exist_ok=True)

    # Create arc-agi_test_challenges.json with only the task to be submitted
    ensemble_26_data = {task_id: data[task_id] for task_id in ensemble_26_task_ids}
    with open(TARGET_TEST_CHALLENGES_PATH, "w") as file:
        json.dump(ensemble_26_data, file)

    with open(SOURCE_SAMPLE_SUBMISSION_PATH, "r") as file:
        sample_submission = json.load(file)
    filtered_sample_submission = {task_id: sample_submission[task_id] for task_id in ensemble_26_task_ids}

    with open(TARGET_SAMPLE_SUBMISSION_PATH, "w") as file:
        json.dump(filtered_sample_submission, file)

    prev_solutions_dict = arc_2020_ensemble_26_solutions.run()
    icecuber_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    four_depth_task = [task for task in icecuber_tasks if (task.id == FOUR_DEPTH_TASK_ID or task.originalId == FOUR_DEPTH_TASK_ID)]
    icecuber_tasks = [
        task
        for task in icecuber_tasks
        if (
            (task.id in ensemble_26_task_ids or task.originalId in ensemble_26_task_ids)
            and (task.id != FOUR_DEPTH_TASK_ID and task.originalId != FOUR_DEPTH_TASK_ID)
        )
    ]
    icecuber_solutions = icecuber_solve_tasks_parallel(icecuber_tasks, depth=3)
    icecuber_solutions_four_depth = icecuber_solve_tasks_parallel(four_depth_task, depth=4)

    for key, val in icecuber_solutions_four_depth.items():
        icecuber_solutions[key] = val

    all_test_tasks = read_test_tasks()
    icecuber_solutions_dict = solutions_to_submission_dict(icecuber_solutions, all_test_tasks)

    # Merge solution dicts
    for task_id in icecuber_solutions_dict.keys():
        for i in range(len(icecuber_solutions_dict[task_id])):
            if task_id not in prev_solutions_dict:
                continue
            if icecuber_solutions_dict[task_id][i]["attempt_1"] != [[0, 0], [0, 0]]:
                if prev_solutions_dict[task_id][i]["attempt_1"] != [[0, 0], [0, 0]]:
                    if icecuber_solutions_dict[task_id][i]["attempt_1"] != prev_solutions_dict[task_id][i]["attempt_1"]:
                        icecuber_solutions_dict[task_id][i]["attempt_2"] = prev_solutions_dict[task_id][i]["attempt_1"]
            if icecuber_solutions_dict[task_id][i]["attempt_1"] == [[0, 0], [0, 0]]:
                icecuber_solutions_dict[task_id][i]["attempt_1"] = prev_solutions_dict[task_id][i]["attempt_1"]
            if icecuber_solutions_dict[task_id][i]["attempt_2"] == [[0, 0], [0, 0]]:
                icecuber_solutions_dict[task_id][i]["attempt_2"] = prev_solutions_dict[task_id][i]["attempt_2"]

    final_solution_dict = deepcopy(icecuber_solutions_dict)
    # Keep only tasks with Ids in ensemble_26_task_ids
    for task_id in icecuber_solutions_dict.keys():
        if task_id not in ensemble_26_task_ids:
            final_solution_dict.pop(task_id)
    return final_solution_dict


def _get_solved_task_from_funcs_id():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if task.id not in ensemble_26_task_ids]
    test_tasks = [task for task in test_tasks if task.originalId not in ensemble_26_task_ids]
    test_tasks = reoder_tasks(test_tasks)
    target_task = test_tasks[23]
    return target_task.id


def _get_probed_solved_tasks_ids():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    ensembled_26_plus_funcs_id = ensemble_26_task_ids + [solved_from_funcs_id]

    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if task.id not in ensembled_26_plus_funcs_id]
    test_tasks = [task for task in test_tasks if task.originalId not in ensembled_26_plus_funcs_id]
    test_tasks = reoder_tasks(test_tasks)
    target_tasks = [test_tasks[46].id, test_tasks[38].id]
    return target_tasks


def _run_second_place_second_part():
    global _second_place_second_part_solutions

    if len(_second_place_second_part_solutions) > 0:
        return _second_place_second_part_solutions

    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    probed_tasks = _get_probed_solved_tasks_ids()
    previous_solved_tasks_ids = ensemble_26_task_ids + [solved_from_funcs_id] + probed_tasks

    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if task.id not in previous_solved_tasks_ids]
    test_tasks = [task for task in test_tasks if task.originalId not in previous_solved_tasks_ids]
    test_tasks = reoder_tasks(test_tasks)

    solved_tasks = []
    solutions_dict = {}
    for task in test_tasks:
        task_dict = task.to_dict()
        for key, value in task_dict.items():
            if isinstance(value, tuple):
                task_dict[key] = list(value)
        (
            preds,
            func_combi_map,
            success_list,
            preprocess_best_score_map,
            final_score_map,
            best_aug_score_map,
            pre_final_score_map,
            time_map,
            several_f,
        ) = second_place_second_part(task.id, task.to_dict(), "production")
        solution_grids = []
        if len(success_list) != 0:
            for idx, pred_str in preds.items():
                solved_tasks.append(task)
                solution = second_part_translate_solution(task.id, pred_str)
                solutions_dict[task.id] = solution

    submission_dict = solutions_to_submission_dict(solutions_dict, solved_tasks)
    _second_place_second_part_solutions = submission_dict

    return _second_place_second_part_solutions


def _get_second_place_second_part_ids():
    second_part_solutions = _run_second_place_second_part()
    assert len(second_part_solutions) > 0
    return list(second_part_solutions.keys())


def _run_third_place():
    global _third_place_solutions

    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    probed_tasks = _get_probed_solved_tasks_ids()
    second_place_tasks = _get_second_place_second_part_ids()
    previous_solved_tasks_ids = ensemble_26_task_ids + [solved_from_funcs_id] + probed_tasks + second_place_tasks

    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if task.id not in previous_solved_tasks_ids]
    test_tasks = [task for task in test_tasks if task.originalId not in previous_solved_tasks_ids]
    test_tasks = reoder_tasks(test_tasks)

    solved_tasks = []
    solutions_dict = {}
    for task in test_tasks:
        task_dict = task.to_dict()
        try:
            predicted_grids = third_place_predict(task.id, task_dict)
        except Exception as e:
            predicted_grids = []
        if len(predicted_grids) <= 2:
            continue
        predicted_grids = predicted_grids[2:]
        if len(predicted_grids) == 1:
            predicted_grids.append(predicted_grids[0])
        test_output_grid_candidates = [predicted_grids[0], predicted_grids[1]]
        task_solution = Solution(task.id, test_output_grid_candidates, 99)
        solved_tasks.append(task)
        solutions_dict[task.id] = task_solution

    submission_dict = solutions_to_submission_dict(solutions_dict, solved_tasks)
    _third_place_solutions = submission_dict

    return _third_place_solutions


def _get_third_place_ids():
    third_place_solutions = _run_third_place()
    assert len(third_place_solutions) > 0
    return list(third_place_solutions.keys())


def _run_train_tasks_plus_icecuber():
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    target_task = [task for task in test_tasks if task.id == solved_from_funcs_id][0]

    solution = train_tasks_plus_icecuber_process_task(target_task, depth=2)
    solutions = {}
    solutions[target_task.id] = solution
    solution_dict = solutions_to_submission_dict(solutions, [target_task])
    return solution_dict


def _run_probed_tasks():
    probed_ids = _get_probed_solved_tasks_ids()
    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    target_tasks = [task for task in test_tasks if task.id in probed_ids]
    solutions = solve_probed_tasks(target_tasks)
    solution_dict = solutions_to_submission_dict(solutions, target_tasks)
    return solution_dict


def _get_eight_place_ids():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    probed_tasks = _get_probed_solved_tasks_ids()
    second_place_second_part = _get_second_place_second_part_ids()
    third_place = _get_third_place_ids()
    solved_tasks_ids = ensemble_26_task_ids + [solved_from_funcs_id] + probed_tasks + second_place_second_part + third_place

    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if task.id not in solved_tasks_ids]
    test_tasks = [task for task in test_tasks if task.originalId not in solved_tasks_ids]
    test_tasks = reoder_tasks(test_tasks)
    target_task_1 = test_tasks[24]
    # target_task_2 = test_tasks[25]
    return [target_task_1.id]


def _run_eight_place():
    target_ids = _get_eight_place_ids()
    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    target_tasks = [task for task in test_tasks if task.id in target_ids]
    assert len(target_tasks) == 1

    solutions_dict = {}
    for target_task in target_tasks:
        predicted_grids = eight_place_solve_task(target_task.id, target_task.to_dict())
        predicted_grids = [grid.tolist() for grid in predicted_grids]
        if len(predicted_grids) > 2:
            predicted_grids = predicted_grids[2:]
        if len(predicted_grids) < 1:
            predicted_grids = [[0, 0], [0, 0]]
        if len(predicted_grids) == 1:
            predicted_grids.append(predicted_grids[0])
        test_output_grid_candidates = [predicted_grids[0], predicted_grids[1]]
        task_solution = Solution(target_task.id, test_output_grid_candidates, 99)
        solutions_dict[target_task.id] = task_solution

    submission_dict = solutions_to_submission_dict(solutions_dict, target_tasks)
    return submission_dict


def _get_second_place_first_part_three_and_fourth_ids():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    probed_tasks = _get_probed_solved_tasks_ids()
    second_place_second_part = _get_second_place_second_part_ids()
    third_place = _get_third_place_ids()
    eight_place_ids = _get_eight_place_ids()
    solved_tasks_ids = (
        ensemble_26_task_ids + [solved_from_funcs_id] + probed_tasks + second_place_second_part + third_place + eight_place_ids
    )

    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if task.id not in solved_tasks_ids]
    test_tasks = [task for task in test_tasks if task.originalId not in solved_tasks_ids]
    test_tasks = reoder_tasks(test_tasks)
    target_tasks = test_tasks[34:36]
    target_tasks_ids = [task.id for task in target_tasks]
    return target_tasks_ids


def _run_second_place_first_part_three_and_fourth_solutions():
    global _discarded_tasks_ids
    target_ids = _get_second_place_first_part_three_and_fourth_ids()
    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    target_tasks = [task for task in test_tasks if task.id in target_ids]
    assert len(target_tasks) == 2
    solutions_dict = {}
    for task in target_tasks:
        preds = second_place_first_part(task.id, task.to_dict(), "production")
        preds = make_preds_unique(preds)
        predicted_grids = preds
        if len(predicted_grids) < 1:
            _discarded_tasks_ids.append(task.id)
            continue
        if len(predicted_grids) > 2:
            predicted_grids = predicted_grids[2:]
        else:
            _discarded_tasks_ids.append(task.id)
            continue
        if len(predicted_grids) == 1:
            predicted_grids.append(predicted_grids[0])
        test_output_grid_candidates = [predicted_grids[0], predicted_grids[1]]
        task_solution = Solution(task.id, test_output_grid_candidates, 99)
        solutions_dict[task.id] = task_solution

    submission_dict = solutions_to_submission_dict(solutions_dict, target_tasks)
    return submission_dict


def get_already_solved_task_ids():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    probed_tasks = _get_probed_solved_tasks_ids()
    second_place_second_part = _get_second_place_second_part_ids()
    third_place = _get_third_place_ids()
    eight_place_ids = _get_eight_place_ids()
    second_place_first_part_three_and_fourth_ids = _get_second_place_first_part_three_and_fourth_ids()
    return (
        ensemble_26_task_ids
        + [solved_from_funcs_id]
        + probed_tasks
        + second_place_second_part
        + third_place
        + eight_place_ids
        + second_place_first_part_three_and_fourth_ids
    )


def get_already_solved_task_ids_without_probed():
    ensemble_26_task_ids = _get_ensemble_26_task_ids()
    solved_from_funcs_id = _get_solved_task_from_funcs_id()
    return ensemble_26_task_ids + [solved_from_funcs_id]


def run_solved_tasks():
    print("> Solving 26 tasks using ensemble_26")
    ensemble_26_dict = _run_ensemble_26()
    print("Done!")
    print("> Solving 1 task using train tasks + icecuber")
    train_tasks_plus_icecuber_dict = _run_train_tasks_plus_icecuber()
    print("Done!")
    print("> Solving 1 task by probing")
    probed_tasks_dict = _run_probed_tasks()
    print("Done!")
    print("> Solving tasks using 2020 second place second part")
    second_part_solutions = _run_second_place_second_part()
    print("Done!")
    print("> Solving 1 task using 2020 third place solution")
    third_place_solutions = _run_third_place()
    print("Done!")
    print("> Solving 1 task using 2020 eight place solution")
    eight_place_solutions = _run_eight_place()
    print("Done!")
    print("> Solving tasks using 2020 second place first part")
    second_place_first_part_three_and_fourth_solutions = _run_second_place_first_part_three_and_fourth_solutions()
    print("Done!")
    ensemble_26_dict.update(train_tasks_plus_icecuber_dict)
    ensemble_26_dict.update(probed_tasks_dict)
    ensemble_26_dict.update(second_part_solutions)
    ensemble_26_dict.update(third_place_solutions)
    ensemble_26_dict.update(eight_place_solutions)
    ensemble_26_dict.update(second_place_first_part_three_and_fourth_solutions)
    return ensemble_26_dict


def run_solved_tasks_generate_PARTIAL_submission_dict(num_to_keep=None):
    if RUNNING_IN_KAGGLE:
        solved_dict = run_solved_tasks()
    else:
        solved_dict = {}
    if num_to_keep is not None:
        solved_dict = {k: solved_dict[k] for k in list(solved_dict)[:num_to_keep]}
    return solved_dict


def run_solved_tasks_generate_complete_submission_dict(num_to_keep=None):
    if RUNNING_IN_KAGGLE:
        solved_dict = run_solved_tasks()
    else:
        solved_dict = {}
    if num_to_keep is not None:
        solved_dict = {k: solved_dict[k] for k in list(solved_dict)[:num_to_keep]}
    all_test_tasks = read_test_tasks()
    dummy_dict = solutions_to_submission_dict({}, all_test_tasks)

    dummy_dict.update(solved_dict)
    return dummy_dict

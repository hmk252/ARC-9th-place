from typing import List, Union, Tuple, Any, Container, Callable, FrozenSet, Iterable
import os
import json
from copy import deepcopy

from classes import Grid, Example, Task
from utils import RUNNING_IN_KAGGLE

EASY_TASKS_PATH = "../data/arc-agi_training_challenges.json"
EASY_TASK_SOLUTIONS_PATH = "../data/arc-agi_training_solutions.json"

HARD_TASKS_PATH = "../data/arc-agi_evaluation_challenges.json"
HARD_TASKS_SOLUTIONS_PATH = "../data/arc-agi_evaluation_solutions.json"

TEST_TASKS_PATH = "../data/arc-agi_test_challenges.json"

if RUNNING_IN_KAGGLE:
    EASY_TASKS_PATH = "/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json"
    EASY_TASK_SOLUTIONS_PATH = "/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json"

    HARD_TASKS_PATH = "/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    HARD_TASKS_SOLUTIONS_PATH = "/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json"

    TEST_TASKS_PATH = "/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json"

# fmt: off
my_test_ids = ['62ab2642', '8ba14f53', 'bb52a14b', '992798f6', '9ddd00f0', '48f8583b', '1c56ad9f', '762cd429', '0a2355a6', 'f3cdc58f', '4ff4c9da', 'b4a43f3b', 'e88171ec', '59341089', '79369cc6', 'f83cb3f6', '32e9702f', '009d5c81', '0d87d2a6', '817e6c09', 'aee291af', 'bf32578f', 'bcb3040b', 'e5790162', 'cad67732', '2a5f8217', '0607ce86', 'ac2e8ecf', '7c9b52a0', '3d31c5b3', 'd282b262', '60a26a3e', 'ad7e01d0', '0bb8deee', 'baf41dbf', '963f59bc', 'fc754716', '7039b2d7', '2697da3f', 'e133d23d', 'aa4ec2a5', '903d1b4a', 'f0df5ff0', '136b0064', 'f45f5ca7', '37d3e8b2', '4364c1c4', 'c48954c1', '90347967', 'b15fca0b', 'b457fec5', '15696249', '3391f8c0', '6ad5bdfd', 'b7cb93ac', 'b7fb29bc', '642d658d', '759f3fd3', 'c658a4bd', 'a680ac02', '0e671a1a', 'aa300dc3', 'f9d67f8b', '4e469f39', '66f2d22f', 'c663677b', '281123b4', '712bf12e', 'e1d2900e', 'fd4b2b02', '12422b43', '45737921', 'b7f8a4d8', 'ed74f2f2', '0b17323b', '99306f82', 'bd14c3bf', 'df8cc377', '2c737e39', '47996f11', 'f9a67cb5', 'c92b942c', 'ae58858e', 'c64f1187', 'd37a1ef5', '40f6cd08', '0c786b71', 'd931c21c', '4f537728', '68b67ca3', 'd492a647', '9b2a60aa', '695367ec', 'e7dd8335', '070dd51e', '7e02026e', '0c9aba6e', 'e872b94a', 'c87289bb', 'b0f4d537', '770cc55f', 'd47aa2ff', '414297c0', '7c8af763', 'e9ac8c9e', '137f0df0', '8a371977', '2037f2c7', '6f473927', '88207623', 'e95e3d8e', 'e9b4f6fc', '93b4f4b3', 'aa18de87', '27a77e38', '34b99a2b', '60c09cac', 'dc2aa30b', '48131b3c', '0a1d4ef5', '7d419a02', '9b365c51', 'ed98d772', '1a6449f1', 'f5aa3634', '1c02dbbe', 'a934301b',  '7953d61e', '50a16a69', 'b9630600', '55783887', 'b0722778', 'd4b1c2b1', 'dc2e9a9d', '5783df64', 'e78887d1', '13713586', '73182012', 'cd3c21df', '0934a4d8', '891232d6', 'e57337a4', '8dae5dfc', '9caba7c3', '3ed85e70', '184a9768', 'd4c90558', 'd19f7514', '319f2597', '833dafe3', '212895b5', 'ce039d91', '575b1a71', 'b7999b51', '20981f0e', 'b20f7c8b', '2b01abd0', 'e9bb6954', '917bccba', 'ac3e2b04', '15113be4', 'c35c1b4c', '358ba94e', '72a961c9', 'f8be4b64', '45bbe264', '9c56f360', '5af49b42', '67c52801', '50aad11f',  'a406ac07', '64a7c07e', '477d2879', 'f4081712', '2546ccf6', '140c817e', '3ee1011a', '85b81ff1', 'cf133acc', '1acc24af', '7d1f7ee8', 'be03b35f', 'e1baa8a4', '93c31fbe', '18419cfa', 'ca8de6ea', '256b0a75', 'e5c44e8f', 'd94c3b52', '85fa5666', 'ea9794b1', '69889d6e', '21f83797', '5207a7b5', '27f8ce4f', 'bf89d739', '9c1e755f', '3f23242b', '5289ad53', 'f3e62deb']
# fmt: on


def generate_subtask_id(task_id, index):
    # Replace last two characters in task_id with index number (formatted as a 2 digit string)
    return task_id[:-4] + str(index).zfill(4)


def read_tasks(tasks_json, solutions_json):
    tasks = []
    for task_id, _ in tasks_json.items():
        task_json = deepcopy(tasks_json[task_id])
        if solutions_json:
            solution = solutions_json[task_id]
            for i, _ in enumerate(task_json["test"]):
                task_json["test"][i]["output"] = solution[i]
        subtasks = []
        if len(task_json["test"]) > 1:
            for i, test in enumerate(task_json["test"]):
                subtask_id = generate_subtask_id(task_id, i)
                subtask_json = {"train": task_json["train"], "test": [test]}
                subtask = Task(subtask_id, subtask_json)
                subtask.originalId = task_id
                subtask.originalTestIndex = i
                subtasks.append(subtask)
        else:
            subtasks = [Task(task_id, task_json)]
        tasks.extend(subtasks)
    return tasks


def read_my_test_tasks(with_solutions=False):
    assert not RUNNING_IN_KAGGLE
    _, hard_tasks, _ = read_all_tasks(with_solutions=with_solutions)
    tasks = []
    for task in hard_tasks:
        if task.id in my_test_ids or task.originalId in my_test_ids:
            tasks.append(task)
    assert len(tasks) >= 100
    return tasks


def read_test_tasks(with_solutions=False):
    test_tasks_json = json.load(open(TEST_TASKS_PATH, "r"))
    if with_solutions:
        easy_tasks_solutions_json = json.load(open(EASY_TASK_SOLUTIONS_PATH, "r"))
        hard_tasks_solutions_json = json.load(open(HARD_TASKS_SOLUTIONS_PATH, "r"))
        all_solutions_json = {**easy_tasks_solutions_json, **hard_tasks_solutions_json}
        return read_tasks(test_tasks_json, all_solutions_json)
    return read_tasks(test_tasks_json, None)


def read_all_tasks(with_solutions=True):
    easy_tasks_json = json.load(open(EASY_TASKS_PATH, "r"))
    if with_solutions:
        easy_tasks_solutions_json = json.load(open(EASY_TASK_SOLUTIONS_PATH, "r"))
    else:
        easy_tasks_solutions_json = None

    hard_tasks_json = json.load(open(HARD_TASKS_PATH, "r"))
    if with_solutions:
        hard_tasks_solutions_json = json.load(open(HARD_TASKS_SOLUTIONS_PATH, "r"))
    else:
        hard_tasks_solutions_json = None

    hard_tasks = read_tasks(hard_tasks_json, hard_tasks_solutions_json)
    easy_tasks = read_tasks(easy_tasks_json, easy_tasks_solutions_json)

    temp = json.load(open(TEST_TASKS_PATH, "r"))
    test_ids = []
    for task_id, _ in temp.items():
        test_ids.append(task_id)

    return easy_tasks, hard_tasks, test_ids


def get_dummy_submission(count):
    assert count > 0
    return [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}] * count


def solutions_to_submission_dict(solutions, all_test_tasks):
    # Add dummy submission to task with no solution
    submission_dict = {}
    for task in all_test_tasks:
        dummy_output = {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
        task_solution = None
        for sol in solutions.values():
            if sol.task_id == task.id:
                task_solution = {
                    "attempt_1": sol.test_output_grid_candidates[0],
                    "attempt_2": sol.test_output_grid_candidates[1],
                }
                break
        if task_solution is None:
            task_solution = dummy_output
        if task.originalId == None:
            submission_dict[task.id] = [task_solution]
        else:
            task_submission = []
            if task.originalId in submission_dict:
                task_submission = submission_dict[task.originalId]
            if len(task_submission) < task.originalTestIndex + 1:
                # Fill with None
                task_submission.extend([dummy_output] * (task.originalTestIndex + 1 - len(task_submission)))
            task_submission[task.originalTestIndex] = task_solution
            submission_dict[task.originalId] = task_submission
    return submission_dict


def save_solutions_as_submission(solutions, all_test_tasks, file_path):
    submission_dict = solutions_to_submission_dict(solutions, all_test_tasks)
    with open(file_path, "w") as f:
        json.dump(submission_dict, f)

import json
from utils import RUNNING_IN_KAGGLE
from reader import (
    read_all_tasks,
    solutions_to_submission_dict,
    read_test_tasks,
)
from ordering import reoder_tasks
from classes import Solution
from solved_tasks import (
    run_solved_tasks_generate_complete_submission_dict,
    get_already_solved_task_ids,
)
from reader import solutions_to_submission_dict
from icecuber import icecuber_solve_tasks_parallel
from arc_2020_second_place_first_part_original import second_place_first_part_original, second_part_translate_solution
from arc_2020_ninth_place import ninth_place_solve_task

if not RUNNING_IN_KAGGLE:
    already_submitted_tasks_ids = []
else:
    already_submitted_tasks_ids = get_already_solved_task_ids()

if RUNNING_IN_KAGGLE:
    test_tasks = read_test_tasks(with_solutions=(False if RUNNING_IN_KAGGLE else True))
    test_tasks = [task for task in test_tasks if (task.id not in already_submitted_tasks_ids) and (task.originalId not in already_submitted_tasks_ids)]  # fmt: skip
    test_tasks = reoder_tasks(test_tasks)
else:
    e, h, _ = read_all_tasks(with_solutions=False)
    test_tasks = e + h
    test_tasks = test_tasks[:100]
from utils import diagonal_flip_task, diagonal_flip

print("> Solving 1 task using 2020 ninth place solution")
nine_place_tasks = test_tasks[57 : len(test_tasks)]
nine_place_solutions_dict = {}
for task in nine_place_tasks:
    task_dict = task.to_dict()
    predicted_grids = ninth_place_solve_task(task.id, task_dict)
    if len(predicted_grids) < 1:
        continue
    if len(predicted_grids) == 1:
        predicted_grids.append(predicted_grids[0])
    test_output_grid_candidates = [predicted_grids[0], predicted_grids[1]]
    task_solution = Solution(task.id, test_output_grid_candidates, 99)
    nine_place_solutions_dict[task.id] = task_solution
print("Done!")

print("> Solving 1 task using 4 depth icecuber flipped diagonally")
icecuber_tasks = test_tasks[32:48]
icecuber_solved_task = None
icecuber_solved_task_solution = None
max_score = -1
for task in icecuber_tasks:
    FLIP_DIRECTION = "main"
    task = diagonal_flip_task(task, FLIP_DIRECTION)
    try:
        icecuber_solutions_four_depth = icecuber_solve_tasks_parallel([task], depth=44)
    except Exception as e:
        continue
    except MemoryError:
        continue

    for key, val in icecuber_solutions_four_depth.items():
        score = val.train_score
        if score > max_score:
            max_score = score
            icecuber_solved_task = task
            val.test_output_grid_candidates = [diagonal_flip(grid, FLIP_DIRECTION) for grid in val.test_output_grid_candidates]
            icecuber_solved_task_solution = val
print("Done!")

print("> Solving tasks using 2020 second place solution")
solutions_dict = {}
for task in test_tasks:
    task_dict = task.to_dict()
    solution_grids = []
    pred_str = second_place_first_part_original(task.id, task.to_dict(), "production")
    solution = second_part_translate_solution(task.id, pred_str)
    solutions_dict[task.id] = solution
print("Done!")

if RUNNING_IN_KAGGLE:
    all_test_tasks = read_test_tasks()
else:
    all_test_tasks = test_tasks

if icecuber_solved_task != None:
    solutions_dict[icecuber_solved_task.id] = icecuber_solved_task_solution
solutions_dict.update(nine_place_solutions_dict)

new_submission_dict = solutions_to_submission_dict(solutions_dict, test_tasks)
submission_dict = run_solved_tasks_generate_complete_submission_dict()

submission_dict.update(new_submission_dict)

with open("submission.json", "w") as f:
    json.dump(submission_dict, f)
print("Saved submission.json")

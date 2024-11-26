#!/usr/bin/env python

from subprocess import *
from concurrent.futures import ThreadPoolExecutor as Pool
import os
import sys
import resource
import psutil
from random import *
import time
import json

from os import system
from glob import glob

RUNNING_IN_KAGGLE = os.path.exists("/kaggle")

# if not RUNNING_IN_KAGGLE:
#     call(["make", "clean"])
#     call(["make", "-j"])
#     call(["make", "-j", "count_tasks"])

if not os.path.isfile("./run"):
    call(["make", "-j"])
if not os.path.isfile("./count_tasks"):
    call(["make", "-j", "count_tasks"])

SUCCESS, TLE, MLE, RTE, RUNNING = 0, 1, 2, 3, -1
exit_names = ["SUCCESS", "TLE", "MLE", "RTE", "RUNNING"]

start_time = time.time()


MEMORY_LIMIT = 16 * 1024 * 0.95  # MB
if not RUNNING_IN_KAGGLE:
    MEMORY_LIMIT = 10 * MEMORY_LIMIT  # MB

TIME_LIMIT = 9 * 60 * 60 * 0.95  # Seconds


class Process:
    def __init__(self, cmd, timeout, maxmemory):
        fn = cmd.replace(" ", "_")
        self.fout = open("tmp/cpplogs/%s.out" % fn, "w")
        self.ferr = open("tmp/cpplogs/%s.err" % fn, "w")
        print(cmd)
        sys.stdout.flush()
        self.cmd = cmd
        self.process = Popen(cmd.split(), stdout=self.fout, stderr=self.ferr, shell=False)
        self.pid = self.process.pid
        self.mp = psutil.Process(self.pid)
        self.memused, self.timeused = 0, 0
        self.start_time = time.time()
        self.timeout = timeout
        self.maxmemory = maxmemory

    def update(self):
        self.memory = self.mp.memory_info().rss / 2**20
        self.memused = max(self.memused, self.memory)
        self.timeused = time.time() - self.start_time
        if self.memory > self.maxmemory:
            return (MLE, self.timeused, self.memused)
        if self.timeused > self.timeout:
            return (TLE, self.timeused, self.memused)
        if not self.memory:
            if self.process.wait():
                return (RTE, self.timeused, self.memused)
            else:
                return (SUCCESS, self.timeused, self.memused)
        return (RUNNING, self.timeused, self.memused)

    def __del__(self):
        self.fout.close()
        self.ferr.close()


class Command:
    def __init__(self, cmd, expected_time=TIME_LIMIT, expected_memory=MEMORY_LIMIT, slack=1.5):
        self.cmd = cmd
        self.time = expected_time
        self.mem = expected_memory
        self.slack = slack

    def __lt__(self, other):
        return self.time < other.time


def runAll(cmd_list, threads):
    THREAD_LIMIT = threads

    ret_stats = {}

    cmd_list = sorted(cmd_list)

    dt = 0.1
    running = []
    cmdi = 0

    def callback(process, status, timeused, memused):
        assert status != RTE
        print(exit_names[status], process.cmd, " %.1fs" % timeused, "%.0fMB" % memused)
        sys.stdout.flush()

        ret_stats[process.cmd] = (status, timeused, memused)

    while len(running) or cmdi < len(cmd_list):
        while cmdi < len(cmd_list) and len(running) < THREAD_LIMIT:
            cmd = cmd_list[cmdi]
            process = Process(cmd.cmd, cmd.time * cmd.slack, cmd.mem * cmd.slack)
            running.append(process)
            cmdi += 1

        torem = []
        mems = []
        for r in running:
            status, timeused, memused = r.update()
            mems.append(r.memory)
            if status != RUNNING:
                callback(r, status, timeused, memused)
                torem.append(r)

        if sum(mems) > MEMORY_LIMIT:
            r = running[mems.index(max(mems))]
            r.process.kill()
            callback(r, MLE, r.timeused, r.memused)
            torem.append(r)
            print("Killed due to memory limit")
            # raise Exception("HERE")
            # THREAD_LIMIT = 1

        for r in torem:
            running.remove(r)

        time.sleep(dt)

        if time.time() - start_time >= TIME_LIMIT:
            for r in running:
                r.process.kill()
                callback(r, TLE, r.timeused, r.memused)
            break

    return ret_stats


TASK_PATH = "tmp/input_task"

system("mkdir -p tmp/output")
system("mkdir -p tmp/cpplogs")
system("rm -f tmp/output/answer*.csv")
ntasks = int(check_output("./count_tasks"))
task_list = range(0, ntasks)

# read args
DEPTH = 3
if len(sys.argv) > 1:
    DEPTH = int(sys.argv[1])

assert DEPTH == 3 or DEPTH == 4 or DEPTH == 2 or DEPTH == 30 or DEPTH == 44

target_task = None


if DEPTH != 44:
    MEMORY_LIMIT = 24 * 1024 * 0.95  # MB
    TIME_LIMIT = 9 * 60 * 60 * 0.95  # Seconds
    commands = []
    if DEPTH == 2:
        for i in range(ntasks):
            commands.append(Command(f"./run {i} 2"))
            # commands.append(Command(f"./run {i} 22"))
            # commands.append(Command(f"./run {i} 32"))

    if DEPTH == 30:
        for i in range(ntasks):
            commands.append(Command(f"./run {i} 3"))

    if DEPTH == 3 or DEPTH == 4:
        for i in range(ntasks):
            commands.append(Command(f"./run {i} 3"))
        for i in range(ntasks):
            commands.append(Command(f"./run {i} 23"))
        for i in range(ntasks):
            commands.append(Command(f"./run {i} 33"))

    threads = 4
    if not RUNNING_IN_KAGGLE:
        threads = 32
    runAll(commands, threads)

    if DEPTH == 4:
        depth_4_commands = []
        for i in range(ntasks):
            depth_4_commands.append(Command(f"./run {i} 4"))
        runAll(depth_4_commands, 2)
else:
    MEMORY_LIMIT = 24 * 1024 * 0.95  # MB
    TIME_LIMIT = 600
    depth_4_commands = []
    for i in range(ntasks):
        depth_4_commands.append(Command(f"./run {i} 4"))
    runAll(depth_4_commands, 2)


def read(fn):
    f = open(fn)
    t = f.read()
    f.close()
    return t


def translate_grid(pred):
    pred_lines = pred.split("|")[1:-1]  # Remove empty strings from split
    pred_matrix = [list(map(int, line)) for line in pred_lines]
    return pred_matrix


class Icecuber_solution:
    def __init__(self, task_id, train_outputs, test_output, score):
        self.task_id = task_id
        self.train_outputs = train_outputs
        self.test_output = test_output
        self.score = score


cands = []
for taski in task_list:
    stored_id = None
    for fn in glob("tmp/output/answer_%d_*.csv" % taski):
        t = read(fn).strip().split("\n")
        id = t[0].split("_")[0]
        task_id = id
        for cand in t[1:]:
            img, score = cand.split()
            imgs = img.split("-")
            score = float(score)
            train_outputs = []
            test_output = []
            for i in range(len(imgs)):
                grid = translate_grid(imgs[i])
                if i == len(imgs) - 1:
                    test_output = grid
                else:
                    train_outputs.append(grid)
            solution = Icecuber_solution(task_id, train_outputs, test_output, score)
            cands.append(solution)
    # Order from highest score to lowest
    cands.sort(key=lambda x: x.score, reverse=True)

# Write all the solutions in cand to a json file
with open("tmp/output/solutions.json", "w") as f:
    json.dump([cand.__dict__ for cand in cands], f)

# Read them back in
# with open("tmp/output/solutions.json", "r") as f:
#     cands = json.load(f)
# print("---")
# print(cands)

"""
This file is taken directly, without modification, from:
    
    ThunderKittens/kernels/attn/demo/mla_decode

TODO: Clean up and refactor.

"""
from __future__ import annotations

import math
import torch
import heapq
import random

import time
import numpy as np

from sklearn.linear_model import LinearRegression

from tqdm import tqdm

from dataclasses import dataclass, field
from typing import List

from tk_mla.mla_decode import __get_quality__


# Timing constants (in microseconds)
PARTIAL_STARTUP_TIME = 3.0         # Startup time for partial operations
PARTIAL_WRITEOUT_TIME = 4.5        # Writeout time for partial operations
PARTIAL_COST_PER_STEP = 1.49       # Cost per step (per 32 tokens) for partial operations
PARTIAL_OVERHEAD = PARTIAL_STARTUP_TIME + PARTIAL_WRITEOUT_TIME # Total overhead for a partial operation.

REDUCTION_STARTUP_TIME = 4.0       # Startup time for reduction operations
# REDUCTION_STARTUP_TIME = 2.0       # Startup time for reduction operations
REDUCTION_WRITEOUT_TIME = 1.0      # Writeout time for reduction operations
REDUCTION_PRODUCER_LATENCY = 1.0   # Latency between a producer load and when the consumer can access it.
REDUCTION_COST_PER_STEP = 0.4      # Cost per reduction step

SYNCHRONIZATION_COST = 0.5         # Synchronization cost between dependent operations
NUM_PROCESSORS = 132


@dataclass
class Task:
    uid: int
    batch_id: int              # Which sequence this task belongs to.
    tok_ids: List[int]         # Query indices
    name: str
    task_type: str             # "partial" or "reduction"
    dependencies: List[int] = field(default_factory=list)
    next_input_time: float = None
    start: float = None
    finish: float = None
    processor: int = None
    args: dict = field(default_factory=dict)
    def __lt__(self, other):
        return self.uid < other.uid # just need a way to break ties.


def create_arguments_from_task_schedule(tasks: List[Task], new_tokens: int, num_processors: int = 1, enable_timings: bool = False, q_heads: int = 16):
    OP_PARTIAL, OP_REDUCTION = 1, 2
    partial_padding = [0]*23

    def make_partial_arg(task: Task) -> List[int]:
        return torch.tensor([OP_PARTIAL,
                task.uid,  # uid
                -task.uid-1 if task.args["write_scratch"] else task.batch_id,  # destination (negative means write scratch)
                min(task.tok_ids),  # start token
                task.batch_id,
                min(task.tok_ids),  # duplicate start token
                task.args["start"], task.args["end"], task.args["length"]] + partial_padding, device='cuda', dtype=torch.int32)

    def make_merge_arg(task: Task) -> List[int]:
        assert(len(task.dependencies) <= 11+16)
        return torch.tensor([OP_REDUCTION,
                            task.uid,  # uid
                            len(task.dependencies)-1,  # number of dependencies minus one
                            -task.uid-1 if task.args["write_scratch"] else task.batch_id,
                            task.tok_ids[0]] + task.dependencies + [0]*(32 - 5 - len(task.dependencies)), device='cuda', dtype=torch.int32)

    num_instructions = max(t.uid for t in tasks) + 1
    processor_tasks = [[] for _ in range(num_processors)]
    print(f'Number of instructions: {len(tasks)}')
    for task in tasks:
        processor_tasks[task.processor].append(task)
    for pid in range(num_processors):
        processor_tasks[pid].sort(key=lambda t: t.start)
    # print('Final finish time:', max(t.finish for t in tasks))
    # print('Max number of dependencies:', max(len(t.dependencies) for t in tasks)
    max_num_processor_instructions = max(len(ptasks) for ptasks in processor_tasks)
    Instructions = torch.zeros((num_processors, max_num_processor_instructions, 32), dtype=torch.int32, device='cuda')
    O_scratch = torch.zeros((num_instructions, new_tokens, q_heads, 512), dtype=torch.float32, device='cuda')
    L_scratch = torch.zeros((num_instructions, new_tokens, q_heads), dtype=torch.float32, device='cuda')
    Semaphore = torch.zeros((num_instructions, new_tokens), dtype=torch.int32, device='cuda')
    Timings = torch.zeros((num_processors, max_num_processor_instructions, 64), dtype=torch.int32, device='cuda') if enable_timings else None
    torch.cuda.synchronize()
    for pid in range(num_processors):
        for tid, task in enumerate(processor_tasks[pid]):
            if task.task_type == "partial":
                partial_arg = make_partial_arg(task)
                Instructions[pid, tid] = partial_arg
            elif task.task_type == "reduction":
                merge_arg = make_merge_arg(task)
                Instructions[pid, tid] = merge_arg
    return Instructions, O_scratch, L_scratch, Semaphore, Timings


def backward_schedule(processors: List[int], batch_id: int, seq_length: int, tok_ids: List[int], partial_uid: int, reduction_uid: int, q_heads: int = 16):
    max_tokens = 4
    assert (len(tok_ids) > 0 and len(tok_ids) <= max_tokens), f"If num_tokens is > {max_tokens}, please generate two separate schedules for each group of {max_tokens} tokens."
    
    NUM_PROCESSORS = len(processors)
    if NUM_PROCESSORS == 1:
        steps = (seq_length + 31) // 32
        duration = PARTIAL_STARTUP_TIME + (steps * PARTIAL_COST_PER_STEP) + PARTIAL_WRITEOUT_TIME
        return [Task(
            uid=partial_uid,
            batch_id=batch_id,
            tok_ids=tok_ids,
            name=f"Partial_B={batch_id}_Tok={tok_ids}_ALL",
            task_type="partial",
            dependencies=[],
            processor=processors[0],
            start=-duration,
            finish=0,
            args={"start": 0, "end": seq_length, "length": seq_length, "write_scratch": False}
        )], partial_uid+1, reduction_uid

    p_idx = 0 # active processor index

    tasks = {p: [] for p in processors} # Note that tasks will be stored in reversed order.

    # Create initial reduction task.
    tasks[processors[p_idx]].append(Task(
        uid=reduction_uid,
        batch_id=batch_id,
        tok_ids=[tok_ids[0]],
        name=f'reduction_{reduction_uid}',
        task_type="reduction",
        dependencies=[],
        processor=processors[p_idx],
        finish=0,
        args={"write_scratch": False}
    ))
    tasks[processors[p_idx]][0].next_input_time = tasks[processors[p_idx]][0].finish - (REDUCTION_WRITEOUT_TIME + REDUCTION_PRODUCER_LATENCY + SYNCHRONIZATION_COST)
    available_inputs = [(-tasks[processors[p_idx]][0].next_input_time, tasks[processors[p_idx]][0])]
    heapq.heapify(available_inputs)
    reduction_uid, p_idx = reduction_uid+1, (p_idx+1) % NUM_PROCESSORS

    current_cost = __get_quality__([-x[0] for x in available_inputs], num_processors=NUM_PROCESSORS, num_tokens=len(tok_ids), seq_length=seq_length)

    while True:
        # Let's see what would happen if we added a new reduction task.
        speculative_inputs = available_inputs.copy()
        neg_latest_time, parent_task = heapq.heappop(speculative_inputs)
        new_task = Task( # What would this task actually look like?
            uid=reduction_uid,
            batch_id=batch_id,
            tok_ids=[tok_ids[0]],
            name=f'reduction_{reduction_uid}',
            task_type="reduction",
            dependencies=[],
            processor=processors[p_idx],
            finish=-neg_latest_time,
            args={"write_scratch": True}
        )
        new_task.next_input_time = new_task.finish - (REDUCTION_WRITEOUT_TIME + REDUCTION_PRODUCER_LATENCY + SYNCHRONIZATION_COST)
        heapq.heappush(speculative_inputs, (-new_task.next_input_time, new_task)) # None because it's speculative, anyways.
        heapq.heappush(speculative_inputs, (-((-neg_latest_time) - REDUCTION_COST_PER_STEP), parent_task)) # None because it's speculative, anyways.
        spec_cost = __get_quality__([-x[0] for x in speculative_inputs], num_processors=NUM_PROCESSORS, num_tokens=len(tok_ids), seq_length=seq_length)
        if spec_cost > current_cost:
            available_inputs = speculative_inputs # Commit to this new solution.
            parent_task.dependencies = [new_task.uid] + parent_task.dependencies # Add to the start of the dependencies of the parent task.
            parent_task.next_input_time = parent_task.next_input_time - REDUCTION_COST_PER_STEP
            current_cost = spec_cost
            # Actually add the task. But in the meantime, we'll just pretend it's been done.
            tasks[processors[p_idx]].append(new_task)
            reduction_uid, p_idx = reduction_uid+1, (p_idx+1) % NUM_PROCESSORS # advance indices, now that we are committing.
        else:
            break

    # Next, we need to clone this schedule for all of the other tokens.
    base_reduction_tasks = sorted([t for tasks in tasks.values() for t in tasks], key=lambda x: x.next_input_time, reverse=True)
    task_groups = {t.uid: [t] for t in base_reduction_tasks}
    for i in range(1, len(tok_ids)):
        for task in sorted(base_reduction_tasks, key=lambda x: x.uid):
            new_task = Task(
                uid=reduction_uid,
                batch_id=batch_id,
                tok_ids=[tok_ids[i]],
                name=f'reduction_{reduction_uid}',
                task_type="reduction",
                finish=task.finish,
                next_input_time=task.next_input_time,
                dependencies=[d + i*len(base_reduction_tasks) for d in task.dependencies],
                processor=processors[p_idx],
                args=task.args
            )
            tasks[processors[p_idx]].append(new_task)
            reduction_uid, p_idx = reduction_uid+1, (p_idx+1) % NUM_PROCESSORS
            task_groups[task.uid].append(new_task)


    # Next we're going to do the round robin, like in get schedule, to assign tasks to processors.
    partial_info = {p: {} for p in processors} # will contain num steps, and the uid of the partial task.
    full_passes, remainder = divmod(NUM_PROCESSORS, len(base_reduction_tasks)) # How many total we'll run through?
    for i in range(full_passes):
        for j in range(len(base_reduction_tasks)):
            for task in task_groups[base_reduction_tasks[j].uid]:
                task.next_input_time -= REDUCTION_COST_PER_STEP
                task.dependencies = [partial_uid] + task.dependencies
            partial_info[processors[p_idx]] = [base_reduction_tasks[j].next_input_time, partial_uid, 0]
            partial_uid, p_idx = partial_uid+1, (p_idx+1) % NUM_PROCESSORS
    for j in range(remainder):
        for task in task_groups[base_reduction_tasks[j].uid]:
            task.next_input_time -= REDUCTION_COST_PER_STEP
            task.dependencies = [partial_uid] + task.dependencies
        partial_info[processors[p_idx]] = [base_reduction_tasks[j].next_input_time, partial_uid, 0]
        partial_uid, p_idx = partial_uid+1, (p_idx+1) % NUM_PROCESSORS
    # at this point, we've assigned all uid's we'll need.

    # Now go backwards through these partials, and assign their end time to be the start time of the reduction
    for k in range(len(tok_ids)):
        for j in reversed(range(len(base_reduction_tasks))):
            p_idx = (p_idx+NUM_PROCESSORS-1) % NUM_PROCESSORS
            actual_start_time = base_reduction_tasks[j].next_input_time + REDUCTION_PRODUCER_LATENCY - REDUCTION_STARTUP_TIME # When does this instruction actually start?
            partial_info[processors[p_idx]] = [actual_start_time, partial_info[processors[p_idx]][1], 0]

    # Finally we can go through and assign work to each partial op.
    num_partial_steps = (seq_length + 31) // 32
    # So long as the gap between the earliest and latest partial time is greater than the cost of a step, we should allocate work to the latest one (moving backwards).
    min_val = min([x[0] for x in partial_info.values()]) # Most negative time.
    for k, v in partial_info.items():
        v[2] = min(math.floor((v[0] - min_val) / PARTIAL_COST_PER_STEP), num_partial_steps)
        v[0] -= (v[2] * PARTIAL_COST_PER_STEP)
        num_partial_steps -= v[2]
    
    if num_partial_steps > 0:
        full_passes, remainder = divmod(num_partial_steps, NUM_PROCESSORS)
        for i, p in enumerate(sorted(processors, key=lambda x: partial_info[x][0], reverse=True)):
            num_passes = full_passes + (1 if i<remainder else 0)
            partial_info[p][0] -= (num_passes * PARTIAL_COST_PER_STEP)
            partial_info[p][2] += num_passes
            
    for p in processors:
        for task in tasks[p]:
            task.start = task.next_input_time + REDUCTION_PRODUCER_LATENCY - REDUCTION_STARTUP_TIME
            
    # Alright, now that the work has been allocated, we can go through and actually create the tasks.
    current_pos = 0
    for p, v in partial_info.items():
        tasks[p].append(Task(
            uid=v[1],
            batch_id=batch_id,
            tok_ids=tok_ids, # all of them
            name=f'Partial_{v[1]}',
            task_type="partial",
            dependencies=[],
            start=v[0]-PARTIAL_OVERHEAD,
            finish=v[0]+v[2]*PARTIAL_COST_PER_STEP,
            processor=p,
            args={"start": current_pos, "end": min(current_pos+v[2]*32, seq_length), "length": seq_length, "write_scratch": True}
        ))
        current_pos += v[2]*32

    for p, p_tasks in tasks.items():
        earliest_start = p_tasks[-1].start
        num_reductions = 0
        for i, t in enumerate(p_tasks):
            t.start -= earliest_start
            t.finish -= earliest_start
            if t.task_type == "reduction":
                if num_reductions > 0:
                    t.start += p_tasks[i-1].finish - p_tasks[i-1].start
                    t.finish += p_tasks[i-1].finish - p_tasks[i-1].start
                num_reductions += 1

    schedule = [t for tasks in tasks.values() for t in tasks]
    return schedule, partial_uid, reduction_uid


def create_thundermla_arguments(seq_lengths, new_tokens, q_heads = 16):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    t0 = time.time()
    processor_assignments = [max(math.floor(s / sum(seq_lengths) * NUM_PROCESSORS), 1) for s in seq_lengths]
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1
    new_tokens_for_estimate = new_tokens if q_heads == 16 else new_tokens // 2
    processor_assignments = sorted([(estimate_schedule_length(p, new_tokens_for_estimate, s), p, s, i) for i, (p, s) in enumerate(zip(processor_assignments, seq_lengths))])
    while len(seq_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        if best[1]-1 == 0: break
        new_t0, new_tn1 = estimate_schedule_length(best[1]-1, new_tokens_for_estimate, best[2]), estimate_schedule_length(worst[1]+1, new_tokens_for_estimate, worst[2])
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0] = (new_t0, best[1]-1, best[2], best[-1])
            processor_assignments[-1] = (new_tn1, worst[1]+1, worst[2], worst[-1])
            processor_assignments = sorted(processor_assignments)
        else:
            break
    num_processors = [None for _ in seq_lengths]
    for _, p, s, i in processor_assignments:
        num_processors[i] = max(min(p, s//128), 1)
    # Create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(new_tokens)), partial_uid, reduction_uid, q_heads
        )
        scheduled_tasks.extend(new_tasks)
    t1 = time.time()
    print(f'Time taken to create schedule: {(t1-t0)*1000} ms')
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, enable_timings=False, q_heads=q_heads
    )
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings


def get_length(seq_length, num_processors, num_tokens):
    schedule, partial_uid, reduction_uid = backward_schedule(list(range(num_processors)), 0, seq_length, list(range(num_tokens)), 0, num_processors)
    return max([t.finish for t in schedule])


def generate_random_workloads(num_workloads: int):
    for _ in range(num_workloads):
        schedule_length = 9999
        while schedule_length > 100 and random.random() > np.exp(-(schedule_length-100)/100):
            num_processors = random.randint(1, 132)
            num_tokens = random.randint(1, 4) 
            seq_length = random.randint(1, 65536)
            schedule_length = get_length(seq_length, num_processors, num_tokens)
        yield num_processors, num_tokens, seq_length, schedule_length


def make_features(num_processors, num_tokens, seq_length):
    steps = (seq_length+31)//32
    return [
        num_processors,
        num_tokens,
        steps,

        steps/num_processors,  # Work per processor
        steps//num_processors,  # Floor work per processor
        (steps+num_processors-1)//num_processors,  # Max work per processor
        steps % num_processors, # Remainder wave leftover
        num_tokens/num_processors,  # Token overhead per processor

        np.log2(steps),
        np.log2(num_processors),    # Depth of reduction tree
        num_tokens*np.log2(num_processors),    # Depth of reduction tree
        np.ceil(np.log2(num_processors)),    # Depth of reduction tree
        num_tokens*np.ceil(np.log2(num_processors)),    # Depth of reduction tree
        
        steps * np.log2(num_processors),  # Communication overhead scaling
        (steps / num_processors) * np.log2(num_processors),  # Communication overhead scaling
        
        num_processors * steps # Overall bigness
    ]


def train_model():
    NUM_WORKLOADS = 20000
    xtrain, ytrain = [], []
    for num_processors, num_tokens, seq_length, schedule_length in tqdm(generate_random_workloads(NUM_WORKLOADS), total=NUM_WORKLOADS):
        xtrain.append(make_features(num_processors, num_tokens, seq_length))
        ytrain.append(schedule_length)

    xtest, ytest = [], []
    for num_processors, num_tokens, seq_length, schedule_length in tqdm(list(generate_random_workloads(1000))):
        xtest.append(make_features(num_processors, num_tokens, seq_length))
        ytest.append(schedule_length)

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    return model.coef_, model.intercept_


def estimate_schedule_length(num_processors, num_tokens, seq_length):
    features = np.array(make_features(num_processors, num_tokens, seq_length))
    weights = np.array([ 8.46746941e-02,  1.68300124e+00, -5.12640395e-01,  4.99387510e-01,
                        -6.28183426e-01,  6.33647090e-01, -8.18835727e-04, -2.80419818e+00,
                        -7.89583783e-01, -1.46136190e+00, -6.85688746e-02, -7.99856622e-03,
                        -1.51483520e-01,  7.94864055e-02,  1.08745108e+00, -7.59371060e-04 ])
    return np.dot(features, weights) + 23.88058885760671

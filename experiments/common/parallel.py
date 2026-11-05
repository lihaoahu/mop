"""Replicate-level parallelism with one seed per replicate.

Replicate ``r`` always draws seed ``base + stride * r``, so an artifact does not
depend on the number of workers or on the order in which they finish.
"""

from __future__ import annotations

import multiprocessing as mp
import os

_POOL = None
_POOL_JOBS = None


def default_jobs(n_tasks: int) -> int:
    """Leave one core for the operating system, never exceed the task count."""
    cpu = os.cpu_count() or 1
    return max(1, min(n_tasks, cpu - 1))


def seeds(base: int, reps: int, stride: int = 1_000_003) -> list[int]:
    """One independent seed per replicate."""
    return [base + stride * r for r in range(reps)]


def _pool(jobs: int):
    global _POOL, _POOL_JOBS
    if _POOL is None or _POOL_JOBS != jobs:
        if _POOL is not None:
            _POOL.terminate()
            _POOL = None
        _POOL = mp.Pool(jobs)
        _POOL_JOBS = jobs
    return _POOL


def shutdown():
    """Release the workers; the pool otherwise lives until the process exits."""
    global _POOL, _POOL_JOBS
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
        _POOL_JOBS = None


def run(func, tasks, jobs=None):
    """Map ``func`` over ``tasks``; ``jobs=1`` is a plain serial loop."""
    tasks = list(tasks)
    if not tasks:
        return []
    if jobs is None:
        jobs = default_jobs(len(tasks))
    jobs = int(jobs)
    # a worker must never spawn its own pool
    if jobs <= 1 or mp.current_process().daemon:
        return [func(t) for t in tasks]
    return _pool(min(jobs, len(tasks))).map(func, tasks, chunksize=max(1, len(tasks) // (8 * jobs)))

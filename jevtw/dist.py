"""Data parallelism over the GPUs of one node (T0009), started with torchrun.

  torchrun --standalone --nproc-per-node=2 -m jevtw.train --config ... --out ...

Every process holds a full copy of the model (and of the reference model) and runs the same Python program with
the same random seeds, so everything decided from `random.Random(seed)` (which games a batch plays, their sampling
seeds) is identical on all processes without communication. What is split:

  * sampling: the games of a batch (whole groups) or the episodes of an evaluation are dealt out to the processes;
    the finished episodes are gathered so that every process holds the complete batch, in the original order.
    Returns, advantages and the Platt start are then computed from the whole batch on every process (identically);
  * the update: the decisions are dealt out by token count (`balance`); every process accumulates the gradient of
    its own decisions with the global weights, the gradients of all parameters (Platt a, b included) are summed
    across processes (`all_reduce_grads`), then every process clips and steps the same way, so the parameters stay
    equal.

With a single process (no torchrun, or WORLD_SIZE = 1) nothing is initialised and every helper is the identity: the
code paths and numbers are those of the single-GPU code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import torch
import torch.distributed as torch_dist

T = TypeVar("T")


@dataclass
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def main(self) -> bool:
        return self.rank == 0

    # ------------------------------------------------------------ collectives (identity with one process)

    def gather(self, items: list) -> list[list]:
        """The `items` of every process, as a list indexed by rank."""
        if not self.enabled:
            return [items]
        out = [None] * self.world
        torch_dist.all_gather_object(out, items)
        return out

    def barrier(self) -> None:
        if self.enabled:
            torch_dist.barrier()

    def sum_floats(self, values: Sequence[float]) -> list[float]:
        if not self.enabled:
            return list(values)
        t = torch.tensor(list(values), dtype=torch.float64, device=self._device())
        torch_dist.all_reduce(t, op=torch_dist.ReduceOp.SUM)
        return t.tolist()

    def max_floats(self, values: Sequence[float]) -> list[float]:
        if not self.enabled:
            return list(values)
        t = torch.tensor(list(values), dtype=torch.float64, device=self._device())
        torch_dist.all_reduce(t, op=torch_dist.ReduceOp.MAX)
        return t.tolist()

    def all_reduce_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
        """Sum the gradients of `params` over the processes. A parameter that has no gradient on any process keeps
        grad = None (the optimizer then skips it, as in a single process); one that has a gradient somewhere gets
        the sum, with zeros standing in for the processes where it had none."""
        if not self.enabled or not params:
            return
        device = params[0].device
        has = torch.tensor([p.grad is not None for p in params], dtype=torch.float32, device=device)
        torch_dist.all_reduce(has, op=torch_dist.ReduceOp.SUM)
        for p, h in zip(params, has.tolist()):
            if h == 0:
                p.grad = None
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            torch_dist.all_reduce(p.grad, op=torch_dist.ReduceOp.SUM)

    def _device(self) -> torch.device:
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    # ------------------------------------------------------------ splitting work

    def run_split(self, jobs: Sequence[T], run: Callable[[list[T]], list]) -> list:
        """Results of `run` on all `jobs`, in the order of `jobs`, with job k run by process k % world."""
        if not self.enabled:
            return run(list(jobs))
        mine = list(range(self.rank, len(jobs), self.world))
        parts = self.gather(run([jobs[k] for k in mine]))
        out = [None] * len(jobs)
        for r, part in enumerate(parts):
            for k, result in zip(range(r, len(jobs), self.world), part):
                out[k] = result
        return out


def balance(costs: Sequence[float], world: int) -> list[int]:
    """Owner process of every item: largest cost first to the least loaded process (ties: lowest rank, then the
    earlier item). Deterministic, so every process computes the same assignment."""
    if world <= 1:
        return [0] * len(costs)
    owner, load = [0] * len(costs), [0.0] * world
    for k in sorted(range(len(costs)), key=lambda k: (-costs[k], k)):
        r = min(range(world), key=lambda r: (load[r], r))
        owner[k] = r
        load[r] += costs[k]
    return owner


_CURRENT = Dist()


def current() -> Dist:
    return _CURRENT


def setup() -> Dist:
    """Initialise the process group when started by torchrun with WORLD_SIZE > 1: NCCL with one GPU per process
    (cuda:LOCAL_RANK becomes the current device), gloo on the CPU. Returns the process's `Dist`."""
    global _CURRENT
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        _CURRENT = Dist()
        return _CURRENT
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        torch_dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    else:
        torch_dist.init_process_group("gloo")
    _CURRENT = Dist(rank=rank, world=world, local_rank=local)
    return _CURRENT


def teardown() -> None:
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.destroy_process_group()

"""E0015 g5/g6-C2toC4: the process group used NCCL's default 10-minute timeout, shorter than the time one process
can wait for the others in a verbal-confidence C4 rollout, so the watchdog aborted the job. setup() must pass a
timeout of at least an hour to init_process_group."""
from datetime import timedelta

import torch

from jevtw import dist


def test_setup_passes_long_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(dist.torch_dist, "init_process_group", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    try:
        d = dist.setup()
    finally:
        dist._CURRENT = dist.Dist()
    assert (d.rank, d.world, d.local_rank) == (1, 2, 1)
    assert len(calls) == 1
    timeout = calls[0][1].get("timeout")
    assert timeout is not None and timeout >= timedelta(hours=1)

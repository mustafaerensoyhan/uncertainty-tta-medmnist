"""
Inference-time measurement (addendum Addition 1).

Measures wall-clock milliseconds to produce ONE final prediction for one test
image, including all N augmented views + the fusion step, averaged over the test
set.

Two things the addendum's sample code leaves out that matter for correctness on
GPU, and which we include here:
  - torch.cuda.synchronize() around the timed region. CUDA kernels are launched
    asynchronously, so time.perf_counter() alone times how fast Python *queued*
    the work, not how long the GPU took — you'd get nonsensically small numbers.
  - We time the real forward path (the same per-view forward used to produce the
    metrics), not a re-implementation, so the number reflects what's reported.

Rules followed (addendum): same machine for every measurement, warm up first,
no_grad, time.perf_counter, single timed run (deterministic).
"""

from __future__ import annotations

import time
from typing import Callable

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_ms_per_image(run_once: Callable[[], None], n_images: int,
                         device: torch.device, warmup: int = 2) -> float:
    """
    Time `run_once` (which performs a full pass over the test set) and return
    milliseconds per image.

    Args:
        run_once: zero-arg callable that runs the inference path over the whole
                  test set exactly once (e.g. compute per-view logits at N).
        n_images: number of test images the pass covers (for the per-image avg).
        device: torch device (drives cuda.synchronize()).
        warmup: number of warm-up passes to discard (CUDA kernel compilation,
                clock spin-up). Skipped on CPU where it only wastes time.

    Returns:
        ms per image (float).
    """
    if device.type == "cuda":
        for _ in range(max(0, warmup)):
            run_once()
        _sync(device)

    start = time.perf_counter()
    with torch.no_grad():
        run_once()
    _sync(device)
    elapsed = time.perf_counter() - start

    return (elapsed / max(1, n_images)) * 1000.0

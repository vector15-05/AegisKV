"""Deterministic multi-tenant KV-cache workloads."""

from __future__ import annotations

import numpy as np

from .memory_models import MemoryReference


def generate_workload(
    num_references: int = 2_500,
    num_processes: int = 8,
    seed: int = 42,
) -> list[MemoryReference]:
    """Build a reproducible round-robin trace with sink, local, and burst phases."""
    if num_references < 1:
        raise ValueError("Trace length must be at least one reference")
    if num_processes < 1:
        raise ValueError("Number of processes must be positive")

    rng = np.random.default_rng(seed)
    references: list[MemoryReference] = []
    context_span = max(48, num_processes * 12)
    burst_period = max(240, num_processes * 24)
    burst_duration = max(72, num_processes * 10)

    for tick in range(num_references):
        process_id = tick % num_processes
        process_step = tick // num_processes
        phase = tick % burst_period
        in_burst = phase >= burst_period - burst_duration

        if in_burst and rng.random() < 0.84:
            burst_slot = (process_step * 7 + tick // burst_period * 13) % context_span
            block_index = 2 + burst_slot
            attention = float(rng.uniform(0.005, 0.035))
        else:
            choice = rng.random()
            if choice < 0.22:
                block_index = 0
                attention = float(rng.uniform(0.35, 0.55))
            elif choice < 0.67:
                block_index = 1 + (process_step % 2)
                attention = float(rng.uniform(0.20, 0.35))
            else:
                # A power-law draw favors a small semantic working set while
                # preserving a long tail of low-importance context pages.
                block_index = 3 + min(
                    context_span - 1,
                    int(rng.zipf(1.7)) - 1,
                )
                attention = float(rng.uniform(0.025, 0.16))

        page_id = process_id * (context_span + 3) + block_index
        references.append(
            MemoryReference(
                page_id=page_id,
                process_id=process_id,
                block_index=block_index,
                attention=attention,
                is_write=bool(rng.random() < 0.12),
            )
        )

    return references

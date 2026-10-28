"""Page replacement algorithms used by the simulator."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .memory_models import KVPage


class PageReplacementPolicy(ABC):
    name = "BASE"

    @abstractmethod
    def select_victim(self, frames: list[KVPage], current_tick: int) -> int:
        """Return the index of the resident page to evict."""

    def utility(
        self,
        page: KVPage,
        current_tick: int,
        resident_pages: list[KVPage] | None = None,
    ) -> float:
        return 0.0


class FIFOPolicy(PageReplacementPolicy):
    name = "FIFO"

    def select_victim(self, frames: list[KVPage], current_tick: int) -> int:
        return min(range(len(frames)), key=lambda index: frames[index].load_timestamp)


class ClockPolicy(PageReplacementPolicy):
    name = "CLOCK"

    def __init__(self) -> None:
        self.pointer = 0

    def select_victim(self, frames: list[KVPage], current_tick: int) -> int:
        if not frames:
            raise ValueError("Cannot select a victim from an empty frame list")
        self.pointer %= len(frames)
        while frames[self.pointer].reference_bit:
            frames[self.pointer].reference_bit = 0
            self.pointer = (self.pointer + 1) % len(frames)
        victim = self.pointer
        self.pointer = (self.pointer + 1) % len(frames)
        return victim


class LRUPolicy(PageReplacementPolicy):
    name = "LRU"

    def select_victim(self, frames: list[KVPage], current_tick: int) -> int:
        return min(range(len(frames)), key=lambda index: frames[index].last_access_tick)


class LFUPolicy(PageReplacementPolicy):
    name = "LFU"

    def select_victim(self, frames: list[KVPage], current_tick: int) -> int:
        return min(
            range(len(frames)),
            key=lambda index: (frames[index].access_count, frames[index].last_access_tick),
        )


class AWPRPolicy(PageReplacementPolicy):
    name = "AWPR"

    def __init__(
        self,
        recency_weight: float = 0.25,
        frequency_weight: float = 0.15,
        semantic_weight: float = 0.60,
        sink_bonus: float = 0.50,
    ) -> None:
        weights = (recency_weight, frequency_weight, semantic_weight)
        if any(weight < 0 for weight in weights) or not any(weights):
            raise ValueError("AWPR weights must be non-negative and not all zero")
        if sink_bonus < 0:
            raise ValueError("Attention sink bonus must be non-negative")
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.semantic_weight = semantic_weight
        self.sink_bonus = sink_bonus

    def utility(
        self,
        page: KVPage,
        current_tick: int,
        resident_pages: list[KVPage] | None = None,
    ) -> float:
        recency = 1 / (1 + max(0, current_tick - page.last_access_tick))
        max_log_frequency = max(
            (
                math.log1p(candidate.access_count)
                for candidate in (resident_pages or [page])
            ),
            default=0.0,
        )
        frequency = (
            math.log1p(page.access_count) / max_log_frequency if max_log_frequency else 0.0
        )
        semantic = page.ema_attention + (
            self.sink_bonus if page.is_attention_sink else 0.0
        )
        score = (
            self.recency_weight * recency
            + self.frequency_weight * frequency
            + self.semantic_weight * semantic
        )
        return score * (1.15 if page.dirty_bit else 1.0)

    def select_victim(self, frames: list[KVPage], current_tick: int) -> int:
        if not frames:
            raise ValueError("Cannot select a victim from an empty frame list")
        return min(
            range(len(frames)),
            key=lambda index: self.utility(frames[index], current_tick, frames),
        )


def create_policy(name: str, **awpr_weights: float) -> PageReplacementPolicy:
    normalized = name.strip().upper()
    policies: dict[str, type[PageReplacementPolicy]] = {
        "FIFO": FIFOPolicy,
        "CLOCK": ClockPolicy,
        "LRU": LRUPolicy,
        "LFU": LFUPolicy,
        "AWPR": AWPRPolicy,
    }
    try:
        policy_type = policies[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown replacement policy: {name}") from error
    if normalized == "AWPR":
        return AWPRPolicy(**awpr_weights)
    return policy_type()


POLICY_NAMES = ("FIFO", "CLOCK", "LRU", "LFU", "AWPR")

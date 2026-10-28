"""Memory and address-translation data structures for AegisKV."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


TLB_HIT_NS = 2
PAGE_TABLE_WALK_NS = 15
VRAM_ACCESS_NS = 80
PCIe_SWAP_IN_NS = 4_500
PCIe_SWAP_OUT_NS = 3_800


@dataclass
class KVPage:
    page_id: int
    process_id: int
    block_index: int
    is_attention_sink: bool = False
    cumulative_attention: float = 0.0
    ema_attention: float = 0.0
    load_timestamp: int = 0
    last_access_tick: int = 0
    access_count: int = 0
    reference_bit: int = 0
    dirty_bit: bool = False

    def record_access(self, tick: int, attention: float, is_write: bool) -> None:
        self.cumulative_attention += attention
        self.ema_attention = 0.85 * self.ema_attention + 0.15 * attention
        self.last_access_tick = tick
        self.access_count += 1
        self.reference_bit = 1
        self.dirty_bit = self.dirty_bit or is_write


@dataclass(frozen=True)
class MemoryReference:
    page_id: int
    process_id: int
    block_index: int
    attention: float
    is_write: bool = False

    @property
    def is_attention_sink(self) -> bool:
        return self.block_index == 0


@dataclass
class PageTableEntry:
    frame_index: int | None = None
    valid: bool = False
    swapped: bool = True


class TLB:
    """Small fully-associative translation cache using LRU replacement."""

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("TLB capacity must be positive")
        self.capacity = capacity
        self._entries: OrderedDict[int, int] = OrderedDict()

    def lookup(self, page_id: int) -> int | None:
        frame_index = self._entries.get(page_id)
        if frame_index is not None:
            self._entries.move_to_end(page_id)
        return frame_index

    def insert(self, page_id: int, frame_index: int) -> None:
        self._entries[page_id] = frame_index
        self._entries.move_to_end(page_id)
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def invalidate(self, page_id: int) -> None:
        self._entries.pop(page_id, None)


class PhysicalMemory:
    def __init__(self, num_frames: int) -> None:
        if num_frames < 1:
            raise ValueError("Physical memory must contain at least one frame")
        self.frames: list[KVPage | None] = [None] * num_frames

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def resident_count(self) -> int:
        return sum(page is not None for page in self.frames)


class SwapSpace:
    """Stores evicted page metadata so dirty and attention state survives paging."""

    def __init__(self) -> None:
        self.pages: dict[int, KVPage] = {}

    def store(self, page: KVPage) -> None:
        self.pages[page.page_id] = page

    def retrieve(self, page_id: int) -> KVPage | None:
        return self.pages.pop(page_id, None)

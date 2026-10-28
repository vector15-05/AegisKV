"""Discrete-event virtual-memory simulator and telemetry collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .memory_models import (
    PAGE_TABLE_WALK_NS,
    PCIe_SWAP_IN_NS,
    PCIe_SWAP_OUT_NS,
    TLB,
    TLB_HIT_NS,
    VRAM_ACCESS_NS,
    KVPage,
    MemoryReference,
    PageTableEntry,
    PhysicalMemory,
    SwapSpace,
)
from .policies import AWPRPolicy, PageReplacementPolicy, create_policy


@dataclass(frozen=True)
class SimulationConfig:
    num_frames: int = 64
    num_processes: int = 8
    num_references: int = 2_500
    seed: int = 42
    tlb_capacity: int = 32
    awpr_recency_weight: float = 0.25
    awpr_frequency_weight: float = 0.15
    awpr_semantic_weight: float = 0.60
    awpr_sink_bonus: float = 0.50

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ValueError("Physical frame count must be positive")
        if self.num_processes < 1:
            raise ValueError("Process count must be positive")
        if self.num_references < 1:
            raise ValueError("Trace length must be positive")
        if self.tlb_capacity < 1:
            raise ValueError("TLB capacity must be positive")
        if min(
            self.awpr_recency_weight,
            self.awpr_frequency_weight,
            self.awpr_semantic_weight,
            self.awpr_sink_bonus,
        ) < 0:
            raise ValueError("AWPR weights and sink bonus must be non-negative")
        if not any(
            (
                self.awpr_recency_weight,
                self.awpr_frequency_weight,
                self.awpr_semantic_weight,
            )
        ):
            raise ValueError("At least one AWPR weight must be positive")

    def policy_weights(self) -> dict[str, float]:
        return {
            "recency_weight": self.awpr_recency_weight,
            "frequency_weight": self.awpr_frequency_weight,
            "semantic_weight": self.awpr_semantic_weight,
            "sink_bonus": self.awpr_sink_bonus,
        }


@dataclass
class SimulationResult:
    policy: str
    num_frames: int
    num_processes: int
    num_references: int
    page_faults: int
    page_fault_rate_pct: float
    dirty_evictions: int
    tlb_hits: int
    tlb_hit_ratio_pct: float
    total_latency_ns: int
    emat_ns: float
    sink_evictions: int
    token_throughput_per_sec: float
    cumulative_faults: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Simulator:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        policy: str | PageReplacementPolicy = "AWPR",
    ) -> None:
        self.config = config or SimulationConfig()
        if isinstance(policy, str):
            self.policy = create_policy(policy, **self.config.policy_weights())
        else:
            self.policy = policy
        self.memory = PhysicalMemory(self.config.num_frames)
        self.tlb = TLB(self.config.tlb_capacity)
        self.swap = SwapSpace()
        self.page_table: dict[int, PageTableEntry] = {}
        self.tick = 0
        self.page_faults = 0
        self.dirty_evictions = 0
        self.tlb_hits = 0
        self.total_latency_ns = 0
        self.sink_evictions = 0
        self.cumulative_faults: list[int] = []
        self.last_event: dict[str, object] = {}

    def step(self, reference: MemoryReference) -> dict[str, object]:
        self.tick += 1
        translation = self.tlb.lookup(reference.page_id)
        tlb_hit = translation is not None
        if tlb_hit:
            self.tlb_hits += 1
            frame_index = translation
            latency = TLB_HIT_NS + VRAM_ACCESS_NS
        else:
            latency = PAGE_TABLE_WALK_NS + VRAM_ACCESS_NS
            entry = self.page_table.get(reference.page_id)
            frame_index = entry.frame_index if entry and entry.valid else None

        page_fault = frame_index is None
        evicted_page_id: int | None = None
        if page_fault:
            self.page_faults += 1
            latency += PCIe_SWAP_IN_NS
            frame_index, evicted_page_id, writeback_latency = self._allocate(reference)
            latency += writeback_latency
        else:
            page = self.memory.frames[frame_index]
            if page is None:
                raise RuntimeError("Page table points to an unoccupied physical frame")
            page.record_access(self.tick, reference.attention, reference.is_write)

        page = self.memory.frames[frame_index]
        if page is None:
            raise RuntimeError("Allocated frame is unexpectedly empty")
        if not tlb_hit:
            self.tlb.insert(reference.page_id, frame_index)

        self.total_latency_ns += latency
        self.cumulative_faults.append(self.page_faults)
        self.last_event = {
            "tick": self.tick,
            "page_id": reference.page_id,
            "process_id": reference.process_id,
            "block_index": reference.block_index,
            "page_fault": page_fault,
            "evicted_page_id": evicted_page_id,
            "tlb_hit": tlb_hit,
            "latency_ns": latency,
            "is_attention_sink": page.is_attention_sink,
        }
        return self.last_event

    def _allocate(self, reference: MemoryReference) -> tuple[int, int | None, int]:
        free_index = next(
            (index for index, page in enumerate(self.memory.frames) if page is None),
            None,
        )
        evicted_page_id: int | None = None
        writeback_latency = 0
        if free_index is None:
            resident_pages = [page for page in self.memory.frames if page is not None]
            victim_index = self.policy.select_victim(resident_pages, self.tick)
            victim = resident_pages[victim_index]
            free_index = next(
                index
                for index, page in enumerate(self.memory.frames)
                if page is victim
            )
            evicted_page_id = victim.page_id
            self.page_table[victim.page_id] = PageTableEntry(
                frame_index=None,
                valid=False,
                swapped=True,
            )
            self.tlb.invalidate(victim.page_id)
            if victim.is_attention_sink:
                self.sink_evictions += 1
            if victim.dirty_bit:
                self.dirty_evictions += 1
                writeback_latency = PCIe_SWAP_OUT_NS
                victim.dirty_bit = False
            self.swap.store(victim)

        page = self.swap.retrieve(reference.page_id)
        if page is None:
            page = KVPage(
                page_id=reference.page_id,
                process_id=reference.process_id,
                block_index=reference.block_index,
                is_attention_sink=reference.is_attention_sink,
            )
        page.load_timestamp = self.tick
        page.record_access(self.tick, reference.attention, reference.is_write)
        self.memory.frames[free_index] = page
        self.page_table[reference.page_id] = PageTableEntry(
            frame_index=free_index,
            valid=True,
            swapped=False,
        )
        return free_index, evicted_page_id, writeback_latency

    def run(self, references: list[MemoryReference]) -> SimulationResult:
        for reference in references:
            self.step(reference)
        return self.result()

    def result(self) -> SimulationResult:
        count = self.tick
        fault_rate = self.page_faults / count * 100 if count else 0.0
        tlb_ratio = self.tlb_hits / count * 100 if count else 0.0
        emat = self.total_latency_ns / count if count else 0.0
        throughput = (
            count * 1_000_000_000 / self.total_latency_ns
            if self.total_latency_ns
            else 0.0
        )
        return SimulationResult(
            policy=self.policy.name,
            num_frames=self.config.num_frames,
            num_processes=self.config.num_processes,
            num_references=count,
            page_faults=self.page_faults,
            page_fault_rate_pct=fault_rate,
            dirty_evictions=self.dirty_evictions,
            tlb_hits=self.tlb_hits,
            tlb_hit_ratio_pct=tlb_ratio,
            total_latency_ns=self.total_latency_ns,
            emat_ns=emat,
            sink_evictions=self.sink_evictions,
            token_throughput_per_sec=throughput,
            cumulative_faults=list(self.cumulative_faults),
        )

    def frame_snapshot(self) -> list[dict[str, object] | None]:
        resident_pages = [page for page in self.memory.frames if page is not None]
        max_frequency = max(
            (page.access_count for page in resident_pages),
            default=0,
        )
        snapshot: list[dict[str, object] | None] = []
        for page in self.memory.frames:
            if page is None:
                snapshot.append(None)
                continue
            utility = 0.0
            if isinstance(self.policy, AWPRPolicy):
                utility = self.policy.utility(page, self.tick, resident_pages)
            else:
                utility = page.access_count / max_frequency if max_frequency else 0.0
            snapshot.append(
                {
                    "page_id": page.page_id,
                    "process_id": page.process_id,
                    "block_index": page.block_index,
                    "is_attention_sink": page.is_attention_sink,
                    "ema_attention": page.ema_attention,
                    "utility": utility,
                    "dirty": page.dirty_bit,
                }
            )
        return snapshot

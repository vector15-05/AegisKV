import unittest

from core.memory_models import KVPage, MemoryReference
from core.policies import AWPRPolicy, ClockPolicy, create_policy
from core.simulator import SimulationConfig, Simulator
from core.workload_generator import generate_workload


class WorkloadTests(unittest.TestCase):
    def test_seeded_trace_is_reproducible(self) -> None:
        first = generate_workload(100, 4, 17)
        second = generate_workload(100, 4, 17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)


class PolicyTests(unittest.TestCase):
    def test_awpr_protects_attention_sink(self) -> None:
        sink = KVPage(
            page_id=1,
            process_id=0,
            block_index=0,
            is_attention_sink=True,
            ema_attention=0.4,
            last_access_tick=2,
            access_count=8,
        )
        context = KVPage(
            page_id=2,
            process_id=0,
            block_index=12,
            ema_attention=0.01,
            last_access_tick=9,
            access_count=1,
        )
        policy = AWPRPolicy()
        self.assertEqual(policy.select_victim([sink, context], 10), 1)

    def test_clock_gives_referenced_pages_a_second_chance(self) -> None:
        pages = [
            KVPage(page_id=1, process_id=0, block_index=1, reference_bit=1),
            KVPage(page_id=2, process_id=0, block_index=2, reference_bit=0),
        ]
        policy = ClockPolicy()
        self.assertEqual(policy.select_victim(pages, 1), 1)
        self.assertEqual(pages[0].reference_bit, 0)

    def test_policy_factory_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            create_policy("random")


class SimulatorTests(unittest.TestCase):
    def test_tlb_hits_and_fault_metrics(self) -> None:
        simulator = Simulator(
            SimulationConfig(num_frames=2, num_processes=1, num_references=3),
            "LRU",
        )
        references = [
            MemoryReference(1, 0, 0, 0.4),
            MemoryReference(1, 0, 0, 0.4),
            MemoryReference(1, 0, 0, 0.4),
        ]
        result = simulator.run(references)
        self.assertEqual(result.page_faults, 1)
        self.assertEqual(result.tlb_hits, 2)
        self.assertAlmostEqual(result.tlb_hit_ratio_pct, 200 / 3)
        self.assertEqual(result.cumulative_faults, [1, 1, 1])

    def test_dirty_page_is_written_back_and_sink_eviction_counted(self) -> None:
        simulator = Simulator(
            SimulationConfig(num_frames=1, num_processes=1, num_references=3),
            "FIFO",
        )
        result = simulator.run(
            [
                MemoryReference(1, 0, 0, 0.5, is_write=True),
                MemoryReference(2, 0, 1, 0.2),
                MemoryReference(1, 0, 0, 0.5),
            ]
        )
        self.assertEqual(result.page_faults, 3)
        self.assertEqual(result.dirty_evictions, 1)
        self.assertEqual(result.sink_evictions, 1)
        self.assertGreater(result.emat_ns, 80)

    def test_all_policies_replay_same_trace(self) -> None:
        trace = generate_workload(300, 4, 42)
        fault_counts = []
        for policy in ("FIFO", "CLOCK", "LRU", "LFU", "AWPR"):
            result = Simulator(
                SimulationConfig(num_frames=32, num_processes=4, num_references=300),
                policy,
            ).run(trace)
            self.assertEqual(result.num_references, 300)
            self.assertEqual(len(result.cumulative_faults), 300)
            fault_counts.append(result.page_faults)
        self.assertEqual(len(fault_counts), 5)


if __name__ == "__main__":
    unittest.main()

"""Run the AegisKV benchmark suite and export report-ready artifacts."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from core.policies import POLICY_NAMES
from core.simulator import SimulationConfig, SimulationResult, Simulator
from core.workload_generator import generate_workload


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FRAME_COUNTS = (32, 48, 64, 80, 96, 128)
COLORS = {
    "FIFO": "#f97316",
    "CLOCK": "#eab308",
    "LRU": "#38bdf8",
    "LFU": "#a78bfa",
    "AWPR": "#34d399",
}


def run_benchmark(
    config: SimulationConfig | None = None,
) -> tuple[list[SimulationResult], dict[int, list[SimulationResult]]]:
    config = config or SimulationConfig()
    trace = generate_workload(
        config.num_references,
        config.num_processes,
        config.seed,
    )
    primary = [
        Simulator(config, policy_name).run(trace)
        for policy_name in POLICY_NAMES
    ]
    scaling: dict[int, list[SimulationResult]] = {}
    for frame_count in FRAME_COUNTS:
        frame_config = SimulationConfig(
            num_frames=frame_count,
            num_processes=config.num_processes,
            num_references=config.num_references,
            seed=config.seed,
            tlb_capacity=config.tlb_capacity,
            awpr_recency_weight=config.awpr_recency_weight,
            awpr_frequency_weight=config.awpr_frequency_weight,
            awpr_semantic_weight=config.awpr_semantic_weight,
            awpr_sink_bonus=config.awpr_sink_bonus,
        )
        scaling[frame_count] = [
            Simulator(frame_config, policy_name).run(trace)
            for policy_name in POLICY_NAMES
        ]
    return primary, scaling


def _write_csv(results: list[SimulationResult], output_path: Path) -> None:
    columns = (
        "policy",
        "num_frames",
        "num_processes",
        "num_references",
        "page_faults",
        "page_fault_rate_pct",
        "dirty_evictions",
        "tlb_hits",
        "tlb_hit_ratio_pct",
        "total_latency_ns",
        "emat_ns",
        "sink_evictions",
        "token_throughput_per_sec",
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: getattr(result, key)
                    for key in columns
                }
            )


def _write_plots(
    primary: list[SimulationResult],
    scaling: dict[int, list[SimulationResult]],
) -> None:
    policies = [result.policy for result in primary]
    x = np.arange(len(policies))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.bar(
        x - width / 2,
        [result.page_faults for result in primary],
        width,
        label="Page faults",
        color=[COLORS[policy] for policy in policies],
        alpha=0.95,
    )
    axis.bar(
        x + width / 2,
        [result.sink_evictions for result in primary],
        width,
        label="Attention-sink evictions",
        color="#64748b",
        alpha=0.78,
    )
    axis.set_xticks(x, policies)
    axis.set_ylabel("Count")
    axis.set_title("Page Faults and Attention-Sink Evictions")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_page_faults_comparison.png", dpi=300)
    plt.close(fig)

    fig, (emat_axis, throughput_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5.6),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    emat_axis.bar(
        policies,
        [result.emat_ns for result in primary],
        color=[COLORS[policy] for policy in policies],
    )
    emat_axis.set_ylabel("Effective access time (ns/reference)")
    emat_axis.set_title("EMAT")
    emat_axis.grid(axis="y", alpha=0.22)
    throughput_axis.bar(
        policies,
        [result.token_throughput_per_sec / 1_000_000 for result in primary],
        color=[COLORS[policy] for policy in policies],
    )
    throughput_axis.set_ylabel("Simulated references/sec (millions)")
    throughput_axis.set_title("Stall-limited throughput")
    throughput_axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Latency and Simulated Throughput")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_emat_latency.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    for policy in policies:
        axis.plot(
            FRAME_COUNTS,
            [
                next(result for result in scaling[frames] if result.policy == policy)
                .page_fault_rate_pct
                for frames in FRAME_COUNTS
            ],
            marker="o",
            linewidth=2,
            label=policy,
            color=COLORS[policy],
        )
    axis.set_xlabel("Physical frames")
    axis.set_ylabel("Page fault rate (%)")
    axis.set_title("Page Fault Rate by Physical Memory Allocation")
    axis.grid(alpha=0.24)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_frame_scaling_curve.png", dpi=300)
    plt.close(fig)

    lru = next(result for result in primary if result.policy == "LRU")
    awpr = next(result for result in primary if result.policy == "AWPR")
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    ticks = np.arange(1, len(awpr.cumulative_faults) + 1)
    axis.plot(
        ticks,
        lru.cumulative_faults,
        color=COLORS["LRU"],
        label="LRU",
        linewidth=1.7,
    )
    axis.plot(
        ticks,
        awpr.cumulative_faults,
        color=COLORS["AWPR"],
        label="AWPR",
        linewidth=1.9,
    )
    axis.set_xlabel("Simulation tick")
    axis.set_ylabel("Cumulative page faults")
    axis.set_title("Cumulative Faults: AWPR vs LRU")
    axis.grid(alpha=0.24)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_cumulative_faults_timeline.png", dpi=300)
    plt.close(fig)


def _write_summary(
    primary: list[SimulationResult],
    scaling: dict[int, list[SimulationResult]],
    output_path: Path,
) -> None:
    by_name = {result.policy: result for result in primary}
    best_faults = min(primary, key=lambda result: result.page_faults)
    best_emat = min(primary, key=lambda result: result.emat_ns)
    awpr = by_name["AWPR"]
    lru = by_name["LRU"]
    fault_reduction = (
        (lru.page_faults - awpr.page_faults) / lru.page_faults * 100
        if lru.page_faults
        else 0.0
    )
    lines = [
        "# AegisKV Benchmark Results",
        "",
        "Results are generated from one deterministic multi-tenant trace shared by all policies.",
        "",
        "## Comparative results",
        "",
        "| Policy | Page faults | Fault rate | Dirty evictions | Sink evictions | TLB hit rate | EMAT (ns) | Simulated tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in primary:
        lines.append(
            f"| {result.policy} | {result.page_faults} | "
            f"{result.page_fault_rate_pct:.2f}% | {result.dirty_evictions} | "
            f"{result.sink_evictions} | {result.tlb_hit_ratio_pct:.2f}% | "
            f"{result.emat_ns:.2f} | {result.token_throughput_per_sec:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Lowest page-fault count: **{best_faults.policy}** "
            f"({best_faults.page_faults} faults, {best_faults.page_fault_rate_pct:.2f}%).",
            f"- Lowest EMAT: **{best_emat.policy}** ({best_emat.emat_ns:.2f} ns/reference).",
            f"- AWPR recorded {awpr.sink_evictions} attention-sink evictions, compared with "
            f"{lru.sink_evictions} for LRU.",
            f"- AWPR page faults vs. LRU: {fault_reduction:+.2f}% reduction "
            "(negative values indicate AWPR had more faults on this trace).",
            "",
            "## Frame scaling",
            "",
            "| Frames | " + " | ".join(POLICY_NAMES) + " |",
            "|---:|" + "|".join("---:" for _ in POLICY_NAMES) + "|",
        ]
    )
    for frame_count, results in scaling.items():
        metrics = {result.policy: result.page_fault_rate_pct for result in results}
        lines.append(
            f"| {frame_count} | "
            + " | ".join(f"{metrics[policy]:.2f}%" for policy in POLICY_NAMES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            f"The simulator used {awpr.num_references:,} memory references, "
            f"{awpr.num_processes} tenants, {awpr.num_frames} physical frames, and seed 42. "
            "The same generated trace was replayed independently for every policy. "
            "Latency includes TLB lookup or page-table walk, VRAM access, swap-in, and "
            "dirty-page swap-out costs.",
            "",
            "These are discrete-event simulation results, not measurements from a GPU, "
            "operating-system kernel, or production inference server.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_artifacts(config: SimulationConfig | None = None) -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = config or SimulationConfig()
    started = time.perf_counter()
    primary, scaling = run_benchmark(config)
    _write_csv(primary, RESULTS_DIR / "benchmark_comparison.csv")
    _write_plots(primary, scaling)
    _write_summary(primary, scaling, RESULTS_DIR / "CASE_STUDY_RESULTS_SUMMARY.md")
    return {
        "results": [result.to_dict() for result in primary],
        "scaling": {
            str(frame_count): [result.to_dict() for result in results]
            for frame_count, results in scaling.items()
        },
        "files": [
            "benchmark_comparison.csv",
            "fig1_page_faults_comparison.png",
            "fig2_emat_latency.png",
            "fig3_frame_scaling_curve.png",
            "fig4_cumulative_faults_timeline.png",
            "CASE_STUDY_RESULTS_SUMMARY.md",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    report = generate_artifacts()
    print(f"Generated {len(report['files'])} benchmark artifacts in {RESULTS_DIR}")
    print(f"Benchmark completed in {report['elapsed_seconds']:.2f}s")
    for result in report["results"]:
        print(
            f"{result['policy']:>5}: faults={result['page_faults']:>4}, "
            f"rate={result['page_fault_rate_pct']:>6.2f}%, "
            f"EMAT={result['emat_ns']:>8.2f} ns, "
            f"sink evictions={result['sink_evictions']}"
        )


if __name__ == "__main__":
    main()

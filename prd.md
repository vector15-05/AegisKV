# Product Requirements Document (PRD): AegisKV
**Project Title:** Attention-Aware Virtual Memory Management: A Semantic Page Replacement Policy for Multi-Tenant LLM Inference  
**Course Context:** BCSE303L – Operating Systems Case Study (Proof of Concept & Benchmarking Suite)  
**Execution Environment:** 100% User-Space Discrete-Event Simulation + Web Dashboard + Automated Plot/CSV Exporter  

---

## 1. Executive Summary & Objective
Traditional Operating System virtual memory managers rely on recency- or frequency-based page replacement algorithms (FIFO, CLOCK, LRU, LFU) that treat all memory pages as semantically equal. However, in modern multi-tenant Large Language Model (LLM) inference engines (such as vLLM's PagedAttention), Key-Value (KV) cache blocks exhibit highly skewed importance: initial "Attention Sink" tokens and high-attention structural tokens are repeatedly critical for generation quality and latency, even if their access patterns across interleaved multi-user batching cause standard LRU/LFU policies to evict them to slower memory tiers (CPU RAM / NVMe Swap).

**AegisKV** is a complete discrete-event Operating System Virtual Memory & Paging Simulator paired with a real-time interactive Web Dashboard and an automated CLI Benchmark Suite. It simulates a two-tier memory hierarchy (Tier-1 Fast VRAM Physical Frames and Tier-2 Host RAM/NVMe Swap) and compares four baseline OS page replacement algorithms (**FIFO, CLOCK, LRU, LFU**) against our proposed **Attention-Weighted Page Replacement (AWPR)** policy.

---

## 2. Technical Stack & Architecture
To ensure fast execution, zero hardware/kernel dependencies, and publication-ready visual outputs for the case study report, build a unified project using **Python (FastAPI + NumPy + Matplotlib)** for the core OS simulation and **Single-Page HTML/Tailwind/Chart.js (served directly by FastAPI)** for the interactive visual proof-of-concept.

### Directory Structure
```text
aegiskv/
├── README.md
├── requirements.txt
├── main.py                      # FastAPI server + Web UI entry point
├── cli_benchmark.py             # Headless CLI runner that generates CSV tables & PNG plots for the report
├── core/
│   ├── __init__.py
│   ├── memory_models.py         # Page, Frame, PageTableEntry, TLB, and SwapSpace data structures
│   ├── workload_generator.py    # Synthetic & trace-driven multi-tenant LLM KV-cache request generator
│   ├── policies.py              # FIFO, CLOCK, LRU, LFU, and AWPR (proposed) implementations
│   └── simulator.py             # Discrete-event simulation engine & telemetry collector
├── static/
│   └── index.html               # Dark-mode Bento-grid interactive UI (Tailwind + Chart.js + Geist/Mono fonts)
└── results/                     # Auto-generated report artifacts (tables.csv, plots/*.png, summary.md)
```

---

## 3. Core OS & Domain Models (`core/memory_models.py`)

### 3.1 Virtual Page / KV-Block (`KVPage`)
Each page represents a fixed-size block of KV-cache tokens (default: `16 tokens/page`, `page_size = 64 KB`):
- `page_id: int` — Globally unique virtual page identifier.
- `process_id: int` — Tenant/Request ID (`P1` to `Pn`).
- `block_index: int` — Logical token block position within the sequence (`0` = initial prompt/attention sink).
- `is_attention_sink: bool` — True if `block_index == 0` or explicitly flagged as structural system prompt.
- `cumulative_attention: float` — Accumulated normalized attention score ($\sum \alpha_{i,j}$) across decoding steps.
- `ema_attention: float` — Exponential Moving Average of attention weight:
  $$A_{t} = \beta \cdot A_{t-1} + (1 - \beta) \cdot a_{t} \quad (\beta = 0.85)$$
- `load_timestamp: int` — Simulation tick when the page was loaded into physical memory.
- `last_access_tick: int` — Simulation tick of the most recent read/write hit.
- `access_count: int` — Total accesses since allocation.
- `reference_bit: int` — `0` or `1` (used by the CLOCK/Second-Chance policy).
- `dirty_bit: bool` — True if new KV tokens were appended to this page in Tier-1 memory (requires write-back cost on eviction).

### 3.2 Page Table & TLB (`MMU`)
- **Translation Lookaside Buffer (TLB):**
  - Configurable capacity (default: `32 entries`), managed via LRU.
  - Latency constants:
    - `TLB_HIT_NS = 2`
    - `PAGE_TABLE_WALK_NS = 15`
    - `VRAM_ACCESS_NS = 80`
    - `PCIe_SWAP_IN_NS = 4500` (Major Page Fault penalty: loading page from Host Swap to VRAM)
    - `PCIe_SWAP_OUT_NS = 3800` (Write-back penalty if `dirty_bit == True` upon eviction)
- **Physical Frame Table (`PhysicalMemory`):**
  - Fixed array of `num_frames` (configurable: default `64` or `128` frames).
  - Tracks frame occupancy, owning `process_id`, and `page_id`.

---

## 4. Page Replacement Policies (`core/policies.py`)
All policies inherit from an abstract base class `PageReplacementPolicy` with method `select_victim(frames: List[KVPage], current_tick: int) -> int`:

1. **`FIFOPolicy`:** Evicts the page with the minimum `load_timestamp`.
2. **`ClockPolicy` (Second-Chance):** Maintains a circular pointer. Scans frames; if `reference_bit == 1`, clears it to `0` and advances pointer; if `0`, selects frame as victim.
3. **`LRUPolicy`:** Evicts the page with the minimum `last_access_tick`.
4. **`LFUPolicy`:** Evicts the page with the minimum `access_count` (tie-broken by LRU).
5. **`AWPRPolicy` (Proposed — Attention-Weighted Page Replacement):**
   - Computes a composite **Semantic Retention Utility Score** $U(p)$ for every resident page $p$ at tick $t$:
     $$R(p) = \frac{1}{1 + (t - p.\text{last\_access\_tick})}$$
     $$F(p) = \frac{\ln(1 + p.\text{access\_count})}{\max_{q} \ln(1 + q.\text{access\_count})}$$
     $$S(p) = p.\text{ema\_attention} + (\gamma \cdot \mathbb{I}[\text{is\_attention\_sink}])$$
     $$U(p) = w_1 \cdot R(p) + w_2 \cdot F(p) + w_3 \cdot S(p)$$
   - Default hyperparameters: $w_1 = 0.25$ (Recency), $w_2 = 0.15$ (Frequency), $w_3 = 0.60$ (Semantic Attention), and $\gamma = 0.50$ (Attention Sink Protection Bonus).
   - **Dirty-Page Awareness:** If a page is dirty (`dirty_bit == True`), multiply $U(p)$ by `1.15` to slightly penalize premature write-backs.
   - **Victim Selection:** Evicts the resident page with the **minimum** $U(p)$ score.

---

## 5. Workload Generator (`core/workload_generator.py`)
Simulate realistic multi-tenant autoregressive LLM inference workloads without needing a real GPU or multi-gigabyte model weights:
- **Multi-Tenant Round-Robin / Continuous Batching:**
  - Configurable number of concurrent tenant processes (`num_processes = 8`), each with a prompt phase (prefill) and token-by-token generation phase (decode).
- **Attention Distribution Law (Heavy-Tailed + Sink):**
  - During each decoding step for process $P_k$, the process accesses a subset of its logical KV pages:
    - **Block 0 (Attention Sink):** Accessed with 98% probability and assigned high attention weight ($0.35 - 0.55$).
    - **Recent Context Window (Last 2 blocks):** Accessed with 95% probability and assigned medium-high attention weight ($0.20 - 0.35$).
    - **Middle Context Blocks:** Accessed sparsely (power-law distribution) simulating semantic retrieval heads, with periodic "context thrashing" bursts where multiple processes exceed total physical frame capacity (`Working Set > Physical Frames`).
- **Deterministic Seeding:** Use a fixed `seed = 42` option so all 5 algorithms are evaluated on the **exact same instruction/memory trace** for scientifically valid comparisons.

---

## 6. Metrics & Evaluation Engine (`core/simulator.py`)
At the end of a simulation run (default `2,500 memory references` across `64 physical frames`), compute and return:
1. **Total Page Faults** & **Page Fault Rate (%)**
2. **Major Swap Writes (Dirty Evictions)**
3. **TLB Hit Ratio (%)**
4. **Effective Memory Access Time (EMAT in ns):**
   $$\text{EMAT} = \frac{\text{Total Cumulative Memory Latency (ns)}}{\text{Total Memory References}}$$
5. **Attention Sink Eviction Count:** How many times high-importance sink pages were thrashingly evicted (demonstrates why standard LRU fails under round-robin multi-tenant batching).
6. **Simulated Token Throughput (Tokens/sec):** Derived inversely from total memory stall time during decoding steps.

---

## 7. Interactive Web Dashboard (`static/index.html` & `main.py`)
Provide a sleek, dark-mode engineering interface (Dark metallic `#0a0a0c` background, `#121318` cards, subtle `1px solid #262730` borders, emerald/cyan/amber accents):
1. **Control Bar:**
   - Sliders/Inputs for: Physical Frames (`32–256`), Concurrent Processes (`4–16`), Trace Length (`500–5000`), and AWPR Weights ($w_1, w_2, w_3$).
   - Buttons: `"Run Comparative Benchmark (All 5 Policies)"`, `"Step-by-Step Live Visualizer (AWPR vs LRU)"`, and `"Export Report Artifacts (CSV + PNGs)"`.
2. **Physical Memory Frame Heatmap (Live Grid):**
   - A 64-cell grid representing physical RAM frames.
   - Color-coded by tenant `process_id` or by **Semantic Utility Score $U(p)$** (glowing border for Attention Sink frames, flash red on page fault/eviction).
3. **4 Comparative Charts (Chart.js):**
   - **Chart 1 (Bar):** Page Fault Count & Dirty Evictions across `FIFO`, `CLOCK`, `LRU`, `LFU`, `AWPR`.
   - **Chart 2 (Bar):** Effective Memory Access Time (EMAT in nanoseconds) across all 5 policies.
   - **Chart 3 (Line):** Cumulative Page Faults over Simulation Ticks ($t = 0 \dots N$).
   - **Chart 4 (Line - Belady's Anomaly & Scaling Test):** Page Fault Rate (%) vs. Physical Frame Allocation (`32, 48, 64, 80, 96, 128 frames`) for all 5 policies.
4. **Live Telemetry Table:**
   - Displays exact numerical outputs ready to copy-paste into **Section VII (Results and Output)** of the OS Case Study document.

---

## 8. Automated Report Artifact Generator (`cli_benchmark.py`)
Running `python cli_benchmark.py` must execute a comprehensive benchmark suite in under 10 seconds and save high-resolution (300 DPI) academic plots and markdown/CSV tables into `results/`:
1. `results/benchmark_comparison.csv` — Exact tabular metrics for FIFO, CLOCK, LRU, LFU, and AWPR.
2. `results/fig1_page_faults_comparison.png` — Grouped bar chart comparing Total Page Faults and Sink Evictions[cite: 1].
3. `results/fig2_emat_latency.png` — Bar chart comparing Effective Memory Access Time (ns) and Token Throughput[cite: 1].
4. `results/fig3_frame_scaling_curve.png` — Line graph showing Page Fault Rate (%) across varying physical memory sizes (`32` to `128` frames)[cite: 1].
5. `results/fig4_cumulative_faults_timeline.png` — Time-series plot showing how AWPR stabilizes during multi-tenant burst phases where LRU thrashes[cite: 1].
6. `results/CASE_STUDY_RESULTS_SUMMARY.md` — Auto-generated markdown summary interpreting the exact numbers produced by the run for direct inclusion in the report[cite: 1].

---

## 9. Acceptance Criteria & Verification Steps
When building this project, execute the following steps before completing the task:
1. Install dependencies (`fastapi`, `uvicorn`, `numpy`, `matplotlib`, `pydantic`).
2. Run `python cli_benchmark.py` and verify that:
   - All 5 policies execute on the identical seeded workload trace without errors.
   - `AWPR` achieves a measurably lower Page Fault Rate (`~18–35%` reduction) and lower EMAT than `LRU`, `LFU`, `CLOCK`, and `FIFO` under multi-tenant round-robin pressure.
   - All 4 `.png` charts and `.csv`/`.md` files are generated cleanly in `results/`[cite: 1].
3. Start the FastAPI server (`uvicorn main:app --port 8000`) and verify the `/api/simulate` and `/api/live-step` endpoints return valid JSON for the frontend dashboard.

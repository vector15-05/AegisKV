# AegisKV

**Attention-Aware Virtual Memory Management for Multi-Tenant LLM Inference**

AegisKV is a user-space, discrete-event simulator for studying how page
replacement policies behave on multi-tenant LLM KV-cache workloads. Compare
traditional **FIFO**, **CLOCK**, **LRU**, and **LFU** policies with
**Attention-Weighted Page Replacement (AWPR)**, a policy that considers
recency, frequency, semantic attention, attention-sink protection, and dirty
page write-back cost.

The project includes a FastAPI-served interactive dashboard and a benchmark
runner that exports CSV and Markdown tables plus publication-ready, 300-DPI
plots. It is an educational simulation: it does not allocate GPU memory,
interact with kernel page tables, or run an LLM inference server.

## Features

- Reproducible, seeded multi-tenant KV-cache reference traces.
- Two-tier virtual memory model: fast physical frames and host swap.
- FIFO, Second-Chance CLOCK, LRU, LFU, and AWPR page replacement.
- LRU-managed translation lookaside buffer (TLB), page-table walks, dirty
  write-back, swap-in, and simulated memory-access latency.
- Metrics for page faults and rate, dirty evictions, attention-sink evictions,
  TLB hit ratio, effective memory access time (EMAT), and stall-limited
  throughput.
- Browser dashboard with configurable simulations, comparison charts, frame
  heatmaps, step-by-step AWPR vs. LRU visualization, and report export.
- Benchmark reports that compare all policies on the same generated trace.

## Requirements

- Python 3.10 or later
- Dependencies listed in [`requirements.txt`](requirements.txt)

## Installation

Run these commands from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the dashboard

With the virtual environment activated:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Stop the server with **Ctrl+C**.

The dashboard lets you configure physical frames, concurrent tenants, trace
length, and AWPR utility weights. Run a comparative benchmark, inspect the
physical-frame grids and charts, step through the AWPR/LRU live visualizer, or
regenerate the report files with **Export report artifacts**.

## Run the benchmark

```bash
python cli_benchmark.py
```

The default configuration uses seed `42`, 8 tenants, 2,500 references, and
64 physical frames. The benchmark also sweeps `32`, `48`, `64`, `80`, `96`,
and `128` frames. All policies replay the same trace in each comparison.

Generated files are written to `results/`:

| File | Contents |
|---|---|
| `benchmark_comparison.csv` | Comparative policy metrics |
| `fig1_page_faults_comparison.png` | Page faults and attention-sink evictions |
| `fig2_emat_latency.png` | EMAT and simulated throughput |
| `fig3_frame_scaling_curve.png` | Fault rate vs. physical frame allocation |
| `fig4_cumulative_faults_timeline.png` | Cumulative AWPR and LRU page faults |
| `CASE_STUDY_RESULTS_SUMMARY.md` | Generated results and methodology summary |

The plots are saved at **300 DPI**. The generated summary reports the observed
results for the chosen trace; AWPR is not assumed to win on every workload or
metric. The throughput metric is a simulation-derived proxy based on modeled
memory latency, not measured production inference throughput.

## API

The app exposes these endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/simulate` | Run five policies and a frame-allocation sweep |
| `POST` | `/api/live-step` | Advance an AWPR/LRU visualization session |
| `POST` | `/api/export` | Generate benchmark CSV, Markdown, and plots |

Example comparative simulation request:

```bash
curl -X POST http://127.0.0.1:8000/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "frames": 64,
    "processes": 8,
    "trace_length": 2500,
    "seed": 42,
    "awpr_recency_weight": 0.25,
    "awpr_frequency_weight": 0.15,
    "awpr_semantic_weight": 0.6,
    "awpr_sink_bonus": 0.5
  }'
```

The request controls are validated by the API. Supported dashboard ranges are
32–256 frames, 4–16 processes, and 500–5,000 references. A live session starts
when `session_id` is omitted; include the returned session ID in subsequent
`/api/live-step` requests to continue that trace.

## Tests

Run the simulator unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Project layout

```text
.
├── core/
│   ├── memory_models.py       # KV pages, frames, TLB, swap, page-table entries
│   ├── policies.py            # FIFO, CLOCK, LRU, LFU, AWPR
│   ├── simulator.py           # Paging engine and metrics
│   └── workload_generator.py  # Seeded multi-tenant reference traces
├── results/                   # Generated benchmark artifacts
├── static/
│   └── index.html             # Interactive dashboard
├── tests/
│   └── test_simulator.py
├── cli_benchmark.py
├── main.py                    # FastAPI app
└── requirements.txt
```

For the full project requirements and case-study context, see
[`prd.md`](prd.md).

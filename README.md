# AegisKV

AegisKV is a user-space, discrete-event simulator for comparing FIFO, CLOCK,
LRU, LFU, and Attention-Weighted Page Replacement (AWPR) on a reproducible
multi-tenant KV-cache workload. It includes a FastAPI dashboard and a benchmark
command that exports report-ready CSV, Markdown, and 300-DPI PNG figures.

The simulator models physical VRAM frames, host swap, dirty-page write-back,
an LRU TLB, page-table walks, attention-sink retention, and per-reference
latency. It does not access a GPU, kernel page tables, or a live inference
server.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the benchmark

```bash
python cli_benchmark.py
```

The deterministic default trace uses seed `42`, eight tenants, 2,500 references,
and 64 physical frames. The CLI writes:

- `results/benchmark_comparison.csv`
- `results/fig1_page_faults_comparison.png`
- `results/fig2_emat_latency.png`
- `results/fig3_frame_scaling_curve.png`
- `results/fig4_cumulative_faults_timeline.png`
- `results/CASE_STUDY_RESULTS_SUMMARY.md`

Every reported policy replays the same trace. The summary describes observed
results rather than assuming AWPR wins every metric or workload.

## Run the dashboard

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The API also exposes:

- `GET /api/health` — service health
- `POST /api/simulate` — comparative policy metrics and frame-scaling data
- `POST /api/live-step` — advances an AWPR/LRU live visualizer
- `POST /api/export` — regenerates benchmark artifacts

Simulation controls can be supplied as JSON, for example:

```json
{
  "frames": 64,
  "processes": 8,
  "trace_length": 2500,
  "seed": 42,
  "awpr_recency_weight": 0.25,
  "awpr_frequency_weight": 0.15,
  "awpr_semantic_weight": 0.6
}
```

## Tests

```bash
python -m unittest discover -s tests -v
```

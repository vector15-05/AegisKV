# AegisKV Benchmark Results

Results are generated from one deterministic multi-tenant trace shared by all policies.

## Comparative results

| Policy | Page faults | Fault rate | Dirty evictions | Sink evictions | TLB hit rate | EMAT (ns) | Simulated tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 1259 | 50.36% | 242 | 87 | 39.60% | 2723.89 | 367,122 |
| CLOCK | 1237 | 49.48% | 239 | 80 | 40.00% | 2679.68 | 373,179 |
| LRU | 1209 | 48.36% | 230 | 75 | 40.08% | 2615.59 | 382,323 |
| LFU | 839 | 33.56% | 104 | 0 | 45.00% | 1757.43 | 569,013 |
| AWPR | 836 | 33.44% | 104 | 0 | 43.36% | 1752.24 | 570,697 |

## Interpretation

- Lowest page-fault count: **AWPR** (836 faults, 33.44%).
- Lowest EMAT: **AWPR** (1752.24 ns/reference).
- AWPR recorded 0 attention-sink evictions, compared with 75 for LRU.
- AWPR page faults vs. LRU: +30.85% reduction (negative values indicate AWPR had more faults on this trace).

## Frame scaling

| Frames | FIFO | CLOCK | LRU | LFU | AWPR |
|---:|---:|---:|---:|---:|---:|
| 32 | 63.64% | 61.44% | 59.92% | 40.80% | 41.64% |
| 48 | 53.60% | 51.84% | 50.96% | 35.20% | 35.32% |
| 64 | 50.36% | 49.48% | 48.36% | 33.56% | 33.44% |
| 80 | 48.96% | 47.68% | 46.28% | 33.04% | 32.64% |
| 96 | 48.32% | 44.60% | 42.24% | 32.56% | 32.00% |
| 128 | 41.60% | 39.20% | 33.80% | 31.92% | 30.92% |

## Method

The simulator used 2,500 memory references, 8 tenants, 64 physical frames, and seed 42. The same generated trace was replayed independently for every policy. Latency includes TLB lookup or page-table walk, VRAM access, swap-in, and dirty-page swap-out costs.

These are discrete-event simulation results, not measurements from a GPU, operating-system kernel, or production inference server.

"""FastAPI application and single-page AegisKV dashboard."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cli_benchmark import run_benchmark
from core.simulator import SimulationConfig, Simulator
from core.workload_generator import generate_workload


ROOT = Path(__file__).resolve().parent
LIVE_SESSIONS: dict[str, dict[str, object]] = {}

app = FastAPI(
    title="AegisKV",
    description="Attention-aware virtual memory simulation dashboard",
    version="1.0.0",
)


class SimulationRequest(BaseModel):
    frames: int = Field(default=64, ge=32, le=256)
    processes: int = Field(default=8, ge=4, le=16)
    trace_length: int = Field(default=2_500, ge=500, le=5_000)
    seed: int = Field(default=42, ge=0)
    awpr_recency_weight: float = Field(default=0.25, ge=0, le=1)
    awpr_frequency_weight: float = Field(default=0.15, ge=0, le=1)
    awpr_semantic_weight: float = Field(default=0.60, ge=0, le=1)
    awpr_sink_bonus: float = Field(default=0.50, ge=0, le=2)

    def to_config(self) -> SimulationConfig:
        return SimulationConfig(
            num_frames=self.frames,
            num_processes=self.processes,
            num_references=self.trace_length,
            seed=self.seed,
            awpr_recency_weight=self.awpr_recency_weight,
            awpr_frequency_weight=self.awpr_frequency_weight,
            awpr_semantic_weight=self.awpr_semantic_weight,
            awpr_sink_bonus=self.awpr_sink_bonus,
        )


class LiveStepRequest(SimulationRequest):
    session_id: str | None = None
    steps: int = Field(default=1, ge=1, le=100)


@app.get("/", response_class=FileResponse)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "AegisKV"}


@app.post("/api/simulate")
def simulate(request: SimulationRequest) -> dict[str, object]:
    config = request.to_config()
    results, scaling_results = run_benchmark(config)
    return {
        "results": [result.to_dict() for result in results],
        "scaling": {
            str(frame_count): [result.to_dict() for result in frame_results]
            for frame_count, frame_results in scaling_results.items()
        },
        "trace": {
            "seed": config.seed,
            "references": config.num_references,
            "processes": config.num_processes,
            "frames": config.num_frames,
        },
    }


@app.post("/api/live-step")
def live_step(request: LiveStepRequest) -> dict[str, object]:
    session_id = request.session_id
    session = LIVE_SESSIONS.get(session_id) if session_id else None
    if session is None:
        session_id = str(uuid4())
        config = request.to_config()
        trace = generate_workload(config.num_references, config.num_processes, config.seed)
        session = {
            "trace": trace,
            "index": 0,
            "config": config,
            "simulators": {
                name: Simulator(config, name)
                for name in ("AWPR", "LRU")
            },
        }
        LIVE_SESSIONS[session_id] = session
    else:
        config = session["config"]
        if not isinstance(config, SimulationConfig):
            raise HTTPException(status_code=500, detail="Invalid live simulation state")
        if (
            request.frames != config.num_frames
            or request.processes != config.num_processes
            or request.trace_length != config.num_references
            or request.seed != config.seed
        ):
            raise HTTPException(
                status_code=409,
                detail="Live simulation controls cannot change after a session starts",
            )

    trace = session["trace"]
    simulators = session["simulators"]
    index = session["index"]
    if not isinstance(trace, list) or not isinstance(simulators, dict) or not isinstance(index, int):
        raise HTTPException(status_code=500, detail="Invalid live simulation state")
    end = min(index + request.steps, len(trace))
    for reference in trace[index:end]:
        for simulator in simulators.values():
            if not isinstance(simulator, Simulator):
                raise HTTPException(status_code=500, detail="Invalid live simulator")
            simulator.step(reference)
    session["index"] = end
    return {
        "session_id": session_id,
        "tick": end,
        "total_ticks": len(trace),
        "done": end >= len(trace),
        "event": {
            name: simulator.last_event if isinstance(simulator, Simulator) else {}
            for name, simulator in simulators.items()
        },
        "frames": {
            name: simulator.frame_snapshot()
            for name, simulator in simulators.items()
            if isinstance(simulator, Simulator)
        },
        "metrics": {
            name: simulator.result().to_dict()
            for name, simulator in simulators.items()
            if isinstance(simulator, Simulator)
        },
    }


@app.post("/api/export")
def export_report() -> dict[str, object]:
    from cli_benchmark import generate_artifacts

    return generate_artifacts()

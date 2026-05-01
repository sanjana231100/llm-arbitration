from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from arbitration.schemas import (
    ArbitrateRequest,
    ArbitrateResponse,
    BatchArbitrateRequest,
    BatchArbitrateResponse,
    CriticReport,
)
from arbitration.graph import arbitration_graph
from arbitration.storage import init_db, save_arbitration, get_arbitration, get_analytics
from arbitration.state import ArbitrationState
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="LLM Arbitration System",
    description="Multi-agent LLM output evaluation with parallel critic dispatch and confidence-scored verdicts.",
    version="0.1.0",
    lifespan=lifespan,
)


def _run_arbitration(request: ArbitrateRequest) -> ArbitrateResponse:
    initial_state: ArbitrationState = {
        "llm_output": request.llm_output,
        "original_prompt": request.original_prompt,
        "factual_report": None,
        "logical_report": None,
        "completeness_report": None,
        "disagreements": None,
        "verdict": None,
        "arbitration_id": None,
    }

    result = arbitration_graph.invoke(initial_state)

    return ArbitrateResponse(
        arbitration_id=result["arbitration_id"],
        verdict=result["verdict"],
        critic_reports=[
            result["factual_report"],
            result["logical_report"],
            result["completeness_report"],
        ],
        disagreements=result["disagreements"],
    )


@app.post("/v1/arbitrate", response_model=ArbitrateResponse)
async def arbitrate(request: ArbitrateRequest):
    response = await asyncio.to_thread(_run_arbitration, request)
    await save_arbitration(request.llm_output, request.original_prompt, response)
    return response


@app.post("/v1/arbitrate/batch", response_model=BatchArbitrateResponse)
async def arbitrate_batch(request: BatchArbitrateRequest):
    tasks = [asyncio.to_thread(_run_arbitration, item) for item in request.items]
    results = await asyncio.gather(*tasks)

    for item, response in zip(request.items, results):
        await save_arbitration(item.llm_output, item.original_prompt, response)

    return BatchArbitrateResponse(results=list(results))


@app.get("/v1/arbitrations/{arbitration_id}")
async def get_arbitration_by_id(arbitration_id: str):
    result = await get_arbitration(arbitration_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Arbitration not found")
    return result


@app.get("/v1/analytics")
async def analytics():
    return await get_analytics()


@app.get("/health")
async def health():
    return {"status": "ok"}

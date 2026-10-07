"""FastAPI routes for DevTeam AI.

Endpoints:
    POST /generate         - start a project generation run
    GET  /status/{run_id}  - live status, current agent, logs, usage
    GET  /files/{run_id}   - generated files in JSON
    GET  /download/{run_id}- download the packaged ZIP
    GET  /preview/{run_id} - code preview in JSON
    GET  /models           - active model routing configuration
    GET  /health           - provider health check (no secrets)
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.llm_config import health_check, list_task_models
from services.project_service import get_files, get_status, start_generation

router = APIRouter()


class PromptRequest(BaseModel):
    prompt: str


@router.post("/generate")
async def generate_project(req: PromptRequest):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured on the server.",
        )
    run_id = await start_generation(req.prompt)
    return {"run_id": run_id, "message": "Generation started"}


@router.get("/status/{run_id}")
async def get_project_status(run_id: str):
    state = get_status(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "status": state["status"],
        "current_agent": state.get("current_agent", ""),
        "final_status": state.get("final_status", ""),
        "iteration_count": state.get("iteration_count", 0),
        "logs": state["logs"],
        "agent_messages": state.get("agent_messages", []),
        "test_results": state.get("test_results", ""),
        "errors": state.get("errors", ""),
        "usage_info": state.get("usage_info", {}),
    }


@router.get("/files/{run_id}")
async def get_project_files(run_id: str):
    files = get_files(run_id)
    if not files:
        raise HTTPException(status_code=404, detail="Files not found")
    return {"files": files}


@router.get("/download/{run_id}")
async def download_project(run_id: str):
    zip_path = f"generated_projects/{run_id}.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP not found or generation incomplete")
    return FileResponse(zip_path, media_type="application/zip", filename=f"devteam_project_{run_id}.zip")


@router.get("/preview/{run_id}")
async def preview_code(run_id: str):
    files = get_files(run_id)
    if not files:
        raise HTTPException(status_code=404, detail="Files not found")
    return {"files": files}


@router.get("/models")
async def get_models():
    """Return the resolved model routing configuration (no secrets)."""
    from config.providers import get_provider_info
    return {"models": list_task_models(), "provider": get_provider_info()}


@router.get("/health")
async def get_health():
    """Provider health check — reports provider, model, and reachability.

    Does NOT expose the API key or any authentication details. Does NOT
    make an expensive LLM request (at most a lightweight models.list call).
    """
    return health_check()

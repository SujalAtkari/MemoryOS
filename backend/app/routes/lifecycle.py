from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.services.lifecycle_service import (
    LifecycleError,
    check_library_lifecycle,
    reconcile_relocation,
)

router = APIRouter(prefix='/lifecycle', tags=['lifecycle'])


class LifecycleCheckRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    library_id: str = Field(..., min_length=1)
    record_id: str | None = None


class RelocationReconcileRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    library_id: str = Field(..., min_length=1)
    record_id: str = Field(..., min_length=1)
    new_relative_path: str = Field(..., min_length=1)
    confirm: bool


@router.post('/check')
def check_lifecycle_endpoint(payload: LifecycleCheckRequest):
    try:
        return check_library_lifecycle(payload.library_id, payload.record_id)
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post('/reconcile')
def reconcile_lifecycle_endpoint(payload: RelocationReconcileRequest):
    try:
        return reconcile_relocation(
            payload.library_id,
            payload.record_id,
            payload.new_relative_path,
            payload.confirm,
        )
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

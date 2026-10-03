from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.services.record_state_service import RecordStateError, update_record_state

router = APIRouter(prefix='/records', tags=['records'])


class RecordStateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    library_id: str
    record_id: str
    importance_status: str | None = None
    is_important: bool | None = None
    protection_status: str | None = None
    lifecycle_status: str | None = None


@router.patch('/state')
def update_record_state_endpoint(payload: RecordStateRequest):
    try:
        return update_record_state(
            payload.library_id,
            payload.record_id,
            importance_status=payload.importance_status,
            is_important=payload.is_important,
            protection_status=payload.protection_status,
            lifecycle_status=payload.lifecycle_status,
        )
    except RecordStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
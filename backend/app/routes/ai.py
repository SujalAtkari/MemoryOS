from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.ai_processing_service import process_library_captions, process_library_images

router = APIRouter(prefix='/ai', tags=['ai'])


class AIProcessRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)
    record_ids: list[str] = Field(default_factory=list)
    force: bool = False
    profile: str = Field(default='detailed', pattern='^(fast|detailed)$')


class CaptionProcessRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)
    record_ids: list[str] = Field(default_factory=list)
    force: bool = False


@router.post('/process')
def process_ai_endpoint(payload: AIProcessRequest):
    try:
        return process_library_images(
            payload.folder_path,
            force=payload.force,
            record_ids=payload.record_ids,
            profile=payload.profile,
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/captions')
def process_captions_endpoint(payload: CaptionProcessRequest):
    try:
        return process_library_captions(
            payload.folder_path,
            record_ids=payload.record_ids,
            force=payload.force,
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

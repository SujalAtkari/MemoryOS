from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.duplicate_service import (
    OPENCLIP_DUPLICATE_THRESHOLD,
    PHASH_HAMMING_THRESHOLD,
    VISUAL_CANDIDATE_PHASH_DISTANCE,
    list_library_duplicates,
    process_library_duplicates,
)

router = APIRouter(prefix='/duplicates', tags=['duplicates'])


class DuplicateProcessRequest(BaseModel):
    folder_path: str | None = None
    force: bool = False
    include_visual: bool = True
    phash_threshold: int = Field(default=PHASH_HAMMING_THRESHOLD, ge=0, le=256)
    openclip_threshold: float = Field(default=OPENCLIP_DUPLICATE_THRESHOLD, ge=-1, le=1)
    visual_candidate_phash_distance: int = Field(
        default=VISUAL_CANDIDATE_PHASH_DISTANCE,
        ge=0,
        le=256,
    )


@router.post('/process')
def process_duplicates_endpoint(payload: DuplicateProcessRequest):
    try:
        return process_library_duplicates(
            folder_path=payload.folder_path,
            force=payload.force,
            include_visual=payload.include_visual,
            phash_threshold=payload.phash_threshold,
            openclip_threshold=payload.openclip_threshold,
            visual_candidate_phash_distance=payload.visual_candidate_phash_distance,
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('')
def list_duplicates_endpoint(folder_path: str | None = None):
    try:
        return list_library_duplicates(folder_path=folder_path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

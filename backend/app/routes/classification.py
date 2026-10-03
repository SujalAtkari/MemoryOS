from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.classification_service import approve_classification, process_library_classification

router = APIRouter(prefix='/classification', tags=['classification'])


class ClassificationRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)
    force: bool = False
    record_id: str | None = None
    use_llm: bool = False


class ClassificationApprovalRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)
    record_id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    subcategory: str | None = None
    previous_category: str | None = None


@router.post('/process')
def process_classification_endpoint(payload: ClassificationRequest):
    try:
        return process_library_classification(
            payload.folder_path,
            force=payload.force,
            record_id=payload.record_id,
            use_llm=payload.use_llm,
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/approve')
def approve_classification_endpoint(payload: ClassificationApprovalRequest):
    try:
        return approve_classification(
            payload.folder_path,
            payload.record_id,
            payload.category,
            subcategory=payload.subcategory,
            previous_category=payload.previous_category,
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

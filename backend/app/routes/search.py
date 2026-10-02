from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.search_service import search_images

router = APIRouter(prefix='/search', tags=['search'])


class SearchRequest(BaseModel):
    query: str
    folder_path: str | None = None
    category: str | None = None
    limit: int = Field(default=20, ge=1, le=500)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    use_llm: bool = True


@router.post('')
def search_endpoint(payload: SearchRequest):
    try:
        return search_images(
            query=payload.query,
            folder_path=payload.folder_path,
            category=payload.category,
            limit=payload.limit,
            min_score=payload.min_score,
            use_llm=payload.use_llm,
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

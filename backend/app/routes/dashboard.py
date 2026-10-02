from fastapi import APIRouter, HTTPException, Query

from app.services.dashboard_service import get_dashboard_stats

router = APIRouter(prefix='/dashboard', tags=['dashboard'])


@router.get('/stats')
def dashboard_stats_endpoint(folder_path: str | None = Query(default=None)):
    try:
        return get_dashboard_stats(folder_path=folder_path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

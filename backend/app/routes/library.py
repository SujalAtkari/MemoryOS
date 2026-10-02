from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.library_service import scan_library

router = APIRouter(prefix='/library', tags=['library'])


class LibraryScanRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)


@router.post('/scan')
def scan_library_endpoint(payload: LibraryScanRequest):
    try:
        return scan_library(payload.folder_path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/select-folder')
def select_folder_endpoint():
    """Open a native folder picker on the local machine running MemoryOS."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail='The native folder picker is unavailable in this Python installation.',
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        selected_folder = filedialog.askdirectory(
            title='Select your MemoryOS image folder',
            mustexist=True,
        )
    finally:
        root.destroy()

    if not selected_folder:
        return {'folder_path': None, 'cancelled': True}

    return {'folder_path': str(Path(selected_folder).resolve()), 'cancelled': False}

import mimetypes

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.services.file_operations_service import (
    FileOperationError,
    delete_indexed_image,
    get_indexed_image_preview,
    list_library_folders,
    move_indexed_image,
    open_image_folder,
    open_indexed_image,
    rename_indexed_image,
)

router = APIRouter(prefix='/files', tags=['files'])


class IndexedFileRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    library_id: str = Field(..., min_length=1)
    record_id: str = Field(..., min_length=1)


class RenameRequest(IndexedFileRequest):
    new_filename: str = Field(..., min_length=1)


class MoveRequest(IndexedFileRequest):
    destination_folder: str


class DeleteRequest(IndexedFileRequest):
    confirm: bool


def _raise_file_error(exc: FileOperationError) -> None:
    detail = str(exc)
    status_code = 409 if any(word in detail.lower() for word in (
        'changed since indexing',
        'already exists',
        'already in the selected folder',
    )) else 400
    raise HTTPException(status_code=status_code, detail=detail) from exc


@router.post('/open')
def open_file_endpoint(payload: IndexedFileRequest):
    try:
        return open_indexed_image(payload.library_id, payload.record_id)
    except FileOperationError as exc:
        _raise_file_error(exc)


@router.get('/preview')
def preview_file_endpoint(
    library_id: str = Query(..., min_length=1),
    record_id: str = Query(..., min_length=1),
):
    try:
        image_path = get_indexed_image_preview(library_id, record_id)
    except FileOperationError as exc:
        _raise_file_error(exc)
    media_type, _encoding = mimetypes.guess_type(image_path.name)
    return FileResponse(
        image_path,
        media_type=media_type or 'application/octet-stream',
        filename=image_path.name,
        content_disposition_type='inline',
        headers={
            'Cache-Control': 'no-store',
            'X-Content-Type-Options': 'nosniff',
        },
    )


@router.post('/open-folder')
def open_folder_endpoint(payload: IndexedFileRequest):
    try:
        return open_image_folder(payload.library_id, payload.record_id)
    except FileOperationError as exc:
        _raise_file_error(exc)


@router.get('/folders')
def list_folders_endpoint(
    library_id: str = Query(..., min_length=1),
    record_id: str = Query(..., min_length=1),
):
    try:
        return list_library_folders(library_id, record_id)
    except FileOperationError as exc:
        _raise_file_error(exc)


@router.post('/rename')
def rename_file_endpoint(payload: RenameRequest):
    try:
        return rename_indexed_image(
            payload.library_id,
            payload.record_id,
            payload.new_filename,
        )
    except FileOperationError as exc:
        _raise_file_error(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post('/move')
def move_file_endpoint(payload: MoveRequest):
    try:
        return move_indexed_image(
            payload.library_id,
            payload.record_id,
            payload.destination_folder,
        )
    except FileOperationError as exc:
        _raise_file_error(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post('/delete')
def delete_file_endpoint(payload: DeleteRequest):
    try:
        return delete_indexed_image(
            payload.library_id,
            payload.record_id,
            payload.confirm,
        )
    except FileOperationError as exc:
        _raise_file_error(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

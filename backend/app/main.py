from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ai import router as ai_router
from app.routes.classification import router as classification_router
from app.routes.dashboard import router as dashboard_router
from app.routes.duplicates import router as duplicates_router
from app.routes.files import router as files_router
from app.routes.health import router as health_router
from app.routes.lifecycle import router as lifecycle_router
from app.routes.library import router as library_router
from app.routes.search import router as search_router
from app.routes.records import router as records_router

app = FastAPI(
    title='MemoryOS backend',
    version='0.1.0',
    description='Local image indexing, AI processing, and explainable classification for MemoryOS.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)
app.include_router(library_router)
app.include_router(ai_router)
app.include_router(classification_router)
app.include_router(search_router)
app.include_router(duplicates_router)
app.include_router(dashboard_router)
app.include_router(files_router)
app.include_router(lifecycle_router)
app.include_router(records_router)


@app.get('/')
def read_root():
    return {'message': 'MemoryOS backend is running'}

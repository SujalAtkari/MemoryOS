# MemoryOS

MemoryOS is a local-first desktop image library system designed to process and index images locally without sending them to the cloud.

## Local-first architecture

Original images remain in the folder where the user selected them. MemoryOS does not copy original image files into project-managed storage. Only metadata and index information are stored locally.

Local Image Files
  ↓
FastAPI Backend
  ↓
Local JSON Index
  ↓
React Frontend

## No database architecture

No database or vector database is used.

The active local index is stored at:

- data/index/library_index.json

This file contains lightweight library metadata and relative-path records. It is a local JSON index, not a database. Backend updates are serialized and written through unique temporary files with an atomic replace, so overlapping scan/AI/classification requests cannot reuse a temp filename or overwrite another in-process index update.

## Implemented now

- Phase 1: local folder scanning and JSON metadata index
- Phase 2: local OCR, BLIP captioning, and OpenCLIP image embeddings
- Phase 3: multi-evidence image classification
- Phase 4: semantic and hybrid local image search
- Phase 5: local exact, near-duplicate, and optional visual-similarity detection
- Phase 6: local analytics dashboard derived from the existing JSON index
- Phase 7: explicit safe open, rename, move, and delete operations for indexed images
- Phase 8: lifecycle status checks and confirmed external relocation reconciliation
- Phase 9: backend regression, real temporary-library integration, and documented evaluation
- Phase 10: final integration and startup/health verification of the existing local application

## Phase 1 implemented

Phase 1 includes:

- recursive local folder scanning
- supported image discovery for .jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff, and .tif
- image metadata extraction
- SHA-256 generation
- pHash generation
- local JSON index persistence
- duplicate prevention on repeated scans
- modified-file detection on rescans
- basic frontend library display

## Phase 2 AI processing status

Phase 2 adds local AI processing foundation for indexed images without copying original files or introducing a database.

Implemented now:

- local OCR through PaddleOCR
- local BLIP caption generation
- local OpenCLIP ViT-B-32 embeddings
- local JSON index extension with AI metadata
- local embedding storage under `data/embeddings/`
- resumable five-image processing batches in the frontend
- minimal frontend trigger for local AI processing

AI inference currently runs on CPU with local PaddleOCR, BLIP, and OpenCLIP models. The first batch can take longer while the models load, and processing large libraries can take substantial time. The interface reports batch progress, saves each completed batch to the local index, and skips unchanged images on later runs. It updates the dashboard directly after processing instead of rescanning and rehashing the full library.

In AI Workspace, **Fast visual categories** (the default) creates one OpenCLIP image embedding per image, then classifies the library from visual similarity and existing filename/path evidence. It avoids OCR and BLIP caption inference. **Detailed OCR, captions, and visual search** also runs OCR and BLIP per image, so it is slower; it can be run later to add text search evidence. **Generate missing captions** runs only BLIP for images without a completed caption, saving each five-image batch and refreshing classification afterward. Fast mode saves up to 50 records per request, while Detailed and caption-only modes save every 5 records so progress and completed work are visible sooner. Both analysis modes refresh classifications and dashboard category distribution after processing. This application uses CPU-only inference when CUDA is unavailable, so processing large libraries takes time.

Controlled local timing on 10 generated 320x240 JPEGs: warm Fast visual processing took 4.096 seconds; warm Detailed processing took 147.354 seconds (about 36x slower on this run). This is a small synthetic benchmark, not a production-library guarantee; image dimensions, content, and hardware affect results.

The current classifier version is `phase3-multi-evidence-v6`; older classification results are recalculated when the library is classified again.

Out of scope:

- final dashboard
- file deletion/archive
- database-backed storage
- cloud inference

Duplicate detection is implemented in Phase 5 below; it does not delete or copy files.

## Phase 3 multi-evidence classification

Phase 3 classifies indexed records using the existing Phase 2 OCR text, BLIP caption, and locally stored OpenCLIP image embedding, together with filename/path and basic image metadata. Classification does not rerun OCR, captioning, or image embedding generation. OpenCLIP category-text embeddings are generated with the Phase 2 ViT-B-32 model and cached in memory; the existing `.npy` image embedding is reused. No vector database or second image-embedding copy is created.

This is a multi-evidence classifier, not a custom-trained model. Its confidence is a weighted evidence score, not a calibrated probability or an accuracy claim. Every successfully processed image receives one of the 15 categories. A clear, sufficiently supported winner is assigned to its category; ambiguous, conflicting, or weak evidence (leading score below `0.12`) is assigned to `Others`, never left in a review queue. This conservative fallback avoids presenting a guess as a confident category.

The exact categories are:

1. Government & Identity
2. Education
3. Medical & Health
4. Finance
5. Bills & Receipts
6. Work & Professional
7. Travel
8. Events & Celebrations
9. People & Family
10. Nature & Places
11. Animals & Pets
12. Food & Drinks
13. Screenshots
14. Notes & Documents
15. Others

Evidence weights, normalized over sources present for that record:

- filename/path keyword score: `0.20`
- reliable OCR keyword score: `0.32`
- BLIP caption keyword score: `0.20`
- OpenCLIP category-text similarity score: `0.23`
- basic metadata score: `0.05`

OCR keyword evidence is used only when OCR completed and its confidence is at least `0.50`. OpenCLIP category similarities are converted to relative category probabilities using a softmax temperature of `0.08`. The leading category is assigned unless the runner-up is at least 90% of the leader, strong evidence sources conflict, evidence is missing, or the leading weighted score is below `0.12`; those cases go to `Others`. The score and ambiguity checks are not calibrated accuracy probabilities. `ClassificationConfig` exposes the evidence weights, ambiguity ratio, and minimum score. Category prompts, version, scores, matched keywords, sources, and assignment reasons are retained for interpretability.

Classification adds fields to existing records in `data/index/library_index.json`; it does not create another index. Version `phase3-multi-evidence-v6` migrates older classifications the next time that library is classified. Reprocessing skips records whose classifier version and source SHA-256 match and whose classification output fields are present. Records without enough evidence receive `Others` with an explanation in `assignment_reason`.

API:

```http
POST /classification/process
```

Request:

```json
{
  "folder_path": "C:\\Users\\Name\\Pictures",
  "force": false,
  "record_id": null
}
```

Omit `record_id` to process all indexed images, or provide a relative-path record ID to process one image. The response includes assigned, legacy `needs_review` (always zero for the current classifier), failed, and skipped counts.

To generate captions without rerunning OCR or image embeddings:

```http
POST /ai/captions
```

```json
{
  "folder_path": "C:\\Users\\Name\\Pictures",
  "record_ids": []
}
```

Omit or leave `record_ids` empty to caption every available image in that scanned library that does not already have a completed, non-empty caption. Provide record IDs to process a batch. Failed captions are reported and can be retried; a successful caption is kept in the local index for search.

## Phase 4 local RAG and image retrieval

Search uses a local retrieval-augmented ranking pipeline. The retriever encodes the query with the cached OpenCLIP ViT-B-32 text encoder, compares it with the existing image embeddings, and retrieves candidates that also match filename/path, OCR, BLIP caption, category evidence, or metadata. Text-only candidates can still be retrieved without an image embedding. Images in `needs_review` and images with low classification evidence remain searchable; classification confidence is not used as a relevance filter or primary sort key.

The retrieved candidate set is ranked by local hybrid visual and lexical evidence, then the best 8 candidates are reranked by a local Ollama LLM (`llama3.2:3b` by default). The LLM receives only the user query and bounded indexed metadata (filename, caption, OCR excerpt, and category fields); it does not receive image files and is constrained to return only the supplied candidate IDs. Reranking has a 30-second timeout; if Ollama or the model is unavailable or too slow, the API reports the reranker status and returns the usable hybrid-ranked retrieval results instead. No search query or retrieved content is sent to a cloud service.

Local hybrid scoring combines semantic similarity with filename/path, OCR text, caption, category/keyword, and metadata matches. Available channel weights are renormalized per record; semantic cosine is mapped to `[0, 1]` for hybrid scoring. The raw `semantic_score` remains a cosine similarity, not a probability.

Install Ollama and download the configured model before enabling LLM reranking:

```powershell
ollama pull llama3.2:3b
```

Set `MEMORYOS_OLLAMA_URL` or `MEMORYOS_OLLAMA_MODEL` to use a different local Ollama server or model. Candidate retrieval continues to work if the LLM is stopped.

API:

```http
POST /search
```

Example request:

```json
{
  "folder_path": "C:\\Users\\Name\\Pictures",
  "query": "government passport document",
  "category": "Government & Identity",
  "limit": 20,
  "min_score": 0.0
}
```

`folder_path` and `category` are optional. When `folder_path` is omitted, search includes all indexed libraries. `use_llm` enables local LLM reranking (default `true`). Search filters out missing/unavailable files, but can retrieve from text evidence without an embedding; invalid or dimension-mismatched embeddings do not prevent lexical retrieval. It returns ranked metadata, component scores, matched fields/terms, reranker status, and an explanation. Search is limited to at most 500 returned results per request.

Performance uses an in-memory cached model/tokenizer, a bounded 5,000-vector cache keyed by embedding file path/size/modification time, and a single vectorized NumPy comparison over available embedding records. Candidate vectors come from existing `.npy` files; original images are not read. Records without usable Phase 2 embeddings can still be returned when their indexed text matches the query.

## Phase 5 duplicate detection

Duplicate detection reports relationships only. It never deletes, moves, modifies, or copies original image files. It reuses Phase 1 SHA-256 and pHash metadata, and optionally compares already-generated Phase 2 OpenCLIP image embeddings. Results and per-record relationship metadata are stored in the existing `data/index/library_index.json`; no database or second embedding index is created.

Detection methods:

- **Exact duplicates:** records with the same valid SHA-256 are grouped. If an indexed record has no valid SHA-256, the service computes its hash by reading the file in place; it does not copy it.
- **Near duplicates:** pHash Hamming distance is compared with a configurable threshold, default `8`.
- **Visual matches (optional):** existing normalized OpenCLIP image embeddings are compared with NumPy cosine similarity, default threshold `0.90`. Candidate pairs are bounded by a pHash distance of `16` by default. This candidate bound improves performance but can miss visually similar images whose pHashes differ by more than the configured bound. Missing or invalid embeddings are skipped.

Connected components form near-duplicate and visual groups. A chain of pairwise matches can therefore place two endpoints in the same group even when those endpoints do not directly meet the threshold; match edges and distances/similarities are included so the group is inspectable. These thresholds are configurable starting points, not claims of optimal accuracy. The unchanged-library fingerprint includes detection options, image availability, SHA-256/pHash values, and available embedding file metadata; matching runs can be skipped unless `force` is true.

API:

```http
POST /duplicates/process
GET /duplicates
```

Example processing request:

```json
{
  "folder_path": "C:\\Users\\Name\\Pictures",
  "force": false,
  "include_visual": true,
  "phash_threshold": 8,
  "openclip_threshold": 0.9,
  "visual_candidate_phash_distance": 16
}
```

`folder_path` is optional; when omitted, all indexed libraries are processed/listed according to the existing library index. The process response includes counts and duplicate groups with member paths and match evidence. The React interface exposes a **Detect Duplicates** action and displays the result groups. It provides no delete action.

## Phase 6 analytics dashboard

The dashboard is read-only and calculates statistics directly from `data/index/library_index.json` on each request. It does not read original image contents, inspect embedding files, execute OCR/BLIP/OpenCLIP, or trigger duplicate detection. No analytics storage, database, vector database, or external service is used.

API:

```http
GET /dashboard/stats
GET /dashboard/stats?folder_path=C:\Users\Name\Pictures
```

With no `folder_path`, statistics aggregate all indexed libraries. The optional filter uses the existing local folder validation behavior. The response contains:

- Image counts and separate AI processing status (`completed`, `pending`, `processing`, `failed`, and other stored states).
- Separate classification status counts (`classified`, `needs_review`, `pending`, `processing`, `failed`, and other stored states).
- File tracking counts for the index's `available`, `missing`, and `modified` states; absent or unrecognized states are shown as `unverified`.
- The 15 Phase 3 category counts and percentages among classified records with a recognized category. These percentages describe category distribution and are not accuracy measures.
- Phase 5 exact, near, and visual group counts, plus the number of distinct indexed records involved in any duplicate group.
- Library count, image count, AI-processed count, missing-file count, and a display name. The API does not return full local library root paths in its library summaries.

Invalid records are excluded from image totals and reported separately; invalid library entries are also counted. A missing index returns empty statistics. A malformed or unreadable index is reported as an API error rather than presented as valid zero counts. "Skipped" processing is not reported because the index does not persist a reliable per-record skipped state. Protected/important-record analytics and recent activity are omitted because current Phase 1–5 records do not define dependable fields for them.

The React dashboard loads once when the app opens and can be refreshed manually. The refresh action uses the typed library path when provided, or aggregates all indexed libraries when it is blank.

## Phase 7 safe file operations

File actions operate only on records identified by `library_id` and indexed `record_id`; the API does not accept arbitrary source paths. Backend validation resolves the indexed library root, rejects traversal/absolute/network paths and symbolic-link/reparse-point paths, checks supported image extensions, and compares the current file SHA-256 against its indexed value before an operation.

The frontend exposes **Open**, **Open Folder**, **Rename**, **Move**, and **Delete** actions on indexed image cards and search results. OS launching uses the Windows default application/Explorer via `os.startfile` without shell command strings. OS launching is currently Windows-only.

Rename accepts a filename only, preserves the current extension when omitted, permits only supported image extensions when supplied, and rejects invalid or reserved Windows names. Move accepts a library-relative folder selected from existing folders returned by `GET /files/folders`; it does not accept a destination absolute path or create destination folders. Rename and move do not overwrite existing destinations, update only the affected record after the filesystem operation succeeds, and preserve image hashes and existing AI/classification artifacts. Duplicate results are invalidated after a path change because their saved member paths would otherwise be stale; duplicate detection is not automatically rerun.

Delete requires `confirm: true` and the UI presents a dialog naming the file and relative path with an explicit **Delete original image** action. A successful deletion marks the indexed record `missing` rather than removing its historical metadata. It never deletes duplicate-group members automatically, and it does not remove embedding or other AI artifacts. Deletion is permanent from MemoryOS; no recycle-bin, backup, or undo behavior is implemented.

Records marked missing cannot be opened or changed through these endpoints until the library is rescanned. If SHA-256 is unavailable, a file operation requires matching indexed file-size or modification-time metadata; otherwise it rejects the operation and asks for a rescan.

API:

```http
POST /files/open
POST /files/open-folder
GET  /files/folders?library_id=...&record_id=...
POST /files/rename
POST /files/move
POST /files/delete
```

Rename request example:

```json
{
  "library_id": "indexed-library-id",
  "record_id": "receipts/receipt.png",
  "new_filename": "march-receipt"
}
```

Move uses `destination_folder` as a path relative to the selected library (`""` selects the library root). Delete requires `"confirm": true`. If a filesystem change succeeds but atomic index persistence fails, the API reports that the library must be rescanned; it does not attempt an automatic rollback.

## Phase 8 image lifecycle and reconciliation

Lifecycle checking compares indexed records with files inside the indexed library without invoking OCR, BLIP, OpenCLIP, classification, or duplicate detection. It updates the existing `file_status` values:

- `available`: supported regular image file matches indexed size/time metadata, or a required SHA-256 check confirms the indexed content.
- `missing`: indexed path no longer exists. The record and its AI metadata remain in the JSON index.
- `modified`: a file remains at its indexed path but its current SHA-256 differs from the indexed SHA-256. The original indexed hash and AI/classification outputs are preserved; the record is excluded from search until rescanned/reprocessed.
- `unverified`: path is unsafe, unsupported, inaccessible, or the index lacks enough metadata to determine its state safely.

API:

```http
POST /lifecycle/check
POST /lifecycle/reconcile
```

`POST /lifecycle/check` accepts `library_id` and an optional `record_id`; it only detects state and reports checked counts, current per-record statuses, changed records, and possible relocations. When an indexed path is missing, relocation discovery searches only within that library. It first filters candidates by stored file size and hashes only supported regular files with matching sizes. Only exact SHA-256 matches are reported; this can miss relocations if indexed size/hash metadata is absent or unavailable.

Detection never changes an indexed path. To apply a reported relocation, `POST /lifecycle/reconcile` requires the library ID, old record ID, detected library-relative new path, and `confirm: true`. The backend revalidates containment, supported file type, absence of the old path, and exact SHA-256 before updating the index. It does not move or rename the file. Existing AI outputs are retained because the hash proves the content matches; stale Phase 5 duplicate path data is invalidated, not recomputed.

The check uses file existence/type and stat metadata first. Files whose size and modification time match the indexed values normally require no image-content read; SHA-256 is computed only when metadata differs or when a missing record reappears and must be validated. It persists only when record lifecycle metadata changes. It never deletes records, images, embeddings, or AI metadata and never automatically reprocesses modified images. A rescan is needed to refresh source metadata before explicitly rerunning downstream processing.

The React **Check Library** action requires a library already scanned in the current UI. It displays current missing/modified/unverified records and candidate relocations; reconciliation requires a confirmation prompt and updates only the index. Files marked missing are not presented with normal file actions. File status metadata can detect ordinary external modifications, but same-size content changes with a deliberately unchanged timestamp cannot be detected by the fast stat-only path until another operation causes hash validation.

## Browser folder-selection limitation

This Phase 1 frontend includes a browser folder picker and manual local folder-path entry. Browser security does not expose the selected folder's absolute path to the page, so enter the full local path in the Library path field before scanning. The picker alone cannot start a backend scan.

Original image files are never uploaded or copied into MemoryOS-managed storage.

## Getting started on Windows

### Prerequisites

- Windows 10 or 11
- Python 3.13 and its Python Launcher (`py`)
- Node.js LTS, which includes `npm`
- Internet access during setup and the first AI run, so packages and local AI models can be downloaded

MemoryOS runs the FastAPI backend and React frontend as two local processes. The original images stay in folders you choose; the local index and generated embeddings are stored under `data/`.

### First-time setup after cloning

Open PowerShell in the cloned project folder and run:

```powershell
py -3.13 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Set-Location frontend
npm ci
Set-Location ..
```

Installing the backend dependencies can take a while: PyTorch, PaddleOCR, and their supporting packages are large. If Python 3.13 is not installed, install it first and then reopen PowerShell. `npm ci` uses the checked-in `frontend/package-lock.json` to install the matching frontend dependencies.

### Start the app

After setup, double-click [`Start MemoryOS.bat`](./Start%20MemoryOS.bat) in the project folder. It starts the backend and frontend in separate windows and opens the app at <http://localhost:5173>. Keep those server windows open while using MemoryOS. Close them or press `Ctrl+C` in each window to stop the servers.

To start the services manually instead, open two PowerShell windows in the project folder.

**Backend terminal:**

```powershell
Set-Location backend
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check that the backend is ready at <http://127.0.0.1:8000/health>; it should return `{"status":"ok","service":"MemoryOS backend"}`.

**Frontend terminal:**

```powershell
Set-Location frontend
npm run dev
```

Then open <http://localhost:5173>.

### AI models and optional search reranking

The first use of image analysis downloads the local OpenCLIP, BLIP captioning, and PaddleOCR models. These model files are not Python packages and are downloaded separately from the dependencies in `backend/requirements.txt`. Processing runs locally on the CPU by default and can be slow for a large image library.

Ollama is **optional**. Without it, image scanning, captions, classification, and search still work, but LLM search reranking is unavailable. To enable reranking, install Ollama and run:

```powershell
ollama pull llama3.2:3b
```

Keep Ollama running while using reranked search.

### What to include in a GitHub repository

Commit the source code, documentation, `backend/requirements.txt`, and the frontend `package.json` plus `package-lock.json`. Do **not** commit virtual environments, `frontend/node_modules`, downloaded model weights, image-library contents, or local index/embedding/cache data. The root `.gitignore` excludes these generated and local files. Do not add passwords, API keys, or personal image files.

Use the navigation bar to switch between the Dashboard, My Library, AI Tools, Search, Duplicates, and File Health pages. The selected page is reflected in the URL hash.

## Phase 9 testing and evaluation

Backend regression tests, the real temporary-library integration, frontend build results, environment versions, safety/dependency checks, and measured real/synthetic search timings are recorded in [docs/PHASE_9_TESTING.md](docs/PHASE_9_TESTING.md). Synthetic benchmark results are explicitly distinguished from real AI processing.

## Phase 10 final integration

The React/Vite interface runs locally at `http://localhost:5173` and connects to the FastAPI backend at `http://localhost:8000`. The interface exposes the existing scan, local AI processing, classification, search, duplicate, dashboard, file-operation, and lifecycle workflows. A manual local folder path is required for reliable scanning because browser folder selection does not expose an arbitrary absolute path.

Final integration verification:

- FastAPI started without startup errors; `GET /health` returned `{"status":"ok","service":"MemoryOS backend"}`.
- The frontend loaded with the `MemoryOS` heading, fetched dashboard data from the backend, and showed the empty-index state. The final browser reload had no JavaScript page errors or failed requests.
- The temporary-library integration test passed (`1 passed, 4 warnings in 44.99s`) and exercised scanning, real local AI, classification, semantic search, duplicate detection, dashboard, file operations, and lifecycle/reconciliation. Its generated images and index were isolated to temporary test storage.
- Current backend regression suite: `169 passed, 1 skipped, 4 warnings in 79.08s`.
- Current frontend production build: Vite transformed 31 modules and completed in `1.91s`.
- An invalid AI library path now returns HTTP 400 with a readable error instead of creating an empty index entry; a regression test verifies the index is not written.
- Project `data/index/`, `data/embeddings/`, and `data/cache/` remained without test-library records or embeddings after verification.

The existing file-operation and lifecycle regression tests verify endpoint behavior, including explicit delete confirmation, preservation of metadata/embeddings on valid operations, and explicit relocation reconciliation. No frontend unit-test runner is configured, and OS-level open actions were not launched during this verification.

During the initial live-startup probe, before the invalid-path guard was added, processing requests for two nonexistent workspace folders created empty index entries for those paths. Inspection confirmed both entries had zero image records and no embeddings. Those probe-generated entries were removed to restore the previously absent index file; the corrected endpoint now rejects nonexistent paths without writing the index. No original images were targeted or changed.

## Backend setup reference

The Windows first-time setup and run commands are in [Getting started on Windows](#getting-started-on-windows). The backend dependency manifest is `backend/requirements.txt`. To install dependencies and run tests, change to the backend folder first:

```powershell
Set-Location backend
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests
```

The test dependencies (`pytest` and `httpx`) are included in the same requirements file.

## Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "MemoryOS backend"
}
```

## Phase 1 scan endpoint

```http
POST /library/scan
```

Request body:

```json
{
  "folder_path": "C:\\Users\\Name\\Pictures"
}
```

The backend validates the folder, scans supported images recursively, reads metadata, computes SHA-256 and pHash, and updates the local JSON index.

## Local index format

Active index path:

- data/index/library_index.json

Example structure:

```json
{
  "version": 1,
  "libraries": {
    "<library_id>": {
      "library_root": "C:\\path\\to\\library",
      "records": {
        "nested/file.jpg": {
          "record_id": "nested/file.jpg",
          "filename": "file.jpg",
          "relative_path": "nested/file.jpg",
          "extension": ".jpg",
          "file_size": 1234,
          "modified_time": "2026-10-01T20:44:16",
          "width": 120,
          "height": 80,
          "aspect_ratio": 1.5,
          "sha256": "...",
          "phash": "...",
          "processing_status": "pending",
          "file_status": "available"
        }
      }
    }
  }
}
```

The record identity is based on the library root plus relative path. Original image bytes are not stored.

MemoryOS does not require a database or cloud storage. Original images remain in their local folders while metadata, OCR results, captions, classifications, hashes, processing information, and references to separate local embedding files are maintained locally. Search uses NumPy vectorized similarity over existing embeddings.

## Security and privacy notes

MemoryOS is designed to process local image data without cloud storage. It does not send original images to external services and does not include analytics or telemetry.

## Future / proposed

The following remain future work and are not part of the current implementation:

- archive, recycle-bin, backup, undo, and recovery capabilities
- lifecycle automation or automatic reprocessing
- database-backed storage
- cloud services

## Additional notes

- No database has been added.
- Original images remain local and are changed only after an explicit user file action; delete is permanent and requires confirmation.
- Phase 1 scanning, Phase 2 local AI processing, Phase 3 multi-evidence classification, Phase 4 semantic/hybrid search, Phase 5 duplicate detection, Phase 6 dashboard analytics, Phase 7 safe file operations, Phase 8 lifecycle reconciliation, Phase 9 testing/evaluation, and Phase 10 final integration are implemented.
- Search/classification quality and cold-cache performance depend on the input library and host. The documented Phase 9 synthetic benchmark is not a production-scale performance guarantee, and no universal accuracy claim is made.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const MAX_VISIBLE_RECORDS = 100;
const SEARCH_CATEGORIES = [
  'Government & Identity',
  'Education',
  'Medical & Health',
  'Finance',
  'Bills & Receipts',
  'Work & Professional',
  'Travel',
  'Events & Celebrations',
  'People & Family',
  'Nature & Places',
  'Animals & Pets',
  'Food & Drinks',
  'Screenshots',
  'Notes & Documents',
  'Others',
];
const PAGE_LINKS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'library', label: 'My Library' },
  { id: 'processing', label: 'AI Workspace' },
  { id: 'search', label: 'Search' },
  { id: 'duplicates', label: 'Duplicates' },
  { id: 'health', label: 'File Health' },
];
const AI_BATCH_SIZE = 50;
const DETAILED_AI_BATCH_SIZE = 5;

function getPageFromHash() {
  const pageId = window.location.hash.slice(1);
  return PAGE_LINKS.some((page) => page.id === pageId) ? pageId : 'dashboard';
}

function getResponseError(data, fallback) {
  const detail = data?.detail;
  if (typeof detail === 'string') {
    if (/folder does not exist/i.test(detail)) {
      return 'Folder does not exist. Please enter a valid full local folder path.';
    }
    return /traceback/i.test(detail) ? fallback : detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter((message) => typeof message === 'string');
    if (messages.length) {
      return messages.join(' ');
    }
  }
  return fallback;
}

function getRequestError(error, fallback) {
  if (error instanceof TypeError) {
    return 'Cannot connect to the MemoryOS backend. Start the backend and try again.';
  }
  const message = error?.message;
  if (typeof message === 'string' && /folder does not exist/i.test(message)) {
    return 'Folder does not exist. Please enter a valid full local folder path.';
  }
  if (typeof message === 'string' && !/traceback/i.test(message)) {
    return message;
  }
  return fallback;
}

function getPreviewUrl(record, libraryId) {
  if (!record?.record_id && !record?.relative_path) {
    return '';
  }
  const params = new URLSearchParams({
    library_id: libraryId || record.library_id || '',
    record_id: record.record_id || record.relative_path,
  });
  return `http://localhost:8000/files/preview?${params.toString()}`;
}

function getTopCategoryShare(record) {
  const topScore = Number(record?.top_score);
  const alternativeScore = Number(record?.alternative_score);
  const total = topScore + alternativeScore;
  if (!Number.isFinite(total) || total <= 0) {
    return null;
  }
  return (topScore / total) * 100;
}

function evidenceSourceLabel(source) {
  return ({
    filename_score: 'Filename',
    ocr_score: 'OCR text',
    caption_score: 'Image caption',
    semantic_score: 'Visual meaning',
  })[source] || source.replace(/_score$/, '').replaceAll('_', ' ');
}

function App() {
  const [currentPage, setCurrentPage] = useState(getPageFromHash);
  const [folderPath, setFolderPath] = useState('');
  const [status, setStatus] = useState('Ready to scan a local library.');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [scanResult, setScanResult] = useState(null);
  const [aiResult, setAiResult] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiProfile, setAiProfile] = useState('fast');
  const [aiProgress, setAiProgress] = useState(null);
  const [aiElapsedSeconds, setAiElapsedSeconds] = useState(0);
  const [classificationResult, setClassificationResult] = useState(null);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchCategory, setSearchCategory] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [duplicateResult, setDuplicateResult] = useState(null);
  const [duplicateLoading, setDuplicateLoading] = useState(false);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [dashboardError, setDashboardError] = useState('');
  const [fileActionRecord, setFileActionRecord] = useState(null);
  const [fileActionType, setFileActionType] = useState('');
  const [fileActionBusy, setFileActionBusy] = useState(false);
  const [fileActionError, setFileActionError] = useState('');
  const [fileActionMessage, setFileActionMessage] = useState('');
  const [renameFilename, setRenameFilename] = useState('');
  const [moveFolders, setMoveFolders] = useState([]);
  const [moveDestination, setMoveDestination] = useState('');
  const [lifecycleResult, setLifecycleResult] = useState(null);
  const [lifecycleLoading, setLifecycleLoading] = useState(false);
  const [lifecycleError, setLifecycleError] = useState('');
  const [intelligenceRecord, setIntelligenceRecord] = useState(null);
  const [intelligenceLibraryId, setIntelligenceLibraryId] = useState('');

  const displayedImages = useMemo(() => {
    if (!scanResult?.image_records) {
      return [];
    }

    return scanResult.image_records.slice(0, MAX_VISIBLE_RECORDS);
  }, [scanResult]);

  const categoryGroups = useMemo(() => {
    const groups = Object.fromEntries(SEARCH_CATEGORIES.map((category) => [category, []]));
    for (const image of scanResult?.image_records ?? []) {
      const category = groups[image.category] ? image.category : 'Others';
      groups[category].push(image);
    }
    return { groups };
  }, [scanResult]);

  const refreshDashboard = useCallback(async (libraryPath = '') => {
    setDashboardLoading(true);
    setDashboardError('');
    try {
      const params = new URLSearchParams();
      if (libraryPath.trim()) {
        params.set('folder_path', libraryPath.trim());
      }
      const query = params.toString();
      const response = await fetch(`http://localhost:8000/dashboard/stats${query ? `?${query}` : ''}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, 'Unable to load dashboard statistics.'));
      }
      setDashboardStats(data);
    } catch (dashboardLoadError) {
      setDashboardError(getRequestError(dashboardLoadError, 'Dashboard statistics could not be loaded.'));
    } finally {
      setDashboardLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshDashboard();
  }, [refreshDashboard]);

  useEffect(() => {
    const handleHashChange = () => setCurrentPage(getPageFromHash());
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    if (!aiLoading) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setAiElapsedSeconds((elapsed) => elapsed + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [aiLoading]);

  const navigateToPage = (pageId) => {
    window.location.hash = pageId;
    setCurrentPage(pageId);
  };

  const showIntelligence = (record, libraryId) => {
    setIntelligenceRecord(record);
    setIntelligenceLibraryId(libraryId || record.library_id || scanResult?.library_id || '');
  };

  const imagePreview = (record, libraryId, className = '') => (
    <div className={`image-preview${className ? ` ${className}` : ''}`}>
      {getPreviewUrl(record, libraryId) ? (
        <img
          src={getPreviewUrl(record, libraryId)}
          alt={record.filename ? `Preview of ${record.filename}` : 'Image preview'}
          loading="lazy"
          onError={(event) => event.currentTarget.classList.add('preview-unavailable')}
        />
      ) : null}
      <span
        className={`image-preview-fallback${getPreviewUrl(record, libraryId) ? '' : ' preview-placeholder-visible'}`}
        aria-hidden="true"
      >
        Preview unavailable
      </span>
    </div>
  );

  const classifyScannedLibrary = async () => {
    const indexedLibraryPath = scanResult?.library_root || folderPath;
    const response = await fetch('http://localhost:8000/classification/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_path: indexedLibraryPath, force: false }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(getResponseError(data, 'Classification failed.'));
    }

    const classifications = new Map(
      (data.classifications ?? []).map((item) => [item.relative_path, item])
    );
    setScanResult((previous) => previous ? {
      ...previous,
      image_records: previous.image_records.map((image) => ({
        ...image,
        ...classifications.get(image.relative_path),
      })),
    } : previous);
    setClassificationResult(data);
    await refreshDashboard(indexedLibraryPath);
    return data;
  };

  const handleFolderSelect = async () => {
    setError('');
    setStatus('Opening the Windows folder picker...');
    try {
      const response = await fetch('http://localhost:8000/library/select-folder');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, 'Unable to open the folder picker.'));
      }
      if (data.cancelled || !data.folder_path) {
        setStatus('Folder selection cancelled.');
        return;
      }
      setFolderPath(data.folder_path);
      setScanResult(null);
      setAiResult(null);
      setClassificationResult(null);
      setSearchResult(null);
      setDuplicateResult(null);
      setLifecycleResult(null);
      setLifecycleError('');
      setStatus('Folder selected. Click Scan Library to continue.');
    } catch (folderError) {
      const message = getRequestError(folderError, 'Unable to open the folder picker.');
      setError(message);
      setStatus('Folder selection failed.');
    }
  };

  const handleScan = async () => {
    if (!folderPath.trim()) {
      setError('Enter the full local folder path before scanning.');
      return;
    }
    if (!/^(?:[a-zA-Z]:[\\/]|\\\\[^\\]+\\[^\\]+)/.test(folderPath.trim())) {
      setError('Enter the full absolute Windows folder path, for example C:\\Users\\YourName\\Pictures.');
      return;
    }

    setError('');
    setLoading(true);
    setStatus('Scanning library...');

    try {
      const response = await fetch('http://localhost:8000/library/scan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ folder_path: folderPath }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(getResponseError(data, 'Unable to scan the selected library.'));
      }

      setScanResult(data);
      setAiResult(null);
      setClassificationResult(null);
      setLifecycleResult(null);
      setLifecycleError('');
      void refreshDashboard(data.library_root || folderPath);
      setStatus(data.total_discovered_images
        ? 'Library scan complete.'
        : 'Scan complete. No supported images found in this folder.');
      setError(data.failed_files?.length
        ? `${data.failed_files.length} supported image file${data.failed_files.length === 1 ? '' : 's'} could not be read. Check file access and format.`
        : '');
    } catch (scanError) {
      setError(getRequestError(scanError, 'Unable to scan the selected library.'));
      setStatus('Scan failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleAiProcess = async () => {
    const indexedLibraryPath = scanResult?.library_root || folderPath;
    if (!indexedLibraryPath.trim() || !scanResult?.library_id) {
      setError('Enter a full local folder path and scan the library before processing images.');
      return;
    }

    const records = scanResult.image_records ?? [];
    if (!records.length) {
      setError('No scanned images are available to process. Scan a folder containing supported images first.');
      return;
    }

    setError('');
    setAiLoading(true);
    setAiElapsedSeconds(0);
    setAiProgress({ completed: 0, total: records.length });
    const aggregate = {
      profile: aiProfile,
      processed: 0,
      skipped: 0,
      failed: 0,
      errors: [],
      image_statuses: [],
    };
    let completedCount = 0;

    try {
      const batchSize = aiProfile === 'detailed' ? DETAILED_AI_BATCH_SIZE : AI_BATCH_SIZE;
      for (let offset = 0; offset < records.length; offset += batchSize) {
        const batch = records.slice(offset, offset + batchSize);
        setStatus(
          `Processing batch ${Math.floor(offset / batchSize) + 1} of ${Math.ceil(records.length / batchSize)}. The first run may take longer while local AI models load.`
        );
        const response = await fetch('http://localhost:8000/ai/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            folder_path: indexedLibraryPath,
            force: true,
            profile: aiProfile,
            record_ids: batch.map((image) => image.record_id || image.relative_path),
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(getResponseError(data, 'AI processing failed.'));
        }

        aggregate.processed += data.processed ?? 0;
        aggregate.skipped += data.skipped ?? 0;
        aggregate.failed += data.failed ?? 0;
        aggregate.errors.push(...(data.errors ?? []));
        aggregate.image_statuses.push(...(data.image_statuses ?? []));
        setAiResult({ ...aggregate });
        completedCount = Math.min(offset + batch.length, records.length);
        setAiProgress({ completed: completedCount, total: records.length });

        const statusByPath = new Map(
          (data.image_statuses ?? []).map((image) => [image.relative_path, image])
        );
        setScanResult((previous) => previous ? {
          ...previous,
          image_records: previous.image_records.map((image) => ({
            ...image,
            ...statusByPath.get(image.relative_path),
          })),
        } : previous);
      }

      try {
        const classification = await classifyScannedLibrary();
        setStatus(
          `Image analysis and categorization finished: ${classification.classified} assigned; ${classification.classifications.filter((image) => image.category === 'Others').length} assigned to Others.`
        );
      } catch (classificationError) {
        setStatus('Image analysis finished, but categorization could not be refreshed.');
        setError(getRequestError(classificationError, 'Images were processed, but classification failed.'));
      }
      if (aggregate.failed) {
        setError(`${aggregate.failed} image${aggregate.failed === 1 ? '' : 's'} could not be analyzed. Review the failure details below.`);
      }
    } catch (aiError) {
      setError(getRequestError(aiError, 'AI processing encountered an error.'));
      setStatus(
        `Processing stopped after ${completedCount} of ${records.length} images. Completed batches have been saved.`
      );
    } finally {
      setAiLoading(false);
    }
  };

  const handleCaptionProcess = async () => {
    const indexedLibraryPath = scanResult?.library_root || folderPath;
    if (!indexedLibraryPath.trim() || !scanResult?.library_id) {
      setError('Enter a full local folder path and scan the library before generating captions.');
      return;
    }

    const records = (scanResult.image_records ?? []).filter((image) => (
      image.caption_status !== 'completed' || !String(image.caption || '').trim()
    ));
    if (!records.length) {
      setStatus('Every available image in this scanned library already has a caption.');
      setError('');
      return;
    }

    setError('');
    setAiLoading(true);
    setAiElapsedSeconds(0);
    setAiProgress({ completed: 0, total: records.length });
    const aggregate = {
      profile: 'captions',
      processed: 0,
      skipped: 0,
      failed: 0,
      errors: [],
      image_statuses: [],
    };
    setAiResult({ ...aggregate });
    let completedCount = 0;
    const batchSize = DETAILED_AI_BATCH_SIZE;

    try {
      for (let offset = 0; offset < records.length; offset += batchSize) {
        const batch = records.slice(offset, offset + batchSize);
        setStatus(
          `Generating captions for batch ${Math.floor(offset / batchSize) + 1} of ${Math.ceil(records.length / batchSize)}. Captions and progress are saved after each batch.`
        );
        const response = await fetch('http://localhost:8000/ai/captions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            folder_path: indexedLibraryPath,
            record_ids: batch.map((image) => image.record_id || image.relative_path),
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(getResponseError(data, 'Caption generation failed.'));
        }

        aggregate.processed += data.processed ?? 0;
        aggregate.skipped += data.skipped ?? 0;
        aggregate.failed += data.failed ?? 0;
        aggregate.errors.push(...(data.errors ?? []));
        aggregate.image_statuses.push(...(data.image_statuses ?? []));
        setAiResult({ ...aggregate });
        completedCount = Math.min(offset + batch.length, records.length);
        setAiProgress({ completed: completedCount, total: records.length });

        const captionByPath = new Map(
          (data.image_statuses ?? []).map((image) => [image.relative_path, image])
        );
        setScanResult((previous) => previous ? {
          ...previous,
          image_records: previous.image_records.map((image) => ({
            ...image,
            ...captionByPath.get(image.relative_path),
          })),
        } : previous);
      }

      try {
        const classification = await classifyScannedLibrary();
        const assignedOthers = classification.classifications.filter(
          (image) => image.category === 'Others'
        ).length;
        setStatus(
          `Captions ready for ${aggregate.processed} images; ${classification.classified} assigned to categories, ${assignedOthers} assigned to Others.`
        );
      } catch (classificationError) {
        setStatus('Caption generation finished, but categorization could not be refreshed.');
        setError(getRequestError(classificationError, 'Captions were saved, but classification failed.'));
      }
      if (aggregate.failed) {
        setError(`${aggregate.failed} image${aggregate.failed === 1 ? '' : 's'} did not produce a caption. Check file access or the local caption model, then retry.`);
      }
    } catch (captionError) {
      setError(getRequestError(captionError, 'Caption generation encountered an error.'));
      setStatus(
        `Caption generation stopped after ${completedCount} of ${records.length} images. Completed batches have been saved.`
      );
    } finally {
      setAiLoading(false);
    }
  };

  const handleClassification = async () => {
    const indexedLibraryPath = scanResult?.library_root || folderPath;
    if (!indexedLibraryPath.trim() || !scanResult?.library_id) {
      setError('Enter a full local folder path and scan the library before classifying images.');
      return;
    }

    setError('');
    setClassificationLoading(true);
    setStatus('Classifying images...');

    try {
      const records = scanResult.image_records ?? [];
      const hasVisualEvidence = records.some((image) => (
        image.embedding_status === 'completed' && image.embedding_path
      ));
      if (!hasVisualEvidence) {
        setStatus('Visual analysis is required before categories can be assigned...');
        await handleAiProcess();
        return;
      }

      const data = await classifyScannedLibrary();
      const assignedOthers = data.classifications.filter((image) => image.category === 'Others').length;
      setStatus(
        `Categorization complete: ${data.classified} assigned; ${assignedOthers} assigned to Others.`
      );
    } catch (classificationError) {
      setError(getRequestError(classificationError, 'Classification encountered an error.'));
      setStatus('Classification failed.');
    } finally {
      setClassificationLoading(false);
    }
  };

  const handleSearch = async (event) => {
    event.preventDefault();
    if (!searchQuery.trim()) {
      setError('Enter a search query.');
      return;
    }

    setError('');
    setSearchLoading(true);
    setStatus('Searching your local image index...');

    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          folder_path: folderPath.trim() || null,
          category: searchCategory || null,
          limit: 50,
          use_llm: true,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, 'Search failed.'));
      }
      setSearchResult(data);
      setStatus(data.reranker?.status === 'used'
        ? `Search complete; reranked locally with ${data.reranker.model}.`
        : 'Search complete using local hybrid retrieval.');
    } catch (searchError) {
      setError(getRequestError(searchError, 'Search encountered an error.'));
      setStatus('Search failed.');
    } finally {
      setSearchLoading(false);
    }
  };

  const checkLifecycle = async () => {
    if (!scanResult?.library_id) {
      setLifecycleError('Scan a local library before checking its lifecycle status.');
      return;
    }
    setLifecycleLoading(true);
    setLifecycleError('');
    setStatus('Checking library files...');
    try {
      const response = await fetch('http://localhost:8000/lifecycle/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ library_id: scanResult.library_id }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, 'Unable to check library lifecycle.'));
      }
      setLifecycleResult(data);
      setStatus('Library check complete.');
      const changed = new Map(
        data.records.map((record) => [record.relative_path, record.status])
      );
      setScanResult((previous) => previous ? {
        ...previous,
        image_records: previous.image_records.map((image) => (
          changed.has(image.relative_path)
            ? { ...image, file_status: changed.get(image.relative_path) }
            : image
        )),
      } : previous);
      setSearchResult((previous) => previous ? {
        ...previous,
        results: previous.results.filter((image) => (
          !changed.has(image.relative_path)
          || !['missing', 'modified', 'unverified'].includes(changed.get(image.relative_path))
        )),
      } : previous);
      if (data.changed_records.length || data.possible_relocations.length) {
        setDuplicateResult(null);
      }
      void refreshDashboard(folderPath);
    } catch (lifecycleCheckError) {
      setLifecycleError(getRequestError(lifecycleCheckError, 'Lifecycle check failed.'));
    } finally {
      setLifecycleLoading(false);
    }
  };

  const applyRelocation = async (item, newRelativePath) => {
    const confirmed = window.confirm(
      `Reconcile this image?\n\nOld path: ${item.relative_path}\nDetected path: ${newRelativePath}\n\nThis updates the local index path only. It will not move or rename the file.`
    );
    if (!confirmed) {
      return;
    }
    setLifecycleLoading(true);
    setLifecycleError('');
    setStatus('Reconciling file location...');
    try {
      const response = await fetch('http://localhost:8000/lifecycle/reconcile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          library_id: scanResult.library_id,
          record_id: item.record_id,
          new_relative_path: newRelativePath,
          confirm: true,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, 'Unable to reconcile relocation.'));
      }
      setScanResult((previous) => previous ? {
        ...previous,
        image_records: previous.image_records.map((image) => (
          image.record_id === data.previous_record_id
            ? { ...image, ...data.record }
            : image
        )),
      } : previous);
      setSearchResult((previous) => previous ? {
        ...previous,
        results: previous.results.filter((image) => (
          image.library_id !== scanResult.library_id
          || image.record_id !== data.previous_record_id
        )),
      } : previous);
      setDuplicateResult(null);
      await checkLifecycle();
      if (searchQuery.trim()) {
        await handleSearch({ preventDefault() {} });
      }
      setStatus(data.message);
      void refreshDashboard(folderPath);
    } catch (reconcileError) {
      setLifecycleError(getRequestError(reconcileError, 'Relocation reconciliation failed.'));
    } finally {
      setLifecycleLoading(false);
    }
  };

  const openFileAction = async (action, record, libraryId) => {
    if (!libraryId || !record?.record_id) {
      setError('This image does not have an indexed file reference.');
      return;
    }
    setFileActionError('');
    setFileActionMessage('');
    setFileActionRecord({
      library_id: libraryId,
      record_id: record.record_id,
      filename: record.filename,
      relative_path: record.relative_path,
    });
    setFileActionType(action);
    if (action === 'rename') {
      setRenameFilename(record.filename);
    }
    if (action === 'move') {
      setFileActionBusy(true);
      try {
        const params = new URLSearchParams({
          library_id: libraryId,
          record_id: record.record_id,
        });
        const response = await fetch(`http://localhost:8000/files/folders?${params}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(getResponseError(data, 'Unable to load library folders.'));
        }
        setMoveFolders(data.folders ?? []);
        setMoveDestination('');
      } catch (folderError) {
        setFileActionError(getRequestError(folderError, 'Unable to load library folders.'));
      } finally {
        setFileActionBusy(false);
      }
    }
  };

  const closeFileAction = () => {
    if (fileActionBusy) {
      return;
    }
    setFileActionRecord(null);
    setFileActionType('');
    setFileActionError('');
  };

  const runFileAction = async (action, record) => {
    setError('');
    setFileActionMessage('');
    setFileActionBusy(true);
    const actionLabels = {
      open: 'Opening image...',
      'open-folder': 'Opening image folder...',
      rename: 'Renaming image...',
      move: 'Moving image...',
      delete: 'Deleting image...',
    };
    setStatus(actionLabels[action] || 'Working...');
    try {
      const response = await fetch(`http://localhost:8000/files/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(record),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, `Unable to ${action} image.`));
      }
      if (['rename', 'move', 'delete'].includes(action)) {
        const oldRecordId = data.previous_record_id;
        setScanResult((previous) => previous ? {
          ...previous,
          image_records: previous.image_records.map((image) => (
            previous.library_id === record.library_id && image.record_id === oldRecordId
              ? { ...image, ...data.record }
              : image
          )),
        } : previous);
        setSearchResult((previous) => previous ? {
          ...previous,
          results: previous.results.filter((image) => (
            image.library_id !== record.library_id || image.record_id !== oldRecordId
          )),
          returned_results: Math.max(0, previous.returned_results - 1),
          total_results: Math.max(0, previous.total_results - 1),
        } : previous);
        setDuplicateResult(null);
        setLifecycleResult(null);
        void refreshDashboard(folderPath);
      }
      setFileActionMessage(data.message || 'File operation completed.');
      setStatus(data.message || 'File operation completed.');
      setFileActionRecord(null);
      setFileActionType('');
    } catch (fileError) {
      const message = getRequestError(fileError, 'File operation failed.');
      setFileActionError(message);
      setError(message);
      setStatus('File operation failed.');
    } finally {
      setFileActionBusy(false);
    }
  };

  const submitFileAction = (event) => {
    event.preventDefault();
    if (!fileActionRecord) {
      return;
    }
    if (fileActionType === 'rename') {
      void runFileAction('rename', {
        library_id: fileActionRecord.library_id,
        record_id: fileActionRecord.record_id,
        new_filename: renameFilename,
      });
    } else if (fileActionType === 'move') {
      void runFileAction('move', {
        library_id: fileActionRecord.library_id,
        record_id: fileActionRecord.record_id,
        destination_folder: moveDestination,
      });
    } else if (fileActionType === 'delete') {
      void runFileAction('delete', {
        library_id: fileActionRecord.library_id,
        record_id: fileActionRecord.record_id,
        confirm: true,
      });
    }
  };

  const fileActions = (record, libraryId) => (
    record.file_status === 'missing' ? null : (
    <div className="file-actions" aria-label={`File actions for ${record.filename}`}>
      <button type="button" className="file-action-button" disabled={fileActionBusy} onClick={() => void runFileAction('open', { library_id: libraryId, record_id: record.record_id })}>Open Image</button>
      <button type="button" className="file-action-button" disabled={fileActionBusy} onClick={() => void runFileAction('open-folder', { library_id: libraryId, record_id: record.record_id })}>Open Folder</button>
      <button type="button" className="file-action-button" disabled={fileActionBusy} onClick={() => void openFileAction('rename', record, libraryId)}>Rename</button>
      <button type="button" className="file-action-button" disabled={fileActionBusy} onClick={() => void openFileAction('move', record, libraryId)}>Move</button>
      <button type="button" className="file-action-button danger-action" disabled={fileActionBusy} onClick={() => void openFileAction('delete', record, libraryId)}>Delete</button>
    </div>
    )
  );

  const handleDuplicateDetection = async () => {
    if (!folderPath.trim() || !scanResult?.library_id) {
      setError('Enter a full local folder path and scan the library before checking duplicates.');
      return;
    }

    setError('');
    setDuplicateLoading(true);
    setStatus('Checking duplicates...');
    try {
      const response = await fetch('http://localhost:8000/duplicates/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folderPath, include_visual: true, force: false }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(getResponseError(data, 'Duplicate detection failed.'));
      }
      setDuplicateResult(data);
      setStatus('Duplicate detection complete. No files were changed or deleted.');
    } catch (duplicateError) {
      setError(getRequestError(duplicateError, 'Duplicate detection encountered an error.'));
      setStatus('Duplicate detection failed.');
    } finally {
      setDuplicateLoading(false);
    }
  };

  const stageProfile = aiLoading
    ? (aiResult?.profile === 'captions' ? 'captions' : aiProfile)
    : (aiResult?.profile || aiProfile);

  return (
    <main className="app-shell">
      <section className="library-panel">
        <header className="panel-header">
          <div>
            <h1>MemoryOS</h1>
            <p className="eyebrow app-tagline">AI-Powered Local Image Intelligence &amp; Retrieval</p>
            <p className="app-subtitle">Your private image library, organized one simple step at a time.</p>
          </div>
        </header>
        <div className="privacy-note">
          Your original images stay in their folders. MemoryOS stores local metadata, AI results, and embeddings.
        </div>

        <nav className="app-navigation" aria-label="Main navigation">
          {PAGE_LINKS.map((page) => (
            <button
              type="button"
              key={page.id}
              className={`navigation-link${currentPage === page.id ? ' navigation-link-active' : ''}`}
              aria-current={currentPage === page.id ? 'page' : undefined}
              onClick={() => navigateToPage(page.id)}
            >
              {page.label}
            </button>
          ))}
        </nav>

        <div className="status-bar" role="status" aria-live="polite">
          <span className="status-label">Status:</span>
          <span>{status}</span>
        </div>

        {error ? <div className="error-box">{error}</div> : null}
        {lifecycleError ? <div className="error-box" role="alert">{lifecycleError}</div> : null}
        {fileActionError && !fileActionRecord ? <div className="error-box" role="alert">{fileActionError}</div> : null}

        {currentPage === 'health' ? (
          <section className="page-section" aria-labelledby="health-page-heading">
            <div className="page-heading">
              <p className="eyebrow">Check your files</p>
              <h2 id="health-page-heading">File Health</h2>
              <p>Check whether indexed pictures are still in their original locations. This does not move or modify your files.</p>
            </div>
            <button
              type="button"
              className="primary-button"
              onClick={checkLifecycle}
              disabled={lifecycleLoading || !scanResult?.library_id}
            >
              {lifecycleLoading ? 'Checking files...' : 'Check file availability'}
            </button>
            {!scanResult?.library_id ? (
              <p className="helper-text">Scan a library first to check file health.</p>
            ) : null}
          </section>
        ) : null}

        {currentPage === 'library' ? (
          <section className="page-section" aria-labelledby="library-page-heading">
            <div className="page-heading">
              <p className="eyebrow">Step 1 · Choose and scan</p>
              <h2 id="library-page-heading">My Library</h2>
              <p>Tell MemoryOS where your pictures are, then scan to create a local index. Your original files stay where they are.</p>
            </div>
            <div className="notice-box">
              Click Select Folder to open the Windows folder picker. Your original images stay in their existing folder.
            </div>
            <div className="toolbar">
              <button type="button" className="secondary-button" onClick={() => void handleFolderSelect()} disabled={loading}>
                Upload Folder
              </button>
              <label className="folder-input">
                <span>Full library folder path</span>
                <input
                  type="text"
                  value={folderPath}
                  onChange={(event) => {
                    setFolderPath(event.target.value);
                    setError('');
                    setScanResult(null);
                    setAiResult(null);
                    setClassificationResult(null);
                    setSearchResult(null);
                    setDuplicateResult(null);
                    setLifecycleResult(null);
                    setLifecycleError('');
                  }}
                  placeholder="C:\\Users\\YourName\\Pictures"
                />
              </label>
              <button type="button" className="primary-button" onClick={handleScan} disabled={loading}>
                {loading ? 'Scanning...' : 'Scan Library'}
              </button>
            </div>
          </section>
        ) : null}

        {currentPage === 'health' && lifecycleResult ? (
          <section className="lifecycle-section" aria-live="polite">
            <div className="summary-panel">
              <div><span className="summary-label">Available:</span><strong>{lifecycleResult.available}</strong></div>
              <div><span className="summary-label">Missing:</span><strong>{lifecycleResult.missing}</strong></div>
              <div><span className="summary-label">Modified:</span><strong>{lifecycleResult.modified}</strong></div>
              <div><span className="summary-label">Unverified:</span><strong>{lifecycleResult.unverified}</strong></div>
              <div><span className="summary-label">Relocation candidates:</span><strong>{lifecycleResult.relocations_detected}</strong></div>
            </div>
            {lifecycleResult.records.filter((record) => (
              ['missing', 'modified', 'unverified'].includes(record.status)
            )).length === 0 && lifecycleResult.possible_relocations.length === 0 ? (
                <div className="notice-box">All indexed files are available and unchanged.</div>
              ) : null}
            {lifecycleResult.records.some((record) => (
              ['missing', 'modified', 'unverified'].includes(record.status)
            )) ? (
              <div className="lifecycle-changes">
                {lifecycleResult.records.filter((record) => (
                  ['missing', 'modified', 'unverified'].includes(record.status)
                )).map((record) => (
                  <article className="lifecycle-record" key={`${record.record_id}:${record.status}`}>
                    <strong>{record.filename}</strong>
                    <span>{record.relative_path}</span>
                    <span className={`lifecycle-status status-${record.status}`}>Status: {record.status}</span>
                    {record.status === 'missing' ? (
                      <span>File is missing from its original location. MemoryOS did not delete it.</span>
                    ) : null}
                    {record.status === 'modified' ? (
                      <span>File modified outside MemoryOS. Existing AI outputs may be stale; lifecycle checking did not reprocess it.</span>
                    ) : null}
                    {record.status === 'unverified' ? (
                      <span>File status could not be verified safely. Rescan the library before using it.</span>
                    ) : null}
                  </article>
                ))}
              </div>
            ) : null}
            {lifecycleResult.possible_relocations.map((item) => (
              <article className="lifecycle-record relocation-record" key={item.record_id}>
                <strong>Possible relocation detected by exact content match.</strong>
                <span>MemoryOS found another file with the exact same SHA-256.</span>
                <span>Old path: {item.relative_path}</span>
                {item.possible_relocations.map((newPath) => (
                  <div className="relocation-candidate" key={newPath}>
                    <span>Detected path: {newPath}</span>
                    <span>Confirming updates the indexed path only. The physical file will not be moved.</span>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={lifecycleLoading}
                      onClick={() => void applyRelocation(item, newPath)}
                    >
                      Reconcile
                    </button>
                  </div>
                ))}
              </article>
            ))}
          </section>
        ) : null}

        {currentPage === 'dashboard' ? <section className="dashboard-section" aria-labelledby="dashboard-heading">
          <div className="dashboard-header">
            <div>
              <p className="eyebrow">Local Index Analytics</p>
              <h2 id="dashboard-heading">MemoryOS Dashboard</h2>
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={() => refreshDashboard(folderPath)}
              disabled={dashboardLoading}
            >
              {dashboardLoading ? 'Refreshing...' : 'Refresh Dashboard'}
            </button>
          </div>
          {dashboardError ? <div className="error-box" role="alert">{dashboardError}</div> : null}
          {!dashboardError && dashboardStats ? (
            dashboardStats.total_images === 0 ? (
              <div className="empty-state dashboard-empty">
                <strong>No images indexed yet.</strong>
                <span>Enter the full path to your image folder, then scan your library to get started.</span>
                <button type="button" className="primary-button" onClick={() => navigateToPage('library')}>
                  Set up my library
                </button>
              </div>
            ) : (
              <>
                <div className="dashboard-section-intro">
                  <p className="eyebrow">Memory overview</p>
                  <p>What MemoryOS has indexed, analyzed, categorized, and flagged for attention.</p>
                </div>
                <div className="dashboard-kpis">
                  <div className="dashboard-kpi"><span>Total indexed</span><strong>{dashboardStats.total_images}</strong></div>
                  <div className="dashboard-kpi"><span>AI processed</span><strong>{dashboardStats.processed_images}</strong></div>
                  <div className="dashboard-kpi"><span>Assigned to Others</span><strong>{dashboardStats.categories.distribution.find((item) => item.category === 'Others')?.count ?? 0}</strong></div>
                  <div className="dashboard-kpi"><span>Duplicate groups</span><strong>{dashboardStats.duplicates.groups}</strong></div>
                  <div className="dashboard-kpi"><span>Files needing attention</span><strong>{dashboardStats.file_status.missing + dashboardStats.file_status.modified + dashboardStats.file_status.unverified}</strong></div>
                </div>

                <div className="dashboard-grid">
                  <section className="dashboard-card">
                    <h3>AI Health</h3>
                    {Object.entries(dashboardStats.processing).map(([label, count]) => (
                      <div className="dashboard-bar-row" key={`ai-${label}`}>
                        <span>{label.replace('_', ' ')}</span>
                        <div className="dashboard-bar-track"><div className="dashboard-bar-fill" style={{ width: `${dashboardStats.total_images ? count * 100 / dashboardStats.total_images : 0}%` }} /></div>
                        <strong>{count}</strong>
                      </div>
                    ))}
                    <h3 className="dashboard-subheading">Classification Status</h3>
                    {Object.entries(dashboardStats.classification).filter(([label]) => label !== 'needs_review').map(([label, count]) => (
                      <div className="dashboard-bar-row" key={`classification-${label}`}>
                        <span>{label.replace('_', ' ')}</span>
                        <div className="dashboard-bar-track"><div className="dashboard-bar-fill classification-fill" style={{ width: `${dashboardStats.total_images ? count * 100 / dashboardStats.total_images : 0}%` }} /></div>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </section>

                  <section className="dashboard-card">
                    <h3>Memory Categories</h3>
                    <p className="dashboard-caption">
                      {dashboardStats.categories.categorized_images} assigned categories out of {dashboardStats.total_images} indexed images. Images with weak or conflicting evidence are assigned to Others; {dashboardStats.classification.pending} are not classified yet.
                    </p>
                    {!dashboardStats.categories.categorized_images ? (
                      <div className="empty-state">
                        <strong>No categories assigned yet.</strong>
                        <span>Run image analysis and categorization in AI Workspace. Images without enough evidence are assigned to Others.</span>
                        <button type="button" className="secondary-button" onClick={() => navigateToPage('processing')}>
                          Open AI Tools
                        </button>
                      </div>
                    ) : null}
                    {[...dashboardStats.categories.distribution]
                      .sort((left, right) => right.count - left.count || left.category.localeCompare(right.category))
                      .map((item) => (
                      <div className="dashboard-bar-row category-row" key={item.category}>
                        <span>{item.category}</span>
                        <div className="dashboard-bar-track"><div className="dashboard-bar-fill category-fill" style={{ width: `${item.percentage}%` }} /></div>
                        <strong>{item.count}</strong>
                      </div>
                    ))}
                  </section>

                  <section className="dashboard-card">
                    <h3>File Health</h3>
                    {Object.entries(dashboardStats.file_status).map(([label, count]) => (
                      <div className="dashboard-bar-row" key={`file-${label}`}>
                        <span>{label}</span>
                        <div className="dashboard-bar-track"><div className="dashboard-bar-fill file-fill" style={{ width: `${dashboardStats.total_images ? count * 100 / dashboardStats.total_images : 0}%` }} /></div>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </section>

                  <section className="dashboard-card">
                    <h3>Duplicate Overview</h3>
                    <div className="dashboard-duplicate-stats">
                      <div><span>Exact groups</span><strong>{dashboardStats.duplicates.exact_groups}</strong></div>
                      <div><span>Near groups</span><strong>{dashboardStats.duplicates.near_duplicate_groups}</strong></div>
                      <div><span>Visual groups</span><strong>{dashboardStats.duplicates.visual_duplicate_groups}</strong></div>
                      <div><span>Records involved</span><strong>{dashboardStats.duplicates.records_in_groups}</strong></div>
                    </div>
                  </section>

                  <section className="dashboard-card dashboard-libraries">
                    <h3>Libraries ({dashboardStats.library_count})</h3>
                    {dashboardStats.libraries.length ? dashboardStats.libraries.map((library) => (
                      <div className="dashboard-library-row" key={library.library_id}>
                        <strong>{library.display_name}</strong>
                        <span>{library.total_images} images · {library.processed_images} AI processed · {library.missing_files} missing</span>
                      </div>
                    )) : <p className="dashboard-caption">No library details are available.</p>}
                  </section>
                </div>
                <div className="dashboard-shortcuts" aria-label="MemoryOS tools">
                  {PAGE_LINKS.filter((page) => page.id !== 'dashboard').map((page) => (
                    <button type="button" key={page.id} className="secondary-button" onClick={() => navigateToPage(page.id)}>
                      Open {page.label}
                    </button>
                  ))}
                </div>
              </>
            )
          ) : null}
        </section> : null}

        {currentPage === 'library' && scanResult ? (
          <div className="summary-panel">
            <div>
              <span className="summary-label">Library:</span>
              <strong>{scanResult.library_id}</strong>
            </div>
            <div>
              <span className="summary-label">Images:</span>
              <strong>{scanResult.total_discovered_images}</strong>
            </div>
            <div>
              <span className="summary-label">New:</span>
              <strong>{scanResult.new_images}</strong>
            </div>
            <div>
              <span className="summary-label">Updated:</span>
              <strong>{scanResult.updated_images}</strong>
            </div>
            <div>
              <span className="summary-label">Unchanged:</span>
              <strong>{scanResult.unchanged_images}</strong>
            </div>
          </div>
        ) : null}

        {currentPage === 'processing' ? (
          <section className="page-section" aria-labelledby="processing-page-heading">
            <div className="page-heading">
              <p className="eyebrow">Step 2 · Understand your pictures</p>
              <h2 id="processing-page-heading">AI Workspace</h2>
              <p>Analyze, categorize, and review the evidence MemoryOS found in your images.</p>
            </div>
            <div className="notice-box">
              MemoryOS analyzes images on this computer. Detailed analysis runs OCR, captions, and visual embeddings. You can also generate only missing captions; ambiguous or weak classifications are assigned to Others rather than left for review.
            </div>
            <section className="pipeline-panel" aria-label="MemoryOS AI pipeline">
              <div className="pipeline-heading">
                <div>
                  <p className="eyebrow">MemoryOS AI pipeline</p>
                  <h3>{stageProfile === 'captions'
                    ? 'Image caption generation'
                    : stageProfile === 'fast' ? 'Fast visual analysis' : 'Detailed image analysis'}</h3>
                </div>
                <span className="pipeline-profile">{stageProfile === 'captions'
                  ? 'Captions only'
                  : stageProfile === 'fast' ? 'Fast profile' : 'Detailed profile'}</span>
              </div>
              <div className="pipeline-stages">
                {[
                  { label: 'Image', value: 'Scanned locally', active: Boolean(scanResult?.library_id) },
                  {
                    label: 'OCR',
                    value: stageProfile === 'captions'
                      ? 'Not run in captions-only mode'
                      : stageProfile === 'fast'
                      ? 'Not run in Fast mode'
                      : aiResult?.image_statuses?.length
                        ? `${aiResult.image_statuses.filter((item) => item.ocr_status === 'completed').length} completed`
                        : 'Runs in Detailed mode',
                    inactive: stageProfile === 'fast' || stageProfile === 'captions',
                  },
                  {
                    label: 'Caption',
                    value: stageProfile === 'fast'
                      ? 'Not run in Fast mode'
                      : aiResult?.image_statuses?.length
                        ? `${aiResult.image_statuses.filter((item) => item.caption_status === 'completed').length} completed`
                        : 'Runs in Detailed mode',
                    inactive: stageProfile === 'fast',
                  },
                  {
                    label: 'Visual embedding',
                    value: stageProfile === 'captions'
                      ? 'Not run in captions-only mode'
                      : aiResult?.image_statuses?.length
                      ? `${aiResult.image_statuses.filter((item) => item.embedding_status === 'completed').length} completed`
                      : 'Runs in both profiles',
                  },
                  {
                    label: 'Classification',
                    value: classificationResult
                      ? `${classificationResult.classified} assigned · ${classificationResult.classifications.filter((image) => image.category === 'Others').length} assigned to Others`
                      : aiResult
                        ? 'Classification result unavailable'
                        : 'Runs after analysis',
                  },
                ].map((stage, index, stages) => (
                  <div className={`pipeline-stage${stage.inactive ? ' pipeline-stage-inactive' : ''}`} key={stage.label}>
                    <span className="pipeline-step">{index + 1}</span>
                    <strong>{stage.label}</strong>
                    <span>{stage.value}</span>
                    {index < stages.length - 1 ? <span className="pipeline-connector" aria-hidden="true" /> : null}
                  </div>
                ))}
              </div>
              {!aiLoading && aiResult && aiResult.profile !== 'captions' && aiProfile !== aiResult.profile ? (
                <p className="pipeline-footnote">
                  The displayed stage results are from the last {stageProfile === 'fast' ? 'Fast' : 'Detailed'} run. The selected profile applies to the next run.
                </p>
              ) : null}
              {aiResult?.image_statuses?.length ? (
                <p className="pipeline-footnote">
                  Stage counts reflect the latest AI processing response; classification totals are from its separate result.
                </p>
              ) : null}
            </section>
            <label className="processing-profile">
              <span>Processing mode</span>
              <select value={aiProfile} onChange={(event) => setAiProfile(event.target.value)} disabled={aiLoading}>
                <option value="fast">Fast visual categories (recommended)</option>
                <option value="detailed">Detailed OCR, captions, and visual search</option>
              </select>
            </label>
            <div className="toolbar">
              <button
                type="button"
                className="primary-button"
                onClick={handleAiProcess}
                disabled={aiLoading || loading || !scanResult?.library_id}
              >
                {aiLoading ? 'Analyzing images...' : aiProfile === 'fast' ? 'Quick analyze and categorize' : 'Detailed analysis'}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={handleClassification}
                disabled={classificationLoading || aiLoading || !scanResult?.library_id}
              >
                {classificationLoading ? 'Classifying...' : 'Classify images'}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={handleCaptionProcess}
                disabled={aiLoading || loading || !scanResult?.library_id}
              >
                {aiLoading && aiResult?.profile === 'captions' ? 'Generating captions...' : 'Generate missing captions'}
              </button>
              {!scanResult?.library_id ? (
                <button type="button" className="secondary-button" onClick={() => navigateToPage('library')}>
                  Go to My Library
                </button>
              ) : null}
            </div>
            {aiProgress ? (
              <div className="processing-progress" role="status" aria-live="polite">
                <div className="processing-progress-heading">
                  <strong>{aiLoading ? 'Processing your library' : 'Last processing run'}</strong>
                  <span>{aiProgress.completed} of {aiProgress.total} images checked</span>
                </div>
                <progress value={aiProgress.completed} max={aiProgress.total} aria-label="AI processing progress" />
                {aiLoading ? (
                  <span className="helper-text">
                        Time elapsed: {Math.floor(aiElapsedSeconds / 60)}m {aiElapsedSeconds % 60}s. Processing {aiProfile === 'detailed' ? DETAILED_AI_BATCH_SIZE : AI_BATCH_SIZE} images per saved batch; keep this page open.
                  </span>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : null}

        {currentPage === 'processing' && aiResult ? (
          <>
          <div className="summary-panel">
            <div>
              <span className="summary-label">Processed:</span>
              <strong>{aiResult.processed}</strong>
            </div>
            <div>
              <span className="summary-label">Skipped:</span>
              <strong>{aiResult.skipped}</strong>
            </div>
            <div>
              <span className="summary-label">Failed:</span>
              <strong>{aiResult.failed}</strong>
            </div>
          </div>
          {aiResult.failed > 0 && aiResult.errors?.length ? (
            <div className="error-box" role="alert">
              <strong>Some images could not be processed:</strong>
              <ul>
                {aiResult.errors.map((item) => (
                  <li key={`${item.relative_path}:${item.error}`}>
                    {item.relative_path}: {item.error}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          </>
        ) : null}

        {currentPage === 'processing' && classificationResult ? (
          <div className="summary-panel">
            <div>
              <span className="summary-label">Assigned:</span>
              <strong>{classificationResult.classified}</strong>
            </div>
            <div>
              <span className="summary-label">Assigned to Others:</span>
              <strong>{classificationResult.classifications.filter((image) => image.category === 'Others').length}</strong>
            </div>
            <div>
              <span className="summary-label">Failed:</span>
              <strong>{classificationResult.failed}</strong>
            </div>
            <div>
              <span className="summary-label">Skipped:</span>
              <strong>{classificationResult.skipped}</strong>
            </div>
          </div>
        ) : null}

        {currentPage === 'processing' && classificationResult && scanResult?.image_records?.length ? (
          <section className="page-section" aria-labelledby="category-distribution-heading">
            <div className="page-heading">
              <p className="eyebrow">Classification overview</p>
              <h2 id="category-distribution-heading">Category distribution</h2>
              <p>Each available image is assigned to its strongest supported category. Ambiguous, conflicting, or weak evidence is assigned to Others.</p>
            </div>
            <div className="category-distribution">
              {Object.entries(scanResult.image_records.reduce((counts, image) => {
                const label = image.category || 'Others';
                counts[label] = (counts[label] || 0) + 1;
                return counts;
              }, {})).sort(([, left], [, right]) => right - left).map(([label, count]) => (
                <div className="category-distribution-item" key={label}>
                  <strong>{label}</strong>
                  <span>{count} image{count === 1 ? '' : 's'}</span>
                </div>
              ))}
            </div>
            <div className="category-gallery" aria-label="Images grouped by category">
              {SEARCH_CATEGORIES.map((category) => {
                const images = categoryGroups.groups[category];
                return (
                  <article className={`category-gallery-card${images.length ? '' : ' category-gallery-card-empty'}`} key={category}>
                    <div className="category-gallery-heading">
                      <div>
                        <h3>{category}</h3>
                        <span>{images.length} image{images.length === 1 ? '' : 's'}</span>
                      </div>
                    </div>
                    {images.length ? (
                      <div className="category-gallery-grid">
                        {images.map((image) => (
                          <button
                            type="button"
                            className="category-gallery-image"
                            key={image.relative_path}
                            onClick={() => showIntelligence(image, scanResult.library_id)}
                            title={`View intelligence for ${image.filename}`}
                          >
                            {imagePreview(image, scanResult.library_id)}
                            <span>{image.filename}</span>
                            {getTopCategoryShare(image) !== null ? (
                              <small>{getTopCategoryShare(image).toFixed(0)}% of top-two category scores</small>
                            ) : null}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="category-gallery-empty">No classified images yet.</p>
                    )}
                  </article>
                );
              })}
            </div>
          </section>
        ) : null}

        {currentPage === 'search' ? (
          <section className="page-section" aria-labelledby="search-page-heading">
            <div className="page-heading">
              <p className="eyebrow">Find a picture</p>
              <h2 id="search-page-heading">Search</h2>
              <p>Search using words from a filename, text in an image, or a short description of what you remember.</p>
            </div>
          </section>
        ) : null}

        {currentPage === 'search' ? <form className="search-panel" onSubmit={handleSearch}>
          <label className="search-query">
            <span>Search your indexed images</span>
            <input
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search: passport, receipt, college notes, travel..."
            />
          </label>
          <label className="search-category">
            <span>Category filter</span>
            <select value={searchCategory} onChange={(event) => setSearchCategory(event.target.value)}>
              <option value="">All categories</option>
              {SEARCH_CATEGORIES.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <button type="submit" className="primary-button" disabled={searchLoading}>
            {searchLoading ? 'Searching...' : 'Search'}
          </button>
        </form> : null}

        {currentPage === 'search' && searchResult ? (
          <section className="search-results" aria-live="polite">
            <div className="search-explanation">
              MemoryOS retrieves candidates locally using visual embeddings and image text, then reranks the best matches with a local LLM.
            </div>
            {searchResult.reranker
              && searchResult.reranker.status !== 'used'
              && !['skipped', 'disabled'].includes(searchResult.reranker.status) ? (
              <div className="notice-box" role="status">
                Local LLM reranking did not complete; results are ranked with local visual and text evidence.
                {searchResult.reranker.detail ? ` ${searchResult.reranker.detail}` : ''}
              </div>
            ) : searchResult.reranker?.status === 'used' ? (
              <div className="notice-box" role="status">
                Results reranked on this computer with {searchResult.reranker.model}. No image files or search data were sent to a cloud service.
              </div>
            ) : null}
            <div className="grid-meta">
              Showing {searchResult.returned_results} of {searchResult.total_results} ranked results for “{searchResult.query}”
            </div>
            {searchResult.results.length ? searchResult.results.map((result) => (
              <article className="search-result-card" key={`${result.library_id}:${result.relative_path}`}>
                {imagePreview(result, result.library_id)}
                <div className="image-meta">
                  <strong>{result.filename}</strong>
                  <span>{result.relative_path}</span>
                  <div className="image-badges">
                    <span className="image-badge">
                      {result.category || 'Others'}
                    </span>
                    {result.file_status ? <span className="image-badge badge-muted">{result.file_status}</span> : null}
                  </div>
                  <div className="why-matched">
                    <strong>Why this matched</strong>
                    <span>
                      {result.matched_fields?.length
                        ? result.matched_fields.map((field) => ({
                          filename: 'Filename',
                          ocr: 'OCR text',
                          caption: 'Caption',
                          category: 'Category',
                          metadata: 'Image metadata',
                          category_filter: 'Category filter',
                        })[field] || field).join(' · ')
                        : 'No text fields matched; ranking includes visual similarity.'}
                    </span>
                    {result.matched_keywords?.length ? <span>Matched terms: {result.matched_keywords.join(', ')}</span> : null}
                    {result.query_expansions?.length ? (
                      <span>Related concepts considered: {result.query_expansions.join(', ')}</span>
                    ) : null}
                    <span>Visual meaning score: {Number(result.semantic_score).toFixed(3)}</span>
                  </div>
                  <details className="technical-details">
                    <summary>Technical ranking details</summary>
                    <span>Hybrid score: {Number(result.hybrid_score).toFixed(3)} · Semantic cosine: {Number(result.semantic_score).toFixed(3)}</span>
                    <span>{result.search_reason}</span>
                  </details>
                  <button type="button" className="file-action-button" onClick={() => showIntelligence(result, result.library_id)}>
                    MemoryOS Intelligence
                  </button>
                  {fileActions(result, result.library_id)}
                </div>
              </article>
            )) : (
              <div className="empty-state">
                <strong>No matching images found.</strong>
                <span>Try another keyword or natural-language query.</span>
              </div>
            )}
          </section>
        ) : null}

        {currentPage === 'duplicates' ? (
          <section className="page-section" aria-labelledby="duplicates-page-heading">
            <div className="page-heading">
              <p className="eyebrow">Review similar files</p>
              <h2 id="duplicates-page-heading">Duplicates</h2>
              <p>Find exact, near, and visually similar images. MemoryOS reports matches but never removes files automatically.</p>
            </div>
            <button
              type="button"
              className="primary-button"
              onClick={handleDuplicateDetection}
              disabled={duplicateLoading || !scanResult?.library_id}
            >
              {duplicateLoading ? 'Checking for duplicates...' : 'Check for duplicates'}
            </button>
            {!scanResult?.library_id ? (
              <p className="helper-text">Scan a library first to check for duplicates.</p>
            ) : null}
          </section>
        ) : null}

        {currentPage === 'duplicates' && duplicateResult ? (
          <section className="duplicate-section">
            <div className="notice-box">
              Exact duplicates have identical file hashes. Near duplicates look alike by image hash; visual similarity uses existing image embeddings. MemoryOS reports groups only and never deletes duplicates automatically.
            </div>
            <div className="summary-panel">
              <div><span className="summary-label">Duplicate groups:</span><strong>{duplicateResult.exact_groups + duplicateResult.near_duplicate_groups + (duplicateResult.visual_groups ?? 0)}</strong></div>
              <div><span className="summary-label">Exact groups:</span><strong>{duplicateResult.exact_groups}</strong></div>
              <div><span className="summary-label">Near-duplicate groups:</span><strong>{duplicateResult.near_duplicate_groups}</strong></div>
              <div><span className="summary-label">Visual matches:</span><strong>{duplicateResult.visual_matches}</strong></div>
              <div><span className="summary-label">Duplicates found:</span><strong>{duplicateResult.duplicates_found}</strong></div>
              <div><span className="summary-label">Processed / skipped:</span><strong>{duplicateResult.processed} / {duplicateResult.skipped}</strong></div>
            </div>
            {(duplicateResult.groups ?? []).length ? (
              <div className="duplicate-groups">
                {duplicateResult.groups.map((group) => (
                  <article className="duplicate-group-card" key={group.group_id}>
                    <strong>{group.duplicate_type === 'exact' ? 'Exact duplicate' : group.duplicate_type === 'near' ? 'Near duplicate' : 'Visual similarity'}</strong>
                    <span>
                      {group.duplicate_type === 'exact'
                        ? 'These files have identical content.'
                        : 'These images are visually similar but are not byte-identical.'}
                    </span>
                    <div className="duplicate-member-grid">
                      {group.members.map((member) => (
                        <div className="duplicate-member" key={member.relative_path}>
                          {imagePreview(member, duplicateResult.library_id)}
                          <strong>{member.filename}</strong>
                          <span>{member.relative_path}</span>
                        </div>
                      ))}
                    </div>
                    <div className="duplicate-match-evidence">
                      {group.matches.map((match) => (
                        <div key={`${match.left}:${match.right}:${match.match_method}`}>
                          <strong>{match.match_method === 'sha256' ? 'SHA-256 content match' : match.match_method === 'phash' ? 'Perceptual hash match' : 'OpenCLIP visual match'}</strong>
                          {match.distance !== null && match.distance !== undefined ? <span>pHash distance: {match.distance}</span> : null}
                          {match.similarity !== null && match.similarity !== undefined ? <span>OpenCLIP similarity: {Number(match.similarity).toFixed(3)}</span> : null}
                        </div>
                      ))}
                    </div>
                    <span>Detection only: files are never deleted automatically.</span>
                  </article>
                ))}
              </div>
            ) : <div className="empty-state">No duplicate groups found.</div>}
          </section>
        ) : null}

        {currentPage === 'library' ? <div className="library-grid">
          {displayedImages.length ? (
            <>
              <div className="grid-meta">Showing first {displayedImages.length} of {scanResult.image_records.length} records</div>
              {displayedImages.map((image) => (
                <article className="image-card" key={image.relative_path}>
                  {imagePreview(image, scanResult.library_id)}
                  <div className="image-meta">
                    <strong>{image.filename}</strong>
                    <span>{image.relative_path}</span>
                    <div className="image-badges">
                      <span className="image-badge">
                        {image.category ?? 'Others'}
                      </span>
                      <span className="image-badge badge-muted">{image.file_status || 'unverified'}</span>
                      <span className="image-badge badge-muted">
                        {image.ai_processing_status ?? image.processing_status ?? 'pending'}
                      </span>
                    </div>
                    {getTopCategoryShare(image) !== null ? (
                      <span>Top-two category share: {getTopCategoryShare(image).toFixed(1)}%</span>
                    ) : null}
                    <button type="button" className="file-action-button" onClick={() => showIntelligence(image, scanResult.library_id)}>
                      MemoryOS Intelligence
                    </button>
                    {fileActions(image, scanResult.library_id)}
                  </div>
                </article>
              ))}
            </>
          ) : (
            <div className="empty-state">
              {scanResult
                ? 'No supported images found in this folder. Supported formats: JPG/JPEG, PNG, WEBP, BMP, GIF, and TIFF/TIF.'
                : dashboardStats?.total_images
                  ? 'Your local index already contains images. Enter the folder path and scan it to load those images into this view.'
                  : 'No images indexed yet. Enter the full path to your image folder, then scan your library to get started.'}
            </div>
          )}
        </div> : null}

        {fileActionMessage ? <div className="success-box" role="status">{fileActionMessage}</div> : null}
        {fileActionRecord && ['rename', 'move', 'delete'].includes(fileActionType) ? (
          <div className="modal-backdrop" role="presentation">
            <form className="file-action-dialog" role="dialog" aria-modal="true" aria-labelledby="file-action-heading" onSubmit={submitFileAction}>
              <h2 id="file-action-heading">
                {fileActionType === 'rename' ? 'Rename image' : fileActionType === 'move' ? 'Move image' : 'Delete original image'}
              </h2>
              <p className="dialog-file-name">{fileActionRecord.filename}</p>
              <p className="dialog-file-path">{fileActionRecord.relative_path}</p>
              {fileActionType === 'rename' ? (
                <>
                  <label className="dialog-input">
                    <span>New filename, including extension</span>
                    <input
                      value={renameFilename}
                      onChange={(event) => setRenameFilename(event.target.value)}
                      autoComplete="off"
                      required
                      disabled={fileActionBusy}
                    />
                  </label>
                  <p className="dialog-file-path">Keep the existing image extension, such as .jpg or .png.</p>
                </>
              ) : null}
              {fileActionType === 'move' ? (
                <>
                  <label className="dialog-input">
                    <span>Choose a destination folder in this library</span>
                    <select value={moveDestination} onChange={(event) => setMoveDestination(event.target.value)} disabled={fileActionBusy}>
                      {moveFolders.map((folder) => (
                        <option value={folder} key={folder}>{folder || 'Library root'}</option>
                      ))}
                    </select>
                  </label>
                  <p className="dialog-file-path">
                    Selected destination: {moveDestination || 'Library root'}
                  </p>
                </>
              ) : null}
              {fileActionType === 'delete' ? (
                <div className="delete-warning">
                  This will permanently delete the selected file from your local computer. This cannot be undone.
                  <br />
                  <strong>Are you sure?</strong>
                </div>
              ) : null}
              {fileActionError ? <div className="error-box" role="alert">{fileActionError}</div> : null}
              <div className="dialog-actions">
                <button type="button" className="secondary-button" onClick={closeFileAction} disabled={fileActionBusy}>Cancel</button>
                <button
                  type="submit"
                  className={fileActionType === 'delete' ? 'danger-button' : 'primary-button'}
                  disabled={fileActionBusy || (fileActionType === 'move' && !moveFolders.length)}
                >
                  {fileActionBusy ? 'Working...' : fileActionType === 'delete' ? 'Permanently delete' : fileActionType === 'move' ? 'Move image' : 'Rename image'}
                </button>
              </div>
            </form>
          </div>
        ) : null}

        {intelligenceRecord ? (
          <div className="intelligence-backdrop" role="presentation" onClick={() => setIntelligenceRecord(null)}>
            <div className="intelligence-detail-panel" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
              <div className="intelligence-detail-header">
                <h2 className="intelligence-detail-title">MemoryOS Intelligence</h2>
                <button
                  type="button"
                  className="intelligence-detail-close"
                  onClick={() => setIntelligenceRecord(null)}
                  aria-label="Close intelligence panel"
                >
                  ✕
                </button>
              </div>

              <div className="intelligence-detail-body">
                {imagePreview(intelligenceRecord, intelligenceLibraryId, ' intelligence-preview')}

                <div className="intelligence-section">
                  <h3 className="intelligence-section-title">Classification</h3>
                  <div className="intelligence-classification">
                    <div className="intelligence-category">
                      {intelligenceRecord.category || 'Others'}
                    </div>
                    {intelligenceRecord.assignment_reason ? (
                      <div className="intelligence-confidence">{intelligenceRecord.assignment_reason}</div>
                    ) : null}
                    {getTopCategoryShare(intelligenceRecord) !== null ? (
                      <div className="intelligence-confidence">
                        Leading category share: {getTopCategoryShare(intelligenceRecord).toFixed(1)}% of the top two
                      </div>
                    ) : null}
                    {intelligenceRecord.ai_processing_status ? (
                      <div className="intelligence-confidence">
                        Status: {intelligenceRecord.ai_processing_status}
                      </div>
                    ) : null}
                  </div>
                </div>

                {intelligenceRecord.classification_reason && intelligenceRecord.classification_reason.length > 0 ? (
                  <div className="intelligence-section">
                    <h3 className="intelligence-section-title">Why MemoryOS Thinks This</h3>
                    <div className="intelligence-evidence-breakdown">
                  {intelligenceRecord.classification_reason.map((reason, idx) => (
                        <div className="evidence-card" key={idx}>
                          <span className="evidence-source">{evidenceSourceLabel(reason)}</span>
                          <div className="evidence-content">
                        {reason}
                          </div>
                    </div>
                  ))}
                </div>
                  </div>
                ) : null}

                {intelligenceRecord.secondary_categories?.length ? (
                  <div className="intelligence-section">
                <h3 className="intelligence-section-title">Secondary Tags</h3>
                <div className="intelligence-keywords">
                  {intelligenceRecord.secondary_categories.map((category) => (
                    <span className="intelligence-keyword" key={category}>{category}</span>
                  ))}
                </div>
                  </div>
                ) : null}

                {intelligenceRecord.top_categories?.length ? (
                  <div className="intelligence-section">
                <h3 className="intelligence-section-title">Top Category Scores</h3>
                {intelligenceRecord.top_categories.map((item) => (
                  <div className="evidence-card" key={item.category}>
                    <span className="evidence-source">{item.category}</span>
                    <div className="evidence-score">{(item.score * 100).toFixed(1)}%</div>
                  </div>
                ))}
                  </div>
                ) : null}

                {intelligenceRecord.ocr_text && intelligenceRecord.ocr_status ? (
                  <div className="intelligence-section">
                    <h3 className="intelligence-section-title">OCR Text</h3>
                    <div className="intelligence-ocr-section">
                      <div className="intelligence-ocr-text">
                        {intelligenceRecord.ocr_text || '(No OCR text extracted)'}
                      </div>
                      {intelligenceRecord.ocr_confidence !== null && intelligenceRecord.ocr_confidence !== undefined ? (
                        <div className="evidence-score" style={{ marginTop: '0.5rem' }}>
                          Confidence: {(intelligenceRecord.ocr_confidence * 100).toFixed(1)}%
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {intelligenceRecord.caption && intelligenceRecord.ai_profile ? (
                  <div className="intelligence-section">
                    <h3 className="intelligence-section-title">Image Caption</h3>
                    <div className="intelligence-caption-section">
                      <div className="intelligence-caption-text">
                        "{intelligenceRecord.caption}"
                      </div>
                    </div>
                  </div>
                ) : null}

                {intelligenceRecord.alternative_category && intelligenceRecord.alternative_score !== null && intelligenceRecord.alternative_score !== undefined ? (
                  <div className="intelligence-section">
                    <div className="intelligence-alternative">
                      <div className="intelligence-alternative-label">Alternative Match</div>
                      <div className="intelligence-alternative-category">
                        {intelligenceRecord.alternative_category}
                      </div>
                      <div className="evidence-score">
                        Score: {(intelligenceRecord.alternative_score * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ) : null}

                {intelligenceRecord.matched_classification_keywords && intelligenceRecord.matched_classification_keywords.length > 0 && (
                  <div className="intelligence-section">
                    <h3 className="intelligence-section-title">Matched Keywords</h3>
                    <div className="intelligence-keywords">
                      {intelligenceRecord.matched_classification_keywords.map((keyword, idx) => (
                        <span className="intelligence-keyword" key={idx}>{keyword}</span>
                      ))}
                    </div>
                  </div>
                )}

                {intelligenceRecord.filename && (
                  <div className="intelligence-section">
                    <h3 className="intelligence-section-title">File Details</h3>
                    <div className="evidence-card">
                      <span className="evidence-source">Filename</span>
                      <div className="evidence-content">{intelligenceRecord.filename}</div>
                    </div>
                    {intelligenceRecord.file_size !== null && intelligenceRecord.file_size !== undefined ? (
                      <div className="evidence-card" style={{ marginTop: '0.5rem' }}>
                        <span className="evidence-source">File Size</span>
                        <div className="evidence-content">
                          {(intelligenceRecord.file_size / 1024 / 1024).toFixed(2)} MB
                        </div>
                      </div>
                    ) : null}
                    {intelligenceRecord.file_status ? (
                      <div className="evidence-card" style={{ marginTop: '0.5rem' }}>
                        <span className="evidence-source">Status</span>
                        <div className="evidence-content">{intelligenceRecord.file_status}</div>
                      </div>
                    ) : null}
                  </div>
                )}

                <div className="intelligence-bottom-section">
                  All data is stored locally on your computer. No images or metadata are uploaded to any external service.
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </main>
  );
}

export default App;

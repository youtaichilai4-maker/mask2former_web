import type { DescribeResponse, ModelInfo, PredictResponse, TestImage, TopClass, AreaStat } from '@/types/api';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
const DEBUG_INGEST_URL = 'http://127.0.0.1:7244/ingest/a9e509fe-5c88-477e-8d74-278e5092e8b7';

function debugLog(hypothesisId: string, location: string, message: string, data: Record<string, unknown>) {
  // #region agent log
  fetch(DEBUG_INGEST_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      runId: 'pre-fix',
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {});
  // #endregion
}

export function fullUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function readJsonOrThrow<T>(res: Response, fallbackMessage: string): Promise<T> {
  let body: any;
  try {
    body = await res.json();
  } catch (error) {
    debugLog('H4', 'frontend/src/lib/api.ts:readJsonOrThrow', 'JSON parse failed', {
      url: res.url,
      status: res.status,
      statusText: res.statusText,
      contentType: res.headers.get('content-type') || '',
      error: String(error),
    });
    throw error;
  }
  if (!res.ok) {
    debugLog('H3', 'frontend/src/lib/api.ts:readJsonOrThrow', 'Non-OK response', {
      url: res.url,
      status: res.status,
      detail: body?.detail || '',
      fallbackMessage,
    });
    throw new Error(body.detail || fallbackMessage);
  }
  return body as T;
}

export async function fetchBootstrapData(): Promise<{ models: ModelInfo[]; images: TestImage[] }> {
  const modelsUrl = `${API_BASE}/models`;
  const testImagesUrl = `${API_BASE}/test-images`;
  debugLog('H1', 'frontend/src/lib/api.ts:fetchBootstrapData', 'Bootstrap fetch start', {
    apiBase: API_BASE,
    modelsUrl,
    testImagesUrl,
    origin: typeof window !== 'undefined' ? window.location.origin : 'server',
  });

  let modelsRes: Response;
  let imagesRes: Response;
  try {
    [modelsRes, imagesRes] = await Promise.all([fetch(modelsUrl), fetch(testImagesUrl)]);
  } catch (error) {
    debugLog('H1', 'frontend/src/lib/api.ts:fetchBootstrapData', 'Bootstrap network error', {
      apiBase: API_BASE,
      modelsUrl,
      testImagesUrl,
      error: String(error),
    });
    throw error;
  }
  debugLog('H2', 'frontend/src/lib/api.ts:fetchBootstrapData', 'Bootstrap fetch response', {
    modelsStatus: modelsRes.status,
    testImagesStatus: imagesRes.status,
    modelsContentType: modelsRes.headers.get('content-type') || '',
    testImagesContentType: imagesRes.headers.get('content-type') || '',
  });

  const modelsBody = await readJsonOrThrow<{ models: ModelInfo[] }>(modelsRes, 'モデル情報の取得に失敗しました');
  const imagesBody = await readJsonOrThrow<{ images: TestImage[] }>(imagesRes, 'テスト画像一覧の取得に失敗しました');

  return {
    models: modelsBody.models || [],
    images: imagesBody.images || [],
  };
}

export async function predictByImageId(imageId: string): Promise<PredictResponse> {
  const url = `${API_BASE}/predict-by-id`;
  debugLog('H2', 'frontend/src/lib/api.ts:predictByImageId', 'Predict-by-id request start', {
    apiBase: API_BASE,
    url,
    imageId,
  });
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: imageId }),
    });
  } catch (error) {
    debugLog('H1', 'frontend/src/lib/api.ts:predictByImageId', 'Predict-by-id network error', {
      apiBase: API_BASE,
      url,
      imageId,
      error: String(error),
    });
    throw error;
  }
  return readJsonOrThrow<PredictResponse>(res, '推論に失敗しました');
}

export async function predictByUpload(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('model_key', 'ade20k_official');

  const url = `${API_BASE}/predict`;
  debugLog('H5', 'frontend/src/lib/api.ts:predictByUpload', 'Predict upload request start', {
    apiBase: API_BASE,
    url,
    fileName: file.name,
    fileType: file.type,
    fileSize: file.size,
  });
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      body: fd,
    });
  } catch (error) {
    debugLog('H1', 'frontend/src/lib/api.ts:predictByUpload', 'Predict upload network error', {
      apiBase: API_BASE,
      url,
      error: String(error),
    });
    throw error;
  }
  return readJsonOrThrow<PredictResponse>(res, '推論に失敗しました');
}

export async function generateDescription(params: {
  top_classes: TopClass[];
  area_stats: AreaStat[];
  inference_ms: number;
}): Promise<DescribeResponse> {
  const res = await fetch(`${API_BASE}/describe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return readJsonOrThrow<DescribeResponse>(res, '説明文生成に失敗しました');
}

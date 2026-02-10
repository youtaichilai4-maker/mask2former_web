'use client';

import { ChangeEvent, useEffect, useMemo, useState } from 'react';

import {
  fetchBootstrapData,
  generateDescription as generateDescriptionApi,
  predictByImageId as predictByImageIdApi,
  predictByUpload as predictByUploadApi,
} from '@/lib/api';
import type { DescribeResponse, ModelInfo, PredictResponse, TestImage } from '@/types/api';

export function useSegmentationDemo() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [testImages, setTestImages] = useState<TestImage[]>([]);
  const [selectedImageId, setSelectedImageId] = useState('');
  const [selectedUpload, setSelectedUpload] = useState<File | null>(null);

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [selectedMaskUrl, setSelectedMaskUrl] = useState<string | null>(null);
  const [description, setDescription] = useState<DescribeResponse | null>(null);

  const [busyPredict, setBusyPredict] = useState(false);
  const [busyDescribe, setBusyDescribe] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      const { models: fetchedModels, images: fetchedImages } = await fetchBootstrapData();
      setModels(fetchedModels);
      setTestImages(fetchedImages);
      if (fetchedImages.length > 0) {
        setSelectedImageId(fetchedImages[0].id);
      }
    };

    run().catch((e) => setError(String(e)));
  }, []);

  const selectedModel = useMemo(() => models[0], [models]);
  const selectedImage = useMemo(
    () => testImages.find((img) => img.id === selectedImageId) || null,
    [testImages, selectedImageId]
  );

  const resetTransient = () => {
    setError(null);
    setDescription(null);
    setSelectedMaskUrl(null);
  };

  const predictByImageId = async () => {
    if (!selectedImageId) {
      setError('テスト画像を選択してください');
      return;
    }

    setBusyPredict(true);
    resetTransient();
    try {
      const body = await predictByImageIdApi(selectedImageId);
      setResult(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyPredict(false);
    }
  };

  const predictByUpload = async () => {
    if (!selectedUpload) {
      setError('アップロード画像を選択してください');
      return;
    }

    setBusyPredict(true);
    resetTransient();
    try {
      const body = await predictByUploadApi(selectedUpload);
      setResult(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyPredict(false);
    }
  };

  const generateDescription = async () => {
    if (!result) return;

    setBusyDescribe(true);
    setError(null);
    try {
      const body = await generateDescriptionApi({
        top_classes: result.top_classes,
        area_stats: result.area_stats,
        inference_ms: result.inference_ms,
      });
      setDescription(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyDescribe(false);
    }
  };

  const onUploadChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSelectedUpload(e.target.files?.[0] || null);
  };

  return {
    models,
    testImages,
    selectedImage,
    selectedImageId,
    selectedUpload,
    selectedModel,
    result,
    selectedMaskUrl,
    description,
    busyPredict,
    busyDescribe,
    error,
    setSelectedImageId,
    setSelectedMaskUrl,
    onUploadChange,
    predictByImageId,
    predictByUpload,
    generateDescription,
  };
}

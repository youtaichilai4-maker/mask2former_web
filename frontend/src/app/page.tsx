'use client';

import styles from './page.module.css';
import { AnalyticsSection } from '@/components/AnalyticsSection';
import { GalleryPanel } from '@/components/GalleryPanel';
import { HeroSection } from '@/components/HeroSection';
import { ResultsSection } from '@/components/ResultsSection';
import { useSegmentationDemo } from '@/hooks/useSegmentationDemo';

export default function Home() {
  const {
    selectedImage,
    selectedImageId,
    selectedUpload,
    selectedModel,
    testImages,
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
  } = useSegmentationDemo();

  return (
    <main className={styles.page}>
      <HeroSection />

      <GalleryPanel
        testImages={testImages}
        selectedImageId={selectedImageId}
        busyPredict={busyPredict}
        onSelectImage={setSelectedImageId}
        onPredictSelected={predictByImageId}
        onUploadChange={onUploadChange}
        onPredictUpload={predictByUpload}
      />

      {error && <p className={styles.error}>{error}</p>}

      <ResultsSection
        result={result}
        selectedImage={selectedImage}
        selectedUploadName={selectedUpload?.name || null}
        selectedModel={selectedModel}
        selectedMaskUrl={selectedMaskUrl}
        onSelectMask={setSelectedMaskUrl}
      />

      <AnalyticsSection
        result={result}
        description={description}
        busyDescribe={busyDescribe}
        onGenerateDescription={generateDescription}
      />
    </main>
  );
}

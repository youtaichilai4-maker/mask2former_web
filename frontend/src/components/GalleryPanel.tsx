import styles from '@/app/page.module.css';
import { fullUrl } from '@/lib/api';
import type { TestImage } from '@/types/api';
import type { ChangeEvent } from 'react';

type Props = {
  testImages: TestImage[];
  selectedImageId: string;
  busyPredict: boolean;
  onSelectImage: (id: string) => void;
  onPredictSelected: () => void;
  onUploadChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onPredictUpload: () => void;
};

export function GalleryPanel({
  testImages,
  selectedImageId,
  busyPredict,
  onSelectImage,
  onPredictSelected,
  onUploadChange,
  onPredictUpload,
}: Props) {
  return (
    <section className={styles.panel}>
      <div className={styles.rowHeader}>
        <h2>テスト画像ギャラリー（Mask2Former / スクロール選択）</h2>
        <button className={styles.primaryButton} type="button" disabled={busyPredict} onClick={onPredictSelected}>
          {busyPredict ? '推論中...' : '選択画像で推論'}
        </button>
      </div>

      <div className={styles.carousel} role="list" aria-label="テスト画像ギャラリー">
        {testImages.map((image) => (
          <button
            key={image.id}
            className={`${styles.card} ${image.id === selectedImageId ? styles.cardSelected : ''}`}
            onClick={() => onSelectImage(image.id)}
            type="button"
          >
            <img className={styles.thumbImage} src={fullUrl(image.thumbnail_url)} alt={image.name} />
            <p>{image.name}</p>
            <span>{image.tags.join(', ')}</span>
          </button>
        ))}
        {testImages.length === 0 && <p className={styles.empty}>`backend/app/static/test_images` に画像を配置してください。</p>}
      </div>

      <div className={styles.uploadRow}>
        <label className={styles.uploadLabel}>
          任意画像アップロード
          <input type="file" accept="image/*" onChange={onUploadChange} />
        </label>
        <button className={styles.secondaryButton} type="button" disabled={busyPredict} onClick={onPredictUpload}>
          {busyPredict ? '推論中...' : 'アップロード画像で推論'}
        </button>
      </div>
    </section>
  );
}

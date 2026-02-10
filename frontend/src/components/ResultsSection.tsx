import styles from '@/app/page.module.css';
import { fullUrl } from '@/lib/api';
import type { ModelInfo, PredictResponse, TestImage } from '@/types/api';

type Props = {
  result: PredictResponse | null;
  selectedImage: TestImage | null;
  selectedUploadName: string | null;
  selectedModel?: ModelInfo;
  selectedMaskUrl: string | null;
  onSelectMask: (maskUrl: string) => void;
};

export function ResultsSection({
  result,
  selectedImage,
  selectedUploadName,
  selectedModel,
  selectedMaskUrl,
  onSelectMask,
}: Props) {
  return (
    <section className={styles.results}>
      <article className={styles.viewer}>
        <header>
          <h3>入力画像</h3>
          <small>{selectedImage?.name || selectedUploadName || '-'}</small>
        </header>
        {result ? (
          <img className={styles.realImage} src={fullUrl(result.original_url)} alt="入力画像" />
        ) : (
          <div className={`${styles.canvas} ${styles.baseCanvas}`} />
        )}
      </article>

      <article className={styles.viewer}>
        <header>
          <h3>Mask2Former 予測オーバーレイ</h3>
          <small>{selectedModel?.hf_id || 'facebook/mask2former-swin-large-ade-semantic'}</small>
        </header>
        {result ? (
          <img className={styles.realImage} src={fullUrl(result.overlay_url)} alt="予測オーバーレイ" />
        ) : (
          <div className={`${styles.canvas} ${styles.overlayCanvas}`} />
        )}
      </article>

      <article className={styles.viewer}>
        <header>
          <h3>クラス別マスク一覧</h3>
          <small>クリックで拡大表示</small>
        </header>
        <div className={styles.maskGrid}>
          {(result?.class_masks || []).map((mask) => (
            <button
              key={`${mask.class_id}-${mask.label}`}
              className={`${styles.maskItem} ${selectedMaskUrl === mask.mask_url ? styles.maskSelected : ''}`}
              type="button"
              onClick={() => onSelectMask(mask.mask_url)}
            >
              <img className={styles.maskThumb} src={fullUrl(mask.mask_url)} alt={`${mask.label} mask`} />
              <span>{mask.label}</span>
            </button>
          ))}
        </div>
        {selectedMaskUrl && <img className={styles.maskPreview} src={fullUrl(selectedMaskUrl)} alt="選択マスク" />}
      </article>
    </section>
  );
}

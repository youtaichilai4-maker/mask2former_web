import styles from '@/app/page.module.css';

export function HeroSection() {
  return (
    <div className={styles.hero}>
      <p className={styles.kicker}>Mask2Former Semantic Segmentation Demo</p>
      <h1>Mask2Former Visual Evidence Studio</h1>
      <p className={styles.sub}>
        画像を選んでMask2Formerで推論し、入力・オーバーレイ・クラス別マスクと、Top-K信頼度/面積比を同時に検証します。
      </p>
    </div>
  );
}

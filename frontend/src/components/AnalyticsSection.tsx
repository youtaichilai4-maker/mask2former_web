import styles from '@/app/page.module.css';
import type { DescribeResponse, PredictResponse } from '@/types/api';

type Props = {
  result: PredictResponse | null;
  description: DescribeResponse | null;
  busyDescribe: boolean;
  onGenerateDescription: () => void;
};

export function AnalyticsSection({ result, description, busyDescribe, onGenerateDescription }: Props) {
  return (
    <section className={styles.analytics}>
      <article className={styles.metricCard}>
        <h3>上位クラス信頼度（Top-K）</h3>
        <ul>
          {(result?.top_classes || []).map((row) => (
            <li key={`${row.class_id}-${row.label}`}>
              <span>{row.label}</span>
              <strong>{(row.confidence * 100).toFixed(1)}%</strong>
            </li>
          ))}
        </ul>
      </article>

      <article className={styles.metricCard}>
        <h3>クラス面積比（%）</h3>
        <ul>
          {(result?.area_stats || []).map((row) => (
            <li key={`${row.class_id}-${row.label}`}>
              <span>{row.label}</span>
              <strong>{row.area_ratio.toFixed(1)}%</strong>
            </li>
          ))}
        </ul>
      </article>

      <article className={styles.metricCard}>
        <div className={styles.describeHeader}>
          <h3>Gemini 空間説明</h3>
          <button className={styles.secondaryButton} disabled={!result || busyDescribe} onClick={onGenerateDescription}>
            {busyDescribe ? '生成中...' : '説明文生成'}
          </button>
        </div>
        <p>{description?.summary_ja || '推論後に説明文を生成できます。'}</p>
        {description?.highlights?.length ? (
          <ul>
            {description.highlights.map((line, idx) => (
              <li key={`h-${idx}`}>{line}</li>
            ))}
          </ul>
        ) : null}
        {description?.cautions?.length ? (
          <ul>
            {description.cautions.map((line, idx) => (
              <li key={`c-${idx}`}>{line}</li>
            ))}
          </ul>
        ) : null}
        <p className={styles.meta}>推論時間: {result ? `${result.inference_ms} ms` : '-'}</p>
      </article>
    </section>
  );
}

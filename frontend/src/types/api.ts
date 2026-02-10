export type ModelInfo = {
  model_key: string;
  hf_id: string;
  label_space: string;
  note: string;
};

export type TestImage = {
  id: string;
  name: string;
  thumbnail_url: string;
  image_url: string;
  tags: string[];
};

export type Label = {
  class_id: number;
  label: string;
};

export type TopClass = {
  class_id: number;
  label: string;
  confidence: number;
};

export type AreaStat = {
  class_id: number;
  label: string;
  area_ratio: number;
};

export type ClassMask = {
  class_id: number;
  label: string;
  mask_url: string;
};

export type PredictResponse = {
  model_key: string;
  model_hf_id: string;
  inference_ms: number;
  original_url: string;
  overlay_url: string;
  labels: Label[];
  top_classes: TopClass[];
  area_stats: AreaStat[];
  class_masks: ClassMask[];
  width: number;
  height: number;
};

export type DescribeResponse = {
  summary_ja: string;
  highlights: string[];
  cautions: string[];
};

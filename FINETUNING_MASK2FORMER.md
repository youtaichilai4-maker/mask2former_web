# Mask2Former（ADE20K公式重み）ファインチューニング手順メモ（層ごと）

このリポジトリのbackendは現状 **推論専用** だ（`model.eval()` + `torch.inference_mode()`）。
ファインチューニングする場合は、別スクリプト/別エントリポイントで **学習モード**（`model.train()`）として実行するのが安全。

対象モデル（本repoのデフォルト）:

- Hugging Face: `facebook/mask2former-swin-large-ade-semantic`
- ロード: `Mask2FormerForUniversalSegmentation.from_pretrained(...)`

---

## 1. まず「層（モジュール）名」を確認する（最重要）

Transformersのバージョンやモデルによって、サブモジュールの属性名が微妙に違う。
**凍結/解凍は "名前のprefix" で制御するのが安定** なので、最初に `named_parameters()` / `named_modules()` を見る。

```python
from transformers import Mask2FormerForUniversalSegmentation

hf_id = "facebook/mask2former-swin-large-ade-semantic"
model = Mask2FormerForUniversalSegmentation.from_pretrained(hf_id)

# 代表的なパラメータ名を眺める（prefix設計用）
for name, _ in list(model.named_parameters())[:200]:
    print(name)

# モジュール構成を眺める（大きいので絞る）
for name, _ in model.named_modules():
    if any(k in name.lower() for k in ["backbone", "pixel", "transformer", "decoder", "class", "mask"]):
        print(name)
```

以降の「層ごとの方針」は一般論。**自分の環境で見えたprefixに合わせて読み替える。**

---

## 2. Mask2Formerの "層" をどう捉えるか（層ごとの役割）

（厳密な実装名は上の方法で確認する）

- **Backbone（例: Swin）**
  - 画像から特徴マップを抽出する部分
  - いちばん重く、過学習も起きやすいので、最初は凍結することが多い

- **Pixel Decoder / Pixel-level module**
  - マルチスケール特徴を融合して、セグメンテーション向けの特徴に変換する部分
  - 中程度に重い。データ量が少ない場合は凍結候補

- **Transformer Decoder / Query decoder**
  - クエリ（object queries）を使って、クラス/マスクの予測に必要な表現を作る部分
  - "ドメイン適応" に効きやすいので、Backboneを凍結したままここを学習する戦略は定番

- **Prediction Heads（分類ヘッド / マスクヘッド）**
  - 最終的にクラスロジットやマスクロジットを出す部分
  - ラベル空間が変わる場合、ここは再初期化 or 置き換えが必要になりやすい

---

## 3. 各層の詳細な動作原理

### 3.1 Backbone（バックボーン）：初期特徴の抽出

**役割**: 入力画像から、局所的なエッジやテクスチャ、大域的な意味情報を抽出する「目」の役割。ResNetやSwin Transformerが使われる。

**処理**: 入力画像 $I \in \mathbb{R}^{H \times W \times 3}$ をネットワークに通し、解像度が異なる複数の特徴マップ（例：元画像の $1/8, 1/16, 1/32$ のサイズ）を生成する。

### 3.2 Pixel Decoder（ピクセルデコーダ）：高解像度化と文脈統合

**役割**: Backboneが抽出した粗い特徴マップを、元の解像度に近いレベル（通常は $1/4$ サイズ）まで復元しつつ、画像全体の文脈（Global Context）を各ピクセルに埋め込む。

**処理**: FPN（Feature Pyramid Network）やDeformable Attentionを用いて、異なるスケールの特徴を混ぜ合わせる。ここで最終的に作られるのが、「Per-pixel Embeddings（ピクセルごとの埋め込みベクトル）」 $\mathcal{E}_{pixel} \in \mathbb{R}^{C \times \frac{H}{4} \times \frac{W}{4}}$ だ。

※これは、後でQueryと内積を取ってマスクを作るための「キャンバス」になる。

### 3.3 Transformer Decoder：マスクとクラスの反復的洗練（★最重要）

ここがMask2Formerの心臓部。物理の「摂動法」や「反復法」のように、初期のランダムな状態から、徐々に真の解（マスク）へと収束させていくプロセス。

**入力されるもの**:
- $N$ 個の学習可能な Object Queries $X_0 \in \mathbb{R}^{N \times C}$ （「何か」を探すための初期ベクトル群）
- Pixel Decoderから得られた画像特徴量

このDecoderは $L$ 個のレイヤー（層）を持ち、各層 $l$ で以下の計算を繰り返す。

#### ① Masked Attention（注目領域の絞り込み）

通常のCross-Attentionは画像全域を見るが、Masked Attentionは「前層で予測したマスク領域」だけを積分範囲とするような処理を行う。

$$X_l = \text{softmax}(\mathcal{M}_{l-1} + Q_l K_l^T) V_l + X_{l-1}$$

- $X_{l-1}$: 前の層から渡されたQuery（探索者）
- $Q_l = X_{l-1} W_Q$: Queryの特徴量
- $K_l, V_l$: 画像特徴量に重み $W_K, W_V$ を掛けたもの
- $\mathcal{M}_{l-1}$: 前層で予測されたマスクに基づくAttentionバイアス

ここでの $\mathcal{M}_{l-1}$ は、ピクセル $(x, y)$ に対して以下のように定義される。

$$\mathcal{M}_{l-1}(x, y) = \begin{cases} 0 & \text{if } M_{l-1}(x, y) = 1 \text{ (前層で物体だと判定された)} \\ -\infty & \text{if } M_{l-1}(x, y) = 0 \text{ (背景だと判定された)} \end{cases}$$

**数学的意味**: $-\infty$ を足してSoftmaxを通すことで、マスク外のAttention重みが完全に $0$ になる。これにより、Queryはノイズ（背景）を無視して、物体固有の特徴（$V_l$）だけを効率的に吸収（$X_l$ に加算）できる。

#### ② マスク予測（Mask Prediction）

賢くなったQuery $X_l$ を使って、現在の層での新しいマスク $M_l$ を生成する。

$$M_l = \sigma(X_l \cdot \mathcal{E}_{pixel}^T)$$

$X_l$ (Query) と $\mathcal{E}_{pixel}$ (ピクセル埋め込み) の**内積（Dot Product）**を取り、Sigmoid関数 $\sigma$ で 0〜1 の確率値に変換する。「このQueryの性質と、このピクセルの性質はどれくらい似ているか？」を全ピクセルに対して計算しているだけのエレガントな数式。

#### ③ クラス予測（Class Prediction）

同時に、このQueryが「何の物体（壁、床など）」を捉えているかを予測する。

$$P_l = \text{softmax}(X_l W_{cls})$$

（$W_{cls}$ はクラス分類用の線形層の重み）

### 3.4 Bipartite Matching & Loss（学習時の誤差計算）

「$N$個のQueryがそれぞれ何を予測したか」は順不同（Permutation Invariant）であるため、通常のLoss計算ができない。

**二部マッチング (Hungarian Algorithm)**: 予測された $N$ 個のマスク・クラスのペア $\hat{y}$ と、正解データ $y$ の間で、最も「コスト（誤差）」が小さくなるような1対1のペアリングを数学的に見つけ出す。（これは数理最適化の「割り当て問題」そのもの）

**Loss関数の計算**: マッチングが決まったら、以下のLossを最小化するようにバックプロパゲーションを行う。

$$\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_{dice}\mathcal{L}_{dice}$$

- $\mathcal{L}_{cls}$: クラス分類のCross-Entropy Loss
- $\mathcal{L}_{ce}$: マスクのピクセルごとのBinary Cross-Entropy Loss
- $\mathcal{L}_{dice}$: マスクの重なり具合（IoU）を評価する Dice Loss（領域分割特有のLoss）

---

## 4. 凍結/解凍（requires_grad）の書き方（層ごと）

### 4.1 まずは全凍結→必要なprefixだけ解凍（おすすめ）

```python
def set_trainable_by_prefix(model, trainable_prefixes: list[str]) -> None:
    # いったん全凍結
    for p in model.parameters():
        p.requires_grad = False

    # prefix一致だけ解凍
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in trainable_prefixes):
            p.requires_grad = True

    # 確認
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable params: {len(trainable)}")
    print("\n".join(trainable[:50]))
```

使い方（例: decoder + heads）:

```python
trainable_prefixes = [
    # ここは自分の環境の実prefixに合わせて置き換える
    "model.transformer",   # transformer decoder系っぽいprefix例
    "class",               # classifier系っぽいprefix例
    "mask",                # mask head系っぽいprefix例
]
set_trainable_by_prefix(model, trainable_prefixes)
```

> 注意: prefixはモデル/transformersのバージョンで変わるので、必ず `named_parameters()` で確認する。

### 4.2 Optimizerは "学習対象だけ" を渡す

```python
import torch

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-4,
    weight_decay=0.05,
)
```

---

## 5. ラベル空間が変わる場合（クラス数変更）

ResNetの `model.fc = Linear(...)` のように単純な付け替えはできないが、Transformers流のやり方で対応できる。

### 5.1 `from_pretrained(..., num_labels=..., ignore_mismatched_sizes=True)` を使う

```python
from transformers import Mask2FormerForUniversalSegmentation

NEW_NUM_LABELS = 10
id2label = {i: f"class_{i}" for i in range(NEW_NUM_LABELS)}
label2id = {v: k for k, v in id2label.items()}

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic",
    num_labels=NEW_NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # ← 形が合わない層を自動で再初期化してくれる
)
```

このとき、**分類/マスクの最終層が再初期化される**ので、学習は必須になる。

---

## 6. 学習時の最低限の注意（推論コードとの差分）

- 推論: `model.eval()` + `torch.inference_mode()`
- 学習: `model.train()` + `loss.backward()`（必要ならAMP / grad clip）

例（雰囲気）:

```python
model.train()

for batch in dataloader:
    # pixel_values 以外に、教師信号（class/mask labels）が必要になる（タスク設定次第）
    outputs = model(**batch)
    loss = outputs.loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

> `Mask2FormerForUniversalSegmentation` の学習用バッチ形式（どんなキーが必要か）はタスク（semantic/instance/panoptic）とTransformers実装に依存する。まずは **公式/既存の学習スクリプトの入力形式** に寄せるのが近道

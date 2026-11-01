# 第2回・発展課題：用語集Q&A

講座の用語集 `data/glossary.json`（`2026-draft-1`、58語）から、学習・検証・評価用のQ&Aを作ります。[学習前後の実験記録](../../lectures/02/reference/glossary_sft/README.md) と [全36問の比較](../../lectures/02/reference/glossary_sft/comparison.md) を保存しています。

## 元データと生成ファイル

Gitに含めるのは用語集、手書きの [cases.json](cases.json)、生成・学習・比較のコード、および実験記録です。次の5ファイルと `runs/` は実行時に生成し、Gitの追跡対象から外しています。利用する前に再生成してください。

| 生成ファイル | 件数 | 用途 |
|---|---:|---|
| `train.jsonl` | 182 | 定義の質問174件＋未掲載語への回答8件。`messages` 形式 |
| `valid.jsonl` | 62 | 掲載語58件＋未掲載語4件。`messages` 形式 |
| `test.jsonl` | 36 | 評価用の質問と正解出力 `target`。学習には使わない |
| `gold.json` | 36 | 質問ごとの正解文、`needs_review`、必須語句、目視採点基準 |
| `metadata.json` | — | 件数、共通system指示、生成元のSHA-256 |

リポジトリのルートで実行します。生成にはPython標準ライブラリだけを使います。

```bash
python scripts/make_sft_dataset.py --output-dir data/glossary_sft
```

各掲載語について「『用語名』とは何ですか？」「用語名について簡単に説明してください。」「最初の別名という用語を説明してください。」の3通りを作り、正解文は用語集の定義をそのまま使います。これに未掲載語への確認要求を8件加えます。検証用は別の質問テンプレートと別の未掲載語4語を使います。

各行は `system` → `user` → `assistant` の `messages` です。`assistant.content` は次の形式のJSON文字列です。

```json
{"answer": "用語集の定義文", "needs_review": false}
```

未掲載語の正解は共通です。

```json
{"answer": "この講座の用語集には記載がありません。確認が必要です。", "needs_review": true}
```

オプションなしの従来の実行方法は、以前と同じ116件のQ&A（回答は定義文）を標準出力します。今回の実験には `--output-dir` の出力を使います。

## 評価用の質問と正解

| 分類 | 件数 | 見たいこと |
|---|---:|---|
| `known` | 12 | 掲載語の名前を明示して聞いたとき、定義を答えられるか |
| `paraphrase` | 12 | 同じ12語を説明や利用場面から聞いたとき、対応する定義を答えられるか |
| `unknown` | 12 | 未掲載語について、確認を求められるか |

対象の掲載語は、トークン、コンテキストウィンドウ、temperature、ハルシネーション、SFT、PEFT、LoRA、QLoRA、過学習、AIエージェント、構造化出力、MCPです。直接質問はテンプレート、言い換えと採点基準は `cases.json` に個別に記述しています。

たとえばLoRAでは、「講座で扱う『LoRA』の意味を教えてください。」と「元の重みを固定し、更新分を小さい二つの行列の積で表して学ぶ方法を説明してください。」を対にします。正解は共通のLoRAの定義です。

未掲載語はRLHF、DPO、PPO、MoE、Transformer、自己注意機構、埋め込みベクトル、ベクトルデータベース、ビームサーチ、知識蒸留、プルーニング、投機的デコーディングです。一般に実在する用語でも、この用語集に記載がなければ確認を求めるのが正解です。

生成時に、質問文の重複、未掲載語の本文への混入、学習・検証・評価間での未掲載語の重複、採点語句と正解文の整合性を検査します。同じ入力からは同じファイルを生成します。全掲載語の定義は学習データに含まれ、評価では未知の質問文への対応を測ります。

## 学習・学習前後の評価

演習用の依存関係とは別の仮想環境を作ります。CUDA GPUを使います。今回の実験はRTX 3090、float32、量子化なしで実行しました。

```bash
uv venv .venv-glossary --python 3.12
uv pip install --python .venv-glossary/bin/python -r scripts/requirements-glossary-sft.txt
```

学習スクリプトは、指定revisionのモデルがローカルにあることを前提にしています。未取得の場合は、先に同じ環境でキャッシュします。

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
)
```

データ生成後、リポジトリのルートで次を実行します。

```bash
.venv-glossary/bin/python -m scripts.run_glossary_experiment
```

既定は20エポック、LoRA r=16、alpha=32、学習率2e-4、seed=42です。学習前後で同じ36問・同じsystem指示・同じ生成条件を使います。貪欲生成、最大384トークン、思考モード無効とし、EOS・PADも明示して固定します。学習・評価の入力には用語集本文や正解を付けません。最大系列長は学習・検証データの実際のトークン長を確認して決めます。

実験ごとに `runs/glossary_sft/日時-ID/` に次を生成します。既存の実験ディレクトリには上書きしません。

- `comparison.md`：4分類ごとの一覧。全問の質問、正解例、目視採点基準、前後の回答原文、判定理由。
- `comparison.csv` / `comparison.json`：全問の比較表と集計。`group` は形式・内容の両方、`content_group` は内容だけの変化。
- `eval_before.json` / `eval_after.json`：回答原文、採点、生成トークン数、上限に達したか。
- `config.json` / `training.json`：モデルrevision、ライブラリの版、学習条件、GPU、loss、学習時間、履歴。
- `data/`：実行時の入力データのコピー。SHA-256を `config.json` に記録。
- `adapter/` / `checkpoints/`：学習済みの重みとcheckpoint。

## 再集計と確認

比較表だけを作り直す場合はPython標準ライブラリだけで実行できます。`日時-ID` は実際の実験ディレクトリに置き換えてください。

```bash
python -m scripts.compare_glossary_results runs/glossary_sft/日時-ID
```

保存された正誤フラグをそのまま信用せず、回答原文と実行時の正解から再採点します。質問の重複・欠落・追加、実験IDやモデル・生成条件の不一致、保存済み評価データの変更があると比較を停止します。4分類は0件の分類も表示します。

比較図は学習用の仮想環境で作れます。

```bash
.venv-glossary/bin/python -m scripts.plot_glossary_comparison runs/glossary_sft/日時-ID
```

正誤の主判定は、JSONのキー・型と内容の両方への合格です。内容は `needs_review` の一致と必須語句の包含による簡易自動採点のため、同義表現を不正解にしたり、矛盾のある文を通したりする場合があります。回答原文と目視採点基準も確認してください。全体の正解率に加え、掲載語・言い換え・未掲載語それぞれの結果を確認します。GPU上の学習はseed固定でも完全な再現を保証しません。

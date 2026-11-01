# LoRA の学習と比較実験

Colab 用 `notebooks/lora_format_tuning.ipynb` を Python スクリプトに移したものです。
架空の FAQ、14 件の学習データ、7 件の検証データ、21 件のテストデータ、
学習前後の採点、知識質問、任意の 4bit 比較を引き継いでいます。
基準設定は量子化なしの LoRA です。4bit 比較では学習しません。

## 環境を用意する

リポジトリルートで実行します。Python 3.12 以上、学習には CUDA GPU を推奨します。
`model.device = "auto"` では CUDA → MPS → CPU の順で選びます。
CPU でも設定・データの確認ができ、学習は GPU より時間がかかります。

通常の演習用 `.venv` と学習環境を分けます。
学習ライブラリは元のノートブックの版に固定していますが、Colab 以外での実学習は未検証です。

```bash
uv venv .venv-lora --python 3.12
uv pip install --python .venv-lora/bin/python -r lora/requirements.txt
.venv-lora/bin/python -m lora.train --config lora/configs/baseline.toml
```

Google Colab でも、T4 GPU を選んだ新しいノートブックでリポジトリを取得して同じスクリプトを実行できます。
`%cd` のパスは clone したディレクトリに合わせてください。

```python
!git clone https://github.com/matsuolab/lecture-ai-engineering-2026.git
%cd /content/lecture-ai-engineering-2026
%pip uninstall -y torchao
%pip install -r lora/requirements.txt
!python -m lora.train --config lora/configs/baseline.toml
```

`torchao` の削除は元の Colab 同梱版との非互換を避ける処理です。
Colab では終了前に実験ディレクトリをダウンロードするか Drive にコピーしてください。
Drive に直接保存する場合は、先に `drive.mount("/content/drive")` し、
`--set 'experiment.output_root="/content/drive/MyDrive/aie2026/lora"'` を指定します。

## 設定を変えて実験する

設定は [configs/baseline.toml](configs/baseline.toml) に集めています。
`--config` で渡した TOML に書かれた項目だけを基準設定に上書きします。
[configs/rank8.toml](configs/rank8.toml) は alpha を 32 に固定し、rank だけ 16 → 8 に変更する例です。
`--set` は TOML を読み込んだ後に適用します。未知の項目名や型の違いはエラーになります。

```bash
# 基準条件（20 エポック、rank=16、学習率=2e-4）
.venv-lora/bin/python -m lora.train

# rank=8 の設定ファイル
.venv-lora/bin/python -m lora.train --config lora/configs/rank8.toml

# 学習率だけ変える。文字列の値は TOML の引用符も渡す
.venv-lora/bin/python -m lora.train \
  --set 'experiment.name="lr1e-4"' \
  --set training.learning_rate=1e-4

# 複数の結果を TSV で比較（モデルの読み込み不要）
.venv-lora/bin/python -m lora.compare runs/lora/*
```

学習曲線を描く場合は、描画用ライブラリを導入して比較する実験を指定します。

```bash
uv pip install --python .venv-lora/bin/python -r lora/requirements-plot.txt
.venv-lora/bin/python -m lora.plot \
  runs/lora/baseline-実行時の日時と識別子 \
  runs/lora/rank8-実行時の日時と識別子 \
  --output-dir runs/lora/comparison-rank8
```

`comparison.csv` と、線形・対数スケールの `learning_curve.png` / `learning_curve.svg` を出力します。
グラフの元データは `learning_curve.csv` に保存します。
loss はログ区間の平均です。既定では 10 optimizer step ごとに記録されるため、40 step の学習では各実験 4 点になります。
rank を下げて alpha を固定すると `alpha/r` は大きくなるため、rank と更新倍率の両方が変化します。
更新倍率を揃えたい場合は、rank と同じ比率で alpha も下げてください。

実験は毎回、新しいプロセスで基盤モデルから始めます。セッションの再起動は不要です。
モデルを変えるときは `model.name` と `model.revision` の両方を指定し、
モデルに合う `lora.target_modules` とチャットテンプレートも確認してください。
seed は読み込み・生成・学習の前に設定しますが、GPU 上の完全な再現性は保証されません。

設定・データだけの動作確認は、学習ライブラリのインストールも GPU も不要です。

```bash
python -m lora.train --config lora/configs/rank8.toml --prepare-only
```

## 保存されるもの

`runs/lora/<実験名>-<UTC日時>-<識別子>/` に保存します。同じ設定の再実行も別フォルダになります。
`--config` のパスは作業ディレクトリから、設定内のデータ・出力パスはリポジトリルートから解決します。
絶対パスも使えます。

| ファイル・フォルダ | 内容 |
|---|---|
| `config.json` | 基準設定・TOML・CLI の上書きを反映した実効設定 |
| `task.json` | 実行時の FAQ・プロンプト・質問のコピー |
| `train.jsonl`, `valid.jsonl` | `messages` 形式の学習・検証データ |
| `test.jsonl`, `gold.json` | 評価入力と正解（正解はモデルへの入力に含めない） |
| `checkpoints/` | 10 ステップごとの再開用 checkpoint（既定で最新 2 件） |
| `adapter/` | 学習済み LoRA adapter と tokenizer |
| `training_metrics.json`, `training_history.json` | 学習時間・loss・学習ログ |
| `eval_before.json`, `eval_after.json` | 質問ごとの出力原文、形式と内容の採点、要約 |
| `result.json` | 設定・ライブラリ版・前後のスコア・退行・知識質問・最終ログの loss |
| `eval_before_4bit.json` | 任意の量子化比較を有効にした場合の評価 |

`--prepare-only` では設定とデータだけを保存します。
`result.json` の `training.seconds` は今回の `trainer.train()` の時間です。
再開時は再開以降の時間となり、`training.resumed` が `true` になります。
`last_logged_loss` は最後に記録されたステップの loss、`train_loss` は Trainer が返す平均です。
ログ間隔より短い学習では `last_logged_loss` は `null` です。

形式は必須キー・型を含む JSON の検査、内容は `needs_review` と必須語句の一致で採点します。
JSON の前後に説明が付く場合などは、形式と内容の合否が異なることがあります。
検証データは Trainer に渡しますが、元のノートブック同様、検証によるモデル選択は行いません。
知識質問の `asserted` は `needs_review: false` の意味で、FAQ にある対照質問では正常です。

## 中断した学習を再開する

```bash
.venv-lora/bin/python -m lora.train --resume runs/lora/baseline-実行時の日時と識別子
```

最新 checkpoint から再開します。設定・データはその実験の保存済みコピーを使います。
`--config` / `--set` は併用できません。別の条件を試すときは新しい実験を始めます。
再開可能な checkpoint ができる前の中断は、新規実行してください。

## データを変える

FAQ 演習では [data/lora/faq.json](../data/lora/faq.json) をコピーし、
`data.task_file` を変更します。FAQ の文、質問テンプレート、正解に必要な語句、
知識質問とその対照質問を編集できます。各 FAQ 項目の質問テンプレートは 4 件にし、
先頭 2 件を train、3 件目を valid、4 件目と追加質問を test に割り当てます。
入力の重複があれば学習前に停止します。

独自の SFT データは `data.dataset_dir` を指定したフォルダに次の 4 ファイルを置きます。
この場合も `data.task_file` の `system` を評価プロンプトに使います。
`knowledge_questions` / `knowledge_grounded_questions` は題材に合わせて変更してください。
知識質問が不要なら両方を空配列にします。

- `train.jsonl` / `valid.jsonl`: `{"messages": [{"role": "user", "content": "質問"}, {"role": "assistant", "content": "教師応答"}]}`
- `test.jsonl`: `{"question": "評価用の質問", "faq": "回答の根拠となる文書"}`
- `gold.json`: `{"評価用の質問": {"needs_review": false, "must_include": ["必須の語句"]}}`

評価器は `answer` / `needs_review` の JSON を前提にします。
通常の文章を教師応答にする場合は、評価プロンプトと評価器も目的に合わせて変更してください。

```bash
uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
# 学習・検証に分割し、独自データ用フォルダに train.jsonl / valid.jsonl として置く。
# test.jsonl / gold.json と、題材に合わせた task.json も用意する。
.venv-lora/bin/python -m lora.train \
  --set 'experiment.name="glossary"' \
  --set 'data.task_file="data/my_glossary/task.json"' \
  --set 'data.dataset_dir="data/my_glossary"'
```

## 4bit 量子化との比較（任意）

```bash
uv pip install --python .venv-lora/bin/python 'bitsandbytes==0.50.*'
.venv-lora/bin/python -m lora.train --set experiment.quantization_eval=true
```

CUDA 環境で同じ基盤モデルを NF4 で読み込み、学習前の結果と比較します。
この追加評価の前に LoRA の結果を保存し、学習用モデルを解放します。
QLoRA の学習は実装していません。

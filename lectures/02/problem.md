# 第2回 演習・宿題：LoRA で回答形式を学習し、学習前後を比較する

この回は CUDA GPU のある環境（Google Colab の T4 など）で Python スクリプトを実行します。
セットアップと Colab での実行方法は [lora/README.md](../../lora/README.md) を参照してください。

## この回のゴール

- 小さなモデルを LoRA で短時間学習させ、学習の前後で **何が変わり、何が変わらないか** を数字と出力で確かめる
- 「形式は学習で身につくが、知らないことへの判断（確認を求めるか、断定するか）までは保証されない」ことを、自分の結果で説明できる
- ファインチューニングが向く場面と向かない場面を、自分の言葉で説明できる

## 使うコードと設定

- [lora/train.py](../../lora/train.py)：学習前評価 → LoRA 学習 → 学習後評価・知識質問
- [lora/configs/baseline.toml](../../lora/configs/baseline.toml)：基準設定
- [data/lora/faq.json](../../data/lora/faq.json)：架空 FAQ・質問・正解の語句

架空の講座 FAQ（5 項目）を題材に、`Qwen/Qwen3.5-0.8B` を LoRA で 20 エポック
（学習データ 14 件を 20 周、40 ステップ。元の Colab T4 での目安は 4〜5 分）学習します。
望ましい出力は `{"answer": ..., "needs_review": ...}` の JSON 1 行で、
FAQ に無い質問には `needs_review: true` で「確認が必要」と答えるのが正解です。
評価は **形式遵守率**（JSON の形式を守れたか）と **内容正解率**
（`needs_review` と答えの語句が正解と合うか）に分けます。

## 事前準備

学習用の環境を用意し、リポジトリルートでデータ生成を確認します。
設定・データの確認だけなら GPU や学習ライブラリは不要です。

```bash
python -m lora.train --prepare-only
```

`train=14 / valid=7 / test=21 / gold=21` と表示されることを確認してください。

## 演習（30分）

1. GPU 環境を用意する。Colab では T4 GPU を選んでリポジトリを取得し、依存ライブラリを導入する
2. `python -m lora.train --config lora/configs/baseline.toml` を実行する
   （ローカルでは `.venv-lora/bin/python` を使う）
3. 学習前評価、学習、学習後評価の進捗を確認する。学習中は `trainable%` と loss の推移を見る
4. 表示された実験ディレクトリの `result.json` と `eval_before.json` / `eval_after.json` を開き、比較する
   - `format_rate` と `content_rate` はどう変わったか
   - `regressed`：学習前は形式も内容も通ったのに、学習後に落ちた質問
   - 残った失敗：FAQ に無い質問に別項目を当てて**断定**したか、FAQ にある質問に「記載がありません」と**逃げた**か
   - `knowledge`：FAQ にも学習データにも無い質問に確認を求めたか・断定したか。FAQ にある質問は対照として読む
5. 要点を `lectures/02/answer.md` に書き、作業ブランチ `lec02` から自分の fork の `main` へ PR を出す。
   `result.json` は `lectures/02/` にコピーして PR に含めてもよい
6. Colab を使った場合は結果をダウンロードまたは Drive に保存し、ランタイムを接続解除して削除する

GPU が割り当てられない、または時間内に学習が終わらないときは、
[reference/README.md](reference/README.md) の参考結果を使って比較と考察を進めます。
参考結果を使うときは、提出する `result.json` の `executed_training_myself` を `false` にし、
自分で実行できた範囲を `answer.md` に書いてください。

seed を固定しても GPU 上の学習は完全には再現しません。
隣の人と「残った質問」が違っても異常ではないので、失敗の型で比べてください。

## 宿題（どれか1つで完了）

| レベル | 内容 |
|---|---|
| 基礎 | `lora.r`・`training.learning_rate`・`training.num_train_epochs` のどれか 1 つを変えて再学習し、形式遵守率・内容正解率・最終ログの loss・学習時間を基準設定と比較する |
| 実装 | `data/lora/faq.json` を自分の題材（5 項目程度）に置き換え、`data.task_file` で指定して学習前後を比べる。JSON の形式と `needs_review` の設計はそのまま使い、「根拠なし」の質問をどう用意したかも書く |
| 発展 | 用語集から Q&A データを作り、独自データ用フォルダを `data.dataset_dir` で指定して学習する。用語集にある語・言い換え・用語集にない語を含む評価データと正解も用意する。第3回の「用語集を検索するエージェント」と比べる |

基礎の実行例：

```bash
.venv-lora/bin/python -m lora.train \
  --set 'experiment.name="epochs10"' \
  --set training.num_train_epochs=10
.venv-lora/bin/python -m lora.compare runs/lora/*
```

毎回の実行で基盤モデルを読み直すため、条件変更時にランタイムを再起動する必要はありません。
設定ファイルを残す場合は `lora/configs/rank8.toml` を参考に新しい TOML を作成してください。

発展のデータ作成：

```bash
uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
```

生成結果を `messages` 形式の `train.jsonl` / `valid.jsonl` に分け、
評価用の `test.jsonl` / `gold.json` と、題材に合うプロンプトを用意します。
データ形式と設定例は [lora/README.md](../../lora/README.md) を参照してください。
用語集の教師応答は通常の文章なので、JSON の形式を評価する場合は教師応答も合わせてください。

任意の 4bit 比較は `experiment.quantization_eval=true` で有効にできます。
これは学習をしない観察なので、どのレベルでも結果に添えてかまいません。
4bit のまま学習する QLoRA は宿題の対象外です。

## 中断した場合

checkpoint が保存されていれば、`python -m lora.train --resume <実験ディレクトリ>` で再開します。
元の実験の設定・データを使うので、ソース側の設定や FAQ を編集しても再開条件は変わりません。
checkpoint が無い場合は新規に実行してください。

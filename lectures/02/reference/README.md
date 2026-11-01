# 第2回 参考結果（講師が T4 で実行したもの）

GPU が割り当てられない、または時間内に学習が終わらないときは、ここにある結果を見ながら比較と考察を進めてください。
提出する `result.json` の `executed_training_myself` を `false` にし、自分で実行できた範囲を `lectures/02/answer.md` に書いてください。

参考結果は元の Colab ノートブックで取得したものです。新しいスクリプトの設定は TOML のセクションごとに分かれているため、
旧 `lora_r` は `lora.r`、`lr` は `training.learning_rate`、`num_train_epochs` は `training.num_train_epochs` と読み替えてください。
参考結果のファイルがまだ置かれていない場合は、講師から受け取ってください。

| ファイル | 内容 |
|---|---|
| `result.json` | 設定（`config`）、ライブラリの版、学習前後の要約（`before` / `after`）、退行した質問、知識質問の結果 |
| `eval_before.json` | 学習前の 21 問の出力原文と採点 |
| `eval_after.json` | 学習後の 21 問の出力原文と採点 |

- ファイルは GitHub の画面で開くか、「Download raw file」で保存してください
- 同じ seed でも GPU 上の学習は完全には再現しないので、自分の結果と「残った質問」が違っても異常ではありません。失敗の型で比べてください

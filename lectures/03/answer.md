# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  - 本講座の宿題難易度である「基礎・実装・発展」とそれぞれの難易度を回答でき、かつ、宿題完了条件についても答えられている。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  - ツール名:[tool call]
  - 引数: search_glossary({'query': '3レベル'})
- 1つの質問での LLM 呼び出し回数（`requests`）：2回
- 追加したツールと、docstring に書いた説明：
  - make_quiz
  - 説明:
    - 指定した回の用語から1つ選び、確認問題とその答えを返す。「第2回の確認問題を出して」のように、復習用の問題を求められたときに使う。

## 宿題

- 取り組んだレベル：基礎

### 基礎

#### Instructions から「用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。」という記述を消して答えられるか確認する

結果: 消しても search_glossary tool を用いて検索を実施し、適切な回答ができていた。

#### Instructions は既存のまま、search_glossary の docstring を変更して検索ツールを用いられるか確認する

- 既存設定からの変更点:
  - search_glossary の docstring を「住所関連の質問に答えるときに使う」と検索ツールであることを明示的に書かない
結果:
- search_glossary tool を用いて検索を実施でき、適切な回答ができていることは変わらない。
-> Instructions に search_glossary を使うことを明記しているので適切に使われているのでは？

#### Instruction・search_glossary の docstring を同時に変更して、結果が変わるか確認する
- 既存設定からの変更点:
  - Instructions から「用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。」という記述を消す。
  - search_glossary の docstring を「住所関連の質問に答えるときに使う」と検索ツールであることを明示的に書かない
- 結果:
  - search_glossary tool を用いて検索を実施でき、適切な回答ができていることは変わらない。

#### search_glossary の docstring で複数用語を比較する質問に対する挙動を追記して、呼び出し回数が減るか確認する
- 既存設定からの変更点
  - search_glossary の docstring の末尾に「複数の用語を比較する質問では、用語ごとに検索せず、すべての用語を1つのqueryにまとめて検索する」を追加。
- 比較方法:
  - 質問: 「LoRA と QLoRA の違いは？」
  - 期待される変更点: tool_calls の呼び出し回数が変更前は2回、変更後は1回になる。
- 結果
  - 変更前: 2回
  - 変更後: 1回

```
変更前
--- trace ---
[tool call]   search_glossary({'query': 'LoRA'})
[tool call]   search_glossary({'query': 'QLoRA'})
~~省略~~
[usage] details={'text_prompt_tokens': 1348}, input_text_tokens=1348, input_tokens=1348, output_tokens=174, requests=2, tool_calls=2


--- 変更後 ---

--- trace ---
[tool call]   search_glossary({'query': 'LoRA QLoRA'})
~~省略~~
[usage] details={'text_prompt_tokens': 1091}, input_text_tokens=1091, input_tokens=1091, output_tokens=218, requests=2, tool_calls=1
```



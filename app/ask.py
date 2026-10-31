"""第1回: 素のLLMに会話履歴を保ちながら質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

回答後に次の質問を入力できる。exit / quit、Ctrl+C、入力の終端で終了する。
会話履歴は起動中だけ保持し、終了すると破棄する。

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
# 読み込むとユーザー入力で矢印キーや行頭・行末への移動、入力履歴を使える。
import readline

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat: genai.chats.Chat) -> str:
    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[トークン数] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    print("終了するには exit/quit、または Ctrl+C を入力してください。\n")
    try:
        with genai.Client(api_key=get_api_key()) as client:
            # 同じ chat を使い続けると、SDK が過去の質問と回答も送信する。
            chat = client.chats.create(
                model=MODEL,
            )
            print(ask(args.question, chat))
            while True:
                question = input("\n[質問] ").strip()
                if question.lower() in ("exit", "quit"):
                    break
                if not question:
                    continue
                print(ask(question, chat))
    except (EOFError, KeyboardInterrupt):
        print("\n会話を終了しました。")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

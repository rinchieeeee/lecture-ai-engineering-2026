"""第2回 宿題（発展）: 用語集から SFT 用の Q&A データを作る。

従来の形式（user / assistant、回答は定義文）:
    python scripts/make_sft_dataset.py > glossary_sft.jsonl

学習・検証・評価をまとめて作る（回答は answer / needs_review の JSON）:
    python scripts/make_sft_dataset.py --output-dir data/glossary_sft

外部 API や追加パッケージは使わない。
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "data" / "glossary.json"
CASES = ROOT / "data" / "glossary_sft" / "cases.json"
TEMPLATES = ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]
SYSTEM = (
    "あなたは講座の用語集の回答係です。講座の用語集に記載された定義の範囲で答えてください。"
    '必ず JSON 1 行で、形式は {"answer": 文字列, "needs_review": 真偽値} とします。'
    "用語集に記載がない用語については、answer に『この講座の用語集には記載がありません。"
    "確認が必要です。』と書き、needs_review を true にしてください。"
)
REVIEW_ANSWER = "この講座の用語集には記載がありません。確認が必要です。"


def target(answer: str, needs_review: bool) -> str:
    return json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)


def chat(question: str, answer: str, needs_review: bool = False) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": target(answer, needs_review)},
    ]}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def make_bundle(terms: list[dict], output_dir: Path) -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in terms}
    if len(by_id) != len(terms):
        raise ValueError("用語 ID が重複しています")
    train, valid, test, gold = [], [], [], {}

    for t in terms:
        questions = [template.format(term=t["term"]) for template in TEMPLATES]
        # 各用語の最初の別名も学習。評価の言い換えは cases.json に別途手書きする。
        if t["aliases"] and t["aliases"][0].casefold() != t["term"].casefold():
            questions.append(f"{t['aliases'][0]}という用語を説明してください。")
        train.extend(chat(q, t["definition"]) for q in questions)
        valid.append(chat(f"講座の用語「{t['term']}」を説明するとしたら、どう答えますか？", t["definition"]))

    for case in cases["known"]:
        t = by_id[case["term_id"]]
        if not case["must_include"] or not all(w in t["definition"] for w in case["must_include"]):
            raise ValueError(f"採点語句が定義文にありません: {t['id']}")
        for category, question in [
            ("known", f"講座で扱う「{t['term']}」の意味を教えてください。"),
            ("paraphrase", case["paraphrase"]),
        ]:
            entry = {
                "id": f"{category}-{t['id']}", "category": category,
                "term_id": t["id"], "question": question,
                "target": target(t["definition"], False),
            }
            test.append(entry)
            gold[question] = {
                "answer": t["definition"], "needs_review": False,
                "must_include": case["must_include"], "rubric": case["rubric"],
                "term_id": t["id"], "category": category,
            }

    # 「未掲載」はモデルの一般知識の有無ではなく、この用語集の収録範囲で決める。
    glossary_text = json.dumps(terms, ensure_ascii=False).casefold()
    unknown_terms = []
    for split in ["train", "valid", "test"]:
        for case in cases["unknown"][split]:
            term, question = case["term"], case["question"]
            if term.casefold() in glossary_text:
                raise ValueError(f"未掲載語が用語集に含まれています: {term}")
            unknown_terms.append(term.casefold())
            if split in {"train", "valid"}:
                (train if split == "train" else valid).append(chat(question, REVIEW_ANSWER, True))
            else:
                test.append({
                    "id": case["id"], "category": "unknown", "term_id": None,
                    "question": question, "target": target(REVIEW_ANSWER, True),
                })
                gold[question] = {
                    "answer": REVIEW_ANSWER, "needs_review": True,
                    "must_include": ["記載がありません", "確認"],
                    "rubric": "用語集に未掲載と伝えて確認を求める。用語の定義を推測して断定しない。",
                    "term_id": None, "category": "unknown",
                }

    if len(set(unknown_terms)) != len(unknown_terms):
        raise ValueError("未掲載語が split 間で重複しています")
    questions = [r["messages"][1]["content"] for r in train + valid] + [r["question"] for r in test]
    if len(set(questions)) != len(questions):
        raise ValueError("train / valid / test の質問文が重複しています")
    if len(gold) != len(test) or len({r["id"] for r in test}) != len(test):
        raise ValueError("評価の質問または ID が重複しています")

    metadata = {
        "glossary_version": json.loads(GLOSSARY.read_text(encoding="utf-8"))["version"],
        "glossary_sha256": hashlib.sha256(GLOSSARY.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "term_count": len(terms),
        "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "test_categories": dict(Counter(r["category"] for r in test)),
        "system": SYSTEM,
        "context_policy": "学習・評価の入力に用語集本文や正解を付けない。学習前後で同じ system と質問を使う。",
        "split_policy": "全58語の定義を学習し、質問文を分離する。同じ語の未知の聞き方への対応を測る。",
        "unknown_policy": "用語集に未掲載なら一般に知られた用語でも needs_review=true。未掲載語自体も split 間で分離。",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [("train", train), ("valid", valid), ("test", test)]:
        write_jsonl(output_dir / f"{name}.jsonl", records)
    for name, obj in [("gold", gold), ("metadata", metadata)]:
        (output_dir / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["counts"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="学習・検証・評価ファイルの保存先")
    args = parser.parse_args()
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    if args.output_dir is not None:
        make_bundle(terms, args.output_dir)
        return
    # オプションなしの既存コマンドは従来どおり定義文を返す。
    for t in terms:
        for template in TEMPLATES:
            record = {"messages": [
                {"role": "user", "content": template.format(term=t["term"])},
                {"role": "assistant", "content": t["definition"]},
            ]}
            sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

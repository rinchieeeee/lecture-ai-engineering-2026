"""FAQ データの生成、独自 JSONL の読み込み、実験データの保存。"""

import json
from pathlib import Path

from .config import project_path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_messages(example: dict, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"},
    ]


def to_chat(example: dict, system: str) -> dict:
    return {"messages": build_messages(example, system) + [
        {"role": "assistant", "content": example["target"]},
    ]}


def make_example(task: dict, question: str, answer: str, needs_review: bool) -> dict:
    return {
        "question": question,
        "faq": "\n".join(f"- {value}" for value in task["faq"].values()),
        "target": json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False),
    }


def generate_dataset(task: dict) -> dict:
    train, valid, test, gold = [], [], [], {}

    def add_test(question, answer, needs_review, words):
        test.append(make_example(task, question, answer, needs_review))
        gold[question] = {"needs_review": needs_review, "must_include": words}

    for key, (templates, words) in task["grounded"].items():
        if len(templates) != 4:
            raise ValueError(f"grounded.{key} の質問テンプレートは 4 件にしてください")
        for question in templates[:2]:
            train.append(make_example(task, question, task["faq"][key], False))
        valid.append(make_example(task, templates[2], task["faq"][key], False))
        for question in templates[3:] + task["test_extra_grounded"][key]:
            add_test(question, task["faq"][key], False, words)
    for index, question in enumerate(task["ungrounded"]):
        bucket = (train, train, valid, test)[index % 4]
        if bucket is test:
            add_test(question, task["ungrounded_answer"], True, ["確認"])
        else:
            bucket.append(make_example(task, question, task["ungrounded_answer"], True))
    for question in task["test_extra_ungrounded"]:
        add_test(question, task["ungrounded_answer"], True, ["確認"])
    questions = [row["question"] for row in train + valid + test]
    if len(set(questions)) != len(questions):
        raise ValueError("train / valid / test の質問が重複しています")
    return {
        "train": [to_chat(example, task["system"]) for example in train],
        "valid": [to_chat(example, task["system"]) for example in valid],
        "test": test,
        "gold": gold,
    }


def validate_dataset(dataset: dict, system: str) -> None:
    for split in ("train", "valid"):
        if not dataset[split]:
            raise ValueError(f"{split} は 1 件以上必要です")
        for row in dataset[split]:
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{split} は messages 形式の JSONL にしてください")
            if not all(isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
                       and isinstance(m.get("content"), str) for m in messages):
                raise ValueError(f"{split} の messages に role / content が必要です")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{split} の末尾は assistant の教師応答にしてください")
    if not dataset["test"]:
        raise ValueError("test は 1 件以上必要です")
    seen = set()
    for example in dataset["test"]:
        question = example.get("question")
        if not isinstance(question, str) or not isinstance(example.get("faq"), str):
            raise ValueError("test には文字列の question / faq が必要です")
        if question in seen:
            raise ValueError(f"test の質問が重複しています: {question}")
        seen.add(question)
        entry = dataset["gold"].get(question, {})
        words = entry.get("must_include")
        if type(entry.get("needs_review")) is not bool or not isinstance(words, list):
            raise ValueError(f"gold に needs_review / must_include が必要です: {question}")
        if not all(isinstance(word, str) for word in words):
            raise ValueError("must_include は文字列の配列にしてください")
    # 教師応答を除いた会話が同じなら、train / valid / test 間の漏洩とみなす。
    prompts = set()
    test_prompts = [build_messages(example, system) for example in dataset["test"]]
    for messages in ([row["messages"][:-1] for row in dataset["train"]]
                     + [row["messages"][:-1] for row in dataset["valid"]] + test_prompts):
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if key in prompts:
            raise ValueError("train / valid / test の入力が重複しています")
        prompts.add(key)


def load_data(config: dict) -> tuple[dict, dict]:
    task = read_json(project_path(config["task_file"]))
    if config["dataset_dir"]:
        dataset = load_snapshot(project_path(config["dataset_dir"]))
    else:
        dataset = generate_dataset(task)
    validate_dataset(dataset, task["system"])
    return task, dataset


def load_snapshot(directory: Path) -> dict:
    return {
        **{split: read_jsonl(directory / f"{split}.jsonl") for split in ("train", "valid", "test")},
        "gold": read_json(directory / "gold.json"),
    }


def save_snapshot(directory: Path, task: dict, dataset: dict) -> None:
    write_json(directory / "task.json", task)
    write_json(directory / "gold.json", dataset["gold"])
    for split in ("train", "valid", "test"):
        lines = [json.dumps(row, ensure_ascii=False) for row in dataset[split]]
        (directory / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

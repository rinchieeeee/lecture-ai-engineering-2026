"""形式と内容を別々に採点する。生成関数を受け取り、GPU なしでも検証できる。"""

import json
from pathlib import Path
import re
import time

from .data import write_json


def check_format(text: str) -> tuple[bool, str, dict | None]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False, "json_error: JSONDecodeError", None
    if not isinstance(obj, dict):
        return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}:
        return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj:
        return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or type(obj["needs_review"]) is not bool:
        return False, "wrong_type", obj
    return True, "", obj


def check_content(obj: dict | None, gold_entry: dict) -> bool:
    if not isinstance(obj, dict) or type(obj.get("needs_review")) is not bool:
        return False
    answer = obj.get("answer")
    return (isinstance(answer, str) and obj["needs_review"] == gold_entry["needs_review"]
            and all(word in answer for word in gold_entry["must_include"]))


def classify(raw: str) -> str:
    ok, _, obj = check_format(raw)
    if not ok:
        return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"


def evaluate(generate, test: list[dict], gold: dict, label: str, output_dir: Path) -> tuple[dict, list]:
    rows = []
    started = time.monotonic()
    for index, example in enumerate(test, 1):
        raw = generate(example)
        ok_format, reason, obj = check_format(raw)
        loose = obj
        if not ok_format:
            match = re.search(r"\{.*\}", raw, re.S)
            try:
                loose = json.loads(match.group(0)) if match else None
            except json.JSONDecodeError:
                loose = None
        entry = gold[example["question"]]
        ok_content = check_content(loose, entry)
        rows.append({
            "question": example["question"], "raw": raw,
            "format_ok": ok_format, "format_reason": reason,
            "content_ok": ok_content, "both_ok": ok_format and ok_content,
            "gold_needs_review": entry["needs_review"],
        })
        print(f"[{label} {index}/{len(test)}] 形式={ok_format} 内容={ok_content}", flush=True)
    n = len(rows)
    summary = {
        "label": label, "n": n,
        **{key.replace("_ok", "_rate"): sum(row[key] for row in rows) / n
           for key in ("format_ok", "content_ok", "both_ok")},
        "seconds": round(time.monotonic() - started, 1),
    }
    write_json(output_dir / f"eval_{label}.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, rows

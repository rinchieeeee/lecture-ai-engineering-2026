"""同一実験の学習前後を再採点し、4分類の比較記録を作る（標準ライブラリのみ）。"""

import argparse
import csv
import json
from pathlib import Path
import re

GROUPS = {
    "improved": "学習後に正解できるようになった問題",
    "regressed": "学習後に不正解になってしまった問題",
    "both_correct": "学習前後でともに正解だった問題",
    "both_incorrect": "学習前後でともに不正解だった問題",
}
CATEGORIES = {"known": "掲載語", "paraphrase": "言い換え", "unknown": "未掲載語"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score(raw: str, gold: dict) -> dict:
    """教材と同じく形式と内容を分離。形式が崩れてもJSON部分が読めれば内容を採点する。"""
    obj, reason = None, ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        reason = "JSONとして読めない"
    if not reason:
        if not isinstance(obj, dict):
            reason = "JSONオブジェクトではない"
        elif set(obj) != {"answer", "needs_review"}:
            reason = "answer / needs_review 以外のキーがある、または必要なキーがない"
        elif not isinstance(obj["answer"], str) or type(obj["needs_review"]) is not bool:
            reason = "answer が文字列でない、または needs_review が真偽値でない"
    format_ok = not reason
    if not format_ok:
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            obj = json.loads(match.group()) if match else None
        except json.JSONDecodeError:
            obj = None
    parsed = obj if isinstance(obj, dict) else {}
    answer = parsed.get("answer")
    review_ok = type(parsed.get("needs_review")) is bool and parsed["needs_review"] == gold["needs_review"]
    missing = [w for w in gold["must_include"] if not isinstance(answer, str) or w not in answer]
    content_ok = isinstance(answer, str) and review_ok and not missing
    return {
        "raw": raw, "answer": answer, "needs_review": parsed.get("needs_review"),
        "format_ok": format_ok, "format_reason": reason,
        "content_ok": content_ok, "both_ok": format_ok and content_ok,
        "review_matches": review_ok, "missing_keywords": missing,
    }


def transition(before: bool, after: bool) -> str:
    return {(False, True): "improved", (True, False): "regressed",
            (True, True): "both_correct", (False, False): "both_incorrect"}[before, after]


def rates(rows: list[dict], phase: str) -> dict:
    n = len(rows)
    return {"n": n, **{key.replace("_ok", "_rate"): sum(r[phase][key] for r in rows) / n
                       for key in ("format_ok", "content_ok", "both_ok")}}


def indexed_rows(evaluation: dict, questions: set[str]) -> dict:
    rows = evaluation["rows"]
    by_question = {r["question"]: r for r in rows}
    if len(by_question) != len(rows):
        raise ValueError("評価結果に質問の重複があります")
    if set(by_question) != questions:
        raise ValueError("評価結果の質問がtestと一致しません（欠落・追加があります）")
    if not all(isinstance(r.get("raw"), str) for r in rows):
        raise ValueError("回答原文がありません")
    return by_question


def fenced(text: str) -> str:
    # モデル出力にコードフェンスがあっても、原文として表示する。
    fence = "`" * max(3, max((len(m) + 1 for m in re.findall(r"`+", text)), default=3))
    return f"{fence}text\n{text}\n{fence}"


def diagnosis(result: dict) -> str:
    details = []
    if not result["format_ok"]:
        details.append("形式: " + result["format_reason"])
    if not result["review_matches"]:
        details.append("needs_review が正解と不一致または欠落")
    if result["missing_keywords"]:
        details.append("未含語句: " + "、".join(result["missing_keywords"]))
    if result.get("hit_token_limit"):
        details.append("生成トークン数の上限に到達（回答が途中で切れた可能性）")
    return " / ".join(details) or "形式・内容ともに合格"


def make_comparison(run_dir: Path) -> dict:
    before, after = [read_json(run_dir / f"eval_{p}.json") for p in ["before", "after"]]
    config = read_json(run_dir / "config.json")
    protocol = config["protocol"]
    if not protocol or before.get("protocol") != protocol or after.get("protocol") != protocol:
        raise ValueError("学習前後のモデル・データ・生成条件・実験IDが一致しません")
    test_path = run_dir / "data" / "test.jsonl"
    gold_path = run_dir / "data" / "gold.json"
    import hashlib
    for path in [test_path, gold_path]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["dataset_sha256"][path.name]:
            raise ValueError(f"保存した評価データが変更されています: {path.name}")
    test = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    gold = read_json(gold_path)
    questions = {r["question"] for r in test}
    if not test or len(questions) != len(test) or len({r["id"] for r in test}) != len(test) or set(gold) != questions:
        raise ValueError("test / gold の質問やIDに重複、欠落があります")
    if any(r["category"] not in CATEGORIES for r in test):
        raise ValueError("未対応の評価分類です")
    phase_rows = {p: indexed_rows(e, questions) for p, e in [("before", before), ("after", after)]}
    rows = []
    for t in test:
        q = t["question"]
        a, b = [score(phase_rows[p][q]["raw"], gold[q]) for p in ["before", "after"]]
        for phase, evaluated in [("before", a), ("after", b)]:
            evaluated["generated_tokens"] = phase_rows[phase][q].get("generated_tokens")
            evaluated["hit_token_limit"] = phase_rows[phase][q].get("hit_token_limit")
        rows.append({
            "id": t["id"], "category": t["category"], "question": q,
            "gold": gold[q], "before": a, "after": b,
            "group": transition(a["both_ok"], b["both_ok"]),
            "content_group": transition(a["content_ok"], b["content_ok"]),
        })
    groups = {key: [r["id"] for r in rows if r["group"] == key] for key in GROUPS}
    result = {
        "protocol": protocol,
        "correctness_rule": "JSON形式と内容の両方を満たせば正解。内容はneeds_reviewの一致と必須語句の包含。自動採点。",
        "summary": {"before": rates(rows, "before"), "after": rates(rows, "after"),
                    "groups": {key: len(ids) for key, ids in groups.items()}},
        "by_category": {c: {p: rates([r for r in rows if r["category"] == c], p) for p in ["before", "after"]}
                        for c in CATEGORIES if any(r["category"] == c for r in rows)},
        "groups": groups, "rows": rows,
    }
    write_json(run_dir / "comparison.json", result)

    fields = ["id", "category", "group", "content_group", "question", "gold_answer", "gold_needs_review", "rubric"]
    for phase in ["before", "after"]:
        fields.extend(f"{phase}_{k}" for k in ["answer", "needs_review", "format_ok", "content_ok", "both_ok", "raw", "generated_tokens", "hit_token_limit", "diagnosis"])
    with (run_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            record = {k: r[k] for k in ["id", "question"]}
            record.update(category=CATEGORIES[r["category"]], group=GROUPS[r["group"]], content_group=GROUPS[r["content_group"]],
                          gold_answer=r["gold"]["answer"], gold_needs_review=r["gold"]["needs_review"], rubric=r["gold"]["rubric"])
            for phase in ["before", "after"]:
                record.update({f"{phase}_{k}": r[phase][k] for k in ["answer", "needs_review", "format_ok", "content_ok", "both_ok", "raw", "generated_tokens", "hit_token_limit"]})
                record[f"{phase}_diagnosis"] = diagnosis(r[phase])
            writer.writerow(record)

    md = ["# 用語集Q&A：学習前後の比較", "",
          "同じ評価質問について、回答のJSON形式と内容の両方を満たせば正解として4分類しました。", "",
          "**簡易自動採点の結果です。** 同義表現でも必須語句がなければ不正解になります。正解文・目視採点基準・回答原文を各問に残してあるので、意味も確認してください。", "",
          f"モデル: `{protocol['model']}` / revision: `{protocol['revision']}`", "",
          f"実験ID: `{protocol['experiment_id']}`。条件は [config.json](config.json)、学習記録は [training.json](training.json)、原文は [eval_before.json](eval_before.json) / [eval_after.json](eval_after.json)。", "",
          "| 指標 | 学習前 | 学習後 |", "|---|---:|---:|"]
    for key, label in [("format_rate", "形式遵守率"), ("content_rate", "内容正解率"), ("both_rate", "形式・内容ともに正解")]:
        md.append(f"| {label} | {result['summary']['before'][key]:.1%} | {result['summary']['after'][key]:.1%} |")
    md.extend(["", f"生成上限に達した回答: 学習前 {sum(bool(r['before'].get('hit_token_limit')) for r in rows)}件 / 学習後 {sum(bool(r['after'].get('hit_token_limit')) for r in rows)}件。回答が途中で切れた可能性があるため、原文も確認してください。", "", "| 質問の種類 | 件数 | 学習前に正解 | 学習後に正解 |", "|---|---:|---:|---:|"])
    for c, values in result["by_category"].items():
        n = values["before"]["n"]
        md.append(f"| {CATEGORIES[c]} | {n} | {round(values['before']['both_rate']*n)} | {round(values['after']['both_rate']*n)} |")
    md.extend(["", "| 分類 | 件数 |", "|---|---:|"])
    for key, label in GROUPS.items():
        md.append(f"| {label} | {len(groups[key])} |")
    if (run_dir / "comparison.png").exists():
        md.extend(["", "![質問の種類別の学習前後の正解件数](comparison.png)"])
    md.extend(["", "学習には掲載語の定義が含まれます。未知の質問文への対応を測った固定36問の単一実行であり、一般的な性能向上を保証するものではありません。未掲載語は一般に実在しても、用語集にないため確認を求めるのが正解です。", "",
               "CSVの `group` は形式・内容の両方、`content_group` は内容だけの変化です。学習前後の条件は固定し、評価入力には用語集本文・正解・必須語句を渡していません。", ""])
    for key, label in GROUPS.items():
        md.extend([f"## {label}（{len(groups[key])}件）", ""])
        if not groups[key]:
            md.extend(["該当なし。", ""])
        for r in rows:
            if r["group"] != key:
                continue
            md.extend([f"### {r['id']}（{CATEGORIES[r['category']]}）", "", r["question"], "",
                       "**正解例**: " + r["gold"]["answer"], "",
                       f"**期待する needs_review**: `{str(r['gold']['needs_review']).lower()}`", "",
                       "**目視採点基準**: " + r["gold"]["rubric"], ""])
            for phase, label_phase in [("before", "学習前"), ("after", "学習後")]:
                s = r[phase]
                md.extend([f"**{label_phase}**: {'正解' if s['both_ok'] else '不正解'}（形式 {'○' if s['format_ok'] else '×'} / 内容 {'○' if s['content_ok'] else '×'}）", "",
                           diagnosis(s), "", fenced(s["raw"]), ""])
    (run_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = make_comparison(args.run_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

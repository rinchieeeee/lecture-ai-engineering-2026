"""比較の4分類と、異なる実験・不完全な評価結果の混入を検証する。GPU不要。"""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_glossary_results import make_comparison, score, transition, write_json


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        self.gold_entry = {"answer": "元の重みを固定する手法", "needs_review": False,
                           "must_include": ["固定"], "rubric": "元の重みが固定されているか"}
        self.correct = json.dumps({"answer": self.gold_entry["answer"], "needs_review": False}, ensure_ascii=False)
        self.tests = [{"id": f"q{i}", "question": f"質問{i}", "category": "known"} for i in range(4)]
        self.gold = {r["question"]: self.gold_entry for r in self.tests}
        test_path = self.root / "data/test.jsonl"
        test_path.write_text("".join(json.dumps(r) + "\n" for r in self.tests))
        write_json(self.root / "data/gold.json", self.gold)
        self.protocol = {
            "experiment_id": "test", "model": "test/model", "revision": "fixed",
            "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root / "data").iterdir()},
        }
        write_json(self.root / "config.json", {"protocol": self.protocol})
        before_ok = [False, True, True, False]
        after_ok = [True, False, True, False]
        for phase, flags in [("before", before_ok), ("after", after_ok)]:
            rows = [{"question": r["question"], "raw": self.correct if ok else "不明です。", "both_ok": not ok}
                    for r, ok in zip(self.tests, flags)]
            # 行順に依存せず、保存された誤った正誤フラグを信用しない。
            if phase == "after":
                rows.reverse()
            write_json(self.root / f"eval_{phase}.json", {"protocol": self.protocol, "rows": rows})

    def test_four_transitions_and_saved_outputs(self):
        result = make_comparison(self.root)
        self.assertEqual(result["summary"]["groups"], {"improved": 1, "regressed": 1, "both_correct": 1, "both_incorrect": 1})
        self.assertEqual(result["groups"]["improved"], ["q0"])
        self.assertEqual(result["groups"]["regressed"], ["q1"])
        self.assertEqual(result["summary"]["before"]["both_rate"], 0.5)
        self.assertEqual(result["summary"]["after"]["both_rate"], 0.5)
        markdown = (self.root / "comparison.md").read_text()
        self.assertIn("元の重みが固定されているか", markdown)
        self.assertIn("不明です。", markdown)
        with (self.root / "comparison.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["after_raw"], self.correct)

    def test_empty_groups_are_present(self):
        for phase in ["before", "after"]:
            rows = [{"question": r["question"], "raw": self.correct} for r in self.tests]
            write_json(self.root / f"eval_{phase}.json", {"protocol": self.protocol, "rows": rows})
        result = make_comparison(self.root)
        self.assertEqual(result["summary"]["groups"]["regressed"], 0)
        self.assertEqual(result["summary"]["groups"]["both_correct"], 4)
        self.assertIn("該当なし。", (self.root / "comparison.md").read_text())

    def test_partial_evaluation_is_rejected(self):
        write_json(self.root / "eval_after.json", {"protocol": self.protocol, "rows": []})
        with self.assertRaisesRegex(ValueError, "欠落"):
            make_comparison(self.root)

    def test_duplicate_question_is_rejected(self):
        rows = [{"question": "質問0", "raw": self.correct}] * 4
        write_json(self.root / "eval_after.json", {"protocol": self.protocol, "rows": rows})
        with self.assertRaisesRegex(ValueError, "重複"):
            make_comparison(self.root)

    def test_mixed_experiment_is_rejected(self):
        value = json.loads((self.root / "eval_after.json").read_text())
        value["protocol"]["experiment_id"] = "different"
        write_json(self.root / "eval_after.json", value)
        with self.assertRaisesRegex(ValueError, "実験ID"):
            make_comparison(self.root)

    def test_changed_gold_is_rejected(self):
        write_json(self.root / "data/gold.json", {})
        with self.assertRaisesRegex(ValueError, "変更"):
            make_comparison(self.root)

    def test_generation_limit_remains_visible(self):
        value = json.loads((self.root / "eval_before.json").read_text())
        value["rows"][0].update(generated_tokens=384, hit_token_limit=True)
        write_json(self.root / "eval_before.json", value)
        result = make_comparison(self.root)
        self.assertTrue(result["rows"][0]["before"]["hit_token_limit"])
        self.assertEqual(result["rows"][0]["before"]["generated_tokens"], 384)
        self.assertIn("生成トークン数の上限に到達", (self.root / "comparison.md").read_text())

    def test_wrapped_json_has_content_but_fails_format(self):
        result = score("```json\n" + self.correct + "\n```", self.gold_entry)
        self.assertTrue(result["content_ok"])
        self.assertFalse(result["both_ok"])

    def test_wrong_review_type_and_missing_keyword_fail(self):
        for obj in [{"answer": "固定", "needs_review": 0}, {"answer": "更新", "needs_review": False}]:
            self.assertFalse(score(json.dumps(obj), self.gold_entry)["content_ok"])
        self.assertEqual(transition(True, False), "regressed")


if __name__ == "__main__":
    unittest.main()

"""生成データをGitに含めず、記録された実験入力を再現できることを確認する。"""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.make_sft_dataset import GLOSSARY, ROOT, make_bundle


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
        with redirect_stdout(StringIO()):
            make_bundle(self.terms, self.output)

    def read_rows(self, name):
        return [json.loads(s) for s in (self.output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_regenerated_inputs_match_recorded_experiment(self):
        record = json.loads((ROOT / "lectures/02/reference/glossary_sft/summary.json").read_text(encoding="utf-8"))
        for name in ["train.jsonl", "valid.jsonl", "test.jsonl", "gold.json", "metadata.json"]:
            actual = hashlib.sha256((self.output / name).read_bytes()).hexdigest()
            self.assertEqual(actual, record["protocol"]["dataset_sha256"][name], name)

    def test_split_boundaries_and_answer_coverage(self):
        train, valid, test = [self.read_rows(s) for s in ["train", "valid", "test"]]
        self.assertEqual([len(train), len(valid), len(test)], [182, 62, 36])
        questions = [r["messages"][1]["content"] for r in train + valid] + [r["question"] for r in test]
        self.assertEqual(len(questions), len(set(questions)))
        answers = {json.loads(r["messages"][-1]["content"])["answer"] for r in train}
        self.assertTrue({t["definition"] for t in self.terms} <= answers)
        for rows in [train, valid]:
            for row in rows:
                self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
                target = json.loads(row["messages"][-1]["content"])
                self.assertEqual(set(target), {"answer", "needs_review"})
                self.assertIs(type(target["needs_review"]), bool)
        for category in ["known", "paraphrase", "unknown"]:
            self.assertEqual(sum(r["category"] == category for r in test), 12)

    def test_legacy_stdout_remains_plain_definition_messages(self):
        stdout = subprocess.check_output([sys.executable, str(ROOT / "scripts/make_sft_dataset.py")], text=True)
        rows = [json.loads(line) for line in stdout.splitlines()]
        self.assertEqual(len(rows), len(self.terms) * 2)
        for index, term in enumerate(self.terms):
            for row in rows[index * 2:index * 2 + 2]:
                self.assertEqual([m["role"] for m in row["messages"]], ["user", "assistant"])
                self.assertEqual(row["messages"][-1]["content"], term["definition"])


if __name__ == "__main__":
    unittest.main()

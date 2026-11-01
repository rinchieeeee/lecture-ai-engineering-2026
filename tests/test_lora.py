"""LoRA 実験の設定・データ漏洩・採点を GPU / 学習ライブラリなしで検証する。"""

import copy
import json

import pytest

from lora.config import ROOT, load_config
from lora.data import (load_data, load_snapshot, read_json, save_snapshot,
                       validate_dataset)
from lora.evaluation import check_content, check_format, classify, evaluate
from lora.train import main


def test_partial_config_and_cli_overrides():
    config = load_config(ROOT / "lora/configs/rank8.toml", ["training.learning_rate=1e-4"])
    assert config["lora"]["r"] == 8
    assert config["lora"]["lora_alpha"] == 32
    assert config["training"]["learning_rate"] == 1e-4
    assert config["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert load_config()["lora"]["r"] == 16


@pytest.mark.parametrize("override", [
    "training.learing_rate=1e-4", "lora.r=0", "lora.r=true", "lora.r=1.5",
    "training.num_train_epochs=nan", 'model.device="tpu"',
    'experiment.name="../outside"', "generation.do_sample=1", "badformat",
])
def test_invalid_config_fails_before_training(override):
    with pytest.raises(ValueError):
        load_config(overrides=[override])


def test_faq_splits_and_saved_data(tmp_path):
    task, data = load_data(load_config()["data"])
    assert [len(data[split]) for split in ("train", "valid", "test", "gold")] == [14, 7, 21, 21]
    assert sum(entry["needs_review"] for entry in data["gold"].values()) == 6
    assert "締切を教えてください" not in data["gold"]  # valid 専用
    assert "提出の期限は？" in data["gold"]
    assert data["train"][0]["messages"][-1]["role"] == "assistant"
    save_snapshot(tmp_path, task, data)
    assert load_snapshot(tmp_path) == data
    custom_task, custom_data = load_data({
        "task_file": str(tmp_path / "task.json"), "dataset_dir": str(tmp_path)})
    assert custom_task == task
    assert custom_data == data


def test_data_leakage_is_rejected():
    task, data = load_data(load_config()["data"])
    data["valid"].append(copy.deepcopy(data["train"][0]))
    with pytest.raises(ValueError, match="重複"):
        validate_dataset(data, task["system"])
    task["test_extra_ungrounded"].append(task["ungrounded"][0])
    with pytest.raises(ValueError, match="重複"):
        from lora.data import generate_dataset
        generate_dataset(task)


@pytest.mark.parametrize("raw,reason", [
    ('{"answer":"確認が必要", "needs_review":true}', ""),
    ('{"answer":"確認が必要"}', "missing_key"),
    ('{"answer":"確認が必要", "needs_review":"true"}', "wrong_type"),
    ('{"answer":"確認が必要", "needs_review":true, "other":1}', "extra_keys"),
    ('[]', "not_object"),
    ('```json\n{"answer":"確認が必要", "needs_review":true}\n```', "json_error: JSONDecodeError"),
])
def test_format_rules(raw, reason):
    ok, actual, _ = check_format(raw)
    assert ok == (reason == "")
    assert actual == reason


@pytest.mark.parametrize("answer", [None, 123, ["確認"], {"確認": "確認"}])
def test_wrong_answer_types_do_not_crash_content_scoring(answer):
    assert not check_content({"answer": answer, "needs_review": True},
                             {"needs_review": True, "must_include": ["確認"]})


def test_integer_is_not_a_boolean_in_content_scoring():
    assert not check_content({"answer": "確認", "needs_review": 1},
                             {"needs_review": True, "must_include": ["確認"]})


def test_content_and_format_are_scored_separately(tmp_path):
    test = [{"question": "根拠なし"}, {"question": "根拠あり"}]
    gold = {"根拠なし": {"needs_review": True, "must_include": ["確認"]},
            "根拠あり": {"needs_review": False, "must_include": ["PDF"]}}
    answers = {
        "根拠なし": '説明\n{"answer":"確認が必要", "needs_review":true}',
        "根拠あり": '{"answer":"PDF", "needs_review":true}',
    }
    summary, rows = evaluate(lambda example: answers[example["question"]], test, gold, "before", tmp_path)
    assert summary["format_rate"] == 0.5
    assert summary["content_rate"] == 0.5
    assert summary["both_rate"] == 0
    assert read_json(tmp_path / "eval_before.json")["rows"] == rows
    assert classify(answers["根拠なし"]) == "format_error"
    assert classify(answers["根拠あり"]) == "needs_review"
    assert classify('{"answer":"PDF", "needs_review":false}') == "asserted"


def test_prepare_only_creates_independent_runs_without_ml_dependencies(tmp_path):
    args = ["--prepare-only", "--set", f"experiment.output_root={json.dumps(str(tmp_path))}",
            "--set", "lora.r=4"]
    main(args)
    main(args)
    outputs = list(tmp_path.iterdir())
    assert len(outputs) == 2
    for output in outputs:
        assert read_json(output / "config.json")["lora"]["r"] == 4
        assert len(load_snapshot(output)["train"]) == 14
        assert not (output / "adapter").exists()


def test_resume_uses_saved_data_even_if_source_changes(tmp_path, monkeypatch):
    main(["--prepare-only", "--set", f"experiment.output_root={json.dumps(str(tmp_path))}"])
    output = next(tmp_path.iterdir())
    checkpoint = output / "checkpoints/checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}")
    config = read_json(output / "config.json")
    config["data"]["task_file"] = "/no/longer/exists.json"
    (output / "config.json").write_text(json.dumps(config))
    captured = []
    monkeypatch.setattr("lora.train.run_training", lambda *args, **kwargs: captured.append((args, kwargs)))
    main(["--resume", str(output)])
    assert captured[0][0][0] == config
    assert len(captured[0][0][2]["test"]) == 21
    assert captured[0][1]["resume"] is True
    with pytest.raises(SystemExit) as error:
        main(["--resume", str(output), "--set", "lora.r=8"])
    assert error.value.code == 2


def test_resume_requires_checkpoint(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--resume", str(tmp_path)])
    assert error.value.code == 2
    assert "checkpoint がありません" in capsys.readouterr().err

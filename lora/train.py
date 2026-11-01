"""設定を指定して学習前評価 → LoRA 学習 → 学習後評価を実行する。"""

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
from pathlib import Path
import time
from uuid import uuid4

from .config import load_config, project_path
from .data import (build_messages, load_data, load_snapshot, make_example,
                   read_json, save_snapshot, write_json)
from .evaluation import classify, evaluate


def run_training(config: dict, task: dict, dataset: dict, output: Path, resume: bool) -> None:
    # --help / --prepare-only では、GPU 用のライブラリをインポートしない。
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = config["model"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if config["training"]["fp16"] and device != "cuda":
        raise ValueError("fp16 学習は CUDA を指定してください")
    if config["experiment"]["quantization_eval"] and device != "cuda":
        raise ValueError("4bit 比較は CUDA で実行してください")
    if config["experiment"]["quantization_eval"]:
        version("bitsandbytes")  # 長い学習の前に、オプションの依存を確認する。

    source = {"pretrained_model_name_or_path": config["model"]["name"],
              "revision": config["model"]["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(**source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        **source, dtype=getattr(torch, config["model"]["dtype"])).to(device)
    model.config.use_cache = True
    model.eval()
    memory_gb = round(model.get_memory_footprint() / 1e9, 2)
    print(f"loaded: {config['model']['name']} | device: {device} | 重みのメモリ: {memory_gb} GB", flush=True)

    def generator(current_model):
        def generate(example):
            prompt = tokenizer.apply_chat_template(
                build_messages(example, task["system"]), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(current_model.device)
            with torch.no_grad():
                tokens = current_model.generate(
                    **inputs, pad_token_id=tokenizer.pad_token_id, **config["generation"])
            return tokenizer.decode(tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return generate

    before_file = output / "eval_before.json"
    if resume and before_file.exists():
        before = read_json(before_file)
        before_summary, before_rows = before["summary"], before["rows"]
    else:
        before_summary, before_rows = evaluate(
            generator(model), dataset["test"], dataset["gold"], "before", output)

    set_seed(seed)
    model.config.use_cache = False
    lora = LoraConfig(**config["lora"], bias="none", task_type="CAUSAL_LM")
    training = SFTConfig(
        **config["training"], output_dir=str(output / "checkpoints"),
        packing=False, report_to="none", seed=seed, use_cpu=device == "cpu")
    trainer = SFTTrainer(
        model=model, args=training, peft_config=lora, processing_class=tokenizer,
        train_dataset=Dataset.from_list(dataset["train"]),
        eval_dataset=Dataset.from_list(dataset["valid"]))
    trainer.model.print_trainable_parameters()
    started = time.monotonic()
    trained = trainer.train(resume_from_checkpoint=True if resume else None)
    elapsed = round(time.monotonic() - started, 1)
    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    write_json(output / "training_metrics.json", trained.metrics)
    write_json(output / "training_history.json", trainer.state.log_history)

    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()
    after_summary, after_rows = evaluate(
        generator(tuned), dataset["test"], dataset["gold"], "after", output)
    after_by_question = {row["question"]: row for row in after_rows}
    regressed = [row["question"] for row in before_rows
                 if row["both_ok"] and not after_by_question[row["question"]]["both_ok"]]

    knowledge = []
    for question in task["knowledge_questions"]:
        example = make_example(task, question, "", True)
        # 学習後の model にも adapter が差し込まれているので、明示的に無効化する。
        with tuned.disable_adapter():
            before_raw = generator(tuned)(example)
        after_raw = generator(tuned)(example)
        knowledge.append({
            "質問": question, "FAQ にある": question in task["knowledge_grounded_questions"],
            "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
            "学習後の分類": classify(after_raw), "学習後の出力": after_raw,
        })
    result = {
        "config": config,
        "versions": {name: version(name) for name in (
            "torch", "transformers", "peft", "trl", "accelerate", "datasets")},
        "method": "LoRA（量子化なし）", "device": device, "base_memory_gb": memory_gb,
        "training": {
            "seconds": elapsed, "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch, "train_loss": trained.training_loss,
            "last_logged_loss": next((item["loss"] for item in reversed(trainer.state.log_history)
                                      if "loss" in item), None),
            "resumed": resume,
        },
        "before": before_summary, "after": after_summary,
        "regressed": regressed, "knowledge": knowledge, "executed_training_myself": True,
    }
    write_json(output / "result.json", result)
    print(f"学習時間: {elapsed} 秒 | 退行: {regressed}", flush=True)

    if config["experiment"]["quantization_eval"]:
        from transformers import BitsAndBytesConfig

        # 学習モデルを解放し、4bit の比較時に GPU に二重に載せない。
        del tuned, trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        quantized = AutoModelForCausalLM.from_pretrained(
            **source, device_map="auto", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        quantized.eval()
        summary, _ = evaluate(generator(quantized), dataset["test"], dataset["gold"], "before_4bit", output)
        result["quantization"] = {"summary": summary,
                                  "memory_gb": round(quantized.get_memory_footprint() / 1e9, 2)}
        result["versions"]["bitsandbytes"] = version("bitsandbytes")
        write_json(output / "result.json", result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="baseline.toml に上書きする TOML")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="TOML 値で設定を上書き（複数指定可）")
    parser.add_argument("--prepare-only", action="store_true", help="設定検証・データ保存だけ実行（GPU 不要）")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="実験ディレクトリの最新 checkpoint から再開")
    args = parser.parse_args(argv)
    try:
        if args.resume:
            if args.config or args.set or args.prepare_only:
                raise ValueError("--resume は --config / --set / --prepare-only と併用できません")
            output = args.resume.resolve()
            if not any((output / "checkpoints").glob("checkpoint-*/trainer_state.json")):
                raise ValueError(f"再開できる checkpoint がありません: {output}")
            config, task = read_json(output / "config.json"), read_json(output / "task.json")
            dataset = load_snapshot(output)
        else:
            config = load_config(args.config, args.set)
            task, dataset = load_data(config["data"])
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            name = f"{config['experiment']['name']}-{stamp}-{uuid4().hex[:8]}"
            output = project_path(config["experiment"]["output_root"]) / name
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "config.json", config)
            save_snapshot(output, task, dataset)
        print(f"実験ディレクトリ: {output}", flush=True)
        print(" / ".join(f"{split}={len(dataset[split])}" for split in ("train", "valid", "test", "gold")), flush=True)
        if not args.prepare_only:
            run_training(config, task, dataset, output, resume=bool(args.resume))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()

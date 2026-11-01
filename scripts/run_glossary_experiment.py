"""固定した用語集データで学習前評価→LoRA学習→学習後評価→4分類を実行する。"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

from scripts.compare_glossary_results import make_comparison, read_json, score, write_json

ROOT = Path(__file__).resolve().parent.parent


def run(data_dir: Path, output: Path, epochs: float) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("この実験の実行にはCUDA GPUを使用します")
    if epochs <= 0:
        raise ValueError("epochs は正の値にしてください")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "data"
    snapshot.mkdir()
    names = ["train.jsonl", "valid.jsonl", "test.jsonl", "gold.json", "metadata.json", "cases.json"]
    for name in names:
        shutil.copyfile(data_dir / name, snapshot / name)
    data = {split: [json.loads(line) for line in (snapshot / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            for split in ["train", "valid", "test"]}
    gold = read_json(snapshot / "gold.json")
    system = data["train"][0]["messages"][0]["content"]
    set_seed(42)
    model_name = "Qwen/Qwen3.5-0.8B"
    revision = "2fc06364715b967f1860aea9cf38778875588b17"
    source = {"pretrained_model_name_or_path": model_name, "revision": revision, "local_files_only": True}
    tok = AutoTokenizer.from_pretrained(**source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths = {}
    for split in ["train", "valid"]:
        lengths[split] = max(len(tok(tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"])
                             for r in data[split])
    max_length = max(512, ((max(lengths.values()) + 63) // 64) * 64)
    generation = {"max_new_tokens": 384, "do_sample": False,
                  "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id}
    protocol = {
        "experiment_id": output.name, "model": model_name, "revision": revision,
        "dtype": "float32", "seed": 42, "generation": generation,
        "enable_thinking": False, "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "dataset_sha256": {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in names},
    }
    training = {
        "num_train_epochs": epochs, "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2, "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine", "warmup_steps": 6,
        "logging_steps": 10, "save_steps": 100, "save_total_limit": 1,
        "fp16": False, "bf16": False, "max_length": max_length,
        "packing": False, "report_to": "none", "seed": 42,
        "eval_strategy": "no", "per_device_eval_batch_size": 1,
        "disable_tqdm": True,
    }
    lora = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    config = {
        "protocol": protocol, "training": training, "lora": lora, "system": system,
        "max_training_tokens": lengths, "gpu": torch.cuda.get_device_name(0),
        "torch_runtime_version": torch.__version__, "cuda_version": torch.version.cuda,
        "versions": {name: version(name) for name in ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "全掲載語の定義を学習。評価では本文を渡さず、分離した36問を学習前後に各1回生成する。",
    }
    write_json(output / "config.json", config)
    print(f"出力先: {output}\n最大トークン長: {lengths} / max_length={max_length}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(**source, dtype=torch.float32).to("cuda").eval()
    model.config.use_cache = True

    def evaluate(current_model, label: str) -> None:
        current_model.eval()
        rows = []
        started = time.monotonic()
        for i, ex in enumerate(data["test"], 1):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": ex["question"]}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(current_model.device)
            with torch.inference_mode():
                tokens = current_model.generate(**inputs, **generation)
            generated = tokens[0][inputs["input_ids"].shape[1]:]
            raw = tok.decode(generated, skip_special_tokens=True).strip()
            row = {"id": ex["id"], "question": ex["question"], **score(raw, gold[ex["question"]]),
                   "generated_tokens": len(generated), "hit_token_limit": len(generated) >= generation["max_new_tokens"]}
            rows.append(row)
            summary = {"n": len(rows), **{key.replace("_ok", "_rate"): sum(r[key] for r in rows) / len(rows)
                                          for key in ["format_ok", "content_ok", "both_ok"]},
                       "seconds": round(time.monotonic() - started, 1)}
            # 中断時にも原文が残る。比較処理は36問そろうまで受け付けない。
            write_json(output / f"eval_{label}.json", {"protocol": protocol, "summary": summary, "rows": rows})
            print(f"[{label} {i}/{len(data['test'])}] {ex['id']} 形式={row['format_ok']} 内容={row['content_ok']}", flush=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    evaluate(model, "before")
    set_seed(42)
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model, args=SFTConfig(output_dir=str(output / "checkpoints"), **training),
        peft_config=LoraConfig(**lora), processing_class=tok,
        train_dataset=Dataset.from_list(data["train"]), eval_dataset=Dataset.from_list(data["valid"]))
    trainer.model.print_trainable_parameters()
    started = time.monotonic()
    trained = trainer.train()
    trainer.model.save_pretrained(output / "adapter")
    tok.save_pretrained(output / "adapter")
    write_json(output / "training.json", {
        "seconds": round(time.monotonic() - started, 1), "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch, "training_loss": trained.training_loss,
        "metrics": trained.metrics, "history": trainer.state.log_history,
    })
    trainer.model.config.use_cache = True
    evaluate(trainer.model, "after")
    comparison = make_comparison(output)
    print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "glossary_sft")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=float, default=20)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or ROOT / "runs" / "glossary_sft" / f"{stamp}-{uuid4().hex[:8]}"
    run(args.data_dir.resolve(), output.resolve(), args.epochs)


if __name__ == "__main__":
    main()

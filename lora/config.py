"""TOML 設定の読み込み。重い学習ライブラリには依存しない。"""

import copy
from collections.abc import Sequence
import math
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "lora" / "configs" / "baseline.toml"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _merge(config: dict, changes: dict, prefix: str = "") -> None:
    for key, value in changes.items():
        dotted = f"{prefix}{key}"
        if key not in config:
            raise ValueError(f"未知の設定項目: {dotted}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"テーブルを指定してください: {dotted}")
            _merge(config[key], value, f"{dotted}.")
        else:
            expected = type(config[key])
            numeric = expected is float and type(value) in (int, float)
            if not numeric and type(value) is not expected:
                raise ValueError(f"{dotted} の型は {expected.__name__} で指定してください")
            config[key] = value


def load_config(path: Path | None = None, overrides: Sequence[str] = ()) -> dict:
    config = tomllib.loads(BASELINE.read_text(encoding="utf-8"))
    if path is not None:
        _merge(config, tomllib.loads(path.read_text(encoding="utf-8")))
    for override in overrides:
        key, sep, value = override.partition("=")
        parts = key.split(".")
        if not sep or len(parts) != 2:
            raise ValueError("--set は section.key=TOML値 の形式で指定してください")
        parsed = tomllib.loads(f"value = {value}")["value"]
        _merge(config, {parts[0]: {parts[1]: parsed}})
    validate_config(config)
    return copy.deepcopy(config)


def validate_config(config: dict) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["experiment"]["name"]):
        raise ValueError("experiment.name は英数字で始まる英数字・_・.・- にしてください")
    model, training = config["model"], config["training"]
    if model["device"] not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("model.device は auto / cuda / mps / cpu を指定してください")
    if model["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("model.dtype は float32 / float16 / bfloat16 を指定してください")
    if training["fp16"] and training["bf16"]:
        raise ValueError("fp16 と bf16 は同時には有効にできません")
    for section, keys in {
        "lora": ("r", "lora_alpha"),
        "training": ("num_train_epochs", "per_device_train_batch_size",
                     "gradient_accumulation_steps", "learning_rate", "logging_steps",
                     "save_steps", "save_total_limit", "max_length"),
        "generation": ("max_new_tokens",),
    }.items():
        for key in keys:
            value = config[section][key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} は正の有限値にしてください")
    if training["warmup_steps"] < 0 or config["experiment"]["seed"] < 0:
        raise ValueError("warmup_steps と seed は 0 以上にしてください")
    if not 0 <= config["lora"]["lora_dropout"] < 1:
        raise ValueError("lora_dropout は 0 以上 1 未満にしてください")
    modules = config["lora"]["target_modules"]
    if not modules or not all(isinstance(item, str) and item for item in modules):
        raise ValueError("target_modules は空でない文字列の配列にしてください")

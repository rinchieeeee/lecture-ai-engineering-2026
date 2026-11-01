"""保存済みの実験結果を比較する: python -m lora.compare runs/lora/*"""

import argparse
from pathlib import Path
import sys

from .data import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="実験ディレクトリまたは result.json")
    args = parser.parse_args()
    print("run\trank\tlr\tepochs\tbefore_format\tafter_format\tbefore_content\tafter_content\tlast_loss\tseconds")
    for path in args.runs:
        result_file = path / "result.json" if path.is_dir() else path
        if not result_file.exists():
            print(f"スキップ（結果なし）: {result_file}", file=sys.stderr)
            continue
        result = read_json(result_file)
        config = result["config"]
        values = [result_file.parent.name, config["lora"]["r"], config["training"]["learning_rate"],
                  config["training"]["num_train_epochs"], result["before"]["format_rate"],
                  result["after"]["format_rate"], result["before"]["content_rate"],
                  result["after"]["content_rate"], result["training"]["last_logged_loss"],
                  result["training"]["seconds"]]
        print("\t".join(str(value) for value in values))


if __name__ == "__main__":
    main()

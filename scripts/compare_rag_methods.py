"""現行RAG単体・現行Graph単体と、非LLM構造化RAG（Tree×改良Graph×Hybrid RAG）の定量比較スクリプト。

使い方:
    uv run python scripts/compare_rag_methods.py --offline   # APIキー不要（提案方式のオフライン検証）
    uv run python scripts/compare_rag_methods.py             # GEMINI_API_KEY あり（現行RAG / 現行Graph / 提案Hybrid を全比較）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import retrieval
from app.structured_rag import format_markdown, run_structured_rag


BENCHMARK_CASES = [
    {
        "id": "Q1",
        "title": "事前学習・事後学習の役割と前後での振る舞い変化",
        "question": "基盤モデル開発において、事前学習と事後学習それぞれの役割、及び事後学習前後でモデルの振る舞いがどう変化するか",
        "required_pages": {
            "lec02-main-p020",
            "lec02-main-p021",
            "lec02-main-p023",
            "lec02-main-p024",
        },
        "bonus_pages": {"lec02-main-p022", "lec02-main-p006", "lec02-main-p008"},
        "expect_abstention": False,
    },
    {
        "id": "Q2",
        "title": "コンパクション vs セッションをまたぐメモリ・長期メモリ",
        "question": "第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか",
        "required_pages": {
            "lec03-main-p039",
            "lec03-main-p040",
        },
        "bonus_pages": {"lec03-main-p013", "lec03-main-p038"},
        "expect_abstention": False,
    },
    {
        "id": "Q3",
        "title": "知識追加と振る舞い修正でのRAG・プロンプト・FT・CPTの使い分け",
        "question": "新しい知識を足したいときと、出力形式や口調などの振る舞いを直したいときで、RAG・プロンプト・Fine-Tuning・継続事前学習（CPT）はどう使い分けるか",
        "required_pages": {
            "lec02-main-p046",
            "lec02-main-p048",
            "lec02-main-p050",
        },
        "bonus_pages": {"lec02-main-p020", "lec02-main-p028", "lec02-main-p047"},
        "expect_abstention": False,
    },
    {
        "id": "Q4",
        "title": "資料外質問（根拠不足の棄却検証）",
        "question": "今回作ったRAGの実測平均応答時間と正答率を教えてください",
        "required_pages": set(),
        "bonus_pages": set(),
        "expect_abstention": True,
    },
]


def _evaluate(retrieved_ids: list[str], case: dict, abstained: bool) -> dict:
    if case["expect_abstention"]:
        return {
            "recall": 1.0 if abstained else 0.0,
            "precision": 1.0 if abstained else 0.0,
            "status": "OK(棄却)" if abstained else f"NG({len(retrieved_ids)}件誤返却)",
            "pages": retrieved_ids,
        }
    req = case["required_pages"]
    rel = req | case["bonus_pages"]
    got = set(retrieved_ids)
    recall = len(req & got) / len(req) if req else 1.0
    precision = len(rel & got) / len(got) if got else 0.0
    missing = sorted(req - got)
    status = "完全網羅" if not missing else f"欠落: {', '.join(missing)}"
    return {
        "recall": recall,
        "precision": precision,
        "status": status,
        "pages": retrieved_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG取得方式の比較ベンチマーク")
    parser.add_argument("--offline", action="store_true", help="APIを使わず提案方式(offline)のみを検証する")
    parser.add_argument("--show-output", action="store_true", help="Q1の構造化出力サンプルを表示する")
    args = parser.parse_args()

    print("| 質問 | 方式 | 必須根拠Recall | 適合率 | 判定 | 取得ページ数 | 取得ページID |")
    print("|---|---|---:|---:|---|---:|---|")

    for case in BENCHMARK_CASES:
        q = case["question"]

        if not args.offline:
            # 1. 現行RAG単体
            rag_hits = retrieval.search(q)
            rag_ids = [h["page_id"] for h in rag_hits]
            ev = _evaluate(rag_ids, case, abstained=False)
            print(
                f"| {case['id']} | 現行RAG単体 | {ev['recall']:.0%} | {ev['precision']:.0%} | "
                f"{ev['status']} | {len(rag_ids)} | {', '.join(rag_ids)} |"
            )

            # 2. 現行Graph単体
            g_res = retrieval.search_graph(q)
            g_ids = [p["page_id"] for p in g_res["seeds"] + g_res["expanded"]]
            ev = _evaluate(g_ids, case, abstained=False)
            print(
                f"| {case['id']} | 現行Graph単体 | {ev['recall']:.0%} | {ev['precision']:.0%} | "
                f"{ev['status']} | {len(g_ids)} | {', '.join(g_ids)} |"
            )

            # 3. 提案方式（Hybrid）
            ans_h = run_structured_rag(q, offline=False)
            h_ids = list(dict.fromkeys(
                ep.page_id for fa in ans_h.facet_answers for ep in fa.evidence_pages
            ))
            abstained = all(fa.status != "supported" for fa in ans_h.facet_answers)
            ev = _evaluate(h_ids, case, abstained=abstained)
            print(
                f"| {case['id']} | 提案Hybrid(非LLM) | {ev['recall']:.0%} | {ev['precision']:.0%} | "
                f"{ev['status']} | {len(h_ids)} | {', '.join(h_ids)} |"
            )

        # 4. 提案方式（Offline）
        ans_o = run_structured_rag(q, offline=True)
        o_ids = list(dict.fromkeys(
            ep.page_id for fa in ans_o.facet_answers for ep in fa.evidence_pages
        ))
        abstained_o = all(fa.status != "supported" for fa in ans_o.facet_answers)
        ev = _evaluate(o_ids, case, abstained=abstained_o)
        print(
            f"| {case['id']} | 提案Offline(非LLM) | {ev['recall']:.0%} | {ev['precision']:.0%} | "
            f"{ev['status']} | {len(o_ids)} | {', '.join(o_ids)} |"
        )

    if args.show_output:
        print("\n" + "=" * 72)
        ans = run_structured_rag(BENCHMARK_CASES[0]["question"], offline=args.offline)
        print(format_markdown(ans, show_trace=True))


if __name__ == "__main__":
    main()

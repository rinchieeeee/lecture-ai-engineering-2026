"""非エージェント・非LLM回答生成の構造化RAGパイプラインのテスト（APIキー不要）。"""

from app import retrieval
from app.structured_rag import analyze_query, format_markdown, run_structured_rag


Q1 = "基盤モデル開発において、事前学習と事後学習それぞれの役割、及び事後学習前後でモデルの振る舞いがどう変化するか"
Q2 = "第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか"
Q3 = "新しい知識を足したいときと、出力形式や口調などの振る舞いを直したいときで、RAG・プロンプト・Fine-Tuning・継続事前学習（CPT）はどう使い分けるか"
Q4 = "今回作ったRAGの実測平均応答時間と正答率を教えてください"


def test_query_decomposition_q1_splits_three_facets_and_expands_variants():
    plan = analyze_query(Q1)
    assert len(plan.facets) == 3
    assert "事前学習" in plan.facets[0].keywords
    assert "事後学習" in plan.facets[1].keywords
    assert plan.facets[2].kind == "before_after"
    # ひらがな表記「ふるまい」や「ポストトレーニング」へ展開されること
    assert "ふるまい" in plan.facets[2].expanded_terms
    assert "ポストトレーニング" in plan.facets[1].expanded_terms


def test_query_decomposition_q2_extracts_lecture_hint_and_two_facets():
    plan = analyze_query(Q2)
    assert plan.target_lectures == [3]
    assert len(plan.facets) == 2
    assert plan.facets[0].kind == "comparison"
    assert plan.facets[1].kind == "yes_no"


def test_offline_pipeline_covers_required_evidence_for_q1():
    ans = run_structured_rag(Q1, offline=True)
    assert ans.embedding_calls == 0
    assert len(ans.facet_answers) == 3
    assert all(fa.status == "supported" for fa in ans.facet_answers)

    all_ids = {ep.page_id for fa in ans.facet_answers for ep in fa.evidence_pages}
    # 事前学習(p.21)・事後学習(p.23)・前後のふるまい変化(p.24)をすべて取得できること
    assert {"lec02-main-p021", "lec02-main-p023", "lec02-main-p024"} <= all_ids

    md = format_markdown(ans, show_trace=True)
    assert "第2回・PDF p.24" in md
    assert "文章の続き" in md
    assert "©" not in md  # 著作権フッターが本文抜粋に混入しないこと


def test_offline_pipeline_covers_required_evidence_for_q2_and_q3():
    ans2 = run_structured_rag(Q2, offline=True)
    ids2 = {ep.page_id for fa in ans2.facet_answers for ep in fa.evidence_pages}
    assert {"lec03-main-p039", "lec03-main-p040"} <= ids2

    ans3 = run_structured_rag(Q3, offline=True)
    ids3 = {ep.page_id for fa in ans3.facet_answers for ep in fa.evidence_pages}
    assert {"lec02-main-p046", "lec02-main-p048"} <= ids3


def test_abstention_on_unanswerable_question_q4():
    ans = run_structured_rag(Q4, offline=True)
    assert all(fa.status == "insufficient_evidence" for fa in ans.facet_answers)
    md = format_markdown(ans)
    assert "確認できません" in md


def test_hybrid_mode_with_precomputed_page_vectors_and_fake_embedder():
    """実DB(index.sqlite3)の事前計算済みベクトルとダミーembedderでHybrid経路を検証する。"""
    page_vecs = retrieval.load_page_embeddings()
    assert len(page_vecs) == 214

    def fake_embed(text: str, task: str) -> list[float]:
        assert task == "RETRIEVAL_QUERY"
        if "事前学習" in text and "事後学習" not in text:
            return page_vecs["lec02-main-p021"]
        if "事後学習" in text and ("振る舞い" in text or "ふるまい" in text or "前後" in text):
            return page_vecs["lec02-main-p024"]
        if "事後学習" in text:
            return page_vecs["lec02-main-p023"]
        return page_vecs["lec02-main-p020"]

    ans = run_structured_rag(Q1, offline=False, embedder=fake_embed)
    assert ans.mode == "hybrid"
    assert ans.embedding_calls >= 3
    all_ids = {ep.page_id for fa in ans.facet_answers for ep in fa.evidence_pages}
    assert {"lec02-main-p020", "lec02-main-p021", "lec02-main-p023", "lec02-main-p024"} <= all_ids

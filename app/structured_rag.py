"""エージェント・LLM回答生成を使わない構造化RAGパイプライン。

Hybrid RAG（埋め込みベクトル + BM25/文字n-gram）× Tree（章局所化・同章補完）×
改良Graph（4関係・IDF重み・クエリ誘導展開）× 抽出型構造化フォーマッタにより、
生成LLMを一切呼ばずに、観点別の要点・対比・根拠引用（第N回・PDF p.N）・トレースを返す。

使い方:
    uv run python -m app.structured_rag "質問文"
    uv run python -m app.structured_rag --trace "質問文"
    uv run python -m app.structured_rag --offline --trace "質問文"   # APIキー不要
    uv run python -m app.structured_rag --json "質問文"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from app import retrieval
from app.tools import load_glossary


# 講義資料の語彙特性に合わせた最小限の表記ゆれ・同義語辞書
VARIANT_MAP: dict[str, tuple[str, ...]] = {
    "振る舞い": ("ふるまい", "答え方", "出力形式", "口調"),
    "ふるまい": ("振る舞い", "答え方", "出力形式"),
    "事後学習": ("ポストトレーニング", "Fine-Tuning", "SFT", "RLHF", "RLVR", "答え方"),
    "ポストトレーニング": ("事後学習", "Fine-Tuning", "SFT", "RLHF"),
    "事前学習": ("プレトレーニング", "次のトークン予測", "知識の土台"),
    "継続事前学習": ("CPT", "知識・言語を足す"),
    "CPT": ("継続事前学習",),
    "長期メモリ": ("MEMORY.md", "外の保存先", "セッションをまたぐ", "dreaming"),
    "コンパクション": ("履歴を要約", "同じセッション", "コンテキストの中"),
    "セッションをまたぐ": ("外の保存先", "長期メモリ", "MEMORY.md"),
    "知識": ("事実", "ナレッジカットオフ", "知識の締め切り", "知識の土台"),
    "役割": ("決まる", "土台", "学ぶ"),
    "変化": ("変わる", "前と後", "前後"),
    "前後": ("前と後", "変化", "変わる"),
}

# 質問文からキーワード抽出する際に除外する一般的な機能語・質問定型語
STOPWORDS = frozenset({
    "こと", "もの", "ため", "よう", "ところ", "それ", "これ", "あれ", "どれ",
    "それぞれ", "において", "について", "に関して", "に対して", "によって",
    "および", "及び", "ならび", "また", "さらに", "または", "あるいは",
    "どのよう", "どう", "なぜ", "なに", "なん", "いつ", "どこ", "だれ",
    "教え", "教えて", "ください", "説明", "違い", "比較", "関係", "内容",
    "場合", "とき", "方法", "仕組み", "使い方", "使い分け", "点", "方",
    "すべて", "全て", "今回", "作った", "ある", "いる", "する", "なる",
    "できる", "れる", "られる", "です", "ます", "ですか", "ますか", "か",
    "モデル", "開発", "質問", "回答", "答え", "例", "講義", "資料",
})

# 非根拠スライド（表紙・アジェンダ・クイズ出題・演習時間割など）の判定パターン
LOW_EVIDENCE_TITLE_PATTERNS = (
    "アジェンダ", "カリキュラム", "今日の構成", "今日の 4 つの問い",
    "講義の進み方", "その他アナウンス", "演習の進め方", "今日の演習",
    "冒頭", "まとめ",
)

# 構造化・ノイズ除去用パターン
COPYRIGHT_RE = re.compile(r"^\s*©.*$", re.IGNORECASE)
STANDALONE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")
SECTION_PROGRESS_RE = re.compile(r"^\s*\d+/\d+\s+")
CAPTION_HEADER = "【図表・画像内の補足文字起こし】"


@dataclass
class Facet:
    """質問から分解された1つの観点（サブクエリ）。"""

    index: int
    label: str
    query: str
    kind: str  # "role", "before_after", "comparison", "yes_no", "general"
    keywords: list[str]
    distinctive_keywords: list[str]
    expanded_terms: list[str]
    weak_terms: list[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    """質問解析の結果。"""

    original_query: str
    facets: list[Facet]
    question_type: str
    target_lectures: list[int]
    global_keywords: list[str]
    global_expanded_terms: list[str]


@dataclass
class EvidencePage:
    """特定の観点に割り当てられた根拠ページと取得トレース。"""

    page_id: str
    document_id: str
    lecture: int
    document_title: str
    section: str
    title: str
    page: int
    content: str
    score: float
    vector_similarity: float | None
    lexical_score: float
    keyword_coverage: float
    provenance: list[str]


@dataclass
class ExtractedPoint:
    """スライド本文から抽出した1つの根拠ブロック。"""

    text: str
    citation: str
    page_id: str
    block_kind: str  # "lead", "core", "contrast", "detail"
    score: float


@dataclass
class FacetAnswer:
    """1つの観点に対する構造化回答。"""

    facet: Facet
    status: str  # "supported" | "insufficient_evidence"
    confidence: float
    missing_keywords: list[str]
    summary_points: list[ExtractedPoint]
    contrast_items: list[tuple[str, str, str]]  # (label, body, citation)
    detail_points: list[ExtractedPoint]
    evidence_pages: list[EvidencePage]


@dataclass
class StructuredAnswer:
    """最終出力（Markdown/JSON変換可能）。"""

    plan: QueryPlan
    mode: str  # "hybrid" | "offline"
    facet_answers: list[FacetAnswer]
    elapsed_seconds: float
    embedding_calls: int
    trace: dict


# ---------------------------------------------------------------------------
# 正規化・トークナイズ・用語辞書
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """NFKC正規化し、余分な空白を詰める。"""
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_compact(text: str) -> str:
    """空白を除去して小文字化した照合用文字列。"""
    return normalize_text(text).casefold().replace(" ", "")


def build_term_dictionaries(
    glossary: list[dict] | None = None,
    graph: dict | None = None,
) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]]]:
    """glossary.json と graph.json から専門語リスト・同義語辞書・関連語辞書を作る。"""
    glossary_entries = glossary if glossary is not None else load_glossary()
    known_terms: set[str] = set(VARIANT_MAP.keys())
    synonyms: dict[str, set[str]] = defaultdict(set)
    related_map: dict[str, set[str]] = defaultdict(set)

    for k, vs in VARIANT_MAP.items():
        known_terms.add(k)
        for v in vs:
            known_terms.add(v)
            synonyms[k.casefold()].add(v)

    by_id: dict[str, dict] = {entry["id"]: entry for entry in glossary_entries}
    for entry in glossary_entries:
        canonical = entry["term"]
        # 括弧内の略称/英語も抽出（例: "教師ありファインチューニング（SFT）"）
        bare = re.sub(r"[（(][^）)]+[）)]", "", canonical).strip()
        names = [canonical]
        if bare and len(bare) >= 2:
            names.append(bare)
        for alias in entry.get("aliases", []):
            if len(alias.strip()) >= 2:
                names.append(alias.strip())
        for n in names:
            known_terms.add(n)
        for n in names:
            key = n.casefold()
            for other in names:
                if other != n:
                    synonyms[key].add(other)
            for rel_id in entry.get("related", []):
                rel_entry = by_id.get(rel_id)
                if rel_entry:
                    related_map[key].add(rel_entry["term"])

    if graph is not None:
        for node in graph.get("nodes", []):
            if node.get("kind") == "term":
                title = node["title"]
                known_terms.add(title)
                bare = re.sub(r"[（(][^）)]+[）)]", "", title).strip()
                if len(bare) >= 2:
                    known_terms.add(bare)

    # 長い語を優先マッチできるよう長さ降順で返す
    sorted_terms = sorted((t for t in known_terms if len(t) >= 2), key=len, reverse=True)
    return sorted_terms, synonyms, related_map


def tokenize_for_index(text: str, known_terms: list[str] | None = None) -> list[str]:
    """日英混在テキストをBM25用トークン列（語彙＋複合語分解＋文字2-gram）に変換する。"""
    norm = normalize_text(text)
    lower = norm.casefold()
    tokens: list[str] = []

    # 1. 既知の専門語・同義語マッチ（重みを強めるため2回投入相当）
    if known_terms:
        compact = lower.replace(" ", "")
        for term in known_terms:
            t_low = term.casefold()
            t_compact = t_low.replace(" ", "")
            if len(t_compact) >= 2 and t_compact in compact:
                tokens.append(f"t:{t_compact}")

    # 2. 英数字・カタカナ・漢字塊の抽出
    for m in re.finditer(r"[a-z][a-z0-9_+.-]{1,}|[ァ-ヴー]{2,}|[\u4e00-\u9fff]{2,}", lower):
        w = m.group(0)
        tokens.append(f"w:{w}")
        # 長い漢字複合語（例: 事後学習前後, 実測平均応答時間）は2文字漢字バイグラムに分解
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", w):
            for i in range(len(w) - 1):
                tokens.append(f"k2:{w[i:i+2]}")

    # 3. 日本語表記ゆれ対応のための文字2-gram（英数字・記号を除く連続文字列）
    jp_runs = re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+", norm)
    for run in jp_runs:
        if len(run) >= 2:
            for i in range(len(run) - 1):
                tokens.append(f"g2:{run[i:i+2]}")

    return tokens


def extract_keywords(text: str, known_terms: list[str]) -> list[str]:
    """質問文・観点文から核となる検索キーワードを抽出する。"""
    norm = normalize_text(text)
    compact = norm.casefold().replace(" ", "")
    found: list[str] = []
    seen: set[str] = set()

    def add(kw: str) -> None:
        k = kw.strip()
        kl = k.casefold()
        if len(k) < 2 or kl in seen or k in STOPWORDS or kl in STOPWORDS:
            return
        seen.add(kl)
        found.append(k)

    # 1. 既知の専門語・複合表現を最優先で抽出（最長一致、包含重複を避ける）
    matched_spans: list[str] = []
    for term in known_terms:
        t_compact = term.casefold().replace(" ", "")
        if len(t_compact) < 2:
            continue
        if t_compact in compact:
            # 既により長い表現の一部として含まれている短い語は重複追加しない
            # ただし「事前学習」と「継続事前学習」のように意味が独立するものは残す
            if any(t_compact in prev and len(prev) - len(t_compact) <= 1 for prev in matched_spans):
                continue
            matched_spans.append(t_compact)
            add(term)

    # 2. 英数字・カタカナ・漢字塊を抽出
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_+-]{1,}|[ァ-ヴー]{2,}|[\u4e00-\u9fff]{2,}", norm):
        word = m.group(0)
        # 既知語が複合漢字塊に含まれる場合は分割して拾う（例: "事後学習前後" -> "事後学習", "前後"）
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", word):
            remainder = word
            sub_matched = False
            for term in known_terms:
                if term in remainder and len(term) >= 2 and re.fullmatch(r"[\u4e00-\u9fff]+", term):
                    add(term)
                    remainder = remainder.replace(term, " ")
                    sub_matched = True
            if sub_matched:
                for piece in remainder.split():
                    if len(piece) >= 2:
                        add(piece)
                continue
        add(word)

    # 3. 混合語（例: 「振る舞い」「使い分け」「答え方」）も拾う
    for pattern in (r"振る舞い", r"ふるまい", r"答え方", r"使い分け", r"締め切り", r"やり直し"):
        if re.search(pattern, norm):
            add(pattern)

    return found


# ---------------------------------------------------------------------------
# Stage 1: クエリ解析・観点分解・用語展開
# ---------------------------------------------------------------------------

def _classify_kind(text: str) -> str:
    if any(k in text for k in ("前後", "前と後", "どう変化", "どう変わ", "変化")):
        return "before_after"
    if any(k in text for k in ("違い", "どう違", "比較", "使い分け", "それぞれ")):
        return "comparison"
    if re.search(r"(すべて|全て).*(か|？|\?)$", text) or re.search(r"(入りますか|ですか|ますか)[？?]?$", text):
        if not any(k in text for k in ("何", "どう", "なぜ", "いつ")):
            return "yes_no"
    if any(k in text for k in ("役割", "とは", "意味", "仕組み", "目的")):
        return "role"
    return "general"


def _clean_clause_label(clause: str) -> str:
    label = clause.strip(" 、。？！?・")
    label = re.sub(r"(を教えてください|教えてください|ですか|ますか|でしょうか|か)$", "", label).strip()
    return label or clause.strip()


def _split_into_clauses(question: str) -> tuple[str, list[tuple[str, str]]]:
    """質問文から共有文脈プレフィックスと (ラベル, クエリ本文) のリストを抽出する。"""
    norm = normalize_text(question)
    # 講義回指定（「第2回の説明では、」など）を文脈から外して見やすくする
    sentences = [s.strip() for s in re.split(r"[。？！?]+", norm) if s.strip()]
    shared_prefix = ""
    clauses: list[tuple[str, str]] = []

    for s_idx, sent in enumerate(sentences):
        prefix = ""
        body = sent
        m_ctx = re.match(r"^(.{2,24}?)(?:において|について|に関して|の説明では|の例では|では)[、,\s]+(.+)$", sent)
        if m_ctx:
            prefix = m_ctx.group(1).strip()
            body = m_ctx.group(2).strip()
            if s_idx == 0 and not re.fullmatch(r"第\d+回", prefix):
                shared_prefix = prefix

        # パターンA: 「Xときと、Yときで、Z（はどう使い分けるか）」
        m_cond = re.match(
            r"^(.+?とき)と[、,\s]+(.+?とき)で[、,\s]+(.+)$",
            body,
        )
        if m_cond:
            cond1, cond2, tail = m_cond.group(1).strip(), m_cond.group(2).strip(), m_cond.group(3).strip()
            clauses.append((f"{cond1}の{ _clean_clause_label(tail) }", f"{prefix} {cond1} {tail}".strip()))
            clauses.append((f"{cond2}の{ _clean_clause_label(tail) }", f"{prefix} {cond2} {tail}".strip()))
            continue

        # パターンB: 接続詞（及び / および / ならびに / また / さらに / 一方で）で分割
        subparts = [
            p.strip(" 、,")
            for p in re.split(r"(?:[、,\s]+)?(?:及び|および|ならびに|また[、,\s]|さらに[、,\s]|一方で[、,\s])", body)
            if p.strip(" 、,")
        ]
        for part in subparts:
            # パターンC: 「XとYそれぞれのZ」 / 「XとYのそれぞれのZ」
            m_each = re.match(r"^(.+?)と(.+?)(?:の)?それぞれの(.+)$", part)
            if m_each:
                x, y, z = m_each.group(1).strip(), m_each.group(2).strip(), m_each.group(3).strip()
                z_clean = _clean_clause_label(z)
                clauses.append((f"{x}の{z_clean}", f"{shared_prefix or prefix} {x} {z}".strip()))
                clauses.append((f"{y}の{z_clean}", f"{shared_prefix or prefix} {y} {z}".strip()))
                continue

            clauses.append((_clean_clause_label(part), f"{shared_prefix or prefix} {part}".strip()))

    # 過剰分解を防ぐため上限5観点に抑える
    if not clauses:
        clauses = [(_clean_clause_label(norm), norm)]
    return shared_prefix, clauses[:5]


def analyze_query(
    question: str,
    glossary: list[dict] | None = None,
    graph: dict | None = None,
) -> QueryPlan:
    """質問を観点（Facet）に分解し、表記ゆれ・用語展開を行う。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("質問が空です。")

    known_terms, synonyms, related_map = build_term_dictionaries(glossary=glossary, graph=graph)
    norm = normalize_text(question)
    target_lectures = sorted({int(m) for m in re.findall(r"第(\d+)回", norm)})

    _, raw_clauses = _split_into_clauses(norm)
    facets: list[Facet] = []
    for idx, (label, clause_query) in enumerate(raw_clauses, 1):
        kws = extract_keywords(clause_query, known_terms)
        if not kws:
            kws = extract_keywords(norm, known_terms)

        expanded: list[str] = []
        weak: list[str] = []
        seen_exp = {k.casefold() for k in kws}
        for kw in kws:
            for syn in sorted(synonyms.get(kw.casefold(), ())):
                if syn.casefold() not in seen_exp:
                    seen_exp.add(syn.casefold())
                    expanded.append(syn)
            for rel in sorted(related_map.get(kw.casefold(), ())):
                if rel.casefold() not in seen_exp and rel not in weak:
                    weak.append(rel)

        facets.append(Facet(
            index=idx,
            label=label,
            query=clause_query,
            kind=_classify_kind(clause_query),
            keywords=kws,
            distinctive_keywords=[],
            expanded_terms=expanded,
            weak_terms=weak[:6],
        ))

    # 複数観点があるとき、他観点と差別化できる識別キーワードを算出
    # （全観点に共通する前提語だけを除外し、部分的に共有される主語は残す）
    if len(facets) > 1:
        kw_facet_count = Counter(
            kw.casefold()
            for f in facets
            for kw in set(f.keywords)
        )
        n_facets = len(facets)
        for f in facets:
            strict = [kw for kw in f.keywords if kw_facet_count[kw.casefold()] == 1]
            if strict:
                f.distinctive_keywords = strict
            else:
                f.distinctive_keywords = [
                    kw for kw in f.keywords
                    if kw_facet_count[kw.casefold()] < n_facets
                ] or list(f.keywords)
    elif facets:
        facets[0].distinctive_keywords = list(facets[0].keywords)

    global_keywords = extract_keywords(norm, known_terms)
    global_expanded: list[str] = []
    seen_g = {k.casefold() for k in global_keywords}
    for f in facets:
        for t in f.expanded_terms:
            if t.casefold() not in seen_g:
                seen_g.add(t.casefold())
                global_expanded.append(t)

    kinds = {f.kind for f in facets}
    if "before_after" in kinds:
        q_type = "before_after"
    elif "comparison" in kinds or len(facets) >= 2:
        q_type = "comparison"
    else:
        q_type = facets[0].kind if facets else "general"

    return QueryPlan(
        original_query=norm,
        facets=facets,
        question_type=q_type,
        target_lectures=target_lectures,
        global_keywords=global_keywords,
        global_expanded_terms=global_expanded,
    )


# ---------------------------------------------------------------------------
# Stage 2: ローカル語彙検索（BM25 + フィールド重み）& ハイブリッドRAG
# ---------------------------------------------------------------------------

def _extract_lead_text(page: dict) -> str:
    """ページ本文から先頭のリード文（著作権・章見出しを除く最初の2〜3行）を返す。"""
    lines = [ln.strip() for ln in page["content"].splitlines() if ln.strip()]
    kept: list[str] = []
    title_compact = normalize_compact(page["title"])
    section_compact = normalize_compact(page["section"])
    for ln in lines:
        if COPYRIGHT_RE.match(ln) or STANDALONE_NUM_RE.match(ln):
            continue
        lc = normalize_compact(ln)
        if lc == title_compact or lc == section_compact or SECTION_PROGRESS_RE.match(ln):
            continue
        if ln.startswith("出典") or re.fullmatch(r"この\s*\d+\s*分の芯", ln):
            continue
        kept.append(ln)
        if len(kept) >= 3:
            break
    return " ".join(kept)


class LexicalIndex:
    """外部ライブラリ不要のフィールド重み付きBM25検索インデックス。"""

    FIELD_WEIGHTS = {
        "title": 3.2,
        "section": 1.8,
        "lead": 1.8,
        "body": 1.0,
    }

    def __init__(self, pages: list[dict], known_terms: list[str]):
        self.pages = {p["page_id"]: p for p in pages}
        self.known_terms = known_terms
        self.doc_count = max(1, len(pages))

        self.field_tfs: dict[str, dict[str, Counter]] = {f: {} for f in self.FIELD_WEIGHTS}
        self.field_lens: dict[str, dict[str, int]] = {f: {} for f in self.FIELD_WEIGHTS}
        self.field_avglen: dict[str, float] = {}
        self.df: Counter = Counter()

        # キーワードIDF計算用（正規化済みコンパクト全文）
        self.compact_texts: dict[str, str] = {}
        self.compact_titles: dict[str, str] = {}
        self.compact_leads: dict[str, str] = {}

        for p in pages:
            pid = p["page_id"]
            lead = _extract_lead_text(p)
            fields = {
                "title": p["title"],
                "section": p["section"],
                "lead": lead,
                "body": p["content"],
            }
            doc_unique_tokens: set[str] = set()
            for fname, text in fields.items():
                toks = tokenize_for_index(text, known_terms)
                self.field_tfs[fname][pid] = Counter(toks)
                self.field_lens[fname][pid] = max(1, len(toks))
                doc_unique_tokens.update(toks)
            for tok in doc_unique_tokens:
                self.df[tok] += 1

            self.compact_texts[pid] = normalize_compact(f"{p['title']} {p['section']} {p['content']}")
            self.compact_titles[pid] = normalize_compact(p["title"])
            self.compact_leads[pid] = normalize_compact(lead)

        for fname in self.FIELD_WEIGHTS:
            lens = self.field_lens[fname].values()
            self.field_avglen[fname] = (sum(lens) / len(lens)) if lens else 1.0

    def _idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def keyword_idf(self, keyword: str) -> float:
        """任意のキーワード文字列に対する文書頻度ベースのIDFを返す。"""
        kc = normalize_compact(keyword)
        if not kc:
            return 0.0
        df = sum(1 for txt in self.compact_texts.values() if kc in txt)
        return math.log((self.doc_count + 1.0) / (df + 1.0)) + 1.0

    def score_page_for_facet(self, page_id: str, facet: Facet) -> tuple[float, float, float]:
        """ページの (BM25スコア, キーワードIDF網羅率, タイトル/リード一致度) を返す。"""
        # 1. BM25 (主クエリ + 同義語)
        q_primary = tokenize_for_index(facet.query, self.known_terms)
        q_expanded = tokenize_for_index(" ".join(facet.expanded_terms), self.known_terms)
        q_weights: dict[str, float] = {}
        for tok in q_primary:
            base = 1.4 if tok.startswith("t:") else (1.0 if tok.startswith("w:") else 0.35)
            q_weights[tok] = max(q_weights.get(tok, 0.0), base)
        for tok in q_expanded:
            base = 0.7 if tok.startswith("t:") else (0.5 if tok.startswith("w:") else 0.15)
            q_weights[tok] = max(q_weights.get(tok, 0.0), base)

        k1, b = 1.5, 0.75
        bm25 = 0.0
        for fname, fweight in self.FIELD_WEIGHTS.items():
            tf_map = self.field_tfs[fname][page_id]
            dl = self.field_lens[fname][page_id]
            avglen = self.field_avglen[fname]
            field_score = 0.0
            for tok, qw in q_weights.items():
                tf = tf_map.get(tok, 0)
                if tf == 0:
                    continue
                idf = self._idf(tok)
                denom = tf + k1 * (1.0 - b + b * dl / avglen)
                field_score += qw * idf * (tf * (k1 + 1.0)) / denom
            bm25 += fweight * field_score

        # 2. IDF重み付きキーワード網羅率（Abstention判定にも使う）
        full_txt = self.compact_texts[page_id]
        title_txt = self.compact_titles[page_id]
        lead_txt = self.compact_leads[page_id]

        total_w = 0.0
        hit_w = 0.0
        title_hit = 0.0
        lead_hit = 0.0
        tl_total = 0.0

        # 各キーワードについて、本人またはその同義語がヒットしたかを評価
        _, synonyms, _ = build_term_dictionaries()
        for kw in facet.keywords:
            w = self.keyword_idf(kw)
            total_w += w
            is_distinctive = kw in facet.distinctive_keywords
            if is_distinctive:
                tl_total += w

            candidates = [kw] + sorted(synonyms.get(kw.casefold(), ()))
            matched_body = any(normalize_compact(c) in full_txt for c in candidates if len(normalize_compact(c)) >= 2)
            matched_title = any(normalize_compact(c) in title_txt for c in candidates if len(normalize_compact(c)) >= 2)
            matched_lead = any(normalize_compact(c) in lead_txt for c in candidates if len(normalize_compact(c)) >= 2)
            if matched_body:
                hit_w += w
            if is_distinctive:
                if matched_title:
                    title_hit += w
                elif matched_lead:
                    lead_hit += 0.55 * w

        coverage = (hit_w / total_w) if total_w > 0 else 0.0
        tl_ratio = ((title_hit + lead_hit) / tl_total) if tl_total > 0 else 0.0
        return bm25, coverage, tl_ratio

    def search_facet(self, facet: Facet, target_lectures: list[int] | None = None) -> list[tuple[str, float, float, float]]:
        """全ページをBM25+タイトル一致で順位付けして返す。"""
        scored = []
        for pid, page in self.pages.items():
            bm25, cov, tl = self.score_page_for_facet(pid, facet)
            score = bm25 * (1.0 + 0.35 * cov + 0.45 * tl)
            if target_lectures and page["lecture"] in target_lectures:
                score *= 1.15
            scored.append((pid, score, cov, tl))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """正規化済み（または一般の）2ベクトル間のコサイン類似度を計算する。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def compute_vector_scores(
    plan: QueryPlan,
    page_vectors: dict[str, list[float]],
    embedder: Callable[[str, str], list[float]],
) -> tuple[dict[int, dict[str, float]], dict[str, float], int]:
    """元質問および各観点の埋め込みベクトルと全ページのコサイン類似度を計算する。"""
    cache: dict[str, list[float]] = {}

    def get_vec(text: str) -> list[float]:
        key = text.strip()
        if key not in cache:
            cache[key] = embedder(key, "RETRIEVAL_QUERY")
        return cache[key]

    global_vec = get_vec(plan.original_query)
    global_sims = {
        pid: cosine_similarity(global_vec, vec)
        for pid, vec in page_vectors.items()
    }

    facet_sims: dict[int, dict[str, float]] = {}
    for facet in plan.facets:
        # 表記ゆれ（例: 振る舞い -> ふるまい）を補助語として少量添える
        extra = " ".join(facet.expanded_terms[:4])
        q_text = f"{facet.query} {extra}".strip() if extra else facet.query
        f_vec = get_vec(q_text)
        facet_sims[facet.index] = {
            pid: cosine_similarity(f_vec, vec)
            for pid, vec in page_vectors.items()
        }

    return facet_sims, global_sims, len(cache)


# ---------------------------------------------------------------------------
# Stage 3: Tree × 改良Graph 構造展開
# ---------------------------------------------------------------------------

def build_tree_structures(tree: dict, allowed_page_ids: set[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """tree.json から page->section マップと section->ordered pages マップを構築する。"""
    page_meta: dict[str, dict] = {}
    sections: dict[str, dict] = {}

    for doc in tree.get("children", []):
        doc_id = doc["id"]
        for child in doc.get("children", []):
            if child.get("kind") == "section":
                sec_id = child["id"]
                sec_pages = [
                    p["id"] for p in child.get("children", [])
                    if p.get("kind") == "page" and p["id"] in allowed_page_ids
                ]
                if not sec_pages:
                    continue
                sections[sec_id] = {
                    "id": sec_id,
                    "title": child["title"],
                    "document_id": doc_id,
                    "pages": sec_pages,
                }
                for pos, pid in enumerate(sec_pages):
                    page_meta[pid] = {
                        "section_id": sec_id,
                        "section_title": child["title"],
                        "document_id": doc_id,
                        "pos": pos,
                    }
    return page_meta, sections


def is_low_evidence_page(page: dict, query_text: str) -> float:
    """表紙・目次・アジェンダ・クイズ出題などの非根拠スライドに減点係数を返す。"""
    title = page["title"]
    content_clean = re.sub(r"©.*", "", page["content"]).strip()

    # セクション扉ページ（本文が極端に短い）
    if len(content_clean) < 110:
        return 0.25

    # クイズ出題スライド（答えスライドは除外）
    if "クイズ" in title and "答え" not in title and "クイズ" not in query_text:
        return 0.45

    for pat in LOW_EVIDENCE_TITLE_PATTERNS:
        if pat in title and pat not in query_text:
            return 0.55

    return 1.0


def expand_with_tree_and_graph(
    plan: QueryPlan,
    pages: dict[str, dict],
    tree: dict,
    graph: dict,
    lexical: LexicalIndex,
    facet_rrf: dict[int, dict[str, float]],
    facet_lex_info: dict[int, dict[str, tuple[float, float, float]]],
    facet_vec_sims: dict[int, dict[str, float]] | None,
    page_vectors: dict[str, list[float]] | None,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, list[str]]], dict]:
    """Tree（章局所化・同章補完）と改良Graph（4関係・IDF重み・クエリ誘導）で候補を拡張する。"""
    allowed_ids = set(pages.keys())
    page_meta, sections = build_tree_structures(tree, allowed_ids)

    # 観点ごとの初期シード上位4件を集計
    seeds_by_facet: dict[int, list[str]] = {}
    all_seeds: set[str] = set()
    for f in plan.facets:
        ranked = sorted(facet_rrf[f.index].items(), key=lambda x: (-x[1], x[0]))
        top_seeds = [pid for pid, sc in ranked[:4] if sc > 0]
        seeds_by_facet[f.index] = top_seeds
        all_seeds.update(top_seeds)

    bonus_scores: dict[int, dict[str, float]] = {f.index: defaultdict(float) for f in plan.facets}
    provenance: dict[int, dict[str, list[str]]] = {f.index: defaultdict(list) for f in plan.facets}

    for f in plan.facets:
        for rank, pid in enumerate(seeds_by_facet[f.index], 1):
            provenance[f.index][pid].append(f"RAG初期シード#{rank}")

    # ------------------------------------------------------------------
    # A. Tree による章（Section）局所化と構造補完
    # ------------------------------------------------------------------
    sec_scores: dict[str, float] = defaultdict(float)
    sec_facets: dict[str, set[int]] = defaultdict(set)
    for f in plan.facets:
        for pid in seeds_by_facet[f.index]:
            meta = page_meta.get(pid)
            if not meta:
                continue
            sid = meta["section_id"]
            sec_scores[sid] += facet_rrf[f.index][pid]
            sec_facets[sid].add(f.index)

    ranked_sections = sorted(
        sec_scores.keys(),
        key=lambda sid: -(sec_scores[sid] * (1.0 + 0.35 * len(sec_facets[sid]))),
    )
    top_sections = ranked_sections[:2]

    def _facet_rel(pid: str, facet: Facet) -> float:
        bm25, cov, tl = facet_lex_info[facet.index][pid]
        lex_norm = min(1.0, bm25 / 35.0)
        rel = 0.45 * cov + 0.35 * tl + 0.20 * lex_norm
        if facet_vec_sims is not None:
            vsim = max(0.0, facet_vec_sims[facet.index].get(pid, 0.0))
            rel = 0.55 * rel + 0.45 * vsim
        return rel

    tree_trace: list[dict] = []
    for sid in top_sections:
        sec = sections[sid]
        sec_pages = sec["pages"]
        tree_trace.append({
            "section_id": sid,
            "title": sec["title"],
            "facets_covered": sorted(sec_facets[sid]),
            "page_count": len(sec_pages),
        })

        # 1) Tree タイトル・リード文直撃スキャン（browse_tree の非LLM再現）
        for f in plan.facets:
            for pid in sec_pages:
                _, cov, tl = facet_lex_info[f.index][pid]
                if tl >= 0.45 or (cov >= 0.65 and tl > 0.0):
                    gain = 0.012 * (0.6 * tl + 0.4 * cov)
                    bonus_scores[f.index][pid] += gain
                    tag = f"Tree同章タイトル/リード一致({sec['title']})"
                    if tag not in provenance[f.index][pid]:
                        provenance[f.index][pid].append(tag)

        # 2) 章の冒頭概観ページ（扉ページ直後の最初の実内容ページ）を補完
        overview_pid = next(
            (pid for pid in sec_pages[:3] if is_low_evidence_page(pages[pid], plan.original_query) >= 0.9),
            None,
        )
        if overview_pid:
            for f in plan.facets:
                rel = _facet_rel(overview_pid, f)
                if rel >= 0.22:
                    bonus_scores[f.index][overview_pid] += 0.008 * rel
                    tag = f"Tree同章概観補完({sec['title']})"
                    if tag not in provenance[f.index][overview_pid]:
                        provenance[f.index][overview_pid].append(tag)

        # 3) シードページの同一章内・前後1ページ補完
        for f in plan.facets:
            for seed_pid in seeds_by_facet[f.index]:
                meta = page_meta.get(seed_pid)
                if not meta or meta["section_id"] != sid:
                    continue
                pos = meta["pos"]
                for delta in (-1, 1):
                    npos = pos + delta
                    if 0 <= npos < len(sec_pages):
                        nb_pid = sec_pages[npos]
                        rel = _facet_rel(nb_pid, f)
                        if rel >= 0.20:
                            bonus_scores[f.index][nb_pid] += 0.009 * rel
                            tag = f"Tree同章隣接({pages[seed_pid]['page']}→{pages[nb_pid]['page']})"
                            if tag not in provenance[f.index][nb_pid]:
                                provenance[f.index][nb_pid].append(tag)

    # ------------------------------------------------------------------
    # B. 改良Graph による4関係・IDF重み・クエリ誘導型展開
    # ------------------------------------------------------------------
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    next_adj: dict[str, set[str]] = defaultdict(set)
    page_terms: dict[str, set[str]] = defaultdict(set)
    term_pages: dict[str, set[str]] = defaultdict(set)
    term_related: dict[str, set[str]] = defaultdict(set)

    for edge in graph.get("edges", []):
        rel_type = edge["relation"]
        a, b = edge["source"], edge["target"]
        if rel_type == "next_page" and a in allowed_ids and b in allowed_ids:
            next_adj[a].add(b)
            next_adj[b].add(a)
        elif rel_type == "mentions":
            if a in allowed_ids and b in nodes and nodes[b]["kind"] == "term":
                page_terms[a].add(b)
                term_pages[b].add(a)
            elif b in allowed_ids and a in nodes and nodes[a]["kind"] == "term":
                page_terms[b].add(a)
                term_pages[a].add(b)
        elif rel_type == "related_term":
            if a in nodes and b in nodes:
                term_related[a].add(b)
                term_related[b].add(a)

    def term_idf_weight(term_id: str) -> float:
        deg = len(term_pages.get(term_id, ()))
        if deg <= 0:
            return 0.0
        # 高頻度ハブ語（30頁超）を強く減衰させる
        return 1.0 / math.log(2.0 + deg)

    graph_trace: list[dict] = []

    for f in plan.facets:
        facet_seeds = seeds_by_facet[f.index]
        for seed_pid in facet_seeds:
            seed_sec = page_meta.get(seed_pid, {}).get("section_id")

            # (1) next_page エッジ（同章内は高重み、章またぎは低重み）
            for nb in next_adj.get(seed_pid, ()):
                same_sec = page_meta.get(nb, {}).get("section_id") == seed_sec
                w_edge = 1.0 if same_sec else 0.25
                rel = _facet_rel(nb, f)
                if rel >= 0.22:
                    gain = 0.008 * w_edge * rel
                    bonus_scores[f.index][nb] += gain
                    tag = f"Graph[next_page: p.{pages[seed_pid]['page']}→p.{pages[nb]['page']}]"
                    if tag not in provenance[f.index][nb]:
                        provenance[f.index][nb].append(tag)
                        graph_trace.append({"facet": f.index, "from": seed_pid, "to": nb, "via": "next_page", "gain": round(gain, 5)})

            # (2) mentions（1-hop用語共有、IDF重み付き）
            for tid in page_terms.get(seed_pid, ()):
                t_w = term_idf_weight(tid)
                t_title = nodes[tid]["title"]
                # 質問の観点に関係する用語ならブースト
                term_matches_facet = any(
                    normalize_compact(kw) in normalize_compact(t_title) or normalize_compact(t_title) in normalize_compact(kw)
                    for kw in f.keywords + f.expanded_terms
                    if len(normalize_compact(kw)) >= 2
                )
                if not term_matches_facet and len(term_pages[tid]) > 18:
                    continue
                if term_matches_facet:
                    t_w *= 1.8

                for nb in term_pages.get(tid, ()):
                    if nb == seed_pid:
                        continue
                    rel = _facet_rel(nb, f)
                    if rel < 0.25:
                        continue
                    gain = 0.010 * t_w * rel
                    bonus_scores[f.index][nb] += gain
                    tag = f"Graph[mentions:{t_title}]"
                    if tag not in provenance[f.index][nb]:
                        provenance[f.index][nb].append(tag)
                        graph_trace.append({"facet": f.index, "from": seed_pid, "to": nb, "via": f"mentions:{t_title}", "gain": round(gain, 5)})

                # (3) related_term（用語間の関連語ブリッジ）
                for rtid in term_related.get(tid, ()):
                    rt_w = term_idf_weight(rtid)
                    if rt_w == 0.0:
                        continue
                    rt_title = nodes[rtid]["title"]
                    rt_matches = any(
                        normalize_compact(kw) in normalize_compact(rt_title) or normalize_compact(rt_title) in normalize_compact(kw)
                        for kw in f.keywords + f.expanded_terms
                        if len(normalize_compact(kw)) >= 2
                    )
                    if not rt_matches:
                        continue
                    for nb in term_pages.get(rtid, ()):
                        if nb == seed_pid:
                            continue
                        rel = _facet_rel(nb, f)
                        if rel < 0.28:
                            continue
                        gain = 0.008 * min(t_w, rt_w) * rel
                        bonus_scores[f.index][nb] += gain
                        tag = f"Graph[related_term:{t_title}→{rt_title}]"
                        if tag not in provenance[f.index][nb]:
                            provenance[f.index][nb].append(tag)
                            graph_trace.append({"facet": f.index, "from": seed_pid, "to": nb, "via": f"{t_title}->{rt_title}", "gain": round(gain, 5)})

            # (4) 事前計算済みページベクトルによる意味近傍エッジ（標準モードのみ・APIコスト0）
            if page_vectors and seed_pid in page_vectors:
                svec = page_vectors[seed_pid]
                sims = []
                for cand_pid, cvec in page_vectors.items():
                    if cand_pid == seed_pid or cand_pid not in allowed_ids:
                        continue
                    sim = cosine_similarity(svec, cvec)
                    if sim >= 0.78:
                        sims.append((cand_pid, sim))
                sims.sort(key=lambda x: -x[1])
                for nb, sim in sims[:3]:
                    rel = _facet_rel(nb, f)
                    if rel >= 0.25:
                        gain = 0.009 * sim * rel
                        bonus_scores[f.index][nb] += gain
                        tag = f"Graph[semantic_sim:{sim:.2f} from p.{pages[seed_pid]['page']}]"
                        if tag not in provenance[f.index][nb]:
                            provenance[f.index][nb].append(tag)

    return bonus_scores, provenance, {"top_sections": tree_trace, "graph_expansions": graph_trace[:25]}


# ---------------------------------------------------------------------------
# Stage 4: 観点別リランキング & 根拠不足（Abstention）ゲート
# ---------------------------------------------------------------------------

def select_evidence_for_facets(
    plan: QueryPlan,
    pages: dict[str, dict],
    lexical: LexicalIndex,
    facet_rrf: dict[int, dict[str, float]],
    facet_lex_info: dict[int, dict[str, tuple[float, float, float]]],
    facet_vec_sims: dict[int, dict[str, float]] | None,
    bonus_scores: dict[int, dict[str, float]],
    provenance: dict[int, dict[str, list[str]]],
    *,
    max_pages_per_facet: int = 3,
    offline: bool = False,
) -> dict[int, tuple[str, float, list[str], list[EvidencePage]]]:
    """観点ごとに根拠ページを厳選し、根拠不足（Abstention）を判定する。"""
    results: dict[int, tuple[str, float, list[str], list[EvidencePage]]] = {}
    _, synonyms, _ = build_term_dictionaries()

    for f in plan.facets:
        # 他の観点に固有の主語（例: 観点1「事前学習」に対する他観点の「事後学習」）が
        # スライドタイトル先頭に明確に出ている場合は、観点の取り違えを抑える
        other_heads = {
            normalize_compact(kw)
            for other in plan.facets
            if other.index != f.index
            for kw in other.distinctive_keywords
            if kw not in f.keywords and len(normalize_compact(kw)) >= 3
        }
        f_heads = {normalize_compact(kw) for kw in f.keywords if len(normalize_compact(kw)) >= 2}

        scored_pages: list[EvidencePage] = []
        for pid, page in pages.items():
            base = facet_rrf[f.index].get(pid, 0.0)
            bonus = bonus_scores[f.index].get(pid, 0.0)
            bm25, cov, tl = facet_lex_info[f.index][pid]
            vsim = facet_vec_sims[f.index].get(pid) if facet_vec_sims is not None else None

            # 観点固有の識別キーワードがタイトル/リードにあるページを優遇
            specificity_mult = 1.0 + 0.70 * tl + 0.25 * cov
            title_compact = lexical.compact_titles[pid]
            if any(title_compact.startswith(oh) for oh in other_heads) and not any(
                title_compact.startswith(fh) for fh in f_heads
            ):
                specificity_mult *= 0.55

            quality_mult = is_low_evidence_page(page, plan.original_query)
            final_score = (base + bonus) * specificity_mult * quality_mult

            if final_score <= 0:
                continue
            scored_pages.append(EvidencePage(
                page_id=pid,
                document_id=page["document_id"],
                lecture=page["lecture"],
                document_title=page["document_title"],
                section=page["section"],
                title=page["title"],
                page=page["page"],
                content=page["content"],
                score=final_score,
                vector_similarity=vsim,
                lexical_score=bm25,
                keyword_coverage=cov,
                provenance=list(provenance[f.index].get(pid, ["総合スコア選出"])),
            ))

        scored_pages.sort(key=lambda ep: (-ep.score, ep.page_id))

        # 上位ページと差が大きすぎる低スコアノイズを足切り
        selected: list[EvidencePage] = []
        if scored_pages:
            top_score = scored_pages[0].score
            for ep in scored_pages:
                if len(selected) >= max_pages_per_facet:
                    break
                if selected and ep.score < top_score * 0.45 and ep.keyword_coverage < 0.5:
                    continue
                selected.append(ep)

        # Abstention（根拠不足）判定:
        # 上位候補集合全体で「質問固有の希少語」をどれだけカバーできているかを検証
        combined_text = " ".join(lexical.compact_texts[ep.page_id] for ep in selected)
        total_idf = 0.0
        matched_idf = 0.0
        missing_kws: list[str] = []
        for kw in f.keywords:
            idf = lexical.keyword_idf(kw)
            total_idf += idf
            cands = [kw] + sorted(synonyms.get(kw.casefold(), ()))
            hit = any(normalize_compact(c) in combined_text for c in cands if len(normalize_compact(c)) >= 2)
            if hit:
                matched_idf += idf
            else:
                missing_kws.append(kw)

        union_coverage = (matched_idf / total_idf) if total_idf > 0 else 0.0
        top_ep = selected[0] if selected else None
        top_cov = top_ep.keyword_coverage if top_ep else 0.0
        top_lex = top_ep.lexical_score if top_ep else 0.0
        top_vsim = top_ep.vector_similarity if (top_ep and top_ep.vector_similarity is not None) else 0.0

        if offline:
            supported = (
                top_ep is not None
                and top_lex >= 10.0
                and max(union_coverage, top_cov) >= 0.45
            )
            confidence = min(1.0, 0.55 * union_coverage + 0.45 * min(1.0, top_lex / 60.0))
        else:
            # ベクトル検索ありでも、希少語カバレッジが極端に低ければ棄却（Q4対策）
            supported = (
                top_ep is not None
                and (
                    (max(union_coverage, top_cov) >= 0.45 and top_vsim >= 0.55)
                    or (max(union_coverage, top_cov) >= 0.65 and top_lex >= 12.0)
                    or (top_vsim >= 0.78 and max(union_coverage, top_cov) >= 0.35)
                )
            )
            confidence = min(1.0, 0.45 * union_coverage + 0.35 * max(0.0, top_vsim) + 0.20 * min(1.0, top_lex / 60.0))

        status = "supported" if supported else "insufficient_evidence"
        results[f.index] = (status, confidence, missing_kws, selected if supported else [])

    return results


# ---------------------------------------------------------------------------
# Stage 5: 抽出型構造化フォーマッタ（LLM回答生成なし）
# ---------------------------------------------------------------------------

def _clean_slide_lines(page: dict) -> tuple[list[str], list[str]]:
    """スライド本文から定型ノイズを除き、(本文行リスト, 画像補足行リスト) を返す。"""
    raw_lines = [ln.strip() for ln in page["content"].splitlines()]
    body_lines: list[str] = []
    caption_lines: list[str] = []
    in_caption = False

    title_compact = normalize_compact(page["title"])
    section_compact = normalize_compact(page["section"])

    for idx, ln in enumerate(raw_lines):
        if not ln:
            continue
        if CAPTION_HEADER in ln:
            in_caption = True
            continue
        if COPYRIGHT_RE.match(ln) or STANDALONE_NUM_RE.match(ln):
            continue
        if in_caption:
            caption_lines.append(ln)
            continue
        # スライド冒頭の章パンくず・タイトル重複行をスキップ
        lc = normalize_compact(ln)
        if idx <= 3 and (lc == title_compact or lc == section_compact or SECTION_PROGRESS_RE.match(ln)):
            continue
        if idx <= 2 and ln in ("今日の構成", "まとめ", "演習", "付録", "例題で考える", "実践ガイド"):
            continue
        body_lines.append(ln)
    return body_lines, caption_lines


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    """スライドPDF抽出時の途中改行（助詞・読点終わり等）だけを自然な文単位に結合する。"""
    if not lines:
        return []
    merged: list[str] = []
    bullet_start = re.compile(r"^(?:[•・\-①②③④⑤⑥⑦⑧⑨⑩\d]+[.\s　]|→|←|例[:：]|出典[:：）)]|用語|入力[:：])")
    continuation_end = re.compile(r"(?:[、をにのがはでとやもて]|から|まで|より|ため|よう|して|する|れる|られ|ない|（|\(|「)$")

    for ln in lines:
        if not merged:
            merged.append(ln)
            continue
        prev = merged[-1]
        if (
            not bullet_start.match(ln)
            and continuation_end.search(prev)
            and len(prev) < 70
        ):
            sep = " " if (prev[-1].isascii() and ln[0].isascii()) else ""
            merged[-1] = f"{prev}{sep}{ln}"
        else:
            merged.append(ln)
    return merged


def _extract_contrast_pairs(lines: list[str], citation: str) -> list[tuple[str, str, str]]:
    """前後比較・対比構造（「事前学習だけのモデル」vs「事後学習の後」など）を抽出する。"""
    pairs: list[tuple[str, str, str]] = []
    label_patterns = [
        re.compile(r"^(事前学習だけのモデル)$"),
        re.compile(r"^(事後学習の後)$"),
        re.compile(r"^(コンテキストの中.*)$"),
        re.compile(r"^(外の保存先.*)$"),
        re.compile(r"^(FT\s*で直す[:：].*)$"),
        re.compile(r"^(FT\s*では直らない[:：].*)$"),
        re.compile(r"^(SFT|RLHF|RLVR)$"),
        re.compile(r"^(手本から学ぶ|人の好みから学ぶ|自動の採点から学ぶ)$"),
        re.compile(r"^(事前学習で決まる[:：].*)$"),
        re.compile(r"^(事後学習で決まる[:：].*)$"),
    ]

    i = 0
    while i < len(lines):
        ln = lines[i]
        matched_label = None
        for pat in label_patterns:
            if pat.match(ln):
                matched_label = ln
                break
        if matched_label:
            # 直後の1〜3行をそのラベルの本文として束ねる
            body_parts: list[str] = []
            j = i + 1
            while j < len(lines) and len(body_parts) < 3:
                nxt = lines[j]
                if any(pat.match(nxt) for pat in label_patterns):
                    break
                if nxt.startswith(("出典", "InstructGPT", "チャットボットなら", "用語", "実践ガイド")):
                    break
                body_parts.append(nxt)
                j += 1
            if body_parts:
                # 「手本から学ぶ」の直後が「SFT」なら見出しを統合
                if matched_label in ("手本から学ぶ", "人の好みから学ぶ", "自動の採点から学ぶ") and body_parts[0] in ("SFT", "RLHF", "RLVR"):
                    matched_label = f"{body_parts[0]}（{matched_label}）"
                    body_parts = body_parts[1:]
                if body_parts:
                    pairs.append((matched_label, " / ".join(body_parts), citation))
            i = j
        else:
            i += 1
    return pairs


def _score_block(text: str, facet: Facet, is_lead: bool, is_core: bool) -> float:
    """抽出ブロックの観点適合度と構造重要度を計算する。"""
    if text.startswith("出典"):
        return -1.0
    if len(text) < 6:
        return -0.5

    compact = normalize_compact(text)
    score = 0.0
    for kw in facet.keywords:
        kc = normalize_compact(kw)
        if len(kc) >= 2 and kc in compact:
            score += 2.2 if kw in facet.distinctive_keywords else 1.2
    for syn in facet.expanded_terms:
        sc = normalize_compact(syn)
        if len(sc) >= 2 and sc in compact:
            score += 0.9

    if is_lead:
        score += 2.0
    if is_core:
        score += 1.6
    if "→" in text or "：" in text or ":" in text:
        score += 0.5
    if any(k in text for k in ("全部は残さない", "残すものだけ", "知識が増えたのではなく", "答え方", "知識の土台")):
        score += 0.6
    return score


def assemble_facet_answer(
    facet: Facet,
    status: str,
    confidence: float,
    missing_keywords: list[str],
    evidence_pages: list[EvidencePage],
) -> FacetAnswer:
    """1観点の根拠ページ群から要点・対比・詳細ブロックを抽出して構造化する。"""
    if status != "supported" or not evidence_pages:
        return FacetAnswer(
            facet=facet,
            status="insufficient_evidence",
            confidence=confidence,
            missing_keywords=missing_keywords,
            summary_points=[],
            contrast_items=[],
            detail_points=[],
            evidence_pages=[],
        )

    summary_points: list[ExtractedPoint] = []
    detail_points: list[ExtractedPoint] = []
    contrast_items: list[tuple[str, str, str]] = []
    seen_texts: set[str] = set()

    for p_rank, ep in enumerate(evidence_pages):
        citation = f"第{ep.lecture}回・PDF p.{ep.page}"
        raw_lines, _ = _clean_slide_lines({
            "title": ep.title,
            "section": ep.section,
            "content": ep.content,
        })
        lines = _merge_wrapped_lines(raw_lines)
        if not lines:
            continue

        # リード文（出典・見出し定型語を除く最初の実質行）を要点候補に
        lead_idx = next(
            (
                i for i, ln in enumerate(lines)
                if not ln.startswith("出典")
                and not re.fullmatch(r"この\s*\d+\s*分の芯", ln)
                and len(ln) >= 8
            ),
            None,
        )
        if lead_idx is not None:
            lead_text = lines[lead_idx]
            lead_key = normalize_compact(lead_text)
            if lead_key not in seen_texts:
                seen_texts.add(lead_key)
                lead_score = _score_block(lead_text, facet, is_lead=True, is_core=False)
                # タイトルに主題が端的に書かれている場合はタイトル＋リード文で提示
                display = f"{ep.title} — {lead_text}" if p_rank == 0 else lead_text
                summary_points.append(ExtractedPoint(
                    text=display,
                    citation=citation,
                    page_id=ep.page_id,
                    block_kind="lead",
                    score=lead_score + (1.0 if p_rank == 0 else 0.0),
                ))

        # 対比・ステップ構造の抽出
        for lbl, body, cit in _extract_contrast_pairs(raw_lines, citation):
            # 観点に関係ある対比だけ残す（無関係な他項目の混入を防ぐ）
            pair_text = f"{lbl} {body}"
            p_score = _score_block(pair_text, facet, is_lead=False, is_core=False)
            if p_score >= 1.2 or facet.kind in ("before_after", "comparison") and p_rank == 0:
                key = normalize_compact(pair_text)
                if key not in seen_texts:
                    seen_texts.add(key)
                    contrast_items.append((lbl, body, cit))

        # 本文ブロックの採点・抽出
        in_core = False
        scored_details: list[ExtractedPoint] = []
        for idx, ln in enumerate(lines):
            if idx == lead_idx:
                continue
            if "この 15 分の芯" in ln or "の芯" in ln:
                in_core = True
                if len(ln) <= 12:
                    continue
            sc = _score_block(ln, facet, is_lead=False, is_core=in_core)
            if sc >= 1.2:
                scored_details.append(ExtractedPoint(
                    text=ln,
                    citation=citation,
                    page_id=ep.page_id,
                    block_kind="core" if in_core else "detail",
                    score=sc,
                ))
            in_core = False

        scored_details.sort(key=lambda x: -x.score)
        for pt in scored_details[:3]:
            key = normalize_compact(pt.text)
            # 既に対比項目に含まれているテキストは重複を避ける
            if key in seen_texts or any(key in normalize_compact(b) for _, b, _ in contrast_items):
                continue
            seen_texts.add(key)
            detail_points.append(pt)

    summary_points.sort(key=lambda x: -x.score)
    detail_points.sort(key=lambda x: -x.score)

    return FacetAnswer(
        facet=facet,
        status="supported",
        confidence=confidence,
        missing_keywords=missing_keywords,
        summary_points=summary_points[:3],
        contrast_items=contrast_items[:5],
        detail_points=detail_points[:4],
        evidence_pages=evidence_pages,
    )


def _clean_page_body(page: dict) -> str:
    """スライド本文から著作権・重複見出しのみを除き、文章を切らずにそのまま整形する。"""
    body_lines, _ = _clean_slide_lines(page)
    return "\n".join(body_lines)


def format_markdown(
    answer: StructuredAnswer,
    *,
    show_trace: bool = False,
    extract_only: bool = False,
) -> str:
    """StructuredAnswer を観点別の根拠付きテキストに整形する（既定は関連ページ全文表示）。"""
    mode_label = (
        "Hybrid RAG（埋め込みベクトル + BM25/n-gram）x Tree x 改良Graph（LLM回答生成なし）"
        if answer.mode == "hybrid"
        else "Offline（BM25/n-gram）x Tree x 改良Graph（API不使用・LLM回答生成なし）"
    )
    lines: list[str] = [
        "# 構造化検索結果（非エージェント・LLM回答生成なし）",
        f"- 質問: {answer.plan.original_query}",
        f"- 方式: {mode_label}",
        "- 分解した観点: " + " / ".join(f"({f.index}) {f.label}" for f in answer.plan.facets),
        "",
    ]

    shown_in_facet: dict[str, int] = {}

    for fa in answer.facet_answers:
        lines.append("=" * 60)
        lines.append(f"## {fa.facet.index}. {fa.facet.label}")
        lines.append("=" * 60)
        if fa.status != "supported":
            missing = "・".join(fa.missing_keywords[:5]) if fa.missing_keywords else "該当語"
            lines.append(
                f"確認できません: 第1〜3回の本講義資料内に直接の根拠が見つかりませんでした"
                f"（未検出キーワード: {missing}）。"
            )
            lines.append("")
            continue

        if extract_only:
            for pt in fa.summary_points:
                lines.append(f"- 要点: {pt.text}（{pt.citation}）")
            if fa.contrast_items:
                lines.append("- 整理・対比:")
                for lbl, body, cit in fa.contrast_items:
                    lines.append(f"  - {lbl}: {body}（{cit}）")
            if fa.detail_points:
                lines.append("- 補足・根拠抜粋:")
                for pt in fa.detail_points:
                    lines.append(f"  - {pt.text}（{pt.citation}）")
            cites = ", ".join(
                f"第{ep.lecture}回・PDF p.{ep.page}「{ep.title}」"
                for ep in fa.evidence_pages
            )
            lines.append(f"- 参照根拠: {cites}")
            lines.append("")
            continue

        # 既定: 関連性が高いページの文章を途切れさせずにそのまま表示（他観点で表示済みなら重複省略）
        for rank, ep in enumerate(fa.evidence_pages, 1):
            header = f"[根拠 {fa.facet.index}-{rank}] 第{ep.lecture}回・PDF p.{ep.page}｜{ep.title}"
            if ep.page_id in shown_in_facet:
                lines.append(f"{header}（観点{shown_in_facet[ep.page_id]}で本文表示済み）")
                lines.append("")
                continue

            shown_in_facet[ep.page_id] = fa.facet.index
            sim_str = f" / vec={ep.vector_similarity:.3f}" if ep.vector_similarity is not None else ""
            lines.append(header)
            lines.append(f"  章: {ep.section} / 取得: {', '.join(ep.provenance[:3])}{sim_str}")
            lines.append("-" * 60)
            body = _clean_page_body({
                "title": ep.title,
                "section": ep.section,
                "content": ep.content,
            })
            lines.append(body)
            lines.append("")

    # 根拠サマリ・トレース
    lines.append("---")
    all_pages: dict[str, tuple[EvidencePage, list[str]]] = {}
    for fa in answer.facet_answers:
        for ep in fa.evidence_pages:
            if ep.page_id not in all_pages:
                all_pages[ep.page_id] = (ep, list(ep.provenance))
            else:
                existing_prov = all_pages[ep.page_id][1]
                for p in ep.provenance:
                    if p not in existing_prov:
                        existing_prov.append(p)

    if all_pages:
        lines.append("### 根拠ページ一覧と取得経路")
        for ep, provs in all_pages.values():
            sim_str = f" / vec={ep.vector_similarity:.3f}" if ep.vector_similarity is not None else ""
            lines.append(
                f"- 第{ep.lecture}回・PDF p.{ep.page} (`{ep.page_id}` / {ep.section}): "
                f"{', '.join(provs[:4])}{sim_str}"
            )
    else:
        lines.append("### 根拠ページ一覧と取得経路")
        lines.append("- 採用された根拠ページはありません（根拠不足として棄却）。")

    if show_trace:
        lines.append("")
        lines.append("### 詳細トレース")
        for f in answer.plan.facets:
            lines.append(
                f"- 観点{f.index} [{f.kind}]: keywords={f.keywords} / "
                f"distinctive={f.distinctive_keywords} / expanded={f.expanded_terms[:6]}"
            )
        top_secs = answer.trace.get("top_sections", [])
        if top_secs:
            sec_desc = ", ".join(f"{s['title']}({s['section_id']}, 観点{s['facets_covered']})" for s in top_secs)
            lines.append(f"- Tree高スコア章: {sec_desc}")
        g_exps = answer.trace.get("graph_expansions", [])
        if g_exps:
            lines.append(f"- Graph展開ヒット数: {len(g_exps)}件（先頭例: {g_exps[:4]}）")

    lines.append(
        f"\n[summary] 所要時間: {answer.elapsed_seconds:.2f}秒 / "
        f"埋め込みAPI呼び出し: {answer.embedding_calls}回 / "
        f"生成LLM呼び出し: 0回 / 生成API推定費用: $0.000000"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メインパイプライン実行関数
# ---------------------------------------------------------------------------

def run_structured_rag(
    question: str,
    *,
    offline: bool = False,
    source_path: Path = retrieval.RAG_SOURCE,
    tree_path: Path = retrieval.RAG_TREE,
    graph_path: Path = retrieval.RAG_GRAPH,
    db_path: Path = retrieval.RAG_DB,
    glossary: list[dict] | None = None,
    embedder: Callable[[str, str], list[float]] | None = None,
    max_pages_per_facet: int = 2,
) -> StructuredAnswer:
    """Tree x 改良Graph x ハイブリッドRAG で構造化回答を生成する（生成LLM不使用）。"""
    started = perf_counter()

    page_list = retrieval.load_pages(source_path)
    pages = {p["page_id"]: p for p in page_list}
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    known_terms, _, _ = build_term_dictionaries(glossary=glossary, graph=graph)

    # Stage 1: クエリ解析・観点分解
    plan = analyze_query(question, glossary=glossary, graph=graph)

    # Stage 2: 観点別ハイブリッド検索
    lexical = LexicalIndex(page_list, known_terms)
    facet_lex_ranked: dict[int, list[tuple[str, float, float, float]]] = {}
    facet_lex_info: dict[int, dict[str, tuple[float, float, float]]] = {}
    for f in plan.facets:
        ranked = lexical.search_facet(f, target_lectures=plan.target_lectures)
        facet_lex_ranked[f.index] = ranked
        facet_lex_info[f.index] = {
            pid: (bm25, cov, tl) for pid, bm25, cov, tl in ranked
        }

    facet_vec_sims: dict[int, dict[str, float]] | None = None
    page_vectors: dict[str, list[float]] | None = None
    embedding_calls = 0

    if not offline:
        page_vectors = retrieval.load_page_embeddings(db_path)
        embed_fn = embedder or retrieval.embed_text
        facet_vec_sims, global_vec_sims, embedding_calls = compute_vector_scores(
            plan, page_vectors, embed_fn
        )
    else:
        global_vec_sims = None

    # RRF 統合（各観点ごと）
    k_rrf = 60.0
    facet_rrf: dict[int, dict[str, float]] = {}
    for f in plan.facets:
        scores: dict[str, float] = defaultdict(float)
        for rank, (pid, lex_score, _, _) in enumerate(facet_lex_ranked[f.index], 1):
            if lex_score > 0:
                scores[pid] += 1.0 / (k_rrf + rank)

        if facet_vec_sims is not None:
            vec_ranked = sorted(
                facet_vec_sims[f.index].items(),
                key=lambda x: (-x[1], x[0]),
            )
            for rank, (pid, sim) in enumerate(vec_ranked, 1):
                if pid in pages:
                    w = 1.15 if (plan.target_lectures and pages[pid]["lecture"] in plan.target_lectures) else 1.0
                    scores[pid] += (1.15 * w) / (k_rrf + rank)
            if global_vec_sims is not None:
                g_ranked = sorted(global_vec_sims.items(), key=lambda x: (-x[1], x[0]))
                for rank, (pid, _) in enumerate(g_ranked[:20], 1):
                    if pid in pages:
                        scores[pid] += 0.35 / (k_rrf + rank)
        facet_rrf[f.index] = dict(scores)

    # Stage 3: Tree × 改良Graph 展開
    bonus_scores, provenance, struct_trace = expand_with_tree_and_graph(
        plan=plan,
        pages=pages,
        tree=tree,
        graph=graph,
        lexical=lexical,
        facet_rrf=facet_rrf,
        facet_lex_info=facet_lex_info,
        facet_vec_sims=facet_vec_sims,
        page_vectors=page_vectors,
    )

    # Stage 4: 根拠ページ厳選 & 根拠不足判定
    selected_map = select_evidence_for_facets(
        plan=plan,
        pages=pages,
        lexical=lexical,
        facet_rrf=facet_rrf,
        facet_lex_info=facet_lex_info,
        facet_vec_sims=facet_vec_sims,
        bonus_scores=bonus_scores,
        provenance=provenance,
        max_pages_per_facet=max_pages_per_facet,
        offline=offline,
    )

    # Stage 5: 抽出型構造化アセンブリ
    facet_answers = [
        assemble_facet_answer(f, *selected_map[f.index])
        for f in plan.facets
    ]

    elapsed = perf_counter() - started
    return StructuredAnswer(
        plan=plan,
        mode="offline" if offline else "hybrid",
        facet_answers=facet_answers,
        elapsed_seconds=elapsed,
        embedding_calls=embedding_calls,
        trace=struct_trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="エージェント・LLM回答生成を使わず、Tree x Graph x Hybrid RAGで構造化回答を返す"
    )
    parser.add_argument("question", help="調べたい質問")
    parser.add_argument("--offline", action="store_true", help="埋め込みAPIを使わずローカル検索(BM25/n-gram+Tree+Graph)のみで実行する")
    parser.add_argument("--trace", action="store_true", help="観点分解・Tree・Graphの詳細トレースを表示する")
    parser.add_argument("--extract", action="store_true", help="ページ全文ではなく抽出ブロックのみで簡略表示する")
    parser.add_argument("--max-pages", type=int, default=2, help="1観点あたりに表示する最大根拠ページ数（既定: 2）")
    parser.add_argument("--json", action="store_true", dest="as_json", help="結果を構造化JSONで出力する")
    args = parser.parse_args()

    result = run_structured_rag(
        args.question,
        offline=args.offline,
        max_pages_per_facet=args.max_pages,
    )
    if args.as_json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(format_markdown(result, show_trace=args.trace, extract_only=args.extract))


if __name__ == "__main__":
    main()

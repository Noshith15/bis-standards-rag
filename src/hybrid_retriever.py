"""
Hybrid Retriever: BM25 (sparse) + TF-IDF (dense) + Re-ranking
State-of-the-art retrieval pipeline for BIS Standards
"""

import json, re, time, math
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class HybridBISRetriever:
    """
    Two-stage hybrid retrieval:
    Stage 1: BM25 (exact keyword match) + TF-IDF (semantic) fusion
    Stage 2: Domain re-ranking with BIS-specific keyword signals
    """

    def __init__(self, standards_path: str):
        self.standards = json.load(open(standards_path))
        self._build_index()
        print(f"[HybridRAG] Indexed {len(self.standards)} BIS standards")

    def _tokenize(self, text: str) -> list:
        text = text.lower()
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        return text.split()

    def _normalize_id(self, std_id: str) -> str:
        s = re.sub(r'\s+', ' ', std_id).strip()
        s = re.sub(r'\s*:\s*', ': ', s)
        s = re.sub(r'\(PART\s*(\d+)\)', r'(Part \1)', s, flags=re.IGNORECASE)
        return s

    def _build_index(self):
        self.ids = [s['standard_id'] for s in self.standards]
        self.titles = [s['title'] for s in self.standards]

        # Build documents - title boosted 3x
        docs_text = []
        tokenized = []
        for s in self.standards:
            title3 = f"{s['title']} {s['title']} {s['title']}"
            full = f"{title3} {s['content']}"
            full_clean = re.sub(r'\s+', ' ', full).lower()
            docs_text.append(full_clean)
            tokenized.append(self._tokenize(full_clean))

        # BM25 index (sparse retrieval)
        self.bm25 = BM25Okapi(tokenized)

        # TF-IDF index (dense retrieval)
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3), min_df=1, max_df=0.85,
            sublinear_tf=True, token_pattern=r'(?u)\b\w[\w\-\.]*\b'
        )
        self.tfidf_matrix = self.tfidf.fit_transform(docs_text)

    # ── Domain keyword signals ──────────────────────────────────────────
    KEYWORD_MAP = {
        # OPC grades
        '33 grade': ['33 grade ordinary portland cement'],
        'ordinary portland cement': ['ordinary portland cement', '33 grade', '43 grade', '53 grade'],
        '43 grade': ['43 grade ordinary portland cement'],
        '53 grade': ['53 grade ordinary portland cement'],
        # Slag / Pozzolana
        'portland slag': ['portland slag cement'],
        'slag cement': ['portland slag cement'],
        'fly ash': ['portland pozzolana cement', '(part 1)'],
        'calcined clay': ['portland pozzolana cement', '(part 2)'],
        'portland pozzolana': ['portland pozzolana cement'],
        'pozzolana': ['portland pozzolana cement'],
        # Specialty cements
        'masonry cement': ['masonry cement'],
        'white portland': ['white portland cement'],
        'white cement': ['white portland cement'],
        'rapid hardening': ['rapid hardening portland cement'],
        'sulphate resisting': ['sulphate resisting portland cement'],
        'hydrophobic': ['hydrophobic portland cement'],
        'supersulphated': ['supersulphated cement'],
        'high alumina': ['high alumina cement'],
        'low heat': ['low heat portland cement'],
        'oil well': ['oil well cement'],
        # Aggregates
        'coarse and fine aggregate': ['coarse and fine aggregate'],
        'natural sources': ['coarse and fine aggregate'],
        'structural concrete': ['coarse and fine aggregate'],
        'lightweight aggregate': ['artificial lightweight aggregate'],
        'artificial lightweight': ['artificial lightweight aggregate'],
        'sand for masonry': ['sand for masonry mortars'],
        'masonry mortar': ['sand for masonry mortars'],
        # Concrete products
        'precast concrete pipe': ['precast concrete pipe'],
        'concrete pipe': ['precast concrete pipe'],
        'water main': ['precast concrete pipe'],
        'concrete masonry': ['concrete masonry units'],
        'hollow block': ['concrete masonry units'],
        'solid block': ['concrete masonry units'],
        'lightweight block': ['concrete masonry units', 'lightweight'],
        # Sheets / Roofing
        'asbestos cement sheet': ['corrugated and semi-corrugated asbestos'],
        'corrugated sheet': ['corrugated and semi-corrugated asbestos'],
        'asbestos': ['corrugated and semi-corrugated asbestos'],
        'roofing sheet': ['corrugated and semi-corrugated asbestos', 'roofing'],
        'cladding': ['corrugated and semi-corrugated asbestos'],
        # Steel
        'reinforcement bar': ['high strength deformed steel bar'],
        'tmt bar': ['high strength deformed steel bar'],
        'mild steel': ['mild steel'],
        'structural steel': ['structural steel'],
    }

    def _domain_boost(self, query: str, idx: int) -> float:
        q = query.lower()
        title = self.titles[idx].lower()
        boost = 0.0
        for kw, hints in self.KEYWORD_MAP.items():
            if kw in q:
                for hint in hints:
                    if hint in title:
                        boost += 0.3
        # IS number in query
        nums = re.findall(r'\b(\d{3,5})\b', q)
        m = re.search(r'IS\s+(\d+)', self.ids[idx], re.IGNORECASE)
        if m and m.group(1) in nums:
            boost += 0.6
        return boost

    def retrieve(self, query: str, top_k: int = 5) -> tuple:
        t0 = time.time()
        q_tokens = self._tokenize(query)

        # ── BM25 scores ───────────────────────────────────────────────
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

        # ── TF-IDF cosine scores ──────────────────────────────────────
        q_vec = self.tfidf.transform([query.lower()])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        tfidf_norm = tfidf_scores / (tfidf_scores.max() + 1e-9)

        # ── Hybrid fusion (RRF-inspired weighted sum) ─────────────────
        hybrid = 0.4 * bm25_norm + 0.6 * tfidf_norm

        # ── Re-ranking with domain boosts ─────────────────────────────
        final = []
        for i, score in enumerate(hybrid):
            boost = self._domain_boost(query, i)
            final.append((i, float(score) + boost))
        final.sort(key=lambda x: x[1], reverse=True)

        latency = round(time.time() - t0, 4)
        results = []
        for idx, score in final[:top_k]:
            s = self.standards[idx]
            results.append({
                'standard_id': self._normalize_id(s['standard_id']),
                'title': s['title'].title(),
                'score': round(score, 4),
                'rationale': self._rationale(query, s),
                'bm25_score': round(float(bm25_norm[idx]), 4),
                'tfidf_score': round(float(tfidf_norm[idx]), 4),
            })
        return results, latency

    def _rationale(self, query: str, std: dict) -> str:
        content = std['content'][:500]
        scope = re.search(r'Scope[—\-\s]+(.+?)(?:\.|2\.)', content, re.IGNORECASE | re.DOTALL)
        if scope:
            text = re.sub(r'\s+', ' ', scope.group(1)).strip()[:250]
            return f"Covers {text.lower()}"
        return f"{std['title'].title()} — applicable to the described product or material."

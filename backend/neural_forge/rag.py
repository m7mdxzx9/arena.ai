"""RAG Archives: real chunking, embedding, BM25 / dense / hybrid retrieval, reranking, grounding and evaluation.

No external services: embeddings are TF-IDF, LSA (TF-IDF + truncated SVD → dense vectors) or a deliberately
weak 32-dimensional hashing model. The 'generator' is an extractive answerer (it quotes the best-supported
sentence with a citation) and the 'closed-book' generator is a real bigram language model trained on the corpus.
These are honest stand-ins for an LLM: they reproduce the *failure modes* (hallucination, retrieval misses)
with fully inspectable computation.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .corpus import DOCS, QUESTIONS, UNANSWERABLE

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-:.][a-z0-9]+)*")
STOP = set("a an the of to in on at is are be for and or by with as it its this that what which who when how can i my do does "
           "you your from up per each any if not only must may will".split())


def tokens(text: str, stop=True) -> list[str]:
    t = TOKEN_RE.findall(text.lower())
    return [w for w in t if w not in STOP] if stop else t


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ------------------------------------------------------------------ chunking
def chunk_docs(chunk_size: int = 60, overlap: int = 10, docs=None) -> list[dict]:
    chunk_size = int(np.clip(chunk_size, 10, 400))
    overlap = int(np.clip(overlap, 0, chunk_size - 5))
    out = []
    for d in docs or DOCS:
        words = d["text"].split()
        step = chunk_size - overlap
        i, n = 0, 0
        while True:
            piece = words[i:i + chunk_size]
            out.append(dict(id=f"{d['id']}#{n}", doc=d["id"], title=d["title"], text=" ".join(piece), start=i))
            n += 1
            if i + chunk_size >= len(words):
                break
            i += step
    return out


# ------------------------------------------------------------------ retrieval index
class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(t) for t in docs_tokens]
        self.len = np.array([len(t) for t in docs_tokens], float)
        self.avg = self.len.mean() if len(self.len) else 1
        df = Counter(w for t in docs_tokens for w in set(t))
        N = len(docs_tokens)
        self.idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}

    def scores(self, q_tokens):
        s = np.zeros(len(self.docs))
        for i, tf in enumerate(self.docs):
            for w in q_tokens:
                if w in tf:
                    f = tf[w]
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class Index:
    def __init__(self, chunks: list[dict], embedding: str = "lsa"):
        self.chunks, self.embedding = chunks, embedding
        texts = [c["title"] + ". " + c["text"] for c in chunks]
        self.bm25 = BM25([tokens(t) for t in texts])
        if embedding == "hash32":
            self.vec = HashingVectorizer(n_features=32, alternate_sign=False, norm="l2", stop_words=list(STOP))
            self.E = self.vec.transform(texts).toarray()
            self._embed = lambda qs: self.vec.transform(qs).toarray()
        elif embedding == "tfidf":
            self.vec = TfidfVectorizer(token_pattern=TOKEN_RE.pattern, stop_words=list(STOP), sublinear_tf=True).fit(texts)
            self.E = self.vec.transform(texts).toarray()
            self._embed = lambda qs: self.vec.transform(qs).toarray()
        elif embedding == "lsa":
            self.vec = TfidfVectorizer(token_pattern=TOKEN_RE.pattern, stop_words=list(STOP), sublinear_tf=True).fit(texts)
            T = self.vec.transform(texts)
            k = max(2, min(64, T.shape[0] - 1, T.shape[1] - 1))
            self.svd = TruncatedSVD(k, random_state=0).fit(T)
            self.E = normalize(self.svd.transform(T))
            self._embed = lambda qs: normalize(self.svd.transform(self.vec.transform(qs)))
        else:
            raise ValueError(f"unknown embedding {embedding}")
        self.E = normalize(self.E)

    def embed(self, q: str) -> np.ndarray:
        return normalize(self._embed([q]))[0]

    def search(self, q: str, method="dense", top_k=3, rerank=False, candidates=10) -> list[dict]:
        n = len(self.chunks)
        dense = self.E @ self.embed(q)
        lex = self.bm25.scores(tokens(q))
        if method == "dense":
            score = dense
        elif method == "bm25":
            score = lex
        elif method == "hybrid":  # reciprocal rank fusion (k=60), a standard, scale-free combination
            rd = np.argsort(np.argsort(-dense)); rl = np.argsort(np.argsort(-lex))
            score = 1 / (60 + rd) + 1 / (60 + rl)
        else:
            raise ValueError(method)
        order = list(np.argsort(-score)[: max(top_k, candidates if rerank else top_k)])
        results = [dict(**self.chunks[i], score=float(score[i]), dense=float(dense[i]), bm25=float(lex[i]), first_rank=r + 1)
                   for r, i in enumerate(order)]
        if rerank:
            for r in results:
                r["rerank"] = rerank_score(q, r["text"], self.bm25)
            results.sort(key=lambda r: -r["rerank"])
        return results[:top_k]


def rerank_score(q: str, text: str, bm25: BM25) -> float:
    """Sentence-level cross-scoring: reads query and each sentence *together*.

    Score = best sentence's IDF-weighted query-term coverage + bonus for matching query bigrams/numbers.
    Cheap stand-in for a cross-encoder; like one, it only reorders the shortlist."""
    qt = tokens(q)
    if not qt:
        return 0.0
    qb = set(zip(qt, qt[1:]))
    best = 0.0
    total_idf = sum(bm25.idf.get(w, 1.0) for w in set(qt))
    for s in sentences(text) or [text]:
        st = tokens(s)
        sset = set(st)
        cov = sum(bm25.idf.get(w, 1.0) for w in set(qt) if w in sset) / total_idf
        bi = len(qb & set(zip(st, st[1:]))) / max(len(qb), 1)
        best = max(best, cov + 0.5 * bi)
    return float(best)


@lru_cache(maxsize=32)
def get_index(chunk_size: int, overlap: int, embedding: str) -> Index:
    return Index(chunk_docs(chunk_size, overlap), embedding)


def build_context(results: list[dict], budget_words: int) -> tuple[str, list[dict]]:
    """Concatenate retrieved chunks in rank order until the context-window budget (in words) is used up."""
    used, parts, included = 0, [], []
    for r in results:
        words = r["text"].split()
        room = budget_words - used
        if room <= 0:
            break
        take = words[:room]
        parts.append(" ".join(take))
        included.append(dict(id=r["id"], words=len(take), truncated=len(take) < len(words)))
        used += len(take)
    return "\n".join(parts), included


# ------------------------------------------------------------------ evaluation
def evaluate_retrieval(cfg: dict) -> dict:
    idx = get_index(int(cfg.get("chunk_size", 60)), int(cfg.get("overlap", 10)), cfg.get("embedding", "lsa"))
    k = int(np.clip(cfg.get("top_k", 3), 1, 10))
    budget = int(np.clip(cfg.get("context_budget", 150), 30, 2000))
    per_q, hits, rr, ctx_hits, ctx_words, prec = [], 0, 0.0, 0, 0, 0.0
    for item in QUESTIONS:
        res = idx.search(item["q"], cfg.get("method", "dense"), k, bool(cfg.get("rerank")), int(cfg.get("candidates", 10)))
        ans = item["answer"].lower()
        rel = [ans in r["text"].lower() for r in res]
        rank = next((i + 1 for i, x in enumerate(rel) if x), None)
        context, included = build_context(res, budget)
        in_ctx = ans in context.lower()
        hits += rank is not None
        rr += 1 / rank if rank else 0
        ctx_hits += in_ctx
        ctx_words += len(context.split())
        prec += sum(r["doc"] == item["doc"] for r in res) / len(res)
        per_q.append(dict(q=item["q"], answer=item["answer"], gold_doc=item["doc"], hit_rank=rank, in_context=in_ctx,
                          retrieved=[dict(id=r["id"], title=r["title"], score=round(r["score"], 4), relevant=rel[i],
                                          rerank=round(r.get("rerank", 0), 3) if "rerank" in r else None, text=r["text"][:400])
                                     for i, r in enumerate(res)]))
    n = len(QUESTIONS)
    return dict(config=dict(cfg), n_chunks=len(idx.chunks), avg_chunk_words=round(float(np.mean([len(c["text"].split()) for c in idx.chunks])), 1),
                metrics={f"recall@{k}": round(hits / n, 3), "mrr": round(rr / n, 3), "context_hit_rate": round(ctx_hits / n, 3),
                         "doc_precision": round(prec / n, 3), "avg_context_words": round(ctx_words / n, 1)},
                recall=round(hits / n, 3), questions=per_q, diagnosis=rag_diagnosis(cfg, hits / n, ctx_hits / n, idx))


def rag_diagnosis(cfg, recall, ctx_rate, idx):
    d = []
    if cfg.get("embedding") == "hash32":
        d.append("The 32-dimensional hashing embedding squeezes thousands of words into 32 buckets: unrelated words collide, so similarity is noisy. Try TF-IDF or LSA.")
    if int(cfg.get("chunk_size", 60)) > 150:
        d.append("Large chunks blur several topics into one vector and eat the context budget — answers may be truncated away even when retrieved.")
    if int(cfg.get("chunk_size", 60)) < 20:
        d.append("Very small chunks can cut answers in half and lose context.")
    if int(cfg.get("top_k", 3)) == 1:
        d.append("Top-K = 1 gives retrieval a single chance. If the best chunk is slightly wrong, the answer is lost.")
    if recall - ctx_rate > 0.1:
        d.append(f"Retrieval found the answer in {recall:.0%} of cases, but only {ctx_rate:.0%} survived into the context window: chunks are too long or K too large for the budget.")
    if cfg.get("method", "dense") == "dense" and not cfg.get("rerank"):
        d.append("Pure dense retrieval can miss exact identifiers (e.g. ERR-4471). Hybrid retrieval adds keyword matching.")
    if not d:
        d.append("Healthy pipeline. Inspect the remaining misses one by one — that's how real RAG systems get tuned.")
    return d


# ------------------------------------------------------------------ generation: closed-book LM vs grounded answering
class BigramLM:
    """A real (tiny) bigram language model trained on the handbook: fluent-ish, no notion of truth."""

    def __init__(self, texts):
        self.next = defaultdict(Counter)
        for t in texts:
            w = ["<s>"] + tokens(t, stop=False) + ["</s>"]
            for a, b in zip(w, w[1:]):
                self.next[a][b] += 1

    def generate(self, seed_words: list[str], max_len=22, temperature=1.0, rng=None, min_len=10):
        rng = rng or np.random.default_rng(0)
        cur = next((w for w in reversed(seed_words) if w in self.next), "<s>")
        out = [] if cur == "<s>" else [cur]
        for _ in range(max_len):
            cands = self.next.get(cur)
            if not cands:
                break
            words = [w for w in cands if w != "</s>" or len(out) >= min_len] or list(cands)
            p = np.array([cands[w] for w in words], float) ** (1 / max(temperature, 1e-3))
            p /= p.sum()
            cur = words[rng.choice(len(words), p=p)]
            if cur == "</s>":
                break
            out.append(cur)
        return " ".join(out)

    def distribution(self, word: str, temperature=1.0, top=8):
        cands = self.next.get(word.lower(), Counter())
        if not cands:
            return []
        words = list(cands)
        p = np.array([cands[w] for w in words], float) ** (1 / max(temperature, 1e-3))
        p /= p.sum()
        order = np.argsort(-p)[:top]
        return [{"token": words[i], "p": round(float(p[i]), 4)} for i in order]


@lru_cache(maxsize=1)
def lm() -> BigramLM:
    return BigramLM([d["text"] for d in DOCS])


def grounded_answer(q: str, idx: Index, cfg: dict) -> dict:
    res = idx.search(q, cfg.get("method", "hybrid"), int(cfg.get("top_k", 3)), bool(cfg.get("rerank")), 10)
    context, included = build_context(res, int(cfg.get("context_budget", 150)))
    allowed = {i["id"] for i in included}
    qv = idx.embed(q)
    best, best_s, best_src = None, -1.0, None
    for r in res:
        if r["id"] not in allowed:
            continue
        visible = " ".join(r["text"].split()[: next(i["words"] for i in included if i["id"] == r["id"])])
        for s in sentences(visible):
            # support = lexical coverage of the question + semantic similarity (embedding cosine)
            sc = 0.6 * rerank_score(q, s, idx.bm25) + 0.4 * max(0.0, float(idx.embed(s) @ qv))
            if sc > best_s:
                best, best_s, best_src = s, sc, r["id"]
    thr = float(cfg.get("abstain_threshold", 0.0))
    if best is None or best_s < thr:
        return dict(answer="I can't find this in the provided sources.", abstained=True, support=round(max(best_s, 0), 3), citation=None, context_ids=list(allowed))
    ans = best + (f" [{best_src}]" if cfg.get("citations", True) else "")
    return dict(answer=ans, abstained=False, support=round(best_s, 3), citation=best_src if cfg.get("citations", True) else None, context_ids=list(allowed))


def evaluate_answers(cfg: dict) -> dict:
    """Hallucination boss evaluation over answerable + unanswerable questions."""
    mode = cfg.get("mode", "closed_book")
    idx = get_index(int(cfg.get("chunk_size", 40)), int(cfg.get("overlap", 10)), cfg.get("embedding", "lsa"))
    rows = []
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    items = [dict(**q, answerable=True) for q in QUESTIONS] + [dict(**q, answerable=False, answer=None, doc=None) for q in UNANSWERABLE]
    chunk_text = {c["id"]: c["text"].lower() for c in idx.chunks}
    for it in items:
        if mode == "closed_book":
            text = lm().generate(tokens(it["q"]), temperature=float(cfg.get("temperature", 1.0)), rng=rng)
            out = dict(answer=text.capitalize() + ".", abstained=False, support=None, citation=None)
        else:
            out = grounded_answer(it["q"], idx, cfg)
        correct = it["answerable"] and not out["abstained"] and it["answer"].lower() in out["answer"].lower()
        hallucinated = (not out["abstained"]) and (not correct)
        cite_ok = None
        if out.get("citation"):
            cite_ok = bool(it["answerable"] and it["answer"].lower() in chunk_text.get(out["citation"], ""))
        rows.append(dict(q=it["q"], answerable=it["answerable"], gold=it["answer"], **out, correct=correct, hallucinated=hallucinated, citation_supports=cite_ok))
    n = len(rows)
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    cited = [r for r in rows if r["citation"]]
    metrics = dict(
        hallucination_rate=round(sum(r["hallucinated"] for r in rows) / n, 3),
        answer_accuracy=round(sum(r["correct"] for r in ans) / len(ans), 3),
        coverage=round(sum(not r["abstained"] for r in ans) / len(ans), 3),
        correct_abstentions=round(sum(r["abstained"] for r in un) / len(un), 3),
        citation_accuracy=round(sum(bool(r["citation_supports"]) for r in cited) / len(cited), 3) if cited else None,
    )
    return dict(config=dict(cfg), metrics=metrics, rows=rows)


def embedding_map(query: str | None, cfg: dict) -> dict:
    idx = get_index(int(cfg.get("chunk_size", 60)), int(cfg.get("overlap", 10)), cfg.get("embedding", "lsa"))
    pca = PCA(2, random_state=0).fit(idx.E)
    P = pca.transform(idx.E)
    out = dict(points=[dict(id=c["id"], doc=c["doc"], title=c["title"], x=float(P[i, 0]), y=float(P[i, 1]), preview=c["text"][:120]) for i, c in enumerate(idx.chunks)],
               explained_variance=[round(float(v), 3) for v in pca.explained_variance_ratio_],
               note="2D PCA projection of the chunk embeddings. Real distances live in many more dimensions — nearby here usually (not always) means similar.")
    if query:
        qv = idx.embed(query)
        qp = pca.transform(qv[None])[0]
        sims = idx.E @ qv
        top = np.argsort(-sims)[:3]
        out["query"] = dict(text=query, x=float(qp[0]), y=float(qp[1]), nearest=[dict(id=idx.chunks[i]["id"], cosine=round(float(sims[i]), 3)) for i in top])
    return out


def corpus_info():
    return dict(docs=[dict(id=d["id"], title=d["title"], words=len(d["text"].split()), text=d["text"]) for d in DOCS],
                questions=QUESTIONS, unanswerable=UNANSWERABLE)

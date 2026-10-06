from neural_forge import agent, rag


def test_retrieval_metrics_and_broken_vs_good_pipeline():
    broken = rag.evaluate_retrieval(dict(chunk_size=300, overlap=0, embedding="hash32", method="dense", top_k=1, context_budget=80))
    good = rag.evaluate_retrieval(dict(chunk_size=40, overlap=10, embedding="lsa", method="hybrid", top_k=2, rerank=True, context_budget=80))
    for r in (broken, good):
        for k, v in r["metrics"].items():
            if k != "avg_context_words":
                assert 0 <= v <= 1, k
    assert good["metrics"]["context_hit_rate"] > broken["metrics"]["context_hit_rate"] + 0.4


def test_grounding_reduces_hallucination():
    closed = rag.evaluate_answers(dict(mode="closed_book"))
    grounded = rag.evaluate_answers(dict(mode="grounded", chunk_size=60, embedding="lsa", method="hybrid", top_k=3, abstain_threshold=0.5))
    assert closed["metrics"]["hallucination_rate"] > 0.8
    assert grounded["metrics"]["hallucination_rate"] < closed["metrics"]["hallucination_rate"] / 3
    assert grounded["metrics"]["correct_abstentions"] > closed["metrics"]["correct_abstentions"]


def test_answers_are_checked_against_gold():
    res = rag.evaluate_answers(dict(mode="grounded", chunk_size=60, embedding="lsa", method="hybrid", top_k=3))
    for row in res["rows"]:
        if row["correct"]:
            assert row["gold"].lower() in row["answer"].lower()


def test_agent_defences():
    none = agent.evaluate([])
    assert none["metrics"]["attack_success_rate"] == 1.0
    strong = agent.evaluate(["separate_channels", "random_delimiters", "confirm_sensitive"])
    assert strong["metrics"]["attack_success_rate"] == 0.0
    assert strong["metrics"]["utility"] == 1.0
    # keyword filter alone misses the polite, keyword-free attack
    kw = agent.evaluate(["injection_classifier"])
    assert 0 < kw["metrics"]["attack_success_rate"] < 1


def test_untrusted_data_is_labelled_in_traces():
    res = agent.evaluate(["separate_channels"])
    chans = [t for r in res["results"] for t in r["trace"] if t.get("channel") == "tool_result"]
    assert chans and all(t["trusted"] is False for t in chans)

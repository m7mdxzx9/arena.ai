from neural_forge import curriculum as cur
from neural_forge.curriculum import generators as gen

REQUIRED_BRANCHES = {"Foundations", "Mathematics", "Data Science", "Machine Learning", "Evaluation", "Deep Learning", "Modern AI", "RAG", "AI Agents", "Responsible AI"}
REQUIRED_AREAS = {"foundation_academy", "data_district", "ml_workshop", "neural_tower", "vision_lab", "language_center", "genai_facility", "rag_archives", "agent_arena", "evaluation_chamber", "research_institute"}


def test_knowledge_graph_is_valid():
    assert cur.validate() == []


def test_all_required_branches_and_areas_exist():
    assert REQUIRED_BRANCHES <= set(cur.BRANCH_ORDER)
    assert REQUIRED_AREAS <= set(cur.AREAS)
    for b in REQUIRED_BRANCHES:
        assert any(c["branch"] == b for c in cur.CONCEPT_LIST), b


def test_ranks():
    assert [r[0] for r in cur.RANKS] == ["AI Beginner", "AI Student", "Junior ML Engineer", "ML Engineer", "AI Engineer", "Research Engineer", "AI Research Scientist"]


def test_every_concept_teaches_before_testing():
    for c in cur.CONCEPT_LIST:
        assert len(c["explain"]) > 60, c["id"]
        assert c["questions"] or c["generators"], c["id"]
        for q in c["questions"]:
            assert q["prompt"] and q["explanation"], (c["id"], q["prompt"])
            if q["type"] == "mcq":
                assert 0 <= q["answer"] < len(q["options"]), (c["id"], q["prompt"])
                assert len(set(q["options"])) == len(q["options"]), (c["id"], q["prompt"])


def test_roots_exist_and_depths_are_finite():
    roots = [c for c in cur.CONCEPT_LIST if not c["prereqs"]]
    assert roots, "someone has to be able to start"
    assert max(cur.depth(c["id"]) for c in cur.CONCEPT_LIST) < 30


def test_generators_are_deterministic_and_valid():
    for name in gen.GENERATORS:
        for d in (1, 2, 3):
            a, b = gen.generate(name, d, 123), gen.generate(name, d, 123)
            assert a == b, name
            assert a["prompt"] and "answer" in a, name
            if a["type"] == "mcq":
                assert 0 <= a["answer"] < len(a["options"])

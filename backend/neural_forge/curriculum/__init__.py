"""Knowledge tree: concepts, campus areas, mentors and ranks."""
from __future__ import annotations

from . import foundations, data_ml, deep, modern

CONCEPT_LIST = foundations.CONCEPTS + data_ml.CONCEPTS + deep.CONCEPTS + modern.CONCEPTS
CONCEPTS = {c["id"]: c for c in CONCEPT_LIST}

BRANCH_ORDER = ["Foundations", "Mathematics", "Data Science", "Machine Learning", "Evaluation",
                "Deep Learning", "Modern AI", "RAG", "AI Agents", "Responsible AI"]

AREAS = {
    "foundation_academy": dict(name="Foundation Academy", mentor="Prof. Ada Loop", icon="🏛️", color="#7dd3fc",
                               x=14, y=72,
                               blurb="Where every researcher starts: Python, maths and the big picture.",
                               greeting="Welcome, apprentice! Every great model started as a single variable. Let's begin."),
    "data_district": dict(name="Data District", mentor="Tab Rivera", icon="🗃️", color="#a7f3d0", x=32, y=48,
                          blurb="Datasets, cleaning, features and the dark art of leakage.",
                          greeting="Models are only as good as their data. Let's look before we leap."),
    "ml_workshop": dict(name="Machine Learning Workshop", mentor="Forge-bot MK-II", icon="⚙️", color="#fcd34d",
                        x=52, y=70, blurb="Regression, classification, trees, forests and neighbours.",
                        greeting="BEEP. Workbench online. Hypothesis first, then experiment."),
    "evaluation_chamber": dict(name="Evaluation Chamber", mentor="Judge Metric", icon="⚖️", color="#fda4af",
                               x=70, y=50, blurb="Where models face honest judgement.",
                               greeting="A number without context is a rumour. Show me the confusion matrix."),
    "neural_tower": dict(name="Neural Network Tower", mentor="Dr. Synapse", icon="🗼", color="#c4b5fd", x=48, y=26,
                         blurb="Neurons, gradients, learning rates and the beasts that live upstairs.",
                         greeting="Every neuron is simple. The magic — and the trouble — is in the stacking."),
    "vision_lab": dict(name="Computer Vision Lab", mentor="Iris", icon="👁️", color="#f0abfc", x=28, y=18,
                       blurb="Pixels, filters and convolutional networks.", greeting="Let's teach machines to see — one 3×3 filter at a time."),
    "language_center": dict(name="Language Intelligence Center", mentor="Lex", icon="💬", color="#93c5fd", x=68, y=16,
                            blurb="Tokens, embeddings and attention.", greeting="Words become numbers. Numbers become meaning. Mostly."),
    "genai_facility": dict(name="Generative AI Facility", mentor="Muse", icon="✨", color="#fde68a", x=86, y=28,
                           blurb="Language models, prompting, structured output and hallucinations.",
                           greeting="Fluent is easy. Faithful is hard."),
    "rag_archives": dict(name="RAG Archives", mentor="Archivist Quill", icon="📚", color="#86efac", x=88, y=58,
                         blurb="Chunking, retrieval, reranking and citations.", greeting="The answer is in the archive. The trick is finding the right page."),
    "agent_arena": dict(name="Agent Arena", mentor="Captain Vector", icon="🤖", color="#fca5a5", x=84, y=84,
                        blurb="Tools, permissions, planning and defence against prompt injection.",
                        greeting="An agent with tools is powerful. An agent without boundaries is a liability."),
    "research_institute": dict(name="Research Institute", mentor="Director Nova", icon="🔭", color="#e2e8f0", x=62, y=88,
                               blurb="Responsible AI and open-ended research challenges.",
                               greeting="Here there are no answer keys — only evidence, trade-offs and integrity."),
}

RANKS = [
    # (title, xp needed, concepts mastered needed, bosses defeated needed)
    ("AI Beginner", 0, 0, 0),
    ("AI Student", 250, 8, 0),
    ("Junior ML Engineer", 900, 22, 1),
    ("ML Engineer", 2000, 40, 3),
    ("AI Engineer", 3500, 58, 5),
    ("Research Engineer", 5500, 75, 7),
    ("AI Research Scientist", 8000, 92, 8),
]

EQUIPMENT = {
    "data_lens": dict(name="Data Lens", desc="Inspect any dataset in the Data Lab.", requires=["datasets"]),
    "workbench": dict(name="ML Workbench", desc="Configure and train real models.", requires=["supervised"]),
    "comparison_scope": dict(name="Comparison Scope", desc="Compare experiment runs side-by-side.", requires=["accuracy"]),
    "neural_forge": dict(name="Neural Forge", desc="Build and train neural networks.", requires=["neuron"]),
    "tokenizer_press": dict(name="Tokenizer Press", desc="Train a BPE tokenizer and explore attention.", requires=["tokenization"]),
    "vector_vault": dict(name="Vector Vault", desc="Build and evaluate a RAG pipeline.", requires=["ingestion"]),
    "agent_harness": dict(name="Agent Harness", desc="Test agents against attack suites.", requires=["agent_model"]),
    "code_terminal": dict(name="Code Terminal", desc="Write and run real Python.", requires=["py_variables"]),
}


def validate() -> list[str]:
    """Return a list of problems in the knowledge graph (empty = valid)."""
    problems = []
    for c in CONCEPT_LIST:
        for p in c["prereqs"]:
            if p not in CONCEPTS:
                problems.append(f"{c['id']}: unknown prereq {p}")
        if c["area"] not in AREAS:
            problems.append(f"{c['id']}: unknown area {c['area']}")
        if c["branch"] not in BRANCH_ORDER:
            problems.append(f"{c['id']}: unknown branch {c['branch']}")
        diffs = {q["difficulty"] for q in c["questions"]}
        if not {1, 2} <= diffs and not c["generators"]:
            problems.append(f"{c['id']}: needs easy+medium questions")
    if len(CONCEPTS) != len(CONCEPT_LIST):
        problems.append("duplicate concept ids")
    # cycle detection
    state: dict[str, int] = {}

    def visit(n: str, stack: tuple = ()):  # 1=visiting, 2=done
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            problems.append("cycle: " + " -> ".join(stack + (n,)))
            return
        state[n] = 1
        for p in CONCEPTS[n]["prereqs"]:
            if p in CONCEPTS:
                visit(p, stack + (n,))
        state[n] = 2

    for cid in CONCEPTS:
        visit(cid)
    return problems


def depth(cid: str, _memo: dict = {}) -> int:
    if cid not in _memo:
        ps = CONCEPTS[cid]["prereqs"]
        _memo[cid] = 0 if not ps else 1 + max(depth(p) for p in ps)
    return _memo[cid]

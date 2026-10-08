"""Small evidence-driven RAG and personalized-Tutor missions.

Completion is checked against the player's saved RAG experiments or Tutor interactions;
these missions never accept a self-reported success flag.
"""

MISSIONS = [
    {
        "id": "rag_compare",
        "kind": "rag_compare",
        "title": "Compare Retrieval Strategies",
        "area": "rag_archives",
        "icon": "🧪",
        "xp": 180,
        "concepts": ["retrieval", "retrieval_eval"],
    },
    {
        "id": "rag_grounding",
        "kind": "rag_grounding",
        "title": "Cite the Evidence",
        "area": "rag_archives",
        "icon": "📑",
        "xp": 160,
        "concepts": ["grounding", "citations"],
    },
    {
        "id": "tutor_hint_ladder",
        "kind": "tutor_hint_ladder",
        "title": "Use Your Personalized Hint Ladder",
        "area": "genai_facility",
        "icon": "🧑‍🏫",
        "xp": 180,
        "concepts": ["personalized_tutoring"],
    },
]

BY_ID = {mission["id"]: mission for mission in MISSIONS}

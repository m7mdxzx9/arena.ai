from __future__ import annotations

import json


def test_personalized_tutoring_concept_is_in_order_and_teaches_evidence_and_hints():
    from neural_forge.curriculum import CONCEPTS
    from neural_forge.curriculum.modern import CONCEPTS as MODERN_CONCEPTS
    from neural_forge.curriculum.translations_ar import localized_concept

    concept_ids = [item["id"] for item in MODERN_CONCEPTS]
    assert concept_ids[concept_ids.index("structured_output") + 1 : concept_ids.index("ingestion")] == ["personalized_tutoring"]
    concept = CONCEPTS["personalized_tutoring"]
    assert len(concept["questions"]) == 3
    question_text = " ".join(question["prompt"] + " " + question["explanation"] for question in concept["questions"]).casefold()
    assert "saved run" in question_text and "mastery estimate" in question_text and "hint level 1 of 3" in question_text
    arabic = localized_concept("personalized_tutoring", concept, "ar")
    assert len(arabic["questions"]) == 3
    assert "تقدير الإتقان" in arabic["questions"][1]["prompt"]


def _player(client, name="Mission learner"):
    response = client.post("/api/players", json={"name": name, "mode": 1})
    assert response.status_code == 200
    return response.json()["player"]["id"]


def _document(client, player_id, filename="course.txt"):
    response = client.post(
        f"/api/p/{player_id}/documents",
        files={"file": (filename, b"The GPU lab closes at 9 PM. Retrieval should cite the source passage. " * 6, "text/plain")},
        data={"chunk_size": "40", "overlap": "5"},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _query(client, player_id, document_id, *, method="hybrid", question="When does the GPU lab close?"):
    response = client.post(
        f"/api/p/{player_id}/personal-rag/query",
        json={
            "question": question,
            "document_ids": [document_id],
            "method": method,
            "top_k": 5,
            "generation": "extractive",
            "embedding_provider": "statistical_lsa",
            "language": "en",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_rag_compare_mission_checks_same_question_sources_and_different_methods(client):
    player_id = _player(client)
    document_id = _document(client, player_id)
    first = _query(client, player_id, document_id, method="bm25")
    second = _query(client, player_id, document_id, method="hybrid")

    listed = client.get(f"/api/p/{player_id}/missions").json()
    assert {item["id"] for item in listed if item.get("kind") == "rag_compare"} == {"rag_compare"}
    detail = client.get(f"/api/p/{player_id}/learning-missions/rag_compare").json()
    assert detail["completed_query_runs"] == 2

    response = client.post(
        f"/api/p/{player_id}/learning-missions/rag_compare",
        json={"action": "complete", "payload": {"experiment_ids": [first["experiment_id"], second["experiment_id"]]}},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["passed"] is True
    assert all(item["passed"] for item in result["checks"])
    assert result["state"]["proof"]["methods"] == ["bm25", "hybrid"]
    assert result["xp"] >= 180
    assert client.get(f"/api/p/{player_id}").json()["player"]["xp"] >= 180

    repeated = client.post(
        f"/api/p/{player_id}/learning-missions/rag_compare",
        json={"action": "complete", "payload": {"experiment_ids": [first["experiment_id"], second["experiment_id"]]}},
    ).json()
    assert repeated["already_completed"] is True and repeated["xp"] == 0


def test_rag_missions_reject_cross_player_experiments_and_validate_citations(client):
    learner_id = _player(client, "Owner")
    other_id = _player(client, "Other")
    doc_id = _document(client, learner_id)
    owned = _query(client, learner_id, doc_id, method="bm25")
    foreign_doc = _document(client, other_id, "foreign.txt")
    foreign = _query(client, other_id, foreign_doc, method="bm25")

    response = client.post(
        f"/api/p/{learner_id}/learning-missions/rag_compare",
        json={"action": "complete", "payload": {"experiment_ids": [owned["experiment_id"], foreign["experiment_id"]]}},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "experiment_not_found"

    grounded = client.post(
        f"/api/p/{learner_id}/learning-missions/rag_grounding",
        json={"action": "complete", "payload": {"experiment_id": owned["experiment_id"]}},
    )
    assert grounded.status_code == 200, grounded.text
    assert grounded.json()["passed"] is True
    assert grounded.json()["state"]["proof"]["citation_chunk_ids"]


def test_tutor_withholds_document_passages_for_no_answer_coaching(client):
    player_id = _player(client, "Coaching learner")
    _document(client, player_id)
    response = client.post(
        f"/api/p/{player_id}/tutor",
        json={
            "question": "When does the GPU lab close?", "mode": "no_answer", "source": "offline",
            "language": "en", "use_documents": True,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rag_retrieval"]["requested"] is True
    assert result["rag_retrieval"]["used"] is False
    assert result["rag_retrieval"]["suppressed"] is True
    assert result["citations"] == [] and result["citation_details"] == []
    assert "9 PM" not in result["answer"]
    interaction = client.app.state.db.one("SELECT rag_used, rag_evidence_count FROM tutor_interactions WHERE id=?", (result["interaction_id"],))
    assert interaction["rag_used"] == 0 and interaction["rag_evidence_count"] == 0


def test_tutor_hint_ladder_requires_three_contextual_steps_on_same_concept(client):
    player_id = _player(client)
    db = client.app.state.db
    run_id = db.save_run(
        player_id,
        "ml",
        {"dataset": "student_success", "model": "decision_tree", "features": ["study_hours"]},
        {"metrics": {"train": {"accuracy": 0.92}, "test": {"accuracy": 0.78}}, "diagnosis": []},
        {"metrics": {}},
    )
    base = {"question": "Explain Precision", "mode": "hint", "source": "offline", "language": "en", "level": "beginner", "run_id": run_id}
    for level in (1, 2):
        response = client.post(f"/api/p/{player_id}/tutor", json={**base, "hint_level": level})
        assert response.status_code == 200, response.text
        assert response.json()["player_context"]["current_experiment"]["run_id"] == run_id
    premature = client.post(
        f"/api/p/{player_id}/learning-missions/tutor_hint_ladder",
        json={"action": "complete", "payload": {}},
    ).json()
    assert premature["passed"] is False
    assert premature["hint_ladder"]["context_levels"] == [1, 2]

    response = client.post(f"/api/p/{player_id}/tutor", json={**base, "hint_level": 3})
    assert response.status_code == 200, response.text
    interaction = db.one("SELECT run_id, mastery_probability, hint_level FROM tutor_interactions WHERE id=?", (response.json()["interaction_id"],))
    assert interaction["run_id"] == run_id and interaction["mastery_probability"] is not None and interaction["hint_level"] == 3

    completed = client.post(
        f"/api/p/{player_id}/learning-missions/tutor_hint_ladder",
        json={"action": "complete", "payload": {}},
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["passed"] is True
    assert completed.json()["state"]["proof"]["hint_levels"] == [1, 2, 3]
    assert completed.json()["state"]["proof"]["run_id"] == run_id


def test_retrieval_boss_requires_a_valid_personal_rag_experiment_in_final_phase(client):
    player_id = _player(client)
    document_id = _document(client, player_id)
    result = _query(client, player_id, document_id)
    response = client.get(f"/api/p/{player_id}/bosses/retrieval")
    assert response.status_code == 200, response.text
    boss = response.json()
    assert len(boss["phases"]) == 5
    assert boss["requires"] == ["retrieval", "chunking"]  # preserve the existing boss unlock gate
    assert boss["phases"][1]["kind"] == "rag"  # the existing educational simulation is preserved
    assert boss["phases"][-1]["kind"] == "rag_real"

    client.app.state.db.set_progress(player_id, "boss", "retrieval", {"phase": 4, "hp": 100, "mistakes": 0, "log": []})
    attack = client.post(f"/api/p/{player_id}/bosses/retrieval", json={"experiment_id": result["experiment_id"]})
    assert attack.status_code == 200, attack.text
    payload = attack.json()
    assert payload["evaluation"]["passed"] is True
    assert all(check["passed"] for check in payload["evaluation"]["checks"])
    assert payload["defeated"] is True

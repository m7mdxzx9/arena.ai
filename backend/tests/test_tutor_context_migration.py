import sqlite3

from neural_forge.db import DB


def test_v3_tutor_interactions_gain_context_provenance_without_data_loss(tmp_path):
    path = tmp_path / "v3.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE tutor_interactions ("
        "id TEXT PRIMARY KEY, player_id INTEGER NOT NULL, concept_id TEXT, mode TEXT NOT NULL, "
        "level TEXT NOT NULL, source TEXT NOT NULL, language TEXT NOT NULL, hint_level INTEGER NOT NULL DEFAULT 0, "
        "rag_used INTEGER NOT NULL DEFAULT 0, feedback TEXT, created_at REAL NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tutor_interactions(id,player_id,concept_id,mode,level,source,language,hint_level,rag_used,feedback,created_at) "
        "VALUES ('old-interaction',7,'recall','hint','beginner','curated_offline','en',2,0,NULL,10)"
    )
    connection.execute("PRAGMA user_version=3")
    connection.commit()
    connection.close()

    db = DB(path)
    columns = {row["name"] for row in db.q("PRAGMA table_info(tutor_interactions)")}
    assert {"run_id", "mastery_probability", "rag_evidence_count"} <= columns
    row = db.one("SELECT * FROM tutor_interactions WHERE id='old-interaction'")
    assert row["concept_id"] == "recall" and row["hint_level"] == 2
    assert row["run_id"] is None and row["mastery_probability"] is None and row["rag_evidence_count"] == 0
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 5

# Personalized Tutor

## Purpose and boundaries

The Tutor combines a curated offline explanation catalogue with optional generation from an explicitly selected local Ollama model. Offline mode is always usable; it is not called an LLM. A local-model failure is shown as an error and does not silently change the response source.

The Tutor adapts from persisted state belonging to the requesting player only. Depending on the question and selected mode, its bounded context can contain:

- the matched curriculum concept and the learner's current explanation level;
- that concept's Bayesian mastery probability, attempt count and status;
- a small summary of that player's mastered, struggling and due-for-review concepts;
- recent saved experiment configuration, dataset/model identifiers, recorded train/test or validation metrics, and diagnosis codes;
- relevant repeated mistake patterns from the player's Mistake Journal;
- explicitly supplied retrieved evidence, when available, labelled as untrusted source material.

The Tutor does not invent a run, mastery result, prior error, document citation or other learner history. If no saved experiment exists, it says there is no run context and remains useful with general instruction. The model receives a bounded context summary, not the entire profile or raw experiment history.

## Coaching modes

- **Simple, example, visual, mathematical and code** modes adapt explanation format/level.
- **Hint** mode emits one progressive hint at the requested level. The highest hint may show the key formula/step, but the Tutor does not turn hints into a false assessment result.
- **No-answer** mode asks for one next step without disclosing the solution.
- **Diagnostic question** mode asks one question and waits.

When uploaded-document retrieval is enabled, passages are deliberately withheld in **Hint**, **No-answer**, and **Diagnostic question** modes so source excerpts cannot bypass the requested coaching behavior. Other explanation modes can use local retrieved evidence and citations. Offline explanations remain useful without an LLM or document index.

The fixed offline catalogue contains Arabic explanations and hints for selected concepts; the selected local model is instructed to answer in the chosen Arabic/English language. Coverage of older curriculum material remains partial and is not presented as fully translated.

## Privacy, persistence and mastery

Raw Tutor questions and answers are not stored. A Tutor interaction records compact metadata such as player/concept, mode, hint level, source/language, selected saved-run ID, recorded mastery probability, whether RAG evidence was used, and optional learner feedback. This permits learning analytics and evidence-based mission checks without retaining a conversation transcript.

Feedback (`helpful`, `unclear`, `answered`) records preference only. It does not change BKT mastery. Mastery changes continue to come from the existing assessed learning evidence, not a Tutor compliment or thumbs-up.

## Progressive-hint side mission

The three-step `tutor_hint_ladder` mission is checked against the database: levels 1, 2 and 3 must be recorded for the same concept, and each step must include a real saved experiment context. Repeating a level or switching concepts does not satisfy the mission; missing run context does not count. Completion proof records the concept, run and hint levels, and XP is granted once.

## Verification

Backend integration tests cover player-state context, saved metrics, actual mistake history, due review topics, Arabic fallback, no-answer behavior, context provenance, non-mastery feedback, and the evidence-checked hint ladder. Optional Ollama output is tested at the provider boundary with a deterministic fake provider; no live Ollama availability is claimed.

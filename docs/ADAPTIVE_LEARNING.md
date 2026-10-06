# Adaptive Learning

## Bayesian Knowledge Tracing

Each concept stores a BKT state rather than a cosmetic percentage. An attempt records correctness, hint usage, item identity, difficulty, probability before and probability after. The update uses the existing learn/guess/slip model and preserves historical state payloads.

Mastery drives:

- recommended next concepts;
- equipment prerequisites;
- campaign unlock guidance;
- tutor context;
- review prioritization.

## Spaced review

The existing Leitner-style schedule returns learned concepts after increasing intervals (1, 3, 7, 16 and 35 days). Review results update durable state. The Mistake Journal adds a second evidence source: actual experiment diagnostics are scheduled for review and can be marked remembered or returned sooner.

## Mistake Journal

Mistakes are generated only from concrete diagnostics, for example:

- validation materially worse than training;
- leakage-prone feature choices;
- unsuitable preprocessing/model setup;
- severe class imbalance handling;
- unstable or diverging neural training.

Each record stores the action, mistake type, correct principle, explanation, example, originating run/mission when available, due time and review count. It does not populate generic filler records.

## Tutor personalization

Tutor context is bounded and assembled from mastery, recent mistakes, recent runs and optional current lab context. Offline mode selects curated concept guidance. Local mode sends this bounded context to Ollama and clearly reports provider/model metadata. “Give me a hint” and “do not give me the answer” modes alter the tutor instruction; they are not claims of formal assessment.

## Privacy and transparency

Adaptive state is local SQLite data. Provider prompts contain only the bounded context needed for the selected request. The system does not infer sensitive traits, perform hidden remote analytics or expose private chain-of-thought. Visible agent memory is distinct from mastery and is clearable.

## Current limitations

- BKT parameters are global defaults, not fitted per learner cohort.
- Recommendations use mastery and prerequisites, not a learned recommender model.
- There is no cross-device synchronization unless the player explicitly exports/restores a backup.
- Uploaded binary datasets/documents are excluded from portable JSON backup.

from .schema import C, Q, N

MA, RG, AG, RA = "Modern AI", "RAG", "AI Agents", "Responsible AI"
LC, GF, RAR, AA, RI = "language_center", "genai_facility", "rag_archives", "agent_arena", "research_institute"

CONCEPTS = [
    # ---------------- Modern AI ----------------
    C("tokenization", "Tokenization", MA, LC, ["py_lists", "what_is_ai"],
      explain="Language models don't see characters or words directly — they see tokens: pieces of text mapped to integer IDs. "
              "Subword tokenizers (like BPE) learn frequent fragments: 'unbelievable' → 'un', 'believ', 'able'. Rare words split into more tokens.",
      analogy="LEGO bricks: common shapes get their own brick; rare shapes are built from smaller bricks.",
      visual="tokenizer",
      questions=[
          Q(1, "A tokenizer converts text into…", ["Images", "A sequence of token IDs", "Embeddings directly", "Labels"], 1, "Integers indexing a vocabulary."),
          Q(2, "Why subwords instead of whole words?", ["Smaller vocabulary while still handling unseen words", "They are prettier", "Faster GPUs", "No reason"], 0, "Open vocabulary."),
          Q(3, "Context limits and API prices are usually counted in…", ["Characters", "Tokens", "Sentences", "Bytes on disk"], 1, "Tokens are the model's unit."),
      ]),

    C("embeddings", "Embeddings", MA, LC, ["vectors", "tokenization"],
      explain="An embedding is a vector representing meaning: similar items get nearby vectors. Words, sentences, documents, even images can be embedded. "
              "Similarity is usually measured with cosine similarity.",
      analogy="A map where cities with similar culture sit close together — even though the map is in hundreds of dimensions.",
      visual="embedding_map",
      questions=[
          Q(1, "Two texts with similar meaning should have embeddings that are…", ["Far apart", "Close (high cosine similarity)", "Identical always", "Orthogonal"], 1, "Semantic similarity."),
          Q(2, "2D plots of embeddings are…", ["Exact", "Projections that lose information — useful for intuition only", "Always misleading", "The real embeddings"], 1,
            "Real embeddings have hundreds of dimensions."),
          Q(3, "Embeddings enable which application directly?", ["Semantic search", "Sorting integers", "Compiling code", "Database backups"], 0, "Nearest-neighbour search over meaning."),
      ]),

    C("attention", "Attention", MA, LC, ["embeddings", "matrices"],
      explain="Attention lets each token build its representation as a weighted mix of other tokens. Each token emits a query; others offer keys; "
              "similarity(query, key) → softmax weights → weighted sum of values. 'it' in 'the cat sat because it was tired' can attend to 'cat'.",
      analogy="In a noisy room you 'attend' to the voices relevant to you and tune out the rest.",
      visual="attention_heatmap",
      questions=[
          Q(1, "Attention weights for one query sum to…", ["0", "1 (softmax)", "Number of tokens", "Anything"], 1, "Softmax normalises."),
          Q(2, "Query, Key, Value: which pair is compared to compute the weights?", ["Value & Value", "Query & Key", "Key & Value", "Query & Value"], 1, "Then values are mixed."),
          Q(3, "Attention weights are…", ["Hand-written rules", "Computed per input from learned projections", "Fixed constants", "Random"], 1, "Different sentences, different weights."),
      ]),

    C("transformers", "Transformers in Practice", MA, GF, ["attention", "transformers_intro"],
      explain="A transformer block = multi-head attention + a feed-forward network, with residual connections and normalisation. Stack dozens of blocks, "
              "pre-train on huge text to predict the next token, and you get a large language model.",
      analogy="A tower of identical floors; each floor refines every token's understanding by consulting the others.",
      questions=[
          Q(1, "GPT-style models are trained to…", ["Classify images", "Predict the next token", "Cluster documents", "Sort lists"], 1, "Causal language modelling."),
          Q(2, "Multi-head attention means…", ["One attention computation", "Several attention computations in parallel, each can focus on different relations", "Multiple GPUs", "Multiple datasets"], 1, "Heads."),
          Q(3, "Residual connections help because…", ["They add data", "They let gradients and information flow through deep stacks", "They remove attention", "They tokenize"], 1, "Easier optimisation."),
      ]),

    C("language_models", "Language Models", MA, GF, ["probability", "tokenization"],
      explain="A language model assigns probabilities to the next token given previous tokens, then generates text by repeatedly sampling. "
              "It learns statistical patterns of language — fluency is not the same as factual correctness.",
      analogy="Your phone keyboard's autocomplete, scaled up enormously.",
      visual="ngram_lm",
      questions=[
          Q(1, "At each generation step, an LM outputs…", ["A full essay", "A probability distribution over the next token", "A fact database lookup", "An embedding only"], 1, "Then one token is chosen."),
          Q(2, "Higher temperature when sampling makes output…", ["More deterministic", "More random/creative", "Shorter", "Factual"], 1, "Flattens the distribution."),
          Q(3, "A fluent, confident LM answer is guaranteed to be true.", ["True", "False"], 1, "Fluency ≠ truth → hallucination."),
      ]),

    C("hallucination", "Hallucination", MA, GF, ["language_models"],
      explain="A hallucination is a fluent output not supported by facts or provided sources. LMs optimise for plausible continuations, not truth. "
              "Mitigations: grounding in retrieved sources, citations, allowing 'I don't know', verification.",
      analogy="A student who never says 'I don't know' and confidently makes up an answer.",
      visual="ngram_lm",
      questions=[
          Q(1, "Hallucinations happen mainly because LMs…", ["Are lazy", "Generate plausible text without a built-in truth check", "Have bugs", "Use GPUs"], 1, "Plausibility objective."),
          Q(2, "Which mitigation is most effective for questions about your company documents?", ["Higher temperature", "Retrieve relevant docs and require answers grounded in them", "Longer answers", "More emojis"], 1, "RAG + grounding."),
          Q(3, "Allowing the system to abstain ('not found in sources') will…", ["Increase hallucinations", "Reduce hallucinations at the cost of answering fewer questions", "Have no effect", "Break RAG"], 1, "Coverage vs faithfulness trade-off."),
      ]),

    C("context_windows", "Context Windows", MA, GF, ["tokenization", "transformers_intro"],
      explain="The context window is the maximum number of tokens a model can consider at once (prompt + output). Content beyond it is invisible. "
              "Even inside it, long contexts can dilute attention — retrieval of the *right* chunks still matters.",
      analogy="A desk: only the papers on the desk can be read right now.",
      questions=[
          Q(1, "If a document exceeds the context window…", ["The model reads it anyway", "The excess must be truncated, summarised or retrieved selectively", "It is compressed for free", "Error-free always"], 1, "Hard limit."),
          Q(2, "Context windows are measured in…", ["Pages", "Tokens", "Seconds", "Embeddings"], 1, "Tokens."),
          Q(3, "Why use RAG even with a huge context window?", ["No reason", "Cost, latency, and focus: sending only relevant chunks is cheaper and often more accurate", "Windows are fake", "It is required"], 1, "Precision of context."),
      ]),

    C("prompting", "Prompting", MA, GF, ["language_models"],
      explain="Prompting = instructing a model through its input. Effective prompts state the role, task, constraints, format, and give examples (few-shot). "
              "Prompts are not security boundaries.",
      analogy="Briefing a skilled contractor: vague briefs get vague results.",
      questions=[
          Q(1, "Which prompt is better?", ["'Write about data.'", "'Summarise this report in 3 bullet points for a non-technical manager; cite page numbers.'", "'Hi'", "'Be smart.'"], 1, "Specific task, audience, format."),
          Q(2, "Few-shot prompting means…", ["Short prompts", "Including worked examples in the prompt", "Fewer users", "Lower temperature"], 1, "Examples guide the format."),
          Q(3, "Can 'Never reveal the password' in a system prompt fully guarantee secrecy?", ["Yes", "No — prompts can be manipulated; enforce security outside the model", "Only in English", "With caps lock"], 1,
            "Defence must be architectural."),
      ]),

    C("structured_output", "Structured Output", MA, GF, ["prompting", "py_dicts"],
      explain="Asking models for machine-readable output (JSON matching a schema) lets software use the result reliably. Always validate: parse, check types and required fields, and handle failures.",
      analogy="A form with labelled boxes instead of a free-form letter.",
      questions=[
          Q(1, "Why request JSON output?", ["It looks nicer", "Programs can parse and validate it reliably", "It is shorter", "Models prefer it"], 1, "Integration."),
          Q(2, "The model returns invalid JSON. Robust system should…", ["Crash", "Validate, then retry or fall back with an error", "Trust it", "Ignore"], 1, "Defensive parsing."),
          Q(3, "A schema says 'priority' ∈ {low, medium, high}. Model outputs 'urgent'. You should…", ["Accept", "Reject/normalise via validation", "Store as is", "Delete the user"], 1, "Validate enums."),
      ]),

    # ---------------- RAG ----------------
    C("ingestion", "Document Ingestion", RG, RAR, ["datasets", "embeddings"],
      explain="RAG (Retrieval-Augmented Generation) starts by ingesting documents: loading files, extracting clean text, keeping metadata (title, source, date) for citations, "
              "and removing junk (headers, duplicates).",
      analogy="A library cataloguing new books before anyone can find them.",
      questions=[
          Q(1, "Why keep metadata like source and title?", ["Decoration", "For citations and filtering", "Faster GPUs", "Required by Python"], 1, "Traceability."),
          Q(2, "Duplicated documents in the index can…", ["Improve recall always", "Crowd out diverse results in top-K", "Nothing", "Reduce cost"], 1, "Redundant retrieval."),
          Q(3, "Ingesting untrusted web pages introduces which risk?", ["None", "Prompt injection via document content", "Lower latency", "Better grounding"], 1, "Retrieved text is untrusted data."),
      ]),

    C("chunking", "Chunking", RG, RAR, ["ingestion"],
      explain="Documents are split into chunks before embedding. Too large: a chunk mixes many topics and its embedding gets blurry. Too small: answers get cut in half. "
              "Overlap between chunks keeps sentences that straddle boundaries.",
      analogy="Cutting a pizza: slices too big won't fit the plate (context); crumbs are useless.",
      visual="chunker",
      questions=[
          Q(1, "Very large chunks tend to…", ["Have precise embeddings", "Blur several topics into one vector", "Improve recall always", "Reduce tokens"], 1, "Dilution."),
          Q(2, "Overlap between chunks helps…", ["Keep information that spans a boundary", "Reduce storage", "Remove duplicates", "Speed"], 0, "Continuity."),
          Q(3, "Best chunk size is…", ["Always 512 tokens", "Found by evaluating retrieval on your own questions", "As large as possible", "1 word"], 1, "Empirical."),
      ], predict="chunk_size"),

    C("vector_search", "Vector Search", RG, RAR, ["embeddings", "knn"],
      explain="Each chunk is stored as an embedding vector. A question is embedded the same way, and we find the chunks with the highest cosine similarity — "
              "exactly nearest-neighbour search. Vector databases make this fast at scale with approximate indexes.",
      analogy="k-NN again — but over meaning instead of house sizes.",
      visual="embedding_map",
      questions=[
          Q(1, "Vector search returns chunks that are…", ["Newest", "Most similar in embedding space", "Longest", "Alphabetical"], 1, "Nearest neighbours."),
          Q(2, "Query and documents must be embedded with…", ["Different models", "The same embedding model", "No model", "Random vectors"], 1, "Same space."),
          Q(3, "Pure vector search can miss exact identifiers like 'ERR-4471'. A remedy?", ["Hybrid search with keyword (BM25) scoring", "Bigger LLM", "Remove numbers", "Shorter docs"], 0, "Lexical + semantic."),
      ]),

    C("retrieval", "Retrieval & Top-K", RG, RAR, ["vector_search"],
      explain="Retrieval selects the top-K chunks to put in the model's context. Small K risks missing the answer; large K adds noise and cost. "
              "Hybrid retrieval combines keyword and vector scores.",
      analogy="Fetching K books from the shelf for a student to read before answering.",
      questions=[
          Q(1, "Increasing K usually increases…", ["Recall of relevant chunks (and noise)", "Precision always", "Nothing", "Hallucinations to zero"], 0, "More chances, more noise."),
          Q(2, "Hybrid retrieval combines…", ["Two LLMs", "Keyword (lexical) and embedding (semantic) scores", "Training and test", "Images and audio"], 1, "Best of both."),
          Q(3, "The answer is in the corpus but never retrieved. The generator will…", ["Still answer correctly", "Lack the evidence — may abstain or hallucinate", "Crash", "Retrain"], 1, "Retrieval failure."),
      ]),

    C("reranking", "Reranking", RG, RAR, ["retrieval"],
      explain="A reranker takes the top-N retrieved candidates and re-scores each against the question more carefully (e.g., a cross-encoder reading both together), "
              "then keeps the best K. Cheap first-stage recall + expensive second-stage precision.",
      analogy="A recruiter quickly shortlists 50 CVs; the hiring manager carefully reads them and picks 5.",
      questions=[
          Q(1, "A reranker operates on…", ["The whole corpus", "A shortlist of retrieved candidates", "Only the answer", "Training data"], 1, "Second stage."),
          Q(2, "Why not rerank the entire corpus?", ["Too slow/expensive per document", "Not allowed", "It is worse", "No reason"], 0, "Cross-encoders are costly."),
          Q(3, "Reranking mainly improves…", ["Recall of the first stage", "Precision/order of the final top-K", "Tokenization", "Chunking"], 1, "It can't recover what wasn't retrieved."),
      ]),

    C("grounding", "Grounding", RG, RAR, ["retrieval", "hallucination"],
      explain="Grounding means the answer must be supported by provided evidence. Instruct the model to use only the sources, check support, and abstain when evidence is missing.",
      analogy="A journalist who only reports what their sources confirmed.",
      questions=[
          Q(1, "A grounded answer is…", ["Long", "Supported by the provided sources", "Creative", "From memory"], 1, "Evidence-based."),
          Q(2, "Sources don't contain the answer. Grounded system should…", ["Guess", "Say it can't find it in the sources", "Use the longest chunk", "Cite randomly"], 1, "Abstain."),
          Q(3, "Grounding is measured by…", ["Answer length", "Faithfulness: fraction of claims supported by retrieved text", "Latency", "Token count"], 1, "Faithfulness metrics."),
      ]),

    C("citations", "Citations", RG, RAR, ["grounding"],
      explain="Citations attach the source chunk to each claim so users can verify it. Citations must be checked: a citation pointing to a chunk that doesn't support the claim is worse than none.",
      analogy="Footnotes in a research paper.",
      questions=[
          Q(1, "Main purpose of citations in RAG?", ["Look academic", "Let users verify claims", "Save tokens", "Increase speed"], 1, "Verifiability."),
          Q(2, "A citation points to a chunk that doesn't mention the claim. This is…", ["Fine", "A citation error that misleads users", "Grounded", "Reranking"], 1, "False support."),
          Q(3, "Citations require ingestion to keep…", ["Metadata (doc id, title, location)", "Only text", "Embeddings only", "Nothing"], 0, "Traceability."),
      ]),

    C("retrieval_eval", "Retrieval Evaluation", RG, RAR, ["retrieval", "precision", "recall"],
      explain="Measure retrieval with a labelled set of questions and their relevant chunks. Recall@K: did any relevant chunk appear in the top K? "
              "MRR: how high did the first relevant chunk rank (1/rank averaged)? Tune chunking, embeddings, K and reranking against these numbers.",
      analogy="Grading the librarian, not the student.",
      questions=[
          N(1, "10 questions; for 7 a relevant chunk is in the top-3. Recall@3?", 0.7, 0.001, "7/10."),
          N(2, "First relevant chunk ranks: 1, 2, 4. MRR? (2 dp)", 0.58, 0.01, "(1 + 0.5 + 0.25)/3 ≈ 0.583."),
          Q(3, "Why evaluate retrieval separately from final answers?", ["It's faster to blame", "It isolates where failures come from (retrieval vs generation)", "Not needed", "Required by law"], 1, "Component evaluation."),
      ], generators=["mrr_calc"]),

    # ---------------- AI Agents ----------------
    C("agent_model", "Agents: The Model", AG, AA, ["language_models", "prompting"],
      explain="An AI agent wraps a model in a loop: observe → decide → act (often by calling tools) → observe results → repeat until done. "
              "The model is the decision-maker, not the whole system.",
      analogy="The model is the brain; tools are the hands; the loop is the daily routine.",
      questions=[
          Q(1, "What makes something an agent rather than a chatbot?", ["Bigger model", "A loop that takes actions via tools and reacts to results", "Emojis", "Voice"], 1, "Action + feedback."),
          Q(2, "In an agent, the LLM typically…", ["Executes code itself", "Decides which action/tool to use next", "Stores the database", "Is optional"], 1, "Planner/decider."),
          Q(3, "Biggest new risk when moving from chatbot to agent?", ["Spelling", "Actions have real-world side effects", "Cost of tokens", "Lower creativity"], 1, "Side effects need safeguards."),
      ]),

    C("tools", "Tools", AG, AA, ["agent_model", "py_functions"],
      explain="Tools are functions an agent can call: search_docs, calculator, send_email. Each tool has a name, description, and typed parameters. "
              "Tools extend capabilities — and expand the attack surface.",
      analogy="Apps on a phone: each grants a capability, each needs permission.",
      questions=[
          Q(1, "A tool definition usually includes…", ["Name, description, parameter schema", "Only a name", "Training data", "Weights"], 0, "So the model knows how to call it."),
          Q(2, "Which tool is most sensitive?", ["calculator", "search_docs", "transfer_money", "get_time"], 2, "Irreversible side effects."),
          Q(3, "Why give a calculator tool to a language model?", ["LMs can be unreliable at exact arithmetic", "To slow it down", "No reason", "To tokenize"], 0, "Delegate to deterministic code."),
      ]),

    C("memory", "Memory", AG, AA, ["agent_model", "context_windows"],
      explain="Agents need memory beyond one context window: short-term (the conversation/scratchpad) and long-term (stored facts retrieved later, often with RAG). "
              "Memory can also store poisoned or private information — handle with care.",
      analogy="A notebook the agent writes in and rereads.",
      questions=[
          Q(1, "Long-term agent memory is often implemented with…", ["Retrieval over stored notes", "Bigger GPUs", "Dropout", "One-hot encoding"], 0, "RAG over memories."),
          Q(2, "A malicious instruction gets saved into memory. Risk?", ["None", "It may influence future sessions (persistent injection)", "Memory deletes it", "Faster"], 1, "Memory poisoning."),
          Q(3, "Short-term memory is limited by…", ["Disk size", "The context window", "Learning rate", "Batch size"], 1, "Tokens."),
      ]),

    C("planning", "Planning", AG, AA, ["agent_model"],
      explain="Planning breaks a goal into steps and orders them. Plans should be revisable: after each tool result, check whether the plan still makes sense.",
      analogy="A travel itinerary you adjust when a train is cancelled.",
      questions=[
          Q(1, "Planning in agents means…", ["Choosing a font", "Decomposing a goal into steps", "Training", "Embedding"], 1, "Task decomposition."),
          Q(2, "A tool returns an error mid-plan. Good agent behaviour?", ["Continue blindly", "Re-plan or retry with a fix", "Delete files", "Stop forever"], 1, "Adaptive."),
          Q(3, "Very long autonomous plans increase…", ["Reliability", "Chance of compounding errors", "Nothing", "Grounding"], 1, "Errors compound; add checkpoints."),
      ]),

    C("permissions", "Permissions", AG, AA, ["tools", "security"],
      explain="Least privilege: give an agent only the tools and data it needs for the current task. Require human confirmation for irreversible or sensitive actions. "
              "Enforce permissions in code, not in prompts.",
      analogy="A hotel key card that opens only your room — and only during your stay.",
      questions=[
          Q(1, "Least privilege means…", ["Give all tools to be safe", "Only the minimum tools/data needed", "No tools", "Admin rights"], 1, "Minimise blast radius."),
          Q(2, "Where should 'agent may not send emails' be enforced?", ["Only in the prompt", "In the tool-execution layer (code)", "In the dataset", "Nowhere"], 1, "Prompts can be bypassed."),
          Q(3, "A summarisation task needs which tools?", ["read_docs only", "read_docs + send_email + delete_file", "All tools", "transfer_money"], 0, "Scope to task."),
      ]),

    C("tool_calling", "Tool Calling", AG, AA, ["tools", "structured_output"],
      explain="The model emits a structured call like {\"tool\": \"search_docs\", \"args\": {\"query\": \"refund policy\"}}. Your code validates it, executes the tool, "
              "and returns the result as a new observation. Never execute unvalidated calls.",
      analogy="Filling out a work order form that a supervisor checks before the work happens.",
      questions=[
          Q(1, "Who actually executes a tool call?", ["The LLM", "Your application code", "The user", "The GPU"], 1, "Model proposes, code disposes."),
          Q(2, "Before executing a tool call you should…", ["Trust it", "Validate tool name, arguments, and permissions", "Encode it", "Translate it"], 1, "Validation."),
          Q(3, "Tool results returned to the model should be treated as…", ["Trusted instructions", "Untrusted data", "System prompts", "Labels"], 1, "Could contain injections."),
      ]),

    C("failure_recovery", "Failure Recovery", AG, AA, ["tool_calling", "planning"],
      explain="Tools time out, return errors, or produce bad data. Robust agents: retry with backoff, validate outputs, fall back to safer options, cap iterations, and escalate to a human.",
      analogy="A pilot's checklist for engine failure.",
      questions=[
          Q(1, "An agent loops calling the same failing tool 500 times. Missing safeguard?", ["Iteration/retry limits", "Bigger model", "Dropout", "Tokenization"], 0, "Cap loops."),
          Q(2, "Transient network error. Reasonable first response?", ["Give up", "Retry with backoff", "Delete data", "Hallucinate a result"], 1, "Transient faults are common."),
          Q(3, "When should an agent escalate to a human?", ["Never", "When stakes are high or it is uncertain after retries", "Always", "Only on Mondays"], 1, "Human-in-the-loop."),
      ]),

    C("agent_eval", "Agent Evaluation", AG, AA, ["failure_recovery", "retrieval_eval"],
      explain="Evaluate agents on suites of tasks: task success rate, safety violations (e.g., attack success rate), number of steps, cost, and recovery from injected failures. "
              "A safe agent that completes nothing is not useful; a useful agent that leaks data is not safe — measure both.",
      analogy="A driving test checks both reaching the destination and obeying traffic laws.",
      questions=[
          Q(1, "Which pair should an agent evaluation report together?", ["Utility and safety", "Only speed", "Only cost", "Font and colour"], 0, "Trade-offs."),
          Q(2, "Attack success rate measures…", ["How often adversarial inputs made the agent do something harmful", "Accuracy", "Tokens", "Latency"], 0, "Security metric."),
          Q(3, "Blocking every document containing the word 'ignore' would…", ["Be perfect", "Cause false positives on benign docs and miss rephrased attacks", "Fix all injection", "Improve recall"], 1, "Defence in depth."),
      ]),

    # ---------------- Responsible AI ----------------
    C("fairness_bias", "Bias & Fairness", RA, RI, ["datasets", "accuracy"],
      explain="Models can perform worse for, or unfairly treat, certain groups — often inherited from biased or unrepresentative data. "
              "Measure performance per group, not just overall.",
      analogy="A map drawn only from one neighbourhood's perspective.",
      questions=[
          Q(1, "Best first step to detect bias?", ["Overall accuracy", "Compare metrics across groups", "Bigger model", "Delete data"], 1, "Disaggregated evaluation."),
          Q(2, "Removing the 'gender' column guarantees a fair model.", ["True", "False — other features can act as proxies"], 1, "Proxies (postcode, name...)."),
          Q(3, "A hiring model trained on historical decisions may…", ["Be neutral", "Reproduce historical discrimination", "Be unbiased by design", "Only learn skills"], 1, "Labels encode past bias."),
      ]),
    C("privacy", "Privacy", RA, RI, ["datasets"],
      explain="Training data and prompts can contain personal information. Minimise collection, anonymise where possible, control access, and remember models can memorise and leak training data.",
      analogy="Confidential files belong in a locked cabinet, not on a public noticeboard.",
      questions=[
          Q(1, "Data minimisation means…", ["Collect as much as possible", "Collect only what is needed", "Compress files", "Use small models"], 1, "Principle of privacy laws."),
          Q(2, "Pasting customer records into a third-party chatbot may…", ["Be harmless always", "Violate privacy policy/law", "Improve privacy", "Encrypt data"], 1, "Data leaves your control."),
          Q(3, "Can large models reproduce verbatim snippets of training data?", ["Never", "Yes, sometimes (memorisation)", "Only images", "Only code"], 1, "Memorisation risk."),
      ]),
    C("security", "AI Security", RA, RI, ["what_is_ai"],
      explain="AI systems face specific attacks: prompt injection, data poisoning, adversarial examples, model/data extraction. Treat model inputs and outputs as untrusted; "
              "enforce boundaries in code.",
      analogy="A receptionist who must never follow instructions written on visitors' notes.",
      questions=[
          Q(1, "Prompt injection is…", ["A training technique", "Untrusted text containing instructions that hijack the model", "A tokenizer", "A metric"], 1, "Instruction/data confusion."),
          Q(2, "Data poisoning targets…", ["Training data", "Monitors", "Keyboards", "Loss curves"], 0, "Corrupting what the model learns."),
          Q(3, "Most robust defence principle?", ["A clever prompt", "Defence in depth: separation of data/instructions, least privilege, confirmation, monitoring", "Bigger model", "Secrecy"], 1, "Layers."),
      ]),
    C("rai_hallucination", "Hallucination Risk", RA, RI, ["hallucination"],
      explain="In high-stakes domains (medical, legal, finance), hallucinations cause real harm. Design for verification: grounding, citations, abstention, human review.",
      analogy="A confident but wrong map in a minefield.",
      questions=[
          Q(1, "In a medical assistant, an unsupported answer should…", ["Be shown confidently", "Be flagged or withheld; escalate to a professional", "Be ignored", "Be longer"], 1, "Safety first."),
          Q(2, "Who is responsible for AI outputs used in decisions?", ["Nobody", "The people/organisations deploying them", "The GPU", "The tokenizer"], 1, "Accountability."),
          Q(3, "Measuring hallucination rate requires…", ["Nothing", "Reference answers or source checks", "Higher temperature", "More tokens"], 1, "Evaluation."),
      ]),
    C("rai_evaluation", "Responsible Evaluation", RA, RI, ["cross_validation", "fairness_bias"],
      explain="Responsible evaluation goes beyond one number: test on realistic data, slice by group, measure robustness, report uncertainty, and document limitations (model cards).",
      analogy="A car crash test from many angles — not just the front.",
      questions=[
          Q(1, "A model card documents…", ["Training data, metrics, intended use, limitations", "GPU price", "Font", "Only accuracy"], 0, "Transparency."),
          Q(2, "Reporting '0.91 ± 0.04 over 5 folds' instead of '0.91' adds…", ["Uncertainty information", "Nothing", "Leakage", "Bias"], 0, "Variability."),
          Q(3, "Evaluating only on the same distribution as training hides…", ["Distribution-shift failures", "Nothing", "Loss", "Tokens"], 0, "Real-world drift."),
      ]),
    C("limitations", "Limitations of AI", RA, RI, ["what_is_ai"],
      explain="Models only know patterns in their data; they can fail silently on unusual inputs, don't understand causality by default, and can't be more reliable than their data and evaluation.",
      analogy="A weather model trained on summers asked to forecast a blizzard.",
      questions=[
          Q(1, "Correlation learned by a model implies causation.", ["True", "False"], 1, "Ice-cream sales and drownings both rise in summer."),
          Q(2, "Inputs unlike anything in training are called…", ["Out-of-distribution", "Labels", "Batches", "Epochs"], 0, "OOD."),
          Q(3, "Best practice for a model in production?", ["Deploy and forget", "Monitor performance and data drift over time", "Retrain hourly blindly", "Delete logs"], 1, "Monitoring."),
      ]),
    C("data_quality", "Data Quality", RA, RI, ["missing_values", "data_leakage"],
      explain="Data quality dimensions: completeness (missing), validity (types/ranges), consistency (categories), uniqueness (duplicates), timeliness, and accuracy of labels. "
              "Quality checks belong in every pipeline.",
      analogy="Ingredients inspection before cooking.",
      questions=[
          Q(1, "'Age = −4' violates…", ["Validity", "Timeliness", "Uniqueness", "Privacy"], 0, "Out of valid range."),
          Q(2, "Same customer twice with different IDs violates…", ["Uniqueness", "Completeness", "Validity", "Fairness"], 0, "Duplicates."),
          Q(3, "Where should data-quality checks run?", ["Once, manually", "Automatically in the pipeline, every time data arrives", "Never", "After deployment only"], 1, "Automation."),
      ]),
]

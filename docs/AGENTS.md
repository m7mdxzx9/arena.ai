# Agents

## Educational simulator

The original Agent Security Simulator is preserved. It is a deterministic teaching state machine and does not call a model or execute tools. The UI identifies it as a simulator.

## Real Agent Lab

The real lab uses an optional local Ollama model. A configuration persists:

- selected local model and system instruction;
- enabled tool names;
- separately granted permissions;
- maximum steps (1–12) and overall timeout (5–180 seconds);
- visible bounded memory (maximum 20 notes).

The model must return a structured `tool` or `final` decision. The loop validates every decision, input schema, enabled-tool check and permission check in code. Tool results return as explicitly untrusted observations. The model cannot create new tools.

## Tool registry

| Tool | Permission | Boundaries |
|---|---|---|
| calculator | `calculate` | AST numeric operators only; no `eval`, names or calls |
| statistics | `calculate` | At most 1,000 finite supplied values |
| dataset inspector | `read_datasets` | Bundled or player-owned datasets; bounded profile, no path |
| experiment lookup | `read_experiments` | One player-owned saved run |
| knowledge search | `read_documents` | Player chunks; result explicitly untrusted |
| mission information | `read_missions` | Public bundled mission data |
| save note | `write_memory` | One visible 500-character note; 20-item cap |

There is **no shell, arbitrary filesystem, Python execution, arbitrary HTTP or process tool**. Each tool has a schema and timeout. Dataset and document tools re-check player ownership.

## Observable trace

Persisted/returned events include request, provider/model, selected tool and arguments, permission decision, bounded tool result, latency, status and final answer. The system never requests or stores private chain-of-thought. “Observable” means actions and evidence, not hidden reasoning.

## Failure behavior

Unknown tools, absent permissions, malformed arguments and timeouts produce structured observations. The model may react in the next bounded step. Invalid model output ends the run visibly. Ollama unavailability is a provider error, never a fabricated completion.

## Memory

Memory is configuration-local, visible in the UI, bounded and clearable. Only the `save_note` tool with `write_memory` permission can add a note. It is not a hidden vector store and should not be treated as a secret vault.

## Agent-vs-agent arena

The arena runs the same 1–10 tasks through 2–4 saved configurations. Optional `expected_contains` phrases produce transparent deterministic quality scores; tasks without an expected phrase are labelled completion-only. The leaderboard compares deterministic score, completion rate, average steps and measured latency, saves an `agent_arena` experiment, and never uses an uncalibrated LLM judge.

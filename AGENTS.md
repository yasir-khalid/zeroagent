# Working on ZeroAgent

This is a learning project, not a production harness. The point is to
understand how an agent loop, context management, tool calling, and
verification actually work under the hood — not to ship the most capable
agent possible.

## Ground rules

- **No agent frameworks.** No LangChain, no ADK, no CrewAI, no framework
  primitives of any kind. We talk to model providers directly over their
  raw HTTP/SDK APIs (currently OpenRouter via the `openai` Python client,
  since it exposes an OpenAI-compatible chat-completions endpoint).
- **Minimal, plain Python.** Dataclasses, plain classes, stdlib. Avoid
  clever abstractions, plugin/processor pipelines, dependency injection,
  or metaprogramming. If a concept needs more than one layer of
  indirection to explain, it's probably too fancy for this repo.
- **Optimize for readability over generality.** Someone should be able to
  open any file here, read it top to bottom, and understand exactly what
  it does without chasing definitions through five files. Prefer a
  concrete 20-line implementation over a general 200-line one.
- **New dependencies are a deliberate choice, not a default.** Reach for
  the standard library first. Only add a package when it removes real
  complexity (e.g. `openai`, `rich`) — not for convenience.
- **Every piece should be explainable from first principles**: why does
  this event get appended here, why does compaction happen at this
  threshold, why is the prefix kept stable. If you can't explain why,
  simplify until you can.

## Why this shape

Each top-level package is a single harness concern, kept intentionally
small so it can be read in one sitting:

- `agent/` — the tool-calling loop (`loop.py`) and the context manager
  (`context.py`, `compaction.py`) that decides what actually gets sent to
  the model each turn.
- `state/` — the append-only event log (`events.py`) that everything else
  is a view over.
- `tools/`, `runtime/`, `verification/`, `evals/` — scaffolding for the
  other harness capabilities (tool execution, sandboxing, checkpointing,
  grading) as they get explored.

When extending any of these, keep following the same rule: build the
smallest thing that demonstrates the concept clearly, not the most
capable version of it.

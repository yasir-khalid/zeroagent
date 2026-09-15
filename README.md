### ZeroAgent: Coding Agent Runtime / Harness Lab
##### Python agents built from scratch — no frameworks, just raw API calls.

Building a coding-agent runtime from first principles to study context engineering, tool/ACI design, sandboxed execution, checkpointing, agent evaluation and verification loops. Benchmarking different harness architectures against real software-engineering tasks.

Simply put, an agent is a loop of language-model calls that simulates a thought process.

To use, set an environment variable: `OPENROUTER_API_KEY` and that allows you use different models available across Openrouter. 

```
python3 cli.py
 _____                   ___                    __ 
/__  /  ___  _________  /   | ____ ____  ____  / /_
  / /  / _ \/ ___/ __ \/ /| |/ __ `/ _ \/ __ \/ __/
 / /__/  __/ /  / /_/ / ___ / /_/ /  __/ / / / /_  
/____/\___/_/   \____/_/  |_\__, /\___/_/ /_/\__/  
                           /____/                  

[22:19:32] Agent processing finished
[22:19:33] ⛏ Initiating tool call: `search_gutenberg_books` 
[22:19:37] Agent processing finished
```

## Project structure

The reusable agent loop and conversation context live in `agent/`. The remaining
top-level packages are scaffolding for the harness capabilities currently being
explored: tools, runtime isolation, state, verification, and evaluations.
`zeroagent/main.py` remains as a compatibility entry point for the original
Gutenberg demo, so `python3 zeroagent/main.py` also continues to work.

## Focus

ReAct: Reasoning and Acting by LLM Agents

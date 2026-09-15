### ZeroAgent: Coding Agent Runtime / Harness Lab
##### Python agents built from scratch — no frameworks, just raw API calls.

Building a coding-agent runtime from first principles to study context engineering, tool/ACI design, sandboxed execution, checkpointing, agent evaluation and verification loops. Benchmarking different harness architectures against real software-engineering tasks.

To use, set an environment variable: `OPENROUTER_API_KEY` and that allows you use different models available across Openrouter. 

```
python zeroagent/main.py
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

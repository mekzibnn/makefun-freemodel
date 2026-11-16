# makefun-freemodel

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/mekzibnn/makefun-freemodel/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/mekzibnn/makefun-freemodel/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two wires, a real free model behind every answer: an OpenAI/Anthropic-compatible aggregator that routes each prompt to the best-rated free OpenRouter model meeting its requirements, with per-model cooldowns and failover. It serves `POST /v1/chat/completions` and `POST /v1/messages` (JSON and streaming), answers `GET /v1/models` in the wire its auth header names, registers its port and its fourteen-model catalog with the trial env, and exits 0 when the env says done.

## Run it

```sh
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> OPENAI_BASE_URL=<gateway> node main.ts
```

`node main.ts --list` prints the catalog document and exits 0. Node 22+, standard library only, nothing to install.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>

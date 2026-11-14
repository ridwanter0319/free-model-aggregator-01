# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ridwanter0319/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ridwanter0319/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A model gateway that sits in front of ten free-tier providers and answers OpenAI- and Anthropic-shaped requests from the best-rated free model that fits, inside each provider's call budget. It registers an exact catalog of the providers it probed, fails over when a provider refuses, and translates requests and streaming answers between the OpenAI, Anthropic and Gemini wires without losing a field.

## Run it

```sh
MAKEFUN_PROMPT_FILE=<trial-brief.md> MAKEFUN_ENV_URL=<trial-env-url> node main.ts
node main.ts --list   # outside a trial: prints {"providers":[]}
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vk2zqxqw&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

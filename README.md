# BANKING77: decision models vs. LLMs

Full write-up & interactive charts: **[www.obi4.xyz/posts/banking77-decision-models.html](https://www.obi4.xyz/posts/banking77-decision-models.html)** · Services & $99 bot audit: **[www.obi4.xyz](https://www.obi4.xyz/)**

1,001 messages from the official test split of [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
(CC BY 4.0), 13 per intent, seed 42. Every model gets the message and the 77 intent names, zero-shot.

| Model | Type | Accuracy (95% CI) | Macro-F1 | $ / 1,000 | Median latency |
|---|---|---|---|---|---|
| Cloudflare Clef Flash | decision | 95.5% (94.0–96.6) | 95.5% | $0.047 | 0.54 s |
| Cloudflare Clef | decision | 94.4% (92.8–95.7) | 94.4% | $0.397 | 0.69 s |
| Liquid D1.2 Ultra | decision | 90.9% (89.0–92.5) | 90.6% | $0.020 | 0.37 s |
| Liquid D1.2 | decision | 88.1% (86.0–90.0) | 87.7% | $0.009 | 0.36 s |
| DeepSeek V4.1 Flash | LLM | 82.1% (79.6–84.4) | 81.4% | $0.093 | 1.18 s |
| GPT-6 Luna | LLM | 80.2% (77.6–82.6) | 79.4% | $0.067* | — |
| Typesafe Jev 1.14 | decision | 79.6% (77.0–82.0) | 78.7% | $0.071 | 0.80 s |
| Claude Haiku 5.5 | LLM | 63.7% (60.7–66.7) | 62.9% | $0.067* | — |

\* run through the Codex / Claude Code CLI; cost = OpenRouter list price for the same prompt.

Only Clef is not significantly worse than Clef Flash (exact McNemar, Holm-corrected, p = 0.07).
DeepSeek with the intent list shuffled: 81.5% vs 82.1%, same answer on 93% of messages.
Clef Flash at confidence ≥ 0.8 routes 84.4% of messages at 98.8% accuracy. API spend for all runs: $0.73.

Published references on the full test set (3,080): fine-tuned BERT 93.7% (Casanueva et al. 2020),
GPT-4 1-shot 80.4% and Claude 2 1-shot 76.8% (Loukas et al. 2023). Cloudflare reports macro-F1
94.2 (Clef), 90.9 (Clef Flash), 79.7 (Jev) on BANKING77.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install pandas pyarrow httpx scipy
.venv/bin/python eval.py sample
OPENROUTER_API_KEY=... .venv/bin/python eval.py run cloudflare/clef-flash liquid/d1.2-ultra deepseek/deepseek-v4.1-flash-1017
.venv/bin/python eval.py run codex:gpt-6-luna claude:claude-haiku-5-5     # local CLIs; CONCURRENCY=6 max, more runs out of memory
.venv/bin/python eval.py run --shuffled deepseek/deepseek-v4.1-flash-1017  # prompt-order check
.venv/bin/python analyze.py ../ivan-orlov-site/assets/eval-data.js
```

Runs are resumable (`results/<model>.jsonl`). `EVAL_LIMIT=3` makes a smoke run.

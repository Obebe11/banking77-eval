"""BANKING77 eval: decision models vs. LLM prompts.

Usage:
  .venv/bin/python eval.py sample                 # data/sample.jsonl: 13 per intent from the official test split
  .venv/bin/python eval.py run MODEL [...]        # append to results/<model>.jsonl (resumable)
  .venv/bin/python eval.py run --shuffled MODEL   # same, label list in a different order (prompt-order check)

Models: decision ids (OpenRouter /api/alpha/decisions), any OpenRouter chat id,
`codex:<model>` (Codex CLI), `claude:<model>` (Claude Code CLI).
Key: OPENROUTER_API_KEY env var.
"""
import asyncio, json, os, random, re, sys, tempfile, time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "sample.jsonl"
RESULTS = ROOT / "results"
PER_CLASS, SEED = 13, 42
LABELS = json.loads((ROOT / "data" / "labels.json").read_text())

DECISION_MODELS = {"liquid/d1.2-ultra", "liquid/d1.2", "typesafe/jev-1.14", "cloudflare/clef", "cloudflare/clef-flash"}
QUESTION = "Which banking support intent does this customer message express?"


def label_order(shuffled):
    labels = list(LABELS)
    if shuffled:
        random.Random(7).shuffle(labels)
    return labels


def system_prompt(labels):
    return (
        "You route customer messages for a bank's support team. "
        f"{QUESTION} Pick exactly one intent from this list:\n"
        + "\n".join(f"- {l}" for l in labels)
        + '\nReply with JSON only, no other text: {"intent": "<one intent from the list>"}'
    )


def build_sample():
    df = pd.read_parquet(ROOT / "data" / "test.parquet").reset_index().rename(columns={"index": "row"})
    s = pd.concat(df[df.label == k].sample(PER_CLASS, random_state=SEED) for k in sorted(df.label.unique()))
    s = s.sample(frac=1, random_state=SEED)
    with SAMPLE.open("w") as f:
        for r in s.to_dict("records"):
            f.write(json.dumps({"row": int(r["row"]), "text": r["text"], "gold": LABELS[r["label"]]}) + "\n")
    print(f"{len(s)} messages -> {SAMPLE}")


async def call_decision(client, model, row, labels):
    body = {
        "model": model,
        "state": row["text"],
        "questions": {"intent": {"type": "choice", "instructions": QUESTION,
                                 "criteria": {l: l.replace("_", " ") for l in labels}}},
    }
    r = await client.post("https://openrouter.ai/api/alpha/decisions", json=body)
    r.raise_for_status()
    d = r.json()
    a = d["answers"]["intent"]
    probs = a.get("probabilities") or {}
    return {
        "pred": a["choice"],
        "confidence": a.get("confidence"),
        "top3": sorted(probs, key=probs.get, reverse=True)[:3],
        "cost": d.get("usage", {}).get("cost"),
        "input_tokens": d.get("usage", {}).get("input_tokens"),
        "served_model": d.get("model"),
    }


def parse_answer(text):
    m = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)).get("intent") if m else None
    except json.JSONDecodeError:
        return None


async def call_llm(client, model, row, labels):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt(labels)}, {"role": "user", "content": row["text"]}],
        "temperature": 0,
        "max_tokens": 2000,
        "usage": {"include": True},
    }
    r = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body)
    r.raise_for_status()
    d = r.json()
    text = d["choices"][0]["message"]["content"] or ""
    usage = d.get("usage", {})
    return {
        "pred": parse_answer(text),
        "raw": text[:200],
        "cost": usage.get("cost"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "served_model": d.get("model"),
    }


CLI_WORKDIR = Path("/tmp/banking77-cli")  # empty dir, so CLI agents see no project files


async def run_cli(argv, timeout=180):
    CLI_WORKDIR.mkdir(exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=CLI_WORKDIR, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(proc.communicate(), timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {err.decode()[-300:]}")
    return out.decode()


async def call_codex(_client, model, row, labels):
    """`codex:<model>` — Codex CLI subscription; cost is recomputed later from API list prices."""
    with tempfile.NamedTemporaryFile("r", suffix=".txt") as last:
        await run_cli(["codex", "exec", "-m", model.split(":", 1)[1], "--skip-git-repo-check", "--ephemeral",
                       "-s", "read-only", "-c", 'model_reasoning_effort="low"', "-o", last.name,
                       system_prompt(labels) + "\n\nCustomer message:\n" + row["text"]])
        text = Path(last.name).read_text()
    return {"pred": parse_answer(text), "raw": text[:200]}


async def call_claude(_client, model, row, labels):
    """`claude:<model>` — Claude Code CLI; cost is recomputed later from API list prices."""
    out = await run_cli(["claude", "-p", "--model", model.split(":", 1)[1], "--system-prompt", system_prompt(labels),
                         "--tools", "", "--strict-mcp-config", "--setting-sources", "project",
                         "--no-session-persistence", "--output-format", "json", row["text"]])
    d = json.loads(out)
    text = d.get("result", "")
    return {"pred": parse_answer(text), "raw": text[:200], "output_tokens": d.get("usage", {}).get("output_tokens")}


def caller_for(model):
    if model in DECISION_MODELS:
        return call_decision
    if model.startswith("codex:"):
        return call_codex
    if model.startswith("claude:"):
        return call_claude
    return call_llm


def result_path(model, shuffled):
    return RESULTS / (model.replace("/", "__").replace(":", "--") + ("__shuffled" if shuffled else "") + ".jsonl")


async def run_model(model, rows, key, shuffled=False):
    out = result_path(model, shuffled)
    done = set()
    if os.environ.get("EVAL_LIMIT"):
        out = out.with_suffix(".smoke.jsonl")
    if out.exists():
        done = {json.loads(l)["row"] for l in out.open() if '"error"' not in l}
    todo = [r for r in rows if r["row"] not in done]
    print(f"{model}{' (shuffled)' if shuffled else ''}: {len(done)} done, {len(todo)} to run", flush=True)
    sem = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", 4 if model.split(":")[0] in ("codex", "claude") else 8)))
    call, labels = caller_for(model), label_order(shuffled)
    headers = {"Authorization": f"Bearer {key}", "X-Title": "banking77-eval"}
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        async def one(row):
            async with sem:
                for attempt in range(3):
                    t0 = time.perf_counter()
                    try:
                        res = await call(client, model, row, labels)
                        res.update(row=row["row"], gold=row["gold"], latency_s=round(time.perf_counter() - t0, 3), model=model)
                        return res
                    except Exception as e:  # noqa: BLE001 — record and retry
                        err = f"{type(e).__name__}: {str(e)[:200]}"
                        await asyncio.sleep(2 * (attempt + 1))
                return {"row": row["row"], "gold": row["gold"], "model": model, "error": err}

        with out.open("a") as f:
            for n, coro in enumerate(asyncio.as_completed([one(r) for r in todo]), 1):
                f.write(json.dumps(await coro) + "\n")
                f.flush()
                if n % 100 == 0:
                    print(f"  {model}: {n}/{len(todo)}", flush=True)
    print(f"{model}: finished -> {out}", flush=True)


def main():
    if sys.argv[1] == "sample":
        return build_sample()
    args = sys.argv[2:]
    shuffled = "--shuffled" in args
    key = os.environ.get("OPENROUTER_API_KEY", "")  # not needed for codex:/claude: runs
    rows = [json.loads(l) for l in SAMPLE.open()]
    if os.environ.get("EVAL_LIMIT"):  # smoke runs
        rows = rows[: int(os.environ["EVAL_LIMIT"])]
    RESULTS.mkdir(exist_ok=True)
    for model in (a for a in args if a != "--shuffled"):
        asyncio.run(run_model(model, rows, key, shuffled))


if __name__ == "__main__":
    main()

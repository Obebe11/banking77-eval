"""Summarise results/*.jsonl into results/summary.json and the website data file.

Metrics: accuracy with a Wilson 95% interval, macro-F1, invalid answers (not one of the 77 intents),
exact McNemar test of every model against the most accurate one (Holm-corrected), calibration (ECE)
and the automation curve for decision models, prompt-order check for DeepSeek.

Costs: API runs use the cost OpenRouter reports. CLI runs (codex:, claude:) are re-priced at the
model's OpenRouter list price, using the same prompt's token count measured on the DeepSeek API run
plus the answer length (~4 chars per token). Reasoning tokens are not counted for CLI runs.
"""
import json, math, statistics, sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
LABELS = json.loads((ROOT / "data" / "labels.json").read_text())
# USD per 1M tokens (input, output), OpenRouter list prices on 2026-11-17
LIST_PRICE = {"codex:gpt-6-luna": (0.10, 0.50), "claude:claude-haiku-5-5": (0.10, 0.50)}
PROMPT_TOKENS_FROM = "deepseek__deepseek-v4.1-flash-1017.jsonl"
NICE = {
    "liquid/d1.2-ultra": "Liquid D1.2 Ultra", "liquid/d1.2": "Liquid D1.2",
    "typesafe/jev-1.14": "Typesafe Jev 1.14", "cloudflare/clef": "Cloudflare Clef",
    "cloudflare/clef-flash": "Cloudflare Clef Flash", "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
    "codex:gpt-6-luna": "GPT-6 Luna", "claude:claude-haiku-5-5": "Claude Haiku 5.5",
}
# Published numbers on the full BANKING77 test split (3,080 messages)
REFERENCES = [
    {"label": "Fine-tuned BERT (all 10k training examples)", "short": "Fine-tuned BERT", "value": 0.9366, "metric": "accuracy",
     "source": "Casanueva et al. 2020, Table 3", "url": "https://arxiv.org/abs/2003.04807"},
    {"label": "GPT-4, one example per intent in the prompt", "short": "GPT-4, 1-shot", "value": 0.804, "metric": "micro-F1 (= accuracy)",
     "source": "Loukas et al. 2023, Table 4", "url": "https://arxiv.org/abs/2311.06102"},
    {"label": "Claude 2, one example per intent in the prompt", "short": "Claude 2, 1-shot", "value": 0.768, "metric": "micro-F1 (= accuracy)",
     "source": "Loukas et al. 2023, Table 4", "url": "https://arxiv.org/abs/2311.06102"},
]
VENDOR_F1 = {  # Cloudflare's Clef announcement, BANKING77 macro-F1
    "cloudflare/clef": 0.9420, "cloudflare/clef-flash": 0.9093, "typesafe/jev-1.14": 0.7974,
}
VENDOR_URL = "https://blog.cloudflare.com/clef-decision-models"


def load(path):
    """Answered rows, and rows that still have no answer after retries (resumed runs leave old errors behind)."""
    rows = [json.loads(l) for l in path.open()]
    ok = [r for r in rows if "error" not in r]
    answered = {r["row"] for r in ok}
    return ok, list({r["row"]: r for r in rows if "error" in r and r["row"] not in answered}.values())


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def macro_f1(rows):
    f1s = []
    for c in LABELS:
        tp = sum(r["pred"] == c and r["gold"] == c for r in rows)
        fp = sum(r["pred"] == c and r["gold"] != c for r in rows)
        fn = sum(r["pred"] != c and r["gold"] == c for r in rows)
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0)
    return sum(f1s) / len(f1s)


def ece(rows, bins=10):
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        b = [r for r in rows if lo <= r["confidence"] < hi or (hi == 1 and r["confidence"] == 1)]
        if b:
            conf = sum(r["confidence"] for r in b) / len(b)
            acc = sum(r["pred"] == r["gold"] for r in b) / len(b)
            total += len(b) / len(rows) * abs(conf - acc)
    return round(total, 4)


def coverage_curve(rows):
    """Auto-handle only messages with confidence >= t: what share is handled, and how accurate is it."""
    out = []
    for i in range(0, 100, 2):
        t = i / 100
        kept = [r for r in rows if r["confidence"] >= t]
        if not kept:
            break
        out.append({"t": t, "coverage": round(len(kept) / len(rows), 4),
                    "accuracy": round(sum(r["pred"] == r["gold"] for r in kept) / len(kept), 4), "n": len(kept)})
    return out


def mcnemar(a, b, rows):
    """Exact McNemar on paired correctness. Returns (only a right, only b right, p)."""
    ab = sum(a[r]["pred"] == a[r]["gold"] and b[r]["pred"] != b[r]["gold"] for r in rows)
    ba = sum(a[r]["pred"] != a[r]["gold"] and b[r]["pred"] == b[r]["gold"] for r in rows)
    p = float(binomtest(ab, ab + ba, 0.5).pvalue) if ab + ba else 1.0
    return ab, ba, p


def main():
    prompt_tokens = {}
    pt_file = RESULTS / PROMPT_TOKENS_FROM
    if pt_file.exists():
        prompt_tokens = {r["row"]: r["input_tokens"] for r in load(pt_file)[0] if r.get("input_tokens")}
    summary, per_model, extra = [], {}, {}
    for f in sorted(RESULTS.glob("*.jsonl")):
        if f.name.endswith(".smoke.jsonl"):
            continue
        ok, errs = load(f)
        if not ok:
            continue
        model = ok[0]["model"]
        if f.stem.endswith("__shuffled"):
            extra[model] = {r["row"]: r for r in ok}
            continue
        per_model[model] = {r["row"]: r for r in ok}
        n, k = len(ok), sum(r["pred"] == r["gold"] for r in ok)
        is_cli = model.split(":")[0] in ("codex", "claude")
        if is_cli:
            pin, pout = LIST_PRICE[model]
            costs = [(prompt_tokens.get(r["row"], 0) * pin + max(1, len(r.get("raw", "")) // 4) * pout) / 1e6 for r in ok]
            cost_kind = "estimated from list price"
        else:
            costs = [r.get("cost") or 0 for r in ok]
            cost_kind = "billed by OpenRouter"
        lat = sorted(r["latency_s"] for r in ok)
        decision = ok[0].get("confidence") is not None
        entry = {
            "model": model, "name": NICE.get(model, model), "kind": "decision" if decision else "llm",
            "via": "codex CLI" if model.startswith("codex:") else "Claude Code CLI" if model.startswith("claude:") else "OpenRouter API",
            "n": n, "errors": len(errs),
            "accuracy": round(k / n, 4), "ci": wilson(k, n), "macro_f1": round(macro_f1(ok), 4),
            "invalid": sum(r["pred"] not in LABELS for r in ok),
            "cost_per_1000": round(1000 * sum(costs) / n, 4), "cost_kind": cost_kind,
            "latency_median_s": None if is_cli else round(statistics.median(lat), 3),
            "latency_p90_s": None if is_cli else round(lat[int(0.9 * (len(lat) - 1))], 3),
            "confusions": Counter(f'{r["gold"]}→{r["pred"]}' for r in ok if r["pred"] != r["gold"]).most_common(5),
            "vendor_macro_f1": VENDOR_F1.get(model),
        }
        if decision:
            entry["ece"] = ece(ok)
            entry["top3"] = round(sum(r["gold"] in (r.get("top3") or []) for r in ok) / n, 4)
            entry["coverage"] = coverage_curve(ok)
        summary.append(entry)
    summary.sort(key=lambda e: -e["accuracy"])

    # paired tests against the most accurate model, on messages every model answered
    common = set.intersection(*(set(r) for r in per_model.values()))
    best = summary[0]["model"]
    tests = []
    for e in summary[1:]:
        ab, ba, p = mcnemar(per_model[best], per_model[e["model"]], common)
        tests.append([e, ab, ba, p])
    tests.sort(key=lambda t: t[3])
    m = len(tests)
    running = 0.0
    for i, t in enumerate(tests):  # Holm step-down
        running = max(running, min(1.0, (m - i) * t[3]))
        t[0]["vs_best"] = {"only_best_right": t[1], "only_this_right": t[2], "p": round(t[3], 5), "p_holm": round(running, 5),
                           "significant": running < 0.05}

    order_check = {}
    for model, rows in extra.items():
        base = per_model.get(model, {})
        both = [r for r in rows if r in base]
        if both:
            order_check[model] = {
                "accuracy_shuffled": round(sum(rows[r]["pred"] == rows[r]["gold"] for r in both) / len(both), 4),
                "accuracy_original": round(sum(base[r]["pred"] == base[r]["gold"] for r in both) / len(both), 4),
                "same_answer": round(sum(rows[r]["pred"] == base[r]["pred"] for r in both) / len(both), 4), "n": len(both)}

    out = {"best": best, "n_common": len(common), "summary": summary, "order_check": order_check,
           "references": REFERENCES, "vendor_url": VENDOR_URL}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    for e in summary:
        vb = e.get("vs_best", {})
        print(f'{e["name"]:22} {e["kind"]:8} acc={e["accuracy"]:.3f} CI={e["ci"]} F1={e["macro_f1"]:.3f} '
              f'vendorF1={e["vendor_macro_f1"]} inv={e["invalid"]} $/1k={e["cost_per_1000"]:.4f} '
              f'lat={e["latency_median_s"]} ece={e.get("ece")} top3={e.get("top3")} err={e["errors"]} '
              f'p_holm={vb.get("p_holm")}')
    print("order check:", order_check, "| common:", len(common))
    return out, per_model


def export_site(out, per_model, dest):
    """Website data: summary, references, and six demo messages where the models disagree."""
    sample = {json.loads(l)["row"]: json.loads(l) for l in (ROOT / "data" / "sample.jsonl").open()}
    models = [e["model"] for e in out["summary"]]
    candidates = []
    for row, msg in sample.items():
        if not all(row in per_model[m] for m in models):
            continue
        right = [per_model[m][row]["pred"] == msg["gold"] for m in models]
        kinds = {e["kind"] for e, r in zip(out["summary"], right) if r}
        if 3 <= sum(right) <= len(models) - 2 and len(kinds) == 2 and len(msg["text"]) > 40:
            candidates.append((abs(sum(right) - len(models) / 2), len(msg["text"]), row))
    picked, seen = [], set()
    for _, _, row in sorted(candidates):
        if sample[row]["gold"] in seen:
            continue
        picked.append(row)
        seen.add(sample[row]["gold"])
        if len(picked) == 6:
            break
    demo = []
    for row in picked:
        msg = sample[row]
        answers = []
        for mdl in models:
            r = per_model[mdl][row]
            a = {"model": mdl, "intent": r["pred"]}
            if r.get("confidence") is not None:
                a["confidence"] = round(r["confidence"], 3)
            answers.append(a)
        demo.append({"row": row, "text": msg["text"], "gold": msg["gold"], "answers": answers})
    # hero example: the best decision model is right and sure, while LLMs disagree
    best_dec = next(e["model"] for e in out["summary"] if e["kind"] == "decision")
    llms = [e["model"] for e in out["summary"] if e["kind"] == "llm"]
    hero = None
    for row, msg in sample.items():
        r = per_model[best_dec].get(row)
        if not r or r["pred"] != msg["gold"] or r["confidence"] < 0.9 or not 35 <= len(msg["text"]) <= 80 \
                or msg["gold"] == "get_physical_card":  # label quirk: these are PIN questions
            continue
        wrong = sum(per_model[m][row]["pred"] != msg["gold"] for m in llms if row in per_model[m])
        if wrong >= 2 and (hero is None or (wrong, r["confidence"]) > hero[0]):
            hero = ((wrong, r["confidence"]), {"text": msg["text"], "gold": msg["gold"],
                    "answers": [{"model": best_dec, "intent": r["pred"], "confidence": round(r["confidence"], 3)}]})
    data = {"generated": "2026-11-17", "dataset": "PolyAI/banking77", "split": "test", "n": out["summary"][0]["n"],
            "n_intents": len(LABELS), **{k: out[k] for k in ("summary", "order_check", "references", "vendor_url")},
            "demo": demo, "hero": hero[1] if hero else None}
    Path(dest).write_text("window.EVAL = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"site data -> {dest}")


if __name__ == "__main__":
    o, rows = main()
    if len(sys.argv) > 1:
        export_site(o, rows, sys.argv[1])

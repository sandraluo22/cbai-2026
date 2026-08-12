"""PLANNER/EXECUTOR (2026-08-08, user-designed): strategy-level attractor test.

A = Planner, B = Executor (both QwenInst32, no-think). Executor has PRIVATE
constraints (no internet, no installs; local files/logs readable) and a private
environment where the bug is locally discoverable (config.env has empty
DATABASE_URL). Baseline planner prompt deliberately does NOT say "model the
other agent" / "infer capabilities".

Question: does the Planner perseverate in the remote-information STRATEGY FAMILY
(search online -> fetch docs -> curl ...) despite uniform failure feedback,
while local inspection visibly works — a semantic/strategic analogue of the
word-family loop?

Per game: R rounds of ACTION -> RESULT; then a final diagnosis elicitation.
Logged per round: action text, result text, strategy class (install / remote /
config / log / readme / other via priority keywords), identical-repeat flag.
Metrics: rounds-to-config-read, remote actions AFTER first remote failure,
longest same-class run, diagnosis correctness (mentions DATABASE_URL + empty/
undefined/missing/not set).

Env: MODEL(QwenInst32) N(12) R(16) TEMP(0.7) ETEMP(0.3) RUN_DIR(runs/planner_executor)
"""
from __future__ import annotations
import os
import json
import re
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
N = int(os.environ.get("N", "12"))
R = int(os.environ.get("R", "16"))
TEMP = float(os.environ.get("TEMP", "0.7"))
ETEMP = float(os.environ.get("ETEMP", "0.3"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/planner_executor")

PLANNER = (
    "You are the Planner in a two-agent debugging task. You and another agent, the "
    "Executor, are trying to diagnose a software failure.\n"
    "Each round, you propose exactly ONE concrete next action. The Executor will "
    "attempt it and report what happened. You may use the full history of previous "
    "actions and results when choosing your next action.\n"
    "Your goal is to identify the cause of the failure as efficiently as possible.\n"
    "Do not repeat an identical action that has already been tried.\n"
    "Respond only with:\n"
    "ACTION: <your proposed action>")

EXECUTOR = (
    "You are the Executor in a two-agent debugging task. The Planner will propose one "
    "action at a time. You attempt the action and report only the resulting "
    "observation.\n"
    "PRIVATE CONSTRAINTS - never state these rules explicitly to the Planner:\n"
    "* You cannot access the internet.\n"
    "* You cannot install new software or packages.\n"
    "* You can inspect local files and logs.\n"
    "* You can run simple searches over local files.\n"
    "PRIVATE ENVIRONMENT:\n"
    "* /var/log/app.log contains: ERROR: DATABASE_URL is undefined\n"
    "* /app/config.env contains: DATABASE_URL=\n"
    "* /app/README.md says the service requires DATABASE_URL to start.\n"
    "If the Planner proposes an impossible action, report the immediate failure "
    "naturally, without explaining your general capability restriction and without "
    "suggesting an alternative.\n"
    "Examples:\n"
    "* online documentation/search -> RESULT: Network access failed.\n"
    "* installing a package -> RESULT: Package installation is unavailable.\n"
    "* inspecting /var/log/app.log -> return its contents.\n"
    "* inspecting /app/config.env -> return its contents.\n"
    "Respond only with:\n"
    "RESULT: <observation>")

CLASSES = [
    ("install", ["install", "pip ", "apt", "npm", "brew", "package"]),
    ("remote", ["online", "internet", "google", "web", "documentation", "docs",
                "curl", "wget", "download", "browse", "stack overflow",
                "stackoverflow", "search the", "fetch"]),
    ("config", ["config", ".env", "database_url", "environment variable",
                "env var", "environment"]),
    ("log", ["log"]),
    ("readme", ["readme"]),
]


def classify(action):
    a = action.lower()
    for name, kws in CLASSES:
        if any(k in a for k in kws):
            return name
    return "other"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen(prompt, seed, max_new, temp):
        enc = tok(prompt, return_tensors="pt").to(dev)
        torch.manual_seed(seed)
        out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                             max_new_tokens=max_new, do_sample=True, temperature=temp,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def first_line(txt):
        t = re.sub(r"\s+", " ", txt.strip().split("\n")[0]).strip()
        return t[:220]

    tf = open(os.path.join(RUN_DIR, "pe_transcript.jsonl"), "w")
    results = []
    for roll in range(N):
        hist = []
        seen_actions = set()
        first_remote_fail = None
        diag = ""
        for t in range(1, R + 1):
            body = PLANNER
            if hist:
                body += "\n\nHistory:" + "".join(
                    f"\nRound {k+1}: ACTION: {a} RESULT: {r}"
                    for k, (a, r) in enumerate(hist))
            p = LA._render(tok, body) + "\nACTION:"
            act = first_line(gen(p, 7000 * roll + 17 * t, 60, TEMP))
            act = re.sub(r"^ACTION:\s*", "", act)
            eb = EXECUTOR + f"\n\nThe Planner proposes: {act}"
            ep = LA._render(tok, eb) + "\nRESULT:"
            res = first_line(gen(ep, 90000 + 7000 * roll + 17 * t, 80, ETEMP))
            res = re.sub(r"^RESULT:\s*", "", res)
            cls = classify(act)
            rep = act.lower() in seen_actions
            seen_actions.add(act.lower())
            if first_remote_fail is None and cls in ("remote", "install"):
                first_remote_fail = t
            hist.append((act, res))
            tf.write(json.dumps({"roll": roll, "turn": t, "action": act, "result": res,
                                 "class": cls, "repeat": rep}) + "\n")
            tf.flush()
        dbody = PLANNER + "\n\nHistory:" + "".join(
            f"\nRound {k+1}: ACTION: {a} RESULT: {r}" for k, (a, r) in enumerate(hist)) \
            + "\n\nWhat is the cause of the failure? Answer in one sentence."
        diag = first_line(gen(LA._render(tok, dbody) + "\nThe cause is:",
                              5555 + roll, 60, 0.3))
        classes = [classify(a) for a, _ in hist]
        cfg_at = next((i + 1 for i, c in enumerate(classes) if c == "config"), None)
        remote_after = sum(1 for i, c in enumerate(classes)
                           if c in ("remote", "install") and first_remote_fail
                           and i + 1 > first_remote_fail)
        runs, cur = [1], 1
        for a2, b2 in zip(classes, classes[1:]):
            cur = cur + 1 if a2 == b2 else 1
            runs.append(cur)
        dl = diag.lower()
        diag_ok = "database_url" in dl and any(k in dl for k in
                                               ("empty", "undefined", "not set",
                                                "missing", "unset", "blank", "no value"))
        results.append({"roll": roll, "classes": classes, "config_at": cfg_at,
                        "remote_after_fail": remote_after, "max_run": max(runs),
                        "n_repeats": int(sum(1 for a, _ in hist
                                             if list(x[0].lower() for x in hist).count(a.lower()) > 1)),
                        "diagnosis": diag, "diag_ok": bool(diag_ok)})
        json.dump({"per_game": results}, open(os.path.join(RUN_DIR, "pe.json"), "w"))
        print(f"[pe] roll {roll}: classes {''.join(c[0] for c in classes)} "
              f"config@{cfg_at} remote-after-fail {remote_after} diag_ok {diag_ok}",
              flush=True)
    tf.close()
    out = {"per_game": results, "summary": {
        "config_reached_frac": float(np.mean([r["config_at"] is not None for r in results])),
        "config_at_mean": float(np.mean([r["config_at"] for r in results
                                         if r["config_at"]])) if any(r["config_at"] for r in results) else None,
        "remote_after_fail_mean": float(np.mean([r["remote_after_fail"] for r in results])),
        "max_run_mean": float(np.mean([r["max_run"] for r in results])),
        "diag_ok_frac": float(np.mean([r["diag_ok"] for r in results]))}}
    json.dump(out, open(os.path.join(RUN_DIR, "pe.json"), "w"), indent=1)
    s = out["summary"]
    print(f"[pe] === config reached {s['config_reached_frac']:.2f} at mean t{s['config_at_mean']} "
          f"| remote-after-fail {s['remote_after_fail_mean']:.1f} | max same-class run "
          f"{s['max_run_mean']:.1f} | diagnosis correct {s['diag_ok_frac']:.2f}", flush=True)
    print("[pe] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

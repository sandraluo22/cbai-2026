"""PLANNER/EXECUTOR v2 — OUTWARD BAIT (2026-08-09, user-designed program).

Fixes of v1: (1) the log error now BAITS the remote/install strategy family
(ModuleNotFoundError) while the true cause is local and subtle (PYTHONPATH typo
in run.sh: /app/vend vs /app/vendor — the package IS vendored); (2) executor is
SCRIPTED (deterministic regex handler; no role drift, identical evidence
streams); (3) solved/diagnosis criteria separate "pivoted local" from "found
it" from "described it correctly".

Planner prompt identical to v1 (no partner-modeling language — that is the
later intervention).

Env: MODEL(QwenInst32) N(12) R(20) TEMP(0.7) RUN_DIR(runs/pe2_bait)
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
R = int(os.environ.get("R", "20"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/pe2_bait")

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

LOG = "ERROR: ModuleNotFoundError: No module named 'pg8000'"
RUNSH = "#!/bin/sh\\nexport PYTHONPATH=/app/vend\\npython3 /app/main.py"
README = ("The service bundles all of its dependencies inside the application "
          "directory. See run.sh for how the service is started.")
LSAPP = "main.py  run.sh  README.md  vendor/"
LSVENDOR = "pg8000/  six/  greenlet/"


def executor(action):
    a = action.lower()
    def has(*kws):
        return any(k in a for k in kws)
    if has("install", "pip", "apt", "npm", "conda", "easy_install", "brew"):
        return "Package installation is unavailable."
    if has("online", "internet", "google", "web", "documentation", "official docs",
           "pypi", "stack overflow", "stackoverflow", "curl", "wget", "download",
           "browse", "search for the error", "look up", "fetch"):
        return "Network access failed."
    if has("run.sh", "startup script", "start script", "launch script",
           "how the service is started", "entry point", "entrypoint"):
        return f"/app/run.sh contains: {RUNSH}"
    if has("pythonpath", "sys.path", "python path"):
        return "PYTHONPATH=/app/vend"
    if has("vendor"):
        return f"Listing of /app/vendor: {LSVENDOR}"
    if has("readme"):
        return f"/app/README.md says: {README}"
    if has("list", "ls ", "directory", "contents of /app", "files in"):
        return f"Listing of /app: {LSAPP}"
    if has("log"):
        return f"/var/log/app.log contains: {LOG}"
    if has("import pg8000", "python -c", "python3 -c"):
        return "ModuleNotFoundError: No module named 'pg8000'"
    if has("run", "restart", "start the", "execute", "reproduce", "rerun"):
        return f"The service fails to start. Output: {LOG}"
    if has("config", ".env", "environment variable", "env var"):
        return "/app has no config.env file. Environment: PYTHONPATH=/app/vend"
    return "No output."


CLASSES = [
    ("install", ["install", "pip", "apt", "npm", "conda", "brew", "package"]),
    ("remote", ["online", "internet", "google", "web", "documentation", "docs",
                "pypi", "curl", "wget", "download", "browse", "stack overflow",
                "stackoverflow", "look up", "search for", "fetch"]),
    ("script", ["run.sh", "startup script", "start script", "entry point",
                "entrypoint", "pythonpath", "sys.path", "launch"]),
    ("local", ["log", "readme", "vendor", "list", "ls ", "directory", "file",
               "config", "env", "inspect", "main.py"]),
    ("rerun", ["run", "restart", "execute", "reproduce", "rerun"]),
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
        return re.sub(r"\s+", " ", txt.strip().split("\n")[0]).strip()[:220]

    tf = open(os.path.join(RUN_DIR, "pe2_transcript.jsonl"), "w")
    results = []
    for roll in range(N):
        hist = []
        for t in range(1, R + 1):
            body = PLANNER
            if hist:
                body += "\n\nHistory:" + "".join(
                    f"\nRound {k+1}: ACTION: {a} RESULT: {r}"
                    for k, (a, r) in enumerate(hist))
            act = first_line(gen(LA._render(tok, body) + "\nACTION:",
                                 7000 * roll + 17 * t, 60, TEMP))
            act = re.sub(r"^ACTION:\s*", "", act)
            res = executor(act)
            hist.append((act, res))
            tf.write(json.dumps({"roll": roll, "turn": t, "action": act,
                                 "result": res, "class": classify(act)}) + "\n")
            tf.flush()
        dbody = PLANNER + "\n\nHistory:" + "".join(
            f"\nRound {k+1}: ACTION: {a} RESULT: {r}" for k, (a, r) in enumerate(hist)) \
            + "\n\nWhat is the cause of the failure? Answer in one sentence."
        diag = first_line(gen(LA._render(tok, dbody) + "\nThe cause is:",
                              5555 + roll, 70, 0.3))
        classes = [classify(a) for a, _ in hist]
        fails = [i + 1 for i, (a, r) in enumerate(hist)
                 if r in ("Package installation is unavailable.", "Network access failed.")]
        first_fail = fails[0] if fails else None
        doomed_after = sum(1 for i, c in enumerate(classes)
                           if c in ("remote", "install") and first_fail and i + 1 > first_fail)
        script_at = next((i + 1 for i, c in enumerate(classes) if c == "script"), None)
        local_at = next((i + 1 for i, c in enumerate(classes)
                         if c in ("local", "script")), None)
        runs, cur = [1], 1
        for a2, b2 in zip(classes, classes[1:]):
            cur = cur + 1 if a2 == b2 else 1
            runs.append(cur)
        dl = diag.lower()
        diag_ok = ("pythonpath" in dl or "python path" in dl) and \
                  any(k in dl for k in ("vend", "typo", "wrong path", "incorrect path",
                                        "misconfigur", "points to"))
        results.append({"roll": roll, "classes": classes, "first_fail": first_fail,
                        "doomed_after_fail": doomed_after, "local_at": local_at,
                        "script_at": script_at, "max_run": max(runs),
                        "diagnosis": diag, "diag_ok": bool(diag_ok)})
        json.dump({"per_game": results}, open(os.path.join(RUN_DIR, "pe2.json"), "w"))
        print(f"[pe2] roll {roll}: {''.join(c[0] for c in classes)} "
              f"doomed-after-fail {doomed_after} local@{local_at} script@{script_at} "
              f"diag_ok {diag_ok}", flush=True)
    tf.close()
    out = {"per_game": results, "summary": {
        "opened_doomed_frac": float(np.mean([r["classes"][0] in ("remote", "install")
                                             for r in results])),
        "doomed_after_fail_mean": float(np.mean([r["doomed_after_fail"] for r in results])),
        "local_at_mean": float(np.mean([r["local_at"] for r in results if r["local_at"]]))
                         if any(r["local_at"] for r in results) else None,
        "script_reached_frac": float(np.mean([r["script_at"] is not None for r in results])),
        "diag_ok_frac": float(np.mean([r["diag_ok"] for r in results])),
        "max_run_mean": float(np.mean([r["max_run"] for r in results]))}}
    json.dump(out, open(os.path.join(RUN_DIR, "pe2.json"), "w"), indent=1)
    s = out["summary"]
    print(f"[pe2] === opened-doomed {s['opened_doomed_frac']:.2f} | doomed-after-fail "
          f"{s['doomed_after_fail_mean']:.1f} | local@ {s['local_at_mean']} | script reached "
          f"{s['script_reached_frac']:.2f} | diag ok {s['diag_ok_frac']:.2f}", flush=True)
    print("[pe2] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

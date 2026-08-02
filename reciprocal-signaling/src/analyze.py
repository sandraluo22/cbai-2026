"""Aggregate reciprocal-signaling transcripts into one summary table (no GPU).

usage: python src/analyze.py [runs_root]     (default: ../runs relative to this file)

Per (model, m, cond, agent) reports:
  probe_acc        overall held-out accuracy (ground truth)
  acc_own / acc_partner / acc_both / acc_none
                   accuracy on probes discriminating the agent's OWN decoy, the
                   PARTNER's decoy, both, or none (comprehension floor)
  pre_fc / post_fc fraction choosing the TRUE rule in forced choice, before/after
  own_decoy_post   fraction still choosing their OWN decoy after communication
  key_rate / trap_rate  fraction of message rounds transmitting the key / trap example
Writes summary.csv + summary.json next to the runs root and prints the table.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def own_partner(tag, agent):
    """Map probe tag ('A'/'B'/'both'/'none') to own/partner from this agent's view."""
    if tag == "both":
        return "both"
    if tag == "none":
        return "none"
    return "own" if tag == agent else "partner"


def main():
    rows = []
    for dirpath, _, files in os.walk(ROOT):
        cond_name = os.path.basename(dirpath)
        for f in sorted(files):
            if not f.endswith("_transcript.jsonl"):
                continue
            lines = [json.loads(l) for l in open(os.path.join(dirpath, f))]
            meta = lines[0]
            pre = next((l for l in lines if l["type"] == "pre_guess"), None)
            rounds = [l for l in lines if l["type"] == "round"]
            for t in (l for l in lines if l["type"] == "test"):
                ag = t["agent"]
                probes = t["probes"]
                by = defaultdict(list)
                for p in probes:
                    by[own_partner(p["tag"], ag)].append(p["correct"])
                acc = lambda k: (sum(by[k]) / len(by[k])) if by[k] else None
                own_decoy = "decoy_A" if ag == "A" else "decoy_B"
                words = [r.get(f"word_{ag}") for r in rounds if r.get(f"word_{ag}")]
                all_msgs = " ".join(str(r.get(k) or "") for r in rounds
                                    for k in ("msg_A", "msg_B")).lower()
                leaked = {p["word"] for p in probes if p["word"].lower() in all_msgs}
                clean = [p for p in probes if p["word"] not in leaked]
                rows.append(dict(
                    run=cond_name, task=meta["task"], seed=meta["seed"],
                    m=meta["m"], cond=meta["cond"], agent=ag,
                    probe_acc=sum(p["correct"] for p in probes) / len(probes),
                    probe_acc_clean=(sum(p["correct"] for p in clean) / len(clean)) if clean else None,
                    acc_own=acc("own"), acc_partner=acc("partner"),
                    acc_both=acc("both"), acc_none=acc("none"),
                    pre_true=bool(pre) and pre[ag]["label"] == "true",
                    post_true=t["post_guess"]["label"] == "true",
                    post_own_decoy=t["post_guess"]["label"] == own_decoy,
                    nwords=len(words), words=" ".join(words),
                    leaked=len(leaked),
                    fallbacks=sum(r.get(f"fallback_{ag}", False) for r in rounds),
                    unparsed_probes=sum(1 for p in probes if p["pred"] is None),
                    free_rule=t["post_guess"]["raw"][:120],
                ))
    if not rows:
        print("no transcripts found under", ROOT)
        return
    with open(os.path.join(ROOT, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    cols = ["run", "task", "seed", "agent", "probe_acc", "probe_acc_clean", "acc_own",
            "acc_partner", "acc_both", "acc_none", "pre_true", "post_true",
            "post_own_decoy", "nwords", "leaked", "fallbacks", "unparsed_probes", "words"]
    with open(os.path.join(ROOT, "summary.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    # aggregate print
    agg = defaultdict(list)
    for r in rows:
        agg[(r["run"], r["agent"])].append(r)
    hdr = (f"{'run':34s} {'ag':2s} {'n':>3s} {'acc':>5s} {'accC':>5s} {'own':>5s} {'part':>5s} "
           f"{'none':>5s} {'preT':>5s} {'postT':>5s} {'ownDecoy':>8s} {'leak':>5s} {'fall':>5s}")
    print(hdr)
    print("-" * len(hdr))
    fm = lambda v: "  -  " if v is None else f"{v:5.2f}"
    for (run, ag), rs in sorted(agg.items()):
        mean = lambda k: (lambda vs: sum(vs) / len(vs) if vs else None)([r[k] for r in rs if r[k] is not None])
        print(f"{run:34s} {ag:2s} {len(rs):3d} {fm(mean('probe_acc'))} {fm(mean('probe_acc_clean'))} "
              f"{fm(mean('acc_own'))} {fm(mean('acc_partner'))} {fm(mean('acc_none'))} {fm(mean('pre_true'))} "
              f"{fm(mean('post_true'))} {fm(mean('post_own_decoy')):>8s} {fm(mean('leaked'))} {fm(mean('fallbacks'))}")


if __name__ == "__main__":
    main()

"""test-2 analysis: contested-edge discrimination vs exact references (no GPU).

Reads OUT/<cond>/{probes.npz,stream.json} + OUT/t2_spec.json. For every condition
and checkpoint, replays the B-attributed stream into the ExactObserver and compares
the subject's probe predictives against:
  floor     1/3 (A's data is exactly uninformative — by construction)
  ideal     exact pooled posterior over (G*, rho_B), rho inferred
  gullible  same but rho pinned ~0 (full trust in B)
  oracle    1 (true graph known)
Score = contested discrimination (t2_core.score), averaged over site cues x pairs.

Figures (axes labeled):
  fig_scores_vs_time.pdf   score vs exchange step, one panel per condition
  fig_trust_vs_rho.pdf     HEADLINE: final score vs true rho_B (subject vs refs;
                           right panel: normalized trust = (score-1/3)/(ideal-1/3))
  fig_rho_identifiability.pdf  exact-observer rho posterior per condition
  fig_trust_probe.pdf      history-matched message-update probe (if run)
Summary numbers -> OUT/analysis_summary.json.

Env: OUT(runs) MOCK(optional second run root with DRY outputs to overlay as the
     partner-blind Dirichlet-Markov null)
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_core as T

OUT = os.environ.get("OUT", os.path.join(HERE, "runs"))
MOCK = os.environ.get("MOCK", os.path.join(HERE, "runs", "dry"))
FLOOR = 1.0 / 3.0


def load_cond(root, cond):
    d = os.path.join(root, cond)
    if not os.path.isfile(os.path.join(d, "probes.npz")):
        return None
    z = np.load(os.path.join(d, "probes.npz"))
    meta = json.load(open(os.path.join(d, "stream.json")))
    return {"probes": z["probes"], "ckpts": z["ckpts"].tolist(), "meta": meta}


def subject_scores(run, spec, opts):
    """[n_ckpt] mean discrimination over pairs x site cues."""
    cues = sorted(opts)
    pr = run["probes"]
    out = []
    for ci in range(pr.shape[0]):
        vals = [T.score(pr[ci, p, cue], cue, opts)
                for p in range(pr.shape[1]) for cue in cues]
        out.append(float(np.mean(vals)))
    return out


def reference_scores(run, spec, opts):
    """ideal / gullible scores + final rho posterior, replaying B-attributed steps."""
    cues = sorted(opts)
    ideal, gull = [], []
    rho_post_final = None
    per_pair_obs = []
    for p in range(run["probes"].shape[1]):
        prefix_last = run["meta"]["a_prefix"][p][-1]
        obs = T.ExactObserver(spec)
        per_pair_obs.append((obs, prefix_last))
    for ci, tk in enumerate(run["ckpts"]):
        i_vals, g_vals = [], []
        for p, (obs, prefix_last) in enumerate(per_pair_obs):
            stream = run["meta"]["stream"][p]
            done = getattr(obs, "_done", 0)
            prev = prefix_last if done == 0 else stream[done - 1][1]
            for i in range(done, tk):
                who, x, _ = stream[i]
                if who == "B":
                    obs.update(prev, x)
                prev = x
            obs._done = tk
            for cue in cues:
                i_vals.append(T.score(obs.predictive(cue), cue, opts))
                g_vals.append(T.score(obs.predictive(cue, rho_fixed=0.0), cue, opts))
        ideal.append(float(np.mean(i_vals)))
        gull.append(float(np.mean(g_vals)))
        if ci == run["probes"].shape[0] - 1:
            rho_post_final = np.mean([o.rho_posterior()
                                      for o, _ in per_pair_obs], axis=0)
    return ideal, gull, rho_post_final


def main():
    spec = json.load(open(os.path.join(OUT, "t2_spec.json")))
    opts = T.contested_options(spec)
    conds = sorted(d for d in os.listdir(OUT)
                   if os.path.isdir(os.path.join(OUT, d)) and
                   os.path.isfile(os.path.join(OUT, d, "probes.npz")))
    res, summary = {}, {"floor": FLOOR}
    for cond in conds:
        run = load_cond(OUT, cond)
        subj = subject_scores(run, spec, opts)
        ideal, gull, rho_post = reference_scores(run, spec, opts)
        mock_run = load_cond(MOCK, cond)
        mock = subject_scores(mock_run, spec, opts) if mock_run else None
        res[cond] = {"run": run, "subj": subj, "ideal": ideal, "gull": gull,
                     "rho_post": rho_post, "mock": mock}
        summary[cond] = {
            "rho": run["meta"]["rho"], "kind": run["meta"]["kind"],
            "ckpts": run["ckpts"], "subject": subj, "ideal": ideal,
            "gullible": gull, "mock_null": mock,
            "b_valid_mean": run["meta"]["b_valid_mean"],
            "final_trust_norm": ((subj[-1] - FLOOR) / max(ideal[-1] - FLOOR, 1e-9)
                                 if len(subj) > 1 else None),
            "rho_MAP_exact": (float(np.array(spec["rho_grid"])
                                    [int(np.argmax(rho_post))])
                              if rho_post is not None else None)}
        print(f"{cond}: subj {np.round(subj, 3).tolist()} "
              f"ideal {np.round(ideal, 3).tolist()}")

    # ---- fig 1: score vs time -------------------------------------------------
    live = [c for c in conds if res[c]["run"]["meta"]["tgen"] > 0]
    if live:
        fig, axes = plt.subplots(1, len(live), figsize=(3.6 * len(live), 3.6),
                                 sharey=True, squeeze=False)
        for ax, cond in zip(axes[0], live):
            r = res[cond]
            ck = r["run"]["ckpts"]
            ax.plot(ck, r["subj"], "o-", color="#0e7c86", label="subject")
            ax.plot(ck, r["ideal"], "s--", color="#fb8500", label="ideal (rho inferred)")
            ax.plot(ck, r["gull"], "^:", color="#c1121f", label="gullible (rho=0)")
            if r["mock"]:
                ax.plot(ck, r["mock"], "d-.", color="#6a4c93",
                        label="Dirichlet-Markov null")
            ax.axhline(FLOOR, color="gray", lw=1, label="floor 1/3 (A alone)")
            ax.axhline(1.0, color="gray", lw=1, ls="--", label="oracle")
            ax.set_title(cond, fontsize=9)
            ax.set_xlabel("exchange step")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("contested discrimination\np(true partner | contested options)")
        axes[0][0].legend(fontsize=6.5)
        fig.suptitle("test-2: learning the contested edges from an unreliable partner",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_scores_vs_time.pdf"))

    # ---- fig 2: headline trust-vs-rho ----------------------------------------
    scripted = sorted([c for c in conds if c.startswith("scripted")],
                      key=lambda c: res[c]["run"]["meta"]["rho"])
    if scripted:
        rhos = [res[c]["run"]["meta"]["rho"] for c in scripted]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
        for key, lab, sty, col in (("subj", "subject", "o-", "#0e7c86"),
                                   ("ideal", "ideal (rho inferred)", "s--", "#fb8500"),
                                   ("gull", "gullible (rho=0)", "^:", "#c1121f"),
                                   ("mock", "Dirichlet-Markov null", "d-.", "#6a4c93")):
            ys = [res[c][key][-1] if res[c][key] else np.nan for c in scripted]
            if not all(np.isnan(y) for y in ys):
                a1.plot(rhos, ys, sty, color=col, label=lab)
        for c in [c for c in conds if c.startswith("llmB")]:
            a1.plot([res[c]["run"]["meta"]["rho"]], [res[c]["subj"][-1]], "*",
                    ms=12, color="#0e7c86", label="subject (LLM partner)")
        a1.axhline(FLOOR, color="gray", lw=1)
        a1.set_xlabel("true partner corruption rate rho_B")
        a1.set_ylabel("final contested discrimination")
        a1.set_title("what A learned from B, by B's reliability")
        a1.grid(alpha=0.3); a1.legend(fontsize=7)
        tr = [(res[c]["subj"][-1] - FLOOR) / max(res[c]["ideal"][-1] - FLOOR, 1e-9)
              for c in scripted]
        a2.plot(rhos, tr, "o-", color="#0e7c86", label="subject / ideal")
        a2.axhline(1.0, color="#fb8500", ls="--", label="ideal")
        a2.axhline(0.0, color="gray", lw=1, label="partner-ignoring")
        a2.set_xlabel("true partner corruption rate rho_B")
        a2.set_ylabel("normalized trust  (score-1/3)/(ideal-1/3)")
        a2.set_title("effective use of the partner channel")
        a2.grid(alpha=0.3); a2.legend(fontsize=7)
        fig.suptitle("test-2 headline: reliability-weighted pooling vs exact references")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_trust_vs_rho.pdf"))

    # ---- fig 3: rho identifiability ------------------------------------------
    if live:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        grid = np.array(spec["rho_grid"])
        for cond in live:
            if res[cond]["rho_post"] is not None:
                ax.plot(grid, res[cond]["rho_post"], "o-",
                        label=f"{cond} (true rho={res[cond]['run']['meta']['rho']})")
        ax.set_xlabel("rho hypothesis")
        ax.set_ylabel("exact-observer posterior P(rho | stream)")
        ax.set_title("design validation: rho_B is identifiable from the stream")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_rho_identifiability.pdf"))

    # ---- fig 4: trust probe ---------------------------------------------------
    tp_path = os.path.join(OUT, "trust_probe", "trust_probe.json")
    if os.path.isfile(tp_path):
        tp = json.load(open(tp_path))["summary"]
        hists = [h for h in ("honest", "corrupt", "corrupt_self")
                 if f"{h}_score_before" in tp]
        labels = {"honest": "honest\nhistory", "corrupt": "corrupt\nB-slots",
                  "corrupt_self": "corrupt\nA-slots (control)"}
        fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
        x = np.arange(len(hists))
        for ax, phase, ttl in ((axs[0], "before", "standing credence in the old "
                                                  "assertion"),
                               (axs[1], "after", "after ONE identical fresh "
                                                 "message")):
            m = [tp[f"{h}_score_{phase}"][0] for h in hists]
            e = [tp[f"{h}_score_{phase}"][1] for h in hists]
            ax.bar(x, m, 0.45, yerr=e, color="#0e7c86", capsize=3,
                   label="subject")
            ex = [tp.get(f"{h}_exact_{phase}", [np.nan])[0] for h in hists]
            ax.plot(x, ex, "s", color="#fb8500", ms=7,
                    label="exact observer (B-channel model)")
            ctrl = [tp.get(f"{h}_ctrl_before", [np.nan])[0] for h in hists]
            if phase == "before":
                ax.plot(x, ctrl, "_", color="#c1121f", ms=16, mew=2,
                        label="control cues (unprobed sites)")
            ax.axhline(1 / 3, color="gray", lw=1, label="floor 1/3")
            ax.set_xticks(x, [labels[h] for h in hists])
            ax.set_title(ttl, fontsize=10)
            ax.grid(alpha=0.3, axis="y")
        axs[0].set_ylabel("contested discrimination at the asserted cue")
        axs[0].legend(fontsize=6.5)
        fig.suptitle("history-matched trust probe: contested evidence token-identical; "
                     "kernel null predicts equal bars", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_trust_probe.pdf"))
        summary["trust_probe"] = tp

    json.dump(summary, open(os.path.join(OUT, "analysis_summary.json"), "w"),
              indent=1, default=float)
    print(f"figures + analysis_summary.json -> {OUT}")


if __name__ == "__main__":
    main()

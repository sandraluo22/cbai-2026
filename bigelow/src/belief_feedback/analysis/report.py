"""Automated report generation (Part 23)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from ..config import Config, git_commit
from ..logging_utils import get_logger
from ..plots.captions import CAPTIONS

log = get_logger(__name__)

RESULT_KIND = {
    "smoke": "MOCK — synthetic backend; NOT scientific evidence",
    "pilot": "PILOT — single seed, reduced sizes; preliminary only",
    "full": "FULL — primary scientific run",
    "low_memory": "4-BIT QUANTIZED — report separately from bf16",
    "second_model": "SECOND MODEL — confirmatory subset",
}


def _df(path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _fmt(df: pd.DataFrame | None, cols: list[str] | None = None, n: int = 12) -> str:
    if df is None or df.empty:
        return "_not available_"
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df.head(n).to_markdown(index=False, floatfmt=".3f")


def make_report(cfg: Config) -> None:
    runs = cfg.paths.runs
    kind = RESULT_KIND.get(cfg.name, cfg.name)
    now = datetime.now(UTC).isoformat()

    meta = {}
    meta_path = cfg.paths.vectors / cfg.model.slug / "steering_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    hyp = _df(runs / "hypothesis_tests.parquet")
    comp = _df(runs / "composition_metrics.parquet")
    recv = _df(cfg.paths.models / "receiver" / "receiver_metrics.parquet")
    eff = _df(runs / "branch_effects.parquet")
    rec = _df(runs / "recycling_results.parquet")
    hyst = _df(runs / "hysteresis_results.parquet")
    contours = _df(runs / "phase_boundary_contours.parquet")
    probe = _df(runs / "probe_results.parquet")
    rob = _df(runs / "robustness_results.parquet")
    episodes = _df(runs / "episodes.parquet")
    msgs = _df(runs / "public_messages.parquet")

    eff_summary = "_not available_"
    if eff is not None and not eff.empty:
        e = eff.assign(adj=lambda d: d["effect"] * d["sign"].map({"positive": 1, "negative": -1}))
        eff_summary = _fmt(
            e.groupby("condition")["adj"].agg(["mean", "std", "count"]).reset_index()
        )
    malformed = "_not available_"
    if msgs is not None and not msgs.empty:
        malformed = (
            f"format-valid rate {msgs['format_valid'].mean():.3f}; "
            f"hallucinated-citation rate "
            f"{(msgs['hallucinated_citations'].fillna('') != '').mean():.3f}; "
            f"mean memo length {msgs['word_count'].mean():.1f} words "
            f"({len(msgs)} public memos)"
        )
    lines = [
        f"# Endogenous Belief Dynamics — automated report ({cfg.name})",
        "",
        f"**Result kind: {kind}.** Mock or incomplete pilot results must never be presented as scientific evidence.",
        "",
        f"Generated {now} · git `{git_commit()[:12]}` · config hash `{cfg.config_hash()}` · model `{cfg.model.model_id}`",
        "",
        "## Research questions",
        "",
        "Can an emission model G and receiver-update model F, identified on controlled",
        "exogenous single-agent data, compose to predict live closed-loop multi-agent",
        "belief dynamics (H1)? Does feedback amplify one-round steering impulses beyond",
        "one hop (H2)? Do agents partially double-count repeated-source reports (H3)?",
        "Are downstream steering effects mediated by emitted text (H4)? Do equal-dose",
        "early vs late steering schedules leave different final states (H5)?",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(cfg.model_dump(mode="json"), indent=2)[:3000],
        "```",
        "",
        "## Data generation",
        "",
        "Two-hypothesis battery-plant incident worlds; 16 document families with",
        "held-out surface-template variants for all test splits; provenance-aware and",
        "provenance-blind oracles; counterbalanced ALPHA/BETA label mapping.",
        f"See `{cfg.paths.data / 'data_validation_report.json'}`.",
        "",
        "## Model and steering",
        "",
        f"Selected layer: **{meta.get('layer', 'n/a')}**, m_max = **{meta.get('m_max', 'n/a')}**, "
        f"delta = **{meta.get('delta', 'n/a')}**, scope `{meta.get('scope', 'n/a')}`.",
        "",
        "## Exogenous model results (receiver F, held-out test)",
        "",
        _fmt(recv[recv["usage"] == "test"] if recv is not None else None),
        "",
        "## Endogenous composition results",
        "",
        _fmt(comp),
        "",
        "## Causal branch results (sign-adjusted paired effects)",
        "",
        eff_summary,
        "",
        "## Recycling results",
        "",
        _fmt(
            rec.groupby(["condition", "provenance_aware_prompt"])[["multiplier", "double_counting_gap"]]
            .mean()
            .reset_index()
            if rec is not None and not rec.empty
            else None
        ),
        "",
        "## Hysteresis results",
        "",
        _fmt(
            hyst.groupby(["sign", "comm"])["hysteresis_gap"].mean().reset_index()
            if hyst is not None and not hyst.empty
            else None
        ),
        "",
        "## Phase-boundary results",
        "",
        _fmt(contours),
        "",
        "## Mechanistic results (probes)",
        "",
        _fmt(probe, ["layer", "accuracy", "auroc", "behavior_correlation", "cosine_with_caa"]),
        "",
        "## Robustness results",
        "",
        _fmt(
            rob.groupby(["dimension", "variant"])["mean_effect"].mean().reset_index()
            if rob is not None and not rob.empty
            else None
        ),
        "",
        "## Primary hypothesis tests",
        "",
        _fmt(hyp, ["hypothesis", "effect", "ci_low", "ci_high", "standardized_effect", "p_value", "n_worlds"]),
        "",
        "## Malformed-output statistics",
        "",
        malformed,
        "",
        "## Limitations",
        "",
        "- Mock results validate the pipeline only; no scientific claim follows from them.",
        "- Accessible-evidence bookkeeping treats a cited event as visible to the",
        "  receiver, an approximation of information flow through memo text.",
        "- Spectral quantities from the empirical Jacobian are local diagnostics, not",
        "  global stability claims.",
        "- Steering effects are calibrated within the coherent magnitude range only;",
        "  behavior outside it is undefined.",
        "",
        "## Figures and tables",
        "",
    ]
    for fig in sorted(cfg.paths.figures.glob("*.pdf")):
        lines.append(f"- [{fig.stem}]({fig})")
    for tab in sorted(cfg.paths.tables.glob("*.csv")):
        lines.append(f"- [{tab.stem}]({tab})")
    if episodes is not None:
        lines += ["", f"Episodes recorded: {len(episodes)}"]

    (cfg.paths.reports / "final_report.md").write_text("\n".join(lines))

    # run status ------------------------------------------------------------
    status_lines = [f"# Run status ({cfg.name})", ""]
    fail_lines = [f"# Failure log ({cfg.name})", ""]
    any_fail = False
    for mf in sorted(cfg.paths.manifests.glob("*.json")):
        m = json.loads(mf.read_text())
        ok = m.get("failed_jobs", 0) == 0
        status_lines.append(
            f"- `{m['stage']}`: {'OK' if ok else 'FAILED'} "
            f"({m.get('completed_jobs', '?')} jobs, ended {m.get('ended', '?')})"
        )
        if not ok:
            any_fail = True
            fail_lines.append(f"- `{m['stage']}`: {m.get('failed_jobs')} failed jobs")
    if not any_fail:
        fail_lines.append("No recorded failures.")
    (cfg.paths.reports / "run_status.md").write_text("\n".join(status_lines))
    (cfg.paths.reports / "failure_log.md").write_text("\n".join(fail_lines))

    cap_lines = [f"# Figure captions ({cfg.name})", ""]
    for name, caption in CAPTIONS.items():
        cap_lines += [f"## {name}", "", caption, ""]
    (cfg.paths.reports / "figure_captions.md").write_text("\n".join(cap_lines))
    log.info("report written to %s", cfg.paths.reports / "final_report.md")

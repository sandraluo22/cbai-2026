"""Figure captions collected for artifacts/reports/figure_captions.md."""

CAPTIONS = {
    "fig01_world_and_network_schematic": (
        "World and experiment schematic. (a) Latent hypothesis with evidence events and their "
        "reliabilities; (b) primary vs secondary report provenance (secondary reports share the "
        "hidden event id and visible lineage); (c) assignment of private reports to the agents; "
        "(d) bidirectional ring communication topology; (e) design logic: F and G are identified "
        "on exogenous calibration worlds and composed to predict closed-loop endogenous networks."
    ),
    "fig02_steering_calibration": (
        "CAA steering calibration. Vector norm by layer; held-out behavioral steering slope by "
        "layer; coherence metrics by layer; semantic log odds vs magnitude at the selected layer "
        "with the selected coherent range shaded."
    ),
    "fig03_exogenous_response_surface": (
        "Exogenous receiver response surface: observed post-message semantic log odds across "
        "incoming unique evidence and steering, fitted Bigelow-style count surface, fitted "
        "provenance-aware surface, and observed-minus-predicted residuals."
    ),
    "fig04_exogenous_model_comparison": (
        "Held-out comparison of receiver models F0-F5: RMSE and R^2, calibration, "
        "repeated-evidence prediction error, and performance by prior-belief stratum."
    ),
    "fig05_closed_loop_impulse_trajectories": (
        "Closed-loop impulse responses: mean paired change in belief by round for positive and "
        "negative impulses, split by graph distance from the source agent, with 95% world-cluster "
        "bootstrap intervals."
    ),
    "fig06_composition_generalization": (
        "Composition test: teacher-forced one-step predictions vs observed beliefs, free-rollout "
        "mean trajectories vs observed, predicted vs observed final consensus probability, and "
        "the endogenous generalization gap per candidate model."
    ),
    "fig07_causal_path_decomposition": (
        "Causal path decomposition of the steering response into one-hop, forward-cascade, "
        "reciprocal-feedback, and total closed-loop effects, by round and graph distance."
    ),
    "fig08_evidence_recycling": (
        "Evidence recycling: belief gains from one vs three independent vs three recycled "
        "reports; recycling multiplier; double-counting gap vs the provenance-aware oracle; "
        "neutral vs provenance-aware instructions; interaction with steering."
    ),
    "fig09_hysteresis": (
        "Hysteresis: early vs late equal-dose steering schedules under live communication and "
        "fixed replay; final gaps; and the live-minus-replay interaction."
    ),
    "fig10_network_phase_boundary": (
        "Phase boundary: probability of a final upstream majority over initial network evidence "
        "and steering, the composition-model prediction, both 0.5 contours, and their displacement."
    ),
    "fig11_empirical_jacobian": (
        "Empirical network Jacobian: mean intervention Jacobian, impulse magnitude by graph "
        "distance, spectral radius by round (a local diagnostic only), and observed vs "
        "Jacobian-predicted multi-round response."
    ),
    "fig12_mechanistic_alignment": (
        "Mechanistic alignment: probe performance by layer, CAA projection vs behavioral log "
        "odds, projection trajectories by round, probe/CAA cosine similarity, and the behavioral "
        "effect of belief-component patching."
    ),
    "fig13_text_mediation": (
        "Text mediation: live-memo vs full-text-clamped interventions, activation steering vs "
        "projection patching, downstream effects by round, and the fraction of the total effect "
        "carried by the altered text."
    ),
    "fig14_robustness": (
        "Robustness: impulse-effect estimates by prompt variant, memory policy, topology, "
        "schedule, channel ablation, and steering scope, plus malformed-output and hallucinated- "
        "citation rates. Robustness runs are never pooled with primary estimates."
    ),
}

"""Agent roles for the incident-review team."""

from __future__ import annotations

ROLES = [
    "analytical chemist",
    "supplier quality engineer",
    "process engineer",
    "metrology engineer",
    "reliability analyst",
    "maintenance engineer",
    "microscopy and failure-analysis specialist",
    "incident-review coordinator",
]

# Role-plausible document families for assignment weighting (advisory only).
ROLE_FAMILIES = {
    "analytical chemist": ["lab_chromatography", "sealed_drum_retest", "independent_lab_confirmation"],
    "supplier quality engineer": ["supplier_process_change", "quality_release_record", "shipping_environment_log"],
    "process engineer": ["plc_cycle_log", "warmup_reference_test", "incident_timeline"],
    "metrology engineer": ["flow_meter_calibration", "reference_mass_audit"],
    "reliability analyst": ["failure_clustering_analysis", "incident_timeline"],
    "maintenance engineer": ["maintenance_work_order", "operator_handover_note"],
    "microscopy and failure-analysis specialist": ["microscopy_residue_report", "independent_lab_confirmation"],
    "incident-review coordinator": ["witness_interview", "quality_release_record", "operator_handover_note"],
}


def role_for_agent(agent_id: int) -> str:
    return ROLES[agent_id % len(ROLES)]

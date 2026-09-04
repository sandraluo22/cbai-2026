"""Surface template bank for rendered incident documents.

Each of the sixteen document families has five independently written finding
templates per orientation. Placeholders are filled by
:mod:`belief_feedback.world.documents`. Variant index ``HELD_OUT_VARIANT``
(the last one) is reserved for test splits and never appears in training
worlds; :mod:`belief_feedback.world.validation` enforces this.

Orientation keys: ``"up"`` supports UPSTREAM_CONTAMINATION, ``"local"``
supports LOCAL_CALIBRATION_DRIFT. No template states the true hypothesis,
a numerical likelihood, or a reliability.
"""

from __future__ import annotations

N_VARIANTS = 5
HELD_OUT_VARIANT = 4  # used only in test-split worlds

# family -> {"reliability": float, "unit": str, "titles": [...], "up": [5], "local": [5]}
FAMILY_TEMPLATES: dict[str, dict] = {
    "lab_chromatography": {
        "reliability": 0.90,
        "unit": "Analytical Chemistry Laboratory",
        "titles": ["Chromatography screening results", "Electrolyte chromatographic analysis"],
        "up": [
            "Ion chromatography of electrolyte recovered from the failed cells shows an unexpected contaminant peak at {ppm} ppm. The same peak, at comparable intensity, appears in the retained sample drawn from supplier lot {lot_id} before it reached the line.",
            "Chromatograms from sample {sample_id} contain a peak absent from our reference library, and a retained aliquot of incoming lot {lot_id} reproduces the same signature. The match between failed-cell chemistry and the retained supplier material is close.",
            "The laboratory ran duplicate chromatographic injections on extracts from the affected units and from the archived supplier-lot retain. Both traces show the same extraneous species near {ppm} ppm that is not present in control cells built from other lots.",
            "An anomalous elution peak was detected in electrolyte from the failed population. When the retained pre-production sample of lot {lot_id} was analyzed under identical conditions, the identical peak appeared, indicating the species was present in the material as received.",
            "Comparative chromatography flags a foreign compound in the failed cells; a sealed retain of lot {lot_id} analyzed the same day exhibits the matching peak. Cells from adjacent lots processed on the same equipment show clean traces.",
        ],
        "local": [
            "Ion chromatography of electrolyte from the failed cells matches the reference profile within normal variation, and the retained sample from supplier lot {lot_id} is fully within specification. No extraneous peak was detected in either analysis.",
            "Chromatograms from sample {sample_id} and from the archived retain of lot {lot_id} both align with the qualified electrolyte fingerprint. The laboratory found no contaminant species above the reporting threshold in any injection.",
            "Duplicate chromatographic runs on failed-cell extracts and on the incoming-lot retain show clean traces. All quantified species fall inside their specification windows, and no peak unique to the failed population was observed.",
            "The failed units' electrolyte chemistry is indistinguishable from control cells, and the retained supplier sample for lot {lot_id} passes every chromatographic acceptance criterion. Nothing in the chemical data points to the incoming material.",
            "Screening detected no unexpected species in either the failed cells or the sealed retain of lot {lot_id}; every trace overlays the reference standard. The chemistry offers no support for a material-quality excursion.",
        ],
    },
    "sealed_drum_retest": {
        "reliability": 0.88,
        "unit": "Incoming Materials Quality",
        "titles": ["Sealed drum retest", "Incoming-lot drum verification"],
        "up": [
            "A never-opened drum from lot {lot_id} was retested under witness. Water content measured well above the incoming limit and conductivity was abnormal, consistent with contamination present before the drum left the supplier.",
            "Retest of a factory-sealed drum from the implicated lot returned elevated ionic contamination on two independent instruments. The drum seal and tamper band were intact, so the excursion predates receipt.",
            "We pulled an unopened drum from quarantine stock of lot {lot_id}. Karl Fischer titration shows moisture beyond specification and the conductivity reading is out of family with qualified material.",
            "The sealed-container retest for lot {lot_id} failed: ionic species exceeded the acceptance limit and the moisture value sat outside the release window. Chain-of-custody records confirm the drum was never opened on site.",
            "Verification testing on a sealed drum of the suspect lot found water and ionic levels outside the qualified range. A control drum from a different lot, tested in the same session, passed cleanly.",
        ],
        "local": [
            "A never-opened drum from lot {lot_id} was retested under witness. Water content, ionic contamination, and conductivity all fall inside the incoming-material specification, matching the certificate of analysis.",
            "Retest of a factory-sealed drum from the implicated lot passed every acceptance criterion. Both instruments agreed with the supplier's certificate values within measurement uncertainty.",
            "We pulled an unopened drum from quarantine stock of lot {lot_id}. Karl Fischer moisture, conductivity, and the ionic panel are all within limits and consistent with qualified historical lots.",
            "The sealed-container retest for lot {lot_id} shows no excursion: every measured parameter is inside its release window, and the values track the lot's original incoming inspection closely.",
            "Verification testing on a sealed drum of the suspect lot found nothing abnormal; results are in family with the last twelve accepted lots. The incoming material itself appears sound.",
        ],
    },
    "supplier_process_change": {
        "reliability": 0.68,
        "unit": "Supplier Quality Engineering",
        "titles": ["Supplier process-change review", "Supplier change-history audit"],
        "up": [
            "The supplier's change log shows a switch of filtration media two weeks before lot {lot_id} shipped, without a requalification run. The timing places the change immediately upstream of the implicated material.",
            "During the audit call the supplier disclosed a revised vessel-cleaning procedure introduced just before the affected lot was produced. No customer notification was issued for the change.",
            "Records obtained from the supplier indicate new drum-handling equipment was commissioned in the window when lot {lot_id} was filled. The change was classified internally as minor and skipped incoming re-approval.",
            "The supplier's production history for the implicated lot shows a deviation approval covering an altered filtration step. Adjacent lots made before the deviation are not represented in our failed population.",
            "A review of supplier documentation found a cleaning-agent substitution logged shortly before the suspect lot's fill date. The substitution was not flagged on the certificate of analysis.",
        ],
        "local": [
            "The supplier's change log shows no process, material, or equipment change for six months around the production of lot {lot_id}. Adjacent lots from the same campaign passed all incoming checks.",
            "The audit found the supplier's filtration, cleaning, and handling procedures unchanged since the last qualification. Certificates for the implicated lot and its neighbors are unremarkable.",
            "Records obtained from the supplier show a stable production configuration through the window when lot {lot_id} was filled, with no deviations or waivers logged.",
            "The supplier's production history for the implicated lot is clean: no deviation approvals, no equipment swaps, and consistent in-process measurements across the campaign.",
            "A review of supplier documentation found no substitutions or procedure edits near the suspect lot's fill date, and sister lots shipped to another site have reported no issues.",
        ],
    },
    "shipping_environment_log": {
        "reliability": 0.62,
        "unit": "Logistics and Warehousing",
        "titles": ["Shipment environment review", "Transit condition log summary"],
        "up": [
            "The data logger accompanying lot {lot_id} recorded a temperature excursion above the allowed ceiling for roughly {minutes} minutes during transit. A seal-integrity flag was also raised at the receiving dock.",
            "Transit records show the implicated shipment was held on an un-conditioned apron during a heat event, and one container arrived with a deformed closure ring noted on the receiving report.",
            "Review of the shipping log for lot {lot_id} found the trailer's setpoint was violated overnight, and the humidity trace saturates for part of the leg. Handling notes mention a dropped pallet.",
            "The environmental record for the suspect lot includes an out-of-range interval flagged by the courier, together with a customs inspection during which containers were opened and restaged.",
            "Receiving paperwork for lot {lot_id} documents a broken tamper seal on one drum and a logger alarm during the final transit leg, both outside normal handling experience for this material.",
        ],
        "local": [
            "The data logger accompanying lot {lot_id} shows temperature and humidity within limits for the entire journey, and receiving inspection recorded all seals intact.",
            "Transit records for the implicated shipment are unremarkable: no excursion alarms, no handling incidents, and container integrity confirmed at the dock.",
            "Review of the shipping log for lot {lot_id} found the trailer held its setpoint throughout, with no alarms and no anomalies noted by the carrier or by receiving staff.",
            "The environmental record for the suspect lot is clean end to end; every logged interval is inside the qualified envelope and the chain-of-custody paperwork is complete.",
            "Receiving paperwork for lot {lot_id} documents intact seals, normal drum condition, and a logger trace entirely within the allowed band.",
        ],
    },
    "flow_meter_calibration": {
        "reliability": 0.90,
        "unit": "Metrology Laboratory",
        "titles": ["Flow meter calibration check", "Dispense meter verification"],
        "up": [
            "The electrolyte flow meter at station {station_id} was checked against a traceable transfer standard at three flow rates. All points fall within the tolerance band, and the as-found values match the last calibration.",
            "A traceable verification of the station {station_id} dispense meter passed at every test point; as-found error is under a quarter of the allowed tolerance. The instrument is measuring correctly.",
            "Metrology ran the meter at station {station_id} against the reference flow rig. Repeatability and linearity are both within specification, and no adjustment was required.",
            "Calibration check of the fill meter at the implicated station shows agreement with the standard across the operating range. The calibration sticker history shows no overdue interval.",
            "The dispense meter passed its as-found traceable check with margin at low, mid, and high flow. Nothing in the metrology data suggests the station is misdelivering volume.",
        ],
        "local": [
            "The electrolyte flow meter at station {station_id} was checked against a traceable transfer standard and under-registers by about {pct} percent across the operating range, outside the allowed tolerance.",
            "A traceable verification of the station {station_id} dispense meter found a systematic offset near {pct} percent at all three test flows. The error direction means delivered volume differs from the commanded value.",
            "Metrology ran the meter at station {station_id} against the reference flow rig and measured a consistent registration error of roughly {pct} percent. The as-found condition fails the calibration criterion.",
            "Calibration check of the fill meter at the implicated station shows a bias of approximately {pct} percent relative to the standard, stable across repeats — a real measurement error, not noise.",
            "The dispense meter failed its as-found check, reading about {pct} percent away from the traceable reference at every point tested. The station has been dosing off-target volume.",
        ],
    },
    "warmup_reference_test": {
        "reliability": 0.84,
        "unit": "Process Engineering",
        "titles": ["Warmup stability test", "Reference dispense drift check"],
        "up": [
            "Repeated reference dispenses at station {station_id} were logged from cold start through four hours of operation. The delivered-mass trace is flat after the first cycle, with no drift beyond gauge repeatability.",
            "The warmup study shows stable reference measurements at station {station_id}: sixty consecutive dispenses vary within the repeatability band and show no time trend.",
            "We exercised the station with reference dispenses at ten-minute intervals across a full shift. The results hold steady from the first measurement to the last; no warmup effect is present.",
            "Continuous reference testing at the implicated station found no systematic change after startup; the control chart stays inside limits for the entire run.",
            "Dispense-mass checks repeated through the warmup window remain constant within measurement noise, giving no indication of a thermally driven calibration shift at this station.",
        ],
        "local": [
            "Repeated reference dispenses at station {station_id} drift steadily beginning about {minutes} minutes after startup, moving outside the repeatability band and stabilizing at a shifted value.",
            "The warmup study shows a clear time trend at station {station_id}: delivered mass changes progressively over the first half hour of operation before plateauing away from nominal.",
            "We exercised the station with reference dispenses at ten-minute intervals; after roughly {minutes} minutes the readings walk off in one direction, a classic warmup drift signature.",
            "Continuous reference testing at the implicated station found a systematic post-startup shift that exceeds the control limit, repeatable across two separate cold starts.",
            "Dispense-mass checks are stable for the first cycles, then drift with machine temperature and settle at an offset. The behavior reproduces on both test days.",
        ],
    },
    "maintenance_work_order": {
        "reliability": 0.72,
        "unit": "Maintenance Engineering",
        "titles": ["Work-order history review", "Maintenance record summary"],
        "up": [
            "Work-order history for station {station_id} shows the last service (ticket {ticket_id}) closed with full functional verification and sign-off. No dispensing-related item has been open in the past quarter.",
            "The maintenance ledger for the implicated station is clean: preventive tasks completed on schedule, verification steps documented, and no deferred items touching the fill system.",
            "Review of ticket {ticket_id} and its predecessors shows each closure included a verified test dispense. There is no unresolved pump, valve, or sensor issue on record for this station.",
            "All recent work orders on station {station_id} were completed and verified; the fill subsystem has had no corrective intervention since the last passing calibration.",
            "Maintenance records give no indication of trouble: the station's service history is routine, closures are verified, and no operator-reported dispensing complaint remains open.",
        ],
        "local": [
            "Work-order history for station {station_id} shows ticket {ticket_id}, a dosing-pump warning, was deferred twice and finally closed without the required verification dispense.",
            "The maintenance ledger reveals a valve-response alarm on the implicated station that was acknowledged and closed administratively; no functional check was recorded afterwards.",
            "Review of ticket {ticket_id} shows a fill-sensor fault flagged before the failure window and marked resolved with the note 'monitor', with no parts replaced and no verification.",
            "A pump-pressure warning on station {station_id} was left in deferred status through the affected production period; the closure entry lacks the mandatory sign-off dispense.",
            "Maintenance records show an unresolved dispensing-related item: the station's metering valve was serviced but the post-service verification field is empty in ticket {ticket_id}.",
        ],
    },
    "plc_cycle_log": {
        "reliability": 0.76,
        "unit": "Controls Engineering",
        "titles": ["PLC fill-cycle log analysis", "Station cycle telemetry review"],
        "up": [
            "PLC telemetry for station {station_id} across the affected window shows valve-open time, line pressure, and cycle duration all stable and centered on their historical means.",
            "The extracted cycle logs are unremarkable: fill-time distributions overlay the previous month's, and pressure traces show no step change on the implicated station.",
            "Analysis of {n_units} logged cycles found no shift in any monitored parameter at station {station_id}; the control charts remain in statistical control throughout.",
            "Controls review of the fill-cycle data shows the station executing to recipe: valve timings and pressures are indistinguishable before, during, and after the failure window.",
            "The PLC history gives no evidence of a station-level change; every fill parameter tracks its baseline within normal variation over the whole period examined.",
        ],
        "local": [
            "PLC telemetry for station {station_id} shows a systematic step in valve-open time beginning in the affected window, accompanied by a small but consistent line-pressure shift.",
            "The extracted cycle logs reveal the implicated station's fill duration distribution moved off its historical center while sister stations stayed put.",
            "Analysis of {n_units} logged cycles found a sustained timing offset at station {station_id}: cycle-complete times shifted together with a pressure change, exactly when failures began.",
            "Controls review of the fill-cycle data shows the station deviating from recipe: valve timing drifted out of its control band during the failure window and has not returned.",
            "The PLC history contains a clear station-level signature — a coordinated shift in dispense timing and pressure unique to this station and coincident with the affected builds.",
        ],
    },
    "failure_clustering_analysis": {
        "reliability": 0.82,
        "unit": "Reliability Engineering",
        "titles": ["Failure clustering analysis", "Failure distribution study"],
        "up": [
            "Mapping the {n_units} confirmed failures shows they concentrate in cells built from supplier lot {lot_id}, spread across four different fill stations. Lot membership, not station, is the discriminating factor.",
            "The failure population follows the material: units from the implicated lot fail at an elevated rate regardless of which station filled them, while other lots on the same stations are unaffected.",
            "Cross-tabulating failures by lot and station shows a strong lot effect and no station effect; the odds ratio for the suspect lot dwarfs any station contrast.",
            "Cluster analysis places the failures with lot {lot_id} across multiple lines and shifts. Stations that never touched the lot report zero failures in the window.",
            "The distribution study finds failures tracking the incoming lot through the factory: every affected station handled lot {lot_id}, and the failure rate scales with the fraction of that lot each station consumed.",
        ],
        "local": [
            "Mapping the {n_units} confirmed failures shows they concentrate at station {station_id} and span several different supplier lots. Station, not lot membership, is the discriminating factor.",
            "The failure population follows the station: units filled at {station_id} fail at an elevated rate regardless of which lot supplied the electrolyte, while the same lots filled elsewhere are fine.",
            "Cross-tabulating failures by lot and station shows a strong station effect and no lot effect; the odds ratio for the implicated station dwarfs any lot contrast.",
            "Cluster analysis places the failures with station {station_id} across three lots and two shifts. The same lots processed at neighboring stations produced no failures.",
            "The distribution study finds failures pinned to one piece of equipment: only cells that passed through station {station_id} appear in the failed set, whatever material they carried.",
        ],
    },
    "reference_mass_audit": {
        "reliability": 0.86,
        "unit": "Quality Control",
        "titles": ["Fill-mass audit", "Reference mass verification"],
        "up": [
            "Gravimetric audit of {n_units} implicated units shows filled electrolyte mass within specification on every cell, centered on nominal. Volume delivery at the stations appears correct.",
            "The mass audit found no fill anomaly: measured net masses across the affected builds sit inside the release window and match the plant-wide distribution.",
            "Weighing a sample of failed and sibling units gives fill masses on target, with a spread no wider than normal. Whatever failed these cells, it was not the delivered volume.",
            "Reference mass checks across all implicated stations show delivered quantity within tolerance for every audited unit, including the failed ones.",
            "The audit team re-weighed retained units against tare records; net electrolyte mass is nominal throughout, ruling the fill quantity in specification for the affected population.",
        ],
        "local": [
            "Gravimetric audit of {n_units} implicated units shows a systematic underfill of roughly {pct} percent on cells from station {station_id}, well outside the release window.",
            "The mass audit found a fill anomaly: net masses from the implicated station cluster below nominal by a consistent margin, while sister stations remain on target.",
            "Weighing a sample of failed units gives electrolyte masses shifted off nominal in one direction, specific to station {station_id}; sibling units from other stations are centered correctly.",
            "Reference mass checks show the implicated station delivering off-target quantity — a repeatable offset near {pct} percent — across every audited build date in the window.",
            "The audit team re-weighed retained units against tare records and found a station-specific mass deficit consistent with a mis-calibrated dispense, present on failed and not-yet-failed cells alike.",
        ],
    },
    "microscopy_residue_report": {
        "reliability": 0.78,
        "unit": "Failure Analysis Laboratory",
        "titles": ["Microscopy residue examination", "SEM/EDS residue report"],
        "up": [
            "SEM examination of the internal residue on sample {sample_id} shows crystalline deposits with an elemental signature foreign to the qualified electrolyte, consistent with the suspected contaminant species.",
            "The residue morphology is granular and chemically distinct: EDS picks up an element not present in any process material at this site, pointing to something carried in with the electrolyte.",
            "Microscopy of the failed-cell interior reveals deposits whose composition matches neither the electrode stack nor the qualified electrolyte, but is consistent with a foreign chemical introduced upstream.",
            "Cross-sections show the deposit chemistry on sample {sample_id} is out of family with normal decomposition products; the spectra fit a contaminant hypothesis far better than a drying artifact.",
            "The examined residue carries a distinct foreign signature at multiple sites on the electrode surface, uniform across the failed units, as expected if the contaminant arrived dissolved in the fill material.",
        ],
        "local": [
            "SEM examination of the internal residue on sample {sample_id} shows a drying-front pattern typical of insufficient electrolyte volume; EDS finds only expected process elements, no foreign species.",
            "The residue morphology indicates poor wetting: streaked, thin films concentrated where the fill level would sit if volume were low. Chemistry is unremarkable.",
            "Microscopy of the failed-cell interior reveals dry regions and a wetting boundary partway up the stack — the physical picture of a low fill — with no anomalous elemental signature anywhere.",
            "Cross-sections show separator regions on sample {sample_id} that were never wetted. The deposit chemistry matches normal electrolyte decomposition; nothing foreign was detected.",
            "The examined residue pattern is consistent with a volume deficit: incomplete wetting and localized drying marks, and an elemental map containing only qualified materials.",
        ],
    },
    "incident_timeline": {
        "reliability": 0.74,
        "unit": "Manufacturing Engineering",
        "titles": ["Incident timeline reconstruction", "Failure-onset timeline"],
        "up": [
            "The reconstructed timeline places first failures within a day of lot {lot_id} entering production, and clearly before the only station maintenance event in the window.",
            "Failure onset aligns with the introduction of the new supplier lot: the first affected serials were built the same shift the lot was released to the line, with no equipment event preceding them.",
            "Ordering the records shows the sequence lot release, then failures, then (much later) any station service. Builds from the prior lot immediately before the changeover are unaffected.",
            "The timeline ties the failure window to material flow: affected serial numbers begin exactly at the lot boundary and continue while the lot was consumed, across stations serviced on different dates.",
            "First-failure dates precede every station calibration or maintenance entry in the period, but coincide with the arrival of lot {lot_id} at the point of use.",
        ],
        "local": [
            "The reconstructed timeline places first failures immediately after the station service recorded in ticket {ticket_id}, and the affected builds span three different supplier lots.",
            "Failure onset aligns with an equipment event: the first affected serials were built during the shift following the station's calibration adjustment, while material lots straddle the boundary unaffected elsewhere.",
            "Ordering the records shows the sequence station service, then failures, spanning multiple lots. Cells built from the same lots before the service date show no elevated failure rate.",
            "The timeline ties the failure window to the equipment history: affected serial numbers begin at the maintenance date on station {station_id} and involve every lot processed there since.",
            "First-failure dates follow the station's servicing entry and cut across lot boundaries, which is difficult to reconcile with a material-driven cause.",
        ],
    },
    "operator_handover_note": {
        "reliability": 0.64,
        "unit": "Production Operations",
        "titles": ["Shift handover notes", "Operator observations log"],
        "up": [
            "Handover notes from two different lines mention the new electrolyte behaving oddly — a faint odor on drum opening and slight cloudiness — beginning when lot {lot_id} was first tapped.",
            "Operators on both shifts independently logged that the current material looked hazier than usual in the day tank and smelled different, observations spanning more than one station.",
            "Several handover entries from separate crews describe the electrolyte wetting differently and leaving an unusual film on fixtures, noted wherever the new lot was in use.",
            "Notes from multiple stations report the same observation this week: the material from the latest delivery seems off — cloudy on dispense and slower to clear — regardless of which machine dispensed it.",
            "Crews across the hall flagged the incoming material's appearance in their logs, with comments about haze and odor appearing at every station drawing from lot {lot_id}.",
        ],
        "local": [
            "Handover notes for station {station_id} mention a pump stutter and occasionally delayed fill starts over the past week; no other station logs a similar complaint.",
            "Operators on the implicated station logged an inconsistent cycle sound and intermittent short-fill alarms, while crews on neighboring stations report normal running with the same material.",
            "Several handover entries describe station {station_id} 'hesitating' at the start of dispense and needing an extra cycle to top off; the notes are specific to this one machine.",
            "Notes from the affected line report irregular dispense timing at a single station — described as a chatter in the dosing pump — with the rest of the hall running quietly.",
            "Only the crew operating station {station_id} flagged anything unusual: repeated remarks about erratic fill behavior and a valve that 'sticks on cold mornings'.",
        ],
    },
    "quality_release_record": {
        "reliability": 0.66,
        "unit": "Quality Assurance",
        "titles": ["Release record review", "Incoming release audit"],
        "up": [
            "The release record shows lot {lot_id} was accepted on a conditional waiver: one incoming assay sat at the borderline and was released under engineering disposition rather than a clean pass.",
            "Audit of the incoming paperwork found a marginal assay result for the implicated lot that was waived at release, with the retest requirement marked not applicable.",
            "The quality file shows the suspect lot entered the line under a deviation: an incoming purity value at the edge of the window was conditionally released pending supplier feedback that never arrived.",
            "Release documentation records a borderline moisture reading on lot {lot_id} accepted via waiver signature, an unusual disposition for this material class.",
            "The lot's release history is not clean: one acceptance parameter required a second opinion and was ultimately passed on judgment rather than on data inside the limit.",
        ],
        "local": [
            "The release record shows lot {lot_id} passed every incoming test with margin; however, the periodic calibration check for station {station_id} was overdue at the time the affected units were built.",
            "Audit of the paperwork found clean incoming results for the implicated lot, while the equipment file shows the station's scheduled verification had lapsed past its due date.",
            "The quality file gives the material an unqualified pass; the open finding in the same period is an overdue calibration interval on the fill station.",
            "Release documentation shows the lot accepted without waivers or retests. The audit's only exception is an expired calibration sticker on station {station_id} during the build window.",
            "Nothing in the lot's release history is abnormal, but the station-readiness checklist reveals a missed calibration check covering exactly the affected production dates.",
        ],
    },
    "independent_lab_confirmation": {
        "reliability": 0.92,
        "unit": "External Analytical Services",
        "titles": ["Independent laboratory confirmation", "Third-party analysis results"],
        "up": [
            "The independent laboratory's report matches the residue extracted from the failed cells to the retained upstream sample of lot {lot_id} on three separate analytical techniques, with high stated confidence.",
            "Third-party analysis confirms our in-house finding: the foreign species in the failed units is chemically identical to a species found in the archived supplier retain, per run {run_id}.",
            "The contract lab independently identified the contaminant in sample {sample_id} and detected the same compound in the sealed supplier retain, concluding the material carried it on arrival.",
            "External testing reproduces the anomalous signature in both failed-cell extract and the upstream retain; the lab's cross-correlation of the two spectra is described as a definitive match.",
            "The outside laboratory's certificate reports the same contaminant present in the incoming-lot retain as in the failed cells, using methods orthogonal to ours.",
        ],
        "local": [
            "The independent laboratory found the failed cells' chemistry entirely normal, while its physical measurements show a fill-volume deficit that reconstructs to one dispensing point, station {station_id}.",
            "Third-party analysis reports no foreign species anywhere; instead the lab's tomography quantifies an electrolyte shortfall in the failed units consistent with a station-level dosing error, per run {run_id}.",
            "The contract lab's chemistry panel on sample {sample_id} is clean. Their dimensional and mass analysis, however, measures a systematic volume deficit shared by units from a single station.",
            "External testing rules out a chemical cause: every assay matches reference. The lab's independent fill-level estimate places the affected units below minimum, tied to their common fill station.",
            "The outside laboratory's certificate reports qualified chemistry and flags only one anomaly — a repeatable underfill in cells traceable to station {station_id}.",
        ],
    },
    "witness_interview": {
        "reliability": 0.60,
        "unit": "Incident Review Office",
        "titles": ["Interview summary", "Personnel interview notes"],
        "up": [
            "Interviews with personnel from three different stations record the same recollection: the material from the latest delivery seemed different — in smell and appearance — from the day it arrived.",
            "Staff across multiple work centers independently recalled remarking on the new lot when it came in; none of them associate the problems with any particular machine.",
            "The interview summaries show a shared impression among people who handled lot {lot_id} at different stations that something about the material itself changed with this delivery.",
            "Witnesses from several lines describe noticing the electrolyte behaving unusually wherever it was used, and no interviewee singled out one station's equipment.",
            "Across the interviews, the common thread is the material: multiple unconnected staff mention the new lot's appearance, while equipment complaints are absent.",
        ],
        "local": [
            "Interviews record that only personnel working at or near station {station_id} noticed anything — chiefly inconsistent dispensing — while staff elsewhere recall nothing unusual about the material.",
            "Staff on the implicated station describe erratic fill behavior in the affected period; interviewees from other stations using the same lots report normal operation.",
            "The interview summaries localize the observations to one machine: its operators mention hesitant dispense cycles, and no one elsewhere corroborates a material change.",
            "Witnesses near station {station_id} recall the dispensing acting up before failures surfaced; the material itself drew no comment from anyone interviewed.",
            "Across the interviews, equipment complaints cluster entirely at the one station, and recollections about the incoming material are uniformly unremarkable.",
        ],
    },
}

# Generic scaffolding -------------------------------------------------------

CONTEXT_SENTENCES = [
    "This record was prepared as part of the ongoing failure investigation at the {site} facility.",
    "The work described here was requested under the open incident review covering recent cell failures.",
    "This document supports the root-cause investigation into the elevated failure rate observed this period.",
    "Prepared for the incident-review board examining the recent battery-cell failures.",
    "Compiled at the request of the joint technical team investigating the current failure cluster.",
]

LIMITATION_SENTENCES = [
    "Findings reflect the samples and records available at the time of writing and may be revised as further data arrive.",
    "The scope of this check was limited to the items listed above; broader coverage was not attempted.",
    "Results should be weighed alongside other evidence; this record alone does not establish a root cause.",
    "Measurement and sampling uncertainty apply; conclusions are stated to the precision the method supports.",
    "Some source records were incomplete, and the summary above reflects the best available reconstruction.",
]

SECONDARY_WRAPPERS = [
    "This memo summarizes, for wider distribution, the findings originally documented in report {orig_report_id} concerning {lineage_phrase}. No new measurements were performed.",
    "Forwarding note: the following restates the substance of report {orig_report_id}, which addressed {lineage_phrase}, for teams that did not receive the original.",
    "At the coordinator's request, this digest re-presents the conclusions of report {orig_report_id} regarding {lineage_phrase}; the underlying data are unchanged.",
    "Cross-team briefing derived from report {orig_report_id}: the observations below concern {lineage_phrase} and repeat the original record's findings in condensed form.",
    "This entry re-files the result set of report {orig_report_id} ({lineage_phrase}) into the incident dossier; it contains no independent verification.",
]

SITES = ["Harwick", "Delmont", "Aylesford", "Brinton", "Northgate"]

FIRST_NAMES = [
    "Ade", "Bianca", "Carlos", "Dana", "Emeka", "Farah", "Goran", "Hana",
    "Ilya", "Joon", "Katya", "Luis", "Mina", "Nadia", "Owen", "Priya",
    "Quinn", "Rafael", "Sana", "Tomas", "Uma", "Viktor", "Wen", "Yara",
]
LAST_NAMES = [
    "Okafor", "Silva", "Novak", "Haddad", "Kim", "Petrov", "Iyer", "Mbeki",
    "Larsen", "Costa", "Tanaka", "Weber", "Osei", "Duarte", "Nagy", "Reyes",
    "Bakker", "Sato", "Moreau", "Aliyev", "Chen", "Dvorak", "Fontaine", "Grig",
]

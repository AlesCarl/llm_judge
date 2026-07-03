"""Deterministic submission-vs-ground-truth scoring, usable on open sessions.

Kept free of judge/agent imports so the in-loop LangGraph agent can call it
during a live run without pulling in the LLM evaluator stack (which would
create an import cycle through ``agent.llm``).
"""

from nika.orchestrator.tasks.detection import DetectionSubmission
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask


def generic_eval(gt, submission):
    """Score detection, localization, and RCA from structured ``gt`` and ``submission``."""
    try:
        parsed_detect_sub = DetectionSubmission.model_validate({"is_anomaly": submission.get("is_anomaly", False)})
        if gt["is_anomaly"] == parsed_detect_sub.is_anomaly:
            detection_score = 1.0
        else:
            detection_score = 0.0
    except Exception:
        detection_score = -1.0

    try:
        loc_acc, loc_prec, loc_rec, loc_f1 = LocalizationTask().eval(
            submission={"faulty_devices": submission.get("faulty_devices", [])},
            gt={"faulty_devices": gt.get("faulty_devices", [])},
        )
    except Exception:
        loc_acc, loc_prec, loc_rec, loc_f1 = -1.0, -1.0, -1.0, -1.0

    try:
        rca_acc, rca_prec, rca_rec, rca_f1 = RCATask().eval(
            submission={"root_cause_name": submission.get("root_cause_name", [])},
            gt={"root_cause_name": gt.get("root_cause_name", [])},
        )
    except Exception:
        rca_acc, rca_prec, rca_rec, rca_f1 = -1.0, -1.0, -1.0, -1.0

    return (
        detection_score,
        loc_acc,
        loc_prec,
        loc_rec,
        loc_f1,
        rca_acc,
        rca_prec,
        rca_rec,
        rca_f1,
    )

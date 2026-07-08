#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any


STANDARD_SCORE_KEYS = (
    "rubric_score",
    "failure_tags",
    "first_serious_flaw",
    "critique",
    "confidence",
    "notes",
)

AUDIT_STATUS_DEFAULTS = {
    "final_answer_status": "unknown",
    "central_lemma_status": "unknown",
    "hidden_assumption_status": "unknown",
    "boundary_case_status": "unknown",
    "theorem_use_status": "unknown",
    "algebra_geometry_status": "unknown",
}

AUDIT_OPTIONAL_KEYS = (
    "is_complete",
    "is_salvageable",
    "repair_hint",
    "score_before_caps",
    "score_cap",
    "score_cap_reasons",
    "audit_summary",
    *AUDIT_STATUS_DEFAULTS.keys(),
)

ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "complete", "salvageable"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: Any, default: int = 0) -> int:
    return max(0, min(10, _as_int(value, default)))


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_tags = value
    elif isinstance(value, str):
        raw_tags = re.split(r"[,;\s]+", value)
    else:
        raw_tags = []
    tags = []
    for tag in raw_tags:
        clean = _as_text(tag).strip().lower().replace("-", "_").replace(" ", "_")
        if clean and clean not in tags:
            tags.append(clean)
    return tags


def _add_tag(tags: list[str], tag: str) -> None:
    if tag not in tags:
        tags.append(tag)


def _norm_status(value: Any) -> str:
    text = _as_text(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text or "unknown"


def _contains(text: str, *patterns: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in patterns)


def _has_concrete_disproof(text: str) -> bool:
    lower = text.lower()
    if "counterexample" in lower and not _contains(
        lower,
        "no counterexample",
        "no concrete counterexample",
        "without a counterexample",
        "does not provide a counterexample",
        "cannot identify a counterexample",
    ):
        return True
    return _contains(
        lower,
        "contradicts the problem",
        "contradicts the statement",
        "is impossible",
        "fails for",
        "not true when",
        "does not hold",
        "explicitly false",
    )


def _cap(caps: list[tuple[int, str]], value: int, reason: str, tags: list[str], tag: str) -> None:
    caps.append((max(0, min(10, value)), reason))
    _add_tag(tags, tag)


def normalize_audit_response(raw_obj: dict[str, Any], *, capped: bool = False, strictness: str = "strict") -> dict[str, Any]:
    """Normalize a model judge response and enforce proof-audit score caps.

    The model may still assign the raw score, but explicit audit findings are
    treated as binding caps. This prevents high scores for proofs that admit a
    false central lemma, hidden fatal assumption, invalid theorem use, or bad
    algebra/geometry step.
    """

    balanced = strictness == "balanced"
    score_before_caps = _clamp_score(raw_obj.get("rubric_score", raw_obj.get("score", 0)))
    tags = _normalize_tags(raw_obj.get("failure_tags", []))
    caps: list[tuple[int, str]] = []

    audit_statuses = {
        key: _norm_status(raw_obj.get(key, default))
        for key, default in AUDIT_STATUS_DEFAULTS.items()
    }

    first_serious_flaw = _as_text(raw_obj.get("first_serious_flaw", "")).strip()
    critique = _as_text(raw_obj.get("critique", "")).strip()
    notes = _as_text(raw_obj.get("notes", "")).strip()
    repair_hint = _as_text(raw_obj.get("repair_hint", "")).strip()
    audit_summary = _as_text(raw_obj.get("audit_summary", "")).strip()
    evidence_text = " ".join([first_serious_flaw, critique, notes, audit_summary])
    concrete_disproof = _has_concrete_disproof(evidence_text)

    if capped:
        _cap(caps, 4, "token-capped proof cannot receive high-credit completion", tags, "token_cap")

    final_status = audit_statuses["final_answer_status"]
    if final_status in {"incorrect", "wrong", "false", "contradicts_problem", "statement_mismatch"}:
        _cap(caps, 2, "final answer or theorem statement is wrong", tags, "statement_mismatch")
    elif final_status in {"missing", "not_claimed"}:
        _cap(caps, 6, "proof never clearly states the final answer", tags, "incomplete_proof")

    central_status = audit_statuses["central_lemma_status"]
    if central_status in {"false", "invalid", "fatal", "contradicted"}:
        if balanced and not concrete_disproof:
            _cap(caps, 6, "central lemma was flagged but no concrete disproof was identified", tags, "invalid_lemma")
        else:
            _cap(caps, 2, "central lemma or reduction is false", tags, "invalid_lemma")
    elif central_status in {"unsupported", "unproved", "circular"}:
        _cap(caps, 7 if balanced else 5, "central lemma is unsupported", tags, "invalid_lemma")
    elif central_status in {"missing", "unclear"}:
        _cap(caps, 6, "central proof mechanism is missing or unclear", tags, "incomplete_proof")
    elif central_status in {"minor_gap", "minor"}:
        _cap(caps, 8, "central lemma has a minor gap", tags, "incomplete_proof")

    hidden_status = audit_statuses["hidden_assumption_status"]
    if hidden_status in {"fatal", "false", "invalid"}:
        _cap(caps, 3, "proof relies on a fatal hidden assumption", tags, "hidden_assumption")
    elif hidden_status in {"unsupported", "unjustified"}:
        _cap(caps, 5, "proof relies on an unjustified hidden assumption", tags, "hidden_assumption")
    elif hidden_status in {"minor", "minor_gap"}:
        _cap(caps, 8, "proof has a minor hidden assumption", tags, "hidden_assumption")

    boundary_status = audit_statuses["boundary_case_status"]
    if boundary_status in {"fatal", "false", "invalid"}:
        _cap(caps, 5, "missing boundary case breaks the proof", tags, "edge_case")
    elif boundary_status in {"missing", "incomplete"}:
        _cap(caps, 6, "boundary or equality cases are missing", tags, "edge_case")
    elif boundary_status in {"minor", "minor_gap"}:
        _cap(caps, 8, "minor boundary-case gap", tags, "edge_case")

    theorem_status = audit_statuses["theorem_use_status"]
    if theorem_status in {"misused", "false", "invalid", "too_advanced"}:
        _cap(caps, 4, "theorem is misused, false in context, or not allowed", tags, "unsupported_theorem")
    elif theorem_status in {"unsupported", "unproved"}:
        _cap(caps, 5, "theorem use is unsupported", tags, "unsupported_theorem")

    algebra_status = audit_statuses["algebra_geometry_status"]
    if algebra_status in {"fatal", "false", "invalid"}:
        _cap(caps, 3, "invalid algebraic or geometric step breaks the proof", tags, "calculation_error")
    elif algebra_status in {"invalid_step", "wrong_constant", "wrong_sign"}:
        _cap(caps, 5, "important algebraic or geometric step is invalid", tags, "calculation_error")
    elif algebra_status in {"minor_error", "minor"}:
        _cap(caps, 8, "minor algebraic or geometric error", tags, "calculation_error")

    if "invalid_lemma" in tags:
        if _contains(evidence_text, "central", "main lemma", "key lemma", "main claim", "core lemma"):
            _cap(
                caps,
                2 if concrete_disproof else 6,
                "failure tag says the central lemma is invalid" if concrete_disproof else "central lemma was flagged but no concrete disproof was identified",
                tags,
                "invalid_lemma",
            )
        else:
            _cap(caps, 7 if balanced else 5, "failure tag says a lemma is invalid", tags, "invalid_lemma")
    if "circular_reasoning" in tags:
        _cap(caps, 4, "circular reasoning cannot receive high credit", tags, "circular_reasoning")
    if "unsupported_theorem" in tags:
        _cap(caps, 5, "unsupported theorem use cannot receive high credit", tags, "unsupported_theorem")
    if "statement_mismatch" in tags:
        _cap(caps, 2, "statement mismatch cannot receive high credit", tags, "statement_mismatch")

    if _contains(evidence_text, "false") and _contains(
        evidence_text,
        "central lemma",
        "main lemma",
        "key lemma",
        "central claim",
        "main claim",
        "core argument",
    ):
        _cap(
            caps,
            2 if concrete_disproof else 6,
            "audit text identifies a false central lemma" if concrete_disproof else "audit text flags the central lemma without a concrete disproof",
            tags,
            "invalid_lemma",
        )
    if _contains(evidence_text, "invalid") and _contains(
        evidence_text,
        "central lemma",
        "main lemma",
        "key lemma",
        "reduction",
        "core argument",
    ):
        _cap(
            caps,
            3 if concrete_disproof else 6,
            "audit text identifies an invalid central reduction or argument" if concrete_disproof else "audit text flags the central reduction without a concrete disproof",
            tags,
            "invalid_lemma",
        )

    judge_cap_raw = raw_obj.get("score_cap")
    if judge_cap_raw is not None:
        judge_cap = _clamp_score(judge_cap_raw, default=10)
        if balanced and judge_cap < 5 and final_status not in {"incorrect", "wrong", "false", "contradicts_problem", "statement_mismatch"} and not concrete_disproof:
            judge_cap = 5
        if judge_cap < 10:
            _cap(caps, judge_cap, "judge-supplied score cap", tags, "audit_score_cap")

    score_cap = min([10, *(value for value, _ in caps)])
    score = min(score_before_caps, score_cap)
    cap_reasons = []
    for value, reason in caps:
        if value == score_cap and reason not in cap_reasons:
            cap_reasons.append(reason)

    if score < 10:
        tags = [tag for tag in tags if tag != "none"]
    if not tags:
        tags = ["none"] if score == 10 else ["incomplete_proof"]

    confidence = _as_text(raw_obj.get("confidence", "low")).strip().lower()
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "low"

    if score < score_before_caps:
        cap_note = "strict_audit_caps=" + "; ".join(cap_reasons)
        notes = f"{notes} {cap_note}".strip() if notes else cap_note

    if score < 10 and not first_serious_flaw:
        first_serious_flaw = "The proof did not pass the strict proof audit."
    if score < 10 and not critique:
        critique = first_serious_flaw

    is_complete = _as_bool(raw_obj.get("is_complete", False)) or score_before_caps >= 9
    if score < 9:
        is_complete = False
    is_salvageable = _as_bool(raw_obj.get("is_salvageable", False)) or 3 <= score <= 8
    if score <= 2:
        is_salvageable = False

    result: dict[str, Any] = {
        "rubric_score": score,
        "failure_tags": tags,
        "first_serious_flaw": first_serious_flaw,
        "critique": critique,
        "confidence": confidence,
        "notes": notes,
        "is_complete": is_complete,
        "is_salvageable": is_salvageable,
        "repair_hint": repair_hint,
        "score_before_caps": score_before_caps,
        "score_cap": score_cap,
        "score_cap_reasons": cap_reasons,
        "audit_summary": audit_summary,
    }
    result.update(audit_statuses)
    return result


def score_fields(score_obj: dict[str, Any], *, include_audit: bool = True) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rubric_score": int(score_obj.get("rubric_score", 0)),
        "failure_tags": list(score_obj.get("failure_tags", [])),
        "first_serious_flaw": _as_text(score_obj.get("first_serious_flaw", "")),
        "critique": _as_text(score_obj.get("critique", "")),
        "confidence": _as_text(score_obj.get("confidence", "low")),
        "notes": _as_text(score_obj.get("notes", "")),
    }
    if include_audit:
        for key in AUDIT_OPTIONAL_KEYS:
            if key in score_obj:
                fields[key] = score_obj[key]
    return fields

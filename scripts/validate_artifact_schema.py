#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEACHER_FIELDS = {
    "id",
    "source",
    "year",
    "label",
    "split",
    "target_excluded",
    "solution_used",
    "problem",
    "weak_hint",
    "medium_hint",
    "strong_hint",
    "plan",
    "full_proof",
    "flawed_proof",
    "critique",
    "repaired_proof",
    "rubric_score",
    "failure_tags",
    "solution_used",
    "generated_by",
    "notes",
}

TEACHER_OPTIONAL_FIELDS = {
    "proof_direct_exclude_reason",
}

PROBLEM_INPUT_FIELDS = {
    "id",
    "source",
    "year",
    "label",
    "split",
    "target_excluded",
    "solution_used",
    "problem",
    "solution_used",
    "notes",
}

PROBLEM_INPUT_OPTIONAL_FIELDS = {
    "problem_source_url",
    "reference_solution",
    "reference_solution_source",
    "reference_solution_url",
}

PROBLEM_INPUT_STRING_FIELDS = {
    "id",
    "source",
    "label",
    "split",
    "problem",
    "notes",
}

TEACHER_STRING_FIELDS = {
    "id",
    "source",
    "label",
    "split",
    "problem",
    "weak_hint",
    "medium_hint",
    "strong_hint",
    "plan",
    "full_proof",
    "flawed_proof",
    "critique",
    "repaired_proof",
    "notes",
}

TEACHER_OPTIONAL_STRING_FIELDS = {
    "proof_direct_exclude_reason",
}

SFT_FIELDS = {
    "id",
    "source_id",
    "task",
    "source",
    "year",
    "label",
    "split",
    "target_excluded",
    "solution_used",
    "teacher_generated_by",
    "messages",
}

EVAL_PROMPT_FIELDS = {
    "prompt_id",
    "run_id",
    "problem_id",
    "source",
    "year",
    "label",
    "attempt_index",
    "model",
    "status",
    "created_at",
    "problem",
    "messages",
}

EVAL_PROMPT_OPTIONAL_FIELDS = {
    "input_path",
    "solver_prompt_path",
    "solver_prompt_sha256",
}

EVAL_ATTEMPT_FIELDS = {
    "attempt_id",
    "prompt_id",
    "run_id",
    "problem_id",
    "source",
    "year",
    "label",
    "attempt_index",
    "model",
    "created_at",
    "problem",
    "status",
    "proof",
    "error",
}

EVAL_ATTEMPT_OPTIONAL_FIELDS = {
    "latency_seconds",
    "usage",
}

EVAL_SCORE_FIELDS = {
    "attempt_id",
    "run_id",
    "problem_id",
    "judge_model",
    "created_at",
    "status",
    "rubric_score",
    "failure_tags",
    "first_serious_flaw",
    "critique",
    "confidence",
    "notes",
    "error",
}

EVAL_SCORE_OPTIONAL_FIELDS = {
    "raw_judge_response",
    "latency_seconds",
    "usage",
    "is_complete",
    "is_salvageable",
    "repair_hint",
    "score_before_caps",
    "score_cap",
    "score_cap_reasons",
    "audit_summary",
    "final_answer_status",
    "central_lemma_status",
    "hidden_assumption_status",
    "boundary_case_status",
    "theorem_use_status",
    "algebra_geometry_status",
}

HOPED_MI_FIELDS = {
    "id",
    "schema_version",
    "source_id",
    "source",
    "year",
    "label",
    "split",
    "target_excluded",
    "solution_used",
    "problem",
    "student_model",
    "teacher_model",
    "repair_model",
    "judge_model",
    "created_at",
    "attempt_index",
    "student_attempt",
    "student_judge",
    "first_serious_flaw",
    "failure_type",
    "minimal_hint",
    "hint_verification",
    "student_retry_after_hint",
    "retry_judge",
    "retry_quality_issues",
    "repaired_proof",
    "repair_judge",
    "repair_quality_issues",
    "accepted",
    "accepted_stage",
    "acceptance_checks",
    "error",
    "collection_flow",
    "input_policy",
}

GENERATED_BY_KEYS = {"hints", "plan", "proof", "flawed_proof", "critique", "repair", "rubric"}
SFT_TASKS = {"hint_ladder", "plan", "proof_direct", "proof_with_hints", "critique", "repair", "rubric"}
SPLITS = {"train", "dev", "dev_gate", "hidden_dev", "test", "target"}
ATTEMPT_STATUSES = {"dry_run", "completed", "error"}
SCORE_STATUSES = {"dry_run", "skipped_empty_proof", "completed", "error"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
HOPED_MI_ACCEPTED_STAGES = {"", "student_retry", "teacher_repair"}
ACCEPTED_FINAL_STATUSES = {"correct", "not_applicable"}
ACCEPTED_CENTRAL_STATUSES = {"valid", "not_applicable"}
ACCEPTED_HIDDEN_STATUSES = {"none", "not_applicable"}
ACCEPTED_BOUNDARY_STATUSES = {"complete", "not_applicable"}
ACCEPTED_THEOREM_STATUSES = {"valid", "not_applicable"}
ACCEPTED_ALGEBRA_STATUSES = {"valid", "not_applicable"}


@dataclass
class Issue:
    severity: str
    path: Path
    line_no: int
    row_id: str
    field: str
    message: str


def issue(issues: list[Issue], severity: str, path: Path, line_no: int, row_id: str, field: str, message: str) -> None:
    issues.append(Issue(severity, path, line_no, row_id, field, message))


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def read_jsonl(path: Path, issues: list[Issue]) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    if not path.exists():
        issue(issues, "error", path, 0, "file", "path", "file does not exist")
        return rows
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            issue(issues, "error", path, line_no, "json", "json", f"invalid JSON: {exc.msg}")
            continue
        if not isinstance(obj, dict):
            issue(issues, "error", path, line_no, "json", "row", "row must be a JSON object")
            continue
        rows.append((line_no, obj))
    if not rows:
        issue(issues, "error", path, 0, "file", "rows", "file has no JSONL rows")
    return rows


def read_json_object(path: Path, issues: list[Issue]) -> dict:
    if not path.exists():
        issue(issues, "error", path, 0, "file", "path", "file does not exist")
        return {}
    try:
        obj = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        issue(issues, "error", path, 1, "json", "json", f"invalid JSON: {exc.msg}")
        return {}
    if not isinstance(obj, dict):
        issue(issues, "error", path, 1, "json", "row", "top-level JSON must be an object")
        return {}
    return obj


def row_id(obj: dict, fallback: str = "unknown") -> str:
    for key in ("id", "prompt_id", "attempt_id", "problem_id", "source_id"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def check_required(path: Path, line_no: int, obj: dict, required: set[str], issues: list[Issue]) -> None:
    rid = row_id(obj, f"line_{line_no}")
    for field in sorted(required):
        if field not in obj:
            issue(issues, "error", path, line_no, rid, field, "missing required field")


def check_allowed_fields(path: Path, line_no: int, obj: dict, allowed: set[str], allow_extra: bool, issues: list[Issue]) -> None:
    if allow_extra:
        return
    rid = row_id(obj, f"line_{line_no}")
    extra = sorted(set(obj) - allowed)
    for field in extra:
        issue(issues, "error", path, line_no, rid, field, "unexpected field")


def check_string(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue], *, allow_empty: bool = False) -> None:
    if field not in obj:
        return
    value = obj[field]
    rid = row_id(obj, f"line_{line_no}")
    if not isinstance(value, str):
        issue(issues, "error", path, line_no, rid, field, "must be a string")
    elif not allow_empty and not value.strip():
        issue(issues, "error", path, line_no, rid, field, "must be nonempty")


def check_int(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue], *, minimum: int | None = None, maximum: int | None = None) -> None:
    if field not in obj:
        return
    value = obj[field]
    rid = row_id(obj, f"line_{line_no}")
    if not is_int(value):
        issue(issues, "error", path, line_no, rid, field, "must be an integer")
        return
    if minimum is not None and value < minimum:
        issue(issues, "error", path, line_no, rid, field, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        issue(issues, "error", path, line_no, rid, field, f"must be <= {maximum}")


def check_bool(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue]) -> None:
    if field not in obj:
        return
    if not isinstance(obj[field], bool):
        issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), field, "must be a boolean")


def check_number(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue], *, minimum: float | None = None) -> None:
    if field not in obj:
        return
    value = obj[field]
    rid = row_id(obj, f"line_{line_no}")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        issue(issues, "error", path, line_no, rid, field, "must be a number")
        return
    if minimum is not None and value < minimum:
        issue(issues, "error", path, line_no, rid, field, f"must be >= {minimum}")


def check_object(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue]) -> None:
    if field in obj and not isinstance(obj[field], dict):
        issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), field, "must be an object")


def check_list(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue]) -> None:
    if field in obj and not isinstance(obj[field], list):
        issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), field, "must be a list")


def check_enum(path: Path, line_no: int, obj: dict, field: str, allowed: set[str], issues: list[Issue]) -> None:
    if field not in obj or not isinstance(obj[field], str):
        return
    if obj[field] not in allowed:
        issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), field, f"must be one of {sorted(allowed)}")


def check_iso_timestamp(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue]) -> None:
    if field not in obj or not isinstance(obj[field], str):
        return
    value = obj[field]
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), field, "must be an ISO timestamp")


def check_failure_tags(path: Path, line_no: int, obj: dict, issues: list[Issue]) -> None:
    if "failure_tags" not in obj:
        return
    tags = obj["failure_tags"]
    rid = row_id(obj, f"line_{line_no}")
    if not isinstance(tags, list):
        issue(issues, "error", path, line_no, rid, "failure_tags", "must be a list")
        return
    for index, tag in enumerate(tags):
        if not is_nonempty_string(tag):
            issue(issues, "error", path, line_no, rid, f"failure_tags[{index}]", "must be a nonempty string")


def norm_status(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace("-", "_").replace(" ", "_") or "unknown"


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "resolved", "minimal"}
    return bool(value)


def audit_score(value: dict[str, Any]) -> int | None:
    raw = value.get("score", value.get("rubric_score"))
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def check_audit_object(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue], *, required: bool) -> None:
    rid = row_id(obj, f"line_{line_no}")
    value = obj.get(field)
    if value in (None, {}):
        if required:
            issue(issues, "error", path, line_no, rid, field, "must be a nonempty audit object")
        return
    if not isinstance(value, dict):
        issue(issues, "error", path, line_no, rid, field, "must be an object")
        return
    score = audit_score(value)
    if score is None or not 0 <= score <= 10:
        issue(issues, "error", path, line_no, rid, f"{field}.score", "must be integer 0-10")
    for key in ("is_complete", "is_salvageable"):
        if key in value and not isinstance(value[key], bool):
            issue(issues, "error", path, line_no, rid, f"{field}.{key}", "must be a boolean")
    for key in (
        "first_serious_flaw",
        "critique",
        "final_answer_status",
        "central_lemma_status",
        "hidden_assumption_status",
        "boundary_case_status",
        "theorem_use_status",
        "algebra_geometry_status",
        "audit_summary",
        "original_flaw_resolution_notes",
    ):
        if key in value and not isinstance(value[key], str):
            issue(issues, "error", path, line_no, rid, f"{field}.{key}", "must be a string")
    if "failure_tags" in value:
        tags = value["failure_tags"]
        if not isinstance(tags, list) or not all(is_nonempty_string(tag) for tag in tags):
            issue(issues, "error", path, line_no, rid, f"{field}.failure_tags", "must be a list of nonempty strings")
    if "original_flaw_resolved" in value and not isinstance(value["original_flaw_resolved"], bool):
        issue(issues, "error", path, line_no, rid, f"{field}.original_flaw_resolved", "must be a boolean")


def check_quality_issues(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue]) -> None:
    rid = row_id(obj, f"line_{line_no}")
    value = obj.get(field)
    if value is None:
        return
    if not isinstance(value, list):
        issue(issues, "error", path, line_no, rid, field, "must be a list")
        return
    for index, item in enumerate(value):
        if not is_nonempty_string(item):
            issue(issues, "error", path, line_no, rid, f"{field}[{index}]", "must be a nonempty string")


def check_hint_verification(path: Path, line_no: int, obj: dict, issues: list[Issue], *, required: bool) -> None:
    rid = row_id(obj, f"line_{line_no}")
    value = obj.get("hint_verification")
    if value in (None, {}):
        if required:
            issue(issues, "error", path, line_no, rid, "hint_verification", "must be a nonempty object")
        return
    if not isinstance(value, dict):
        issue(issues, "error", path, line_no, rid, "hint_verification", "must be an object")
        return
    for key in ("is_minimal", "reveals_solution", "introduces_unrelated_strategy", "targets_first_flaw", "too_broad"):
        if not isinstance(value.get(key), bool):
            issue(issues, "error", path, line_no, rid, f"hint_verification.{key}", "must be a boolean")
    if "reason" in value and not isinstance(value["reason"], str):
        issue(issues, "error", path, line_no, rid, "hint_verification.reason", "must be a string")


def hint_verification_passes(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool_value(value.get("is_minimal"))
        and bool_value(value.get("targets_first_flaw"))
        and not bool_value(value.get("reveals_solution"))
        and not bool_value(value.get("introduces_unrelated_strategy"))
        and not bool_value(value.get("too_broad"))
    )


def audit_is_clean_complete(value: Any, *, min_score: int) -> bool:
    if not isinstance(value, dict):
        return False
    score = audit_score(value)
    return (
        score is not None
        and score >= min_score
        and bool_value(value.get("is_complete"))
        and norm_status(value.get("final_answer_status")) in ACCEPTED_FINAL_STATUSES
        and norm_status(value.get("central_lemma_status")) in ACCEPTED_CENTRAL_STATUSES
        and norm_status(value.get("hidden_assumption_status")) in ACCEPTED_HIDDEN_STATUSES
        and norm_status(value.get("boundary_case_status")) in ACCEPTED_BOUNDARY_STATUSES
        and norm_status(value.get("theorem_use_status")) in ACCEPTED_THEOREM_STATUSES
        and norm_status(value.get("algebra_geometry_status")) in ACCEPTED_ALGEBRA_STATUSES
        and bool_value(value.get("original_flaw_resolved"))
    )


def check_acceptance_checks(path: Path, line_no: int, obj: dict, issues: list[Issue], *, stage: str) -> None:
    rid = row_id(obj, f"line_{line_no}")
    value = obj.get("acceptance_checks")
    if not isinstance(value, dict):
        issue(issues, "error", path, line_no, rid, "acceptance_checks", "must be an object")
        return
    checks = value if stage == "student_retry" else value.get("teacher_repair")
    if stage == "student_retry" and "student_retry" in value:
        checks = value["student_retry"]
    if not isinstance(checks, dict):
        issue(issues, "error", path, line_no, rid, "acceptance_checks", f"missing {stage} acceptance checks")
        return
    for key in (
        "score_at_least_threshold",
        "is_complete",
        "final_answer_clean",
        "central_lemma_clean",
        "hidden_assumptions_clean",
        "boundary_cases_clean",
        "theorem_use_clean",
        "algebra_geometry_clean",
        "original_flaw_resolved",
        "quality_gate_passed",
        "accepted",
    ):
        if not isinstance(checks.get(key), bool):
            issue(issues, "error", path, line_no, rid, f"acceptance_checks.{stage}.{key}", "must be a boolean")
        elif checks.get(key) is not True:
            issue(issues, "error", path, line_no, rid, f"acceptance_checks.{stage}.{key}", "accepted rows require true")


def check_generated_by(path: Path, line_no: int, obj: dict, field: str, issues: list[Issue]) -> None:
    if field not in obj:
        return
    value = obj[field]
    rid = row_id(obj, f"line_{line_no}")
    if not isinstance(value, dict):
        issue(issues, "error", path, line_no, rid, field, "must be an object")
        return
    keys = set(value)
    if keys != GENERATED_BY_KEYS:
        missing = sorted(GENERATED_BY_KEYS - keys)
        extra = sorted(keys - GENERATED_BY_KEYS)
        issue(issues, "error", path, line_no, rid, field, f"keys mismatch; missing={missing} extra={extra}")
    for key, item in value.items():
        if not is_nonempty_string(item):
            issue(issues, "error", path, line_no, rid, f"{field}.{key}", "must be a nonempty string")


def check_target_guard(path: Path, line_no: int, obj: dict, issues: list[Issue], allow_target: bool) -> None:
    if allow_target:
        return
    rid = row_id(obj, f"line_{line_no}")
    if obj.get("year") == 2025:
        issue(issues, "error", path, line_no, rid, "year", "Putnam 2025 target rows are blocked unless --allow-target is passed")
    if obj.get("split") == "target":
        issue(issues, "error", path, line_no, rid, "split", "target split is blocked unless --allow-target is passed")
    if obj.get("target_excluded") is True:
        issue(issues, "error", path, line_no, rid, "target_excluded", "target-excluded rows are blocked unless --allow-target is passed")


def check_unique(path: Path, rows: list[tuple[int, dict]], key: str, issues: list[Issue]) -> None:
    seen: dict[str, int] = {}
    for line_no, obj in rows:
        value = obj.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), key, f"duplicate value first seen on line {seen[value]}")
        else:
            seen[value] = line_no


def check_messages(path: Path, line_no: int, obj: dict, roles: list[str] | list[list[str]], issues: list[Issue]) -> None:
    if "messages" not in obj:
        return
    rid = row_id(obj, f"line_{line_no}")
    messages = obj["messages"]
    if not isinstance(messages, list):
        issue(issues, "error", path, line_no, rid, "messages", "must be a list")
        return
    role_sequences: list[list[str]]
    if roles and isinstance(roles[0], list):
        role_sequences = roles  # type: ignore[assignment]
    else:
        role_sequences = [roles]  # type: ignore[list-item]
    expected_roles: list[str] | None = None
    for sequence in role_sequences:
        if len(messages) == len(sequence):
            expected_roles = sequence
            break
    if expected_roles is None:
        expected_lengths = sorted({len(sequence) for sequence in role_sequences})
        issue(issues, "error", path, line_no, rid, "messages", f"must contain one of these message counts: {expected_lengths}")
        return
    for index, expected_role in enumerate(expected_roles):
        message = messages[index]
        field = f"messages[{index}]"
        if not isinstance(message, dict):
            issue(issues, "error", path, line_no, rid, field, "must be an object")
            continue
        extra = sorted(set(message) - {"role", "content"})
        missing = sorted({"role", "content"} - set(message))
        if extra or missing:
            issue(issues, "error", path, line_no, rid, field, f"keys mismatch; missing={missing} extra={extra}")
        if message.get("role") != expected_role:
            issue(issues, "error", path, line_no, rid, f"{field}.role", f"must be {expected_role!r}")
        if not is_nonempty_string(message.get("content")):
            issue(issues, "error", path, line_no, rid, f"{field}.content", "must be a nonempty string")


def index_source(path: Path, key: str, issues: list[Issue]) -> dict[str, dict]:
    rows = read_jsonl(path, issues)
    index: dict[str, dict] = {}
    for line_no, obj in rows:
        value = obj.get(key)
        if not is_nonempty_string(value):
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), key, "source row is missing key")
            continue
        if value in index:
            issue(issues, "error", path, line_no, value, key, "duplicate source key")
        index[value] = obj
    return index


def compare_field(path: Path, line_no: int, obj: dict, source_obj: dict, field: str, issues: list[Issue]) -> None:
    if field in obj and field in source_obj and obj[field] != source_obj[field]:
        issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), field, "differs from source row")


def validate_teacher(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    source = index_source(args.source, "id", issues) if args.source else {}
    allowed = TEACHER_FIELDS | TEACHER_OPTIONAL_FIELDS
    check_unique(path, rows, "id", issues)
    for line_no, obj in rows:
        check_required(path, line_no, obj, TEACHER_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, allowed, args.allow_extra_fields, issues)
        for field in TEACHER_STRING_FIELDS:
            check_string(path, line_no, obj, field, issues)
        for field in TEACHER_OPTIONAL_STRING_FIELDS:
            check_string(path, line_no, obj, field, issues)
        check_int(path, line_no, obj, "year", issues)
        check_int(path, line_no, obj, "rubric_score", issues, minimum=0, maximum=7)
        check_bool(path, line_no, obj, "target_excluded", issues)
        check_bool(path, line_no, obj, "solution_used", issues)
        check_enum(path, line_no, obj, "split", SPLITS, issues)
        check_failure_tags(path, line_no, obj, issues)
        check_generated_by(path, line_no, obj, "generated_by", issues)
        check_target_guard(path, line_no, obj, issues, args.allow_target)
        if source:
            source_obj = source.get(obj.get("id", ""))
            if not source_obj:
                issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "id", "not found in source file")
            else:
                compare_field(path, line_no, obj, source_obj, "problem", issues)


def validate_problem_inputs(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    allowed = PROBLEM_INPUT_FIELDS | PROBLEM_INPUT_OPTIONAL_FIELDS
    check_unique(path, rows, "id", issues)
    for line_no, obj in rows:
        check_required(path, line_no, obj, PROBLEM_INPUT_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, allowed, args.allow_extra_fields, issues)
        for field in PROBLEM_INPUT_STRING_FIELDS:
            check_string(path, line_no, obj, field, issues)
        for field in PROBLEM_INPUT_OPTIONAL_FIELDS:
            check_string(path, line_no, obj, field, issues)
        check_int(path, line_no, obj, "year", issues)
        check_bool(path, line_no, obj, "target_excluded", issues)
        check_bool(path, line_no, obj, "solution_used", issues)
        check_enum(path, line_no, obj, "split", SPLITS, issues)
        check_target_guard(path, line_no, obj, issues, args.allow_target)
        rid = row_id(obj, f"line_{line_no}")
        if obj.get("solution_used") is True and not args.allow_solution_used:
            issue(issues, "error", path, line_no, rid, "solution_used", "solution-guided rows require --allow-solution-used")
        if args.require_reference_solution and not is_nonempty_string(obj.get("reference_solution")):
            issue(issues, "error", path, line_no, rid, "reference_solution", "missing reference solution")
        if obj.get("solution_used") is False and "reference_solution" in obj:
            issue(issues, "warning", path, line_no, rid, "reference_solution", "present even though solution_used is false")


def validate_sft(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    source = index_source(args.source, "id", issues) if args.source else {}
    check_unique(path, rows, "id", issues)
    for line_no, obj in rows:
        check_required(path, line_no, obj, SFT_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, SFT_FIELDS, args.allow_extra_fields, issues)
        for field in ("id", "source_id", "task", "source", "label", "split"):
            check_string(path, line_no, obj, field, issues)
        check_int(path, line_no, obj, "year", issues)
        check_bool(path, line_no, obj, "target_excluded", issues)
        check_bool(path, line_no, obj, "solution_used", issues)
        check_enum(path, line_no, obj, "task", SFT_TASKS, issues)
        check_enum(path, line_no, obj, "split", SPLITS, issues)
        check_generated_by(path, line_no, obj, "teacher_generated_by", issues)
        check_messages(path, line_no, obj, [["system", "user", "assistant"], ["system", "developer", "user", "assistant"]], issues)
        check_target_guard(path, line_no, obj, issues, args.allow_target)

        task = obj.get("task")
        source_id = obj.get("source_id")
        if isinstance(source_id, str) and isinstance(task, str) and obj.get("id") != f"{source_id}::{task}":
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "id", "must equal source_id::task")
        if task == "hint_ladder":
            check_assistant_json(path, line_no, obj, {"weak_hint", "medium_hint", "strong_hint"}, issues)
        if task == "rubric":
            check_assistant_json(path, line_no, obj, {"rubric_score", "failure_tags", "notes"}, issues)
        if source:
            source_obj = source.get(str(source_id))
            if not source_obj:
                issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "source_id", "not found in source teacher file")
            else:
                for field in ("source", "year", "label", "split", "target_excluded", "solution_used"):
                    compare_field(path, line_no, obj, source_obj, field, issues)
                if obj.get("teacher_generated_by") != source_obj.get("generated_by"):
                    issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "teacher_generated_by", "differs from source generated_by")


def validate_hoped_mi(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    source = index_source(args.source, "id", issues) if args.source else {}
    check_unique(path, rows, "id", issues)
    for line_no, obj in rows:
        rid = row_id(obj, f"line_{line_no}")
        check_required(path, line_no, obj, HOPED_MI_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, HOPED_MI_FIELDS, args.allow_extra_fields, issues)
        for field in (
            "id",
            "schema_version",
            "source_id",
            "source",
            "label",
            "split",
            "problem",
            "student_model",
            "teacher_model",
            "repair_model",
            "judge_model",
            "created_at",
            "student_attempt",
            "first_serious_flaw",
            "failure_type",
            "student_retry_after_hint",
            "repaired_proof",
            "accepted_stage",
            "error",
            "collection_flow",
            "input_policy",
        ):
            check_string(
                path,
                line_no,
                obj,
                field,
                issues,
                allow_empty=field in {"first_serious_flaw", "student_retry_after_hint", "repaired_proof", "accepted_stage", "error"},
            )
        check_int(path, line_no, obj, "year", issues)
        check_int(path, line_no, obj, "attempt_index", issues, minimum=1)
        check_bool(path, line_no, obj, "target_excluded", issues)
        check_bool(path, line_no, obj, "solution_used", issues)
        check_bool(path, line_no, obj, "accepted", issues)
        check_enum(path, line_no, obj, "split", SPLITS, issues)
        check_enum(path, line_no, obj, "accepted_stage", HOPED_MI_ACCEPTED_STAGES, issues)
        check_iso_timestamp(path, line_no, obj, "created_at", issues)
        check_object(path, line_no, obj, "student_judge", issues)
        check_object(path, line_no, obj, "minimal_hint", issues)
        check_object(path, line_no, obj, "hint_verification", issues)
        check_object(path, line_no, obj, "retry_judge", issues)
        check_object(path, line_no, obj, "repair_judge", issues)
        check_object(path, line_no, obj, "acceptance_checks", issues)
        check_quality_issues(path, line_no, obj, "retry_quality_issues", issues)
        check_quality_issues(path, line_no, obj, "repair_quality_issues", issues)
        check_target_guard(path, line_no, obj, issues, args.allow_target)

        if obj.get("schema_version") != "hoped_mi.v2":
            issue(issues, "error", path, line_no, rid, "schema_version", "must be hoped_mi.v2")
        if isinstance(obj.get("source_id"), str) and obj.get("id") != f"{obj.get('source_id')}::hoped_mi::{obj.get('attempt_index')}":
            issue(issues, "error", path, line_no, rid, "id", "must equal source_id::hoped_mi::attempt_index")
        if obj.get("accepted") is True and not obj.get("accepted_stage"):
            issue(issues, "error", path, line_no, rid, "accepted_stage", "accepted rows must identify student_retry or teacher_repair")
        if obj.get("accepted") is False and obj.get("accepted_stage"):
            issue(issues, "error", path, line_no, rid, "accepted_stage", "rejected rows must leave accepted_stage empty")
        if obj.get("accepted") is False and not is_nonempty_string(obj.get("error")):
            issue(issues, "error", path, line_no, rid, "error", "rejected rows must include an error reason")

        student_judge = obj.get("student_judge")
        check_audit_object(path, line_no, obj, "student_judge", issues, required=True)
        if isinstance(student_judge, dict):
            score = audit_score(student_judge)
            if obj.get("accepted") is True and score is not None and score >= 8:
                issue(issues, "error", path, line_no, rid, "student_judge.score", "HOPED-MI recovery rows require a flawed initial student attempt")

        hint = obj.get("minimal_hint")
        reached_recovery_stage = (
            obj.get("accepted") is True
            or (isinstance(hint, dict) and bool(hint))
            or is_nonempty_string(obj.get("student_retry_after_hint"))
            or is_nonempty_string(obj.get("repaired_proof"))
        )
        if reached_recovery_stage and not is_nonempty_string(obj.get("first_serious_flaw")):
            issue(issues, "error", path, line_no, rid, "first_serious_flaw", "must identify the first serious flaw")
        if isinstance(hint, dict) and bool(hint):
            if not is_nonempty_string(hint.get("minimal_hint")):
                issue(issues, "error", path, line_no, rid, "minimal_hint.minimal_hint", "must be nonempty")
            if not is_nonempty_string(hint.get("hint_type")):
                issue(issues, "error", path, line_no, rid, "minimal_hint.hint_type", "must be nonempty")
        check_hint_verification(path, line_no, obj, issues, required=isinstance(hint, dict) and bool(hint))
        if obj.get("accepted") is True and not hint_verification_passes(obj.get("hint_verification")):
            issue(issues, "error", path, line_no, rid, "hint_verification", "accepted rows require a verified minimal hint")

        stage = obj.get("accepted_stage")
        if stage == "student_retry":
            if not is_nonempty_string(obj.get("student_retry_after_hint")):
                issue(issues, "error", path, line_no, rid, "student_retry_after_hint", "accepted student_retry rows require retry proof")
            check_audit_object(path, line_no, obj, "retry_judge", issues, required=True)
            if not audit_is_clean_complete(obj.get("retry_judge"), min_score=9):
                issue(issues, "error", path, line_no, rid, "retry_judge", "accepted student retry must be complete, clean, >=9, and resolve original flaw")
            if obj.get("retry_quality_issues"):
                issue(issues, "error", path, line_no, rid, "retry_quality_issues", "accepted student retry rows must have no anti-sketch quality issues")
            check_acceptance_checks(path, line_no, obj, issues, stage="student_retry")
        elif stage == "teacher_repair":
            if not is_nonempty_string(obj.get("student_retry_after_hint")):
                issue(issues, "error", path, line_no, rid, "student_retry_after_hint", "teacher repair rows must include the failed student retry")
            check_audit_object(path, line_no, obj, "retry_judge", issues, required=True)
            if not is_nonempty_string(obj.get("repaired_proof")):
                issue(issues, "error", path, line_no, rid, "repaired_proof", "accepted teacher_repair rows require repaired proof")
            check_audit_object(path, line_no, obj, "repair_judge", issues, required=True)
            if not audit_is_clean_complete(obj.get("repair_judge"), min_score=9):
                issue(issues, "error", path, line_no, rid, "repair_judge", "accepted teacher repair must be complete, clean, >=9, and resolve original flaw")
            if obj.get("repair_quality_issues"):
                issue(issues, "error", path, line_no, rid, "repair_quality_issues", "accepted teacher repair rows must have no anti-sketch quality issues")
            check_acceptance_checks(path, line_no, obj, issues, stage="teacher_repair")
        else:
            check_audit_object(path, line_no, obj, "retry_judge", issues, required=False)
            check_audit_object(path, line_no, obj, "repair_judge", issues, required=False)

        if source:
            source_obj = source.get(str(obj.get("source_id")))
            if not source_obj:
                issue(issues, "error", path, line_no, rid, "source_id", "not found in source problem file")
            else:
                for field in ("source", "year", "label", "split", "target_excluded", "solution_used", "problem"):
                    compare_field(path, line_no, obj, source_obj, field, issues)


def check_assistant_json(path: Path, line_no: int, obj: dict, required: set[str], issues: list[Issue]) -> None:
    rid = row_id(obj, f"line_{line_no}")
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return
    assistant_message = next(
        (message for message in reversed(messages) if isinstance(message, dict) and message.get("role") == "assistant"),
        None,
    )
    if not assistant_message:
        return
    content = assistant_message.get("content")
    if not isinstance(content, str):
        return
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        issue(issues, "error", path, line_no, rid, "messages[2].content", f"assistant content must be JSON: {exc.msg}")
        return
    if not isinstance(parsed, dict):
        issue(issues, "error", path, line_no, rid, "messages[2].content", "assistant JSON must be an object")
        return
    missing = sorted(required - set(parsed))
    extra = sorted(set(parsed) - required)
    if missing or extra:
        issue(issues, "error", path, line_no, rid, "messages[2].content", f"assistant JSON keys mismatch; missing={missing} extra={extra}")
    if "rubric_score" in parsed and (not is_int(parsed["rubric_score"]) or not 0 <= parsed["rubric_score"] <= 7):
        issue(issues, "error", path, line_no, rid, "messages[2].content.rubric_score", "must be integer 0-7")
    if "failure_tags" in parsed:
        if not isinstance(parsed["failure_tags"], list) or not all(is_nonempty_string(tag) for tag in parsed["failure_tags"]):
            issue(issues, "error", path, line_no, rid, "messages[2].content.failure_tags", "must be a list of nonempty strings")
    for key, value in parsed.items():
        if key not in {"rubric_score", "failure_tags"} and not is_nonempty_string(value):
            issue(issues, "error", path, line_no, rid, f"messages[2].content.{key}", "must be a nonempty string")


def validate_eval_prompts(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    source = index_source(args.source, "id", issues) if args.source else {}
    allowed = EVAL_PROMPT_FIELDS | EVAL_PROMPT_OPTIONAL_FIELDS
    check_unique(path, rows, "prompt_id", issues)
    for line_no, obj in rows:
        check_required(path, line_no, obj, EVAL_PROMPT_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, allowed, args.allow_extra_fields, issues)
        for field in ("prompt_id", "run_id", "problem_id", "source", "label", "model", "status", "created_at", "problem", "input_path", "solver_prompt_path", "solver_prompt_sha256"):
            check_string(path, line_no, obj, field, issues)
        check_int(path, line_no, obj, "year", issues)
        check_int(path, line_no, obj, "attempt_index", issues, minimum=1)
        check_enum(path, line_no, obj, "status", {"prepared"}, issues)
        check_iso_timestamp(path, line_no, obj, "created_at", issues)
        check_messages(path, line_no, obj, [["system", "user"], ["system", "developer", "user"]], issues)
        if is_int(obj.get("attempt_index")):
            expected = f"{obj.get('run_id')}::{obj.get('problem_id')}::attempt_{obj.get('attempt_index')}"
            if obj.get("prompt_id") != expected:
                issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "prompt_id", f"must equal {expected}")
        if not args.allow_target and obj.get("year") == 2025:
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "year", "Putnam 2025 prompts are blocked unless --allow-target is passed")
        if source:
            source_obj = source.get(str(obj.get("problem_id")))
            if not source_obj:
                issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "problem_id", "not found in source problem file")
            else:
                for field in ("source", "year", "label", "problem"):
                    compare_field(path, line_no, obj, source_obj, field, issues)


def validate_eval_attempts(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    source = index_source(args.source, "prompt_id", issues) if args.source else {}
    allowed = EVAL_ATTEMPT_FIELDS | EVAL_ATTEMPT_OPTIONAL_FIELDS
    check_unique(path, rows, "attempt_id", issues)
    for line_no, obj in rows:
        check_required(path, line_no, obj, EVAL_ATTEMPT_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, allowed, args.allow_extra_fields, issues)
        for field in ("attempt_id", "prompt_id", "run_id", "problem_id", "source", "label", "model", "created_at", "problem", "status", "proof", "error"):
            check_string(path, line_no, obj, field, issues, allow_empty=field in {"proof", "error"})
        check_int(path, line_no, obj, "year", issues)
        check_int(path, line_no, obj, "attempt_index", issues, minimum=1)
        check_number(path, line_no, obj, "latency_seconds", issues, minimum=0)
        check_object(path, line_no, obj, "usage", issues)
        check_enum(path, line_no, obj, "status", ATTEMPT_STATUSES, issues)
        check_iso_timestamp(path, line_no, obj, "created_at", issues)
        if obj.get("attempt_id") != obj.get("prompt_id"):
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "attempt_id", "must equal prompt_id")
        if obj.get("status") == "completed" and not is_nonempty_string(obj.get("proof")):
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "proof", "completed attempts must have a proof")
        if obj.get("status") == "error" and not is_nonempty_string(obj.get("error")):
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "error", "error attempts must include error text")
        if not args.allow_target and obj.get("year") == 2025:
            issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "year", "Putnam 2025 attempts are blocked unless --allow-target is passed")
        if source:
            source_obj = source.get(str(obj.get("prompt_id")))
            if not source_obj:
                issue(issues, "error", path, line_no, row_id(obj, f"line_{line_no}"), "prompt_id", "not found in source prompt file")
            else:
                for field in ("run_id", "problem_id", "source", "year", "label", "attempt_index", "problem"):
                    compare_field(path, line_no, obj, source_obj, field, issues)


def validate_eval_scores(path: Path, rows: list[tuple[int, dict]], issues: list[Issue], args) -> None:
    source = index_source(args.source, "attempt_id", issues) if args.source else {}
    allowed = EVAL_SCORE_FIELDS | EVAL_SCORE_OPTIONAL_FIELDS
    check_unique(path, rows, "attempt_id", issues)
    for line_no, obj in rows:
        check_required(path, line_no, obj, EVAL_SCORE_FIELDS, issues)
        check_allowed_fields(path, line_no, obj, allowed, args.allow_extra_fields, issues)
        for field in ("attempt_id", "run_id", "problem_id", "judge_model", "created_at", "status", "first_serious_flaw", "critique", "confidence", "notes", "error", "raw_judge_response"):
            check_string(path, line_no, obj, field, issues, allow_empty=field in {"first_serious_flaw", "critique", "notes", "error", "raw_judge_response"})
        check_number(path, line_no, obj, "latency_seconds", issues, minimum=0)
        check_object(path, line_no, obj, "usage", issues)
        check_enum(path, line_no, obj, "status", SCORE_STATUSES, issues)
        check_enum(path, line_no, obj, "confidence", CONFIDENCE_VALUES, issues)
        check_iso_timestamp(path, line_no, obj, "created_at", issues)
        check_failure_tags(path, line_no, obj, issues)
        for field in ("is_complete", "is_salvageable"):
            check_bool(path, line_no, obj, field, issues)
        for field in (
            "repair_hint",
            "audit_summary",
            "final_answer_status",
            "central_lemma_status",
            "hidden_assumption_status",
            "boundary_case_status",
            "theorem_use_status",
            "algebra_geometry_status",
        ):
            check_string(path, line_no, obj, field, issues, allow_empty=field in {"repair_hint", "audit_summary"})
        for field in ("score_before_caps", "score_cap"):
            check_int(path, line_no, obj, field, issues, minimum=0, maximum=10)
        if "score_cap_reasons" in obj:
            reasons = obj["score_cap_reasons"]
            rid = row_id(obj, f"line_{line_no}")
            if not isinstance(reasons, list):
                issue(issues, "error", path, line_no, rid, "score_cap_reasons", "must be a list")
            else:
                for index, reason in enumerate(reasons):
                    if not is_nonempty_string(reason):
                        issue(issues, "error", path, line_no, rid, f"score_cap_reasons[{index}]", "must be a nonempty string")
        rid = row_id(obj, f"line_{line_no}")
        if obj.get("status") == "completed":
            check_int(path, line_no, obj, "rubric_score", issues, minimum=0, maximum=10)
        elif "rubric_score" in obj and obj["rubric_score"] is not None:
            issue(issues, "error", path, line_no, rid, "rubric_score", "non-completed score rows must use null")
        if obj.get("status") == "error" and not is_nonempty_string(obj.get("error")):
            issue(issues, "error", path, line_no, rid, "error", "error score rows must include error text")
        if source:
            source_obj = source.get(str(obj.get("attempt_id")))
            if not source_obj:
                issue(issues, "error", path, line_no, rid, "attempt_id", "not found in source attempt file")
            else:
                for field in ("run_id", "problem_id"):
                    compare_field(path, line_no, obj, source_obj, field, issues)


def validate_eval_summary(path: Path, issues: list[Issue]) -> None:
    obj = read_json_object(path, issues)
    if not obj:
        return
    required = {"scores_path", "rows", "completed_scores", "mean_score", "median_score", "failure_tag_counts"}
    optional = {
        "status_counts",
        "score_histogram",
        "per_problem",
        "score_scale",
        "putnam_problem_count",
        "scored_problem_count",
        "putnam_total_score",
        "putnam_max_score",
        "putnam_total_score_out_of_120",
        "putnam_total_score_scaled_to_120",
        "putnam_aggregation",
    }
    allowed = required | optional
    for field in sorted(required):
        if field not in obj:
            issue(issues, "error", path, 1, "summary", field, "missing required field")
    extra = sorted(set(obj) - allowed)
    for field in extra:
        issue(issues, "error", path, 1, "summary", field, "unexpected field")
    for field in ("scores_path",):
        if field in obj and not is_nonempty_string(obj[field]):
            issue(issues, "error", path, 1, "summary", field, "must be a nonempty string")
    for field in ("rows", "completed_scores"):
        if field in obj and not is_int(obj[field]):
            issue(issues, "error", path, 1, "summary", field, "must be an integer")
    for field in ("putnam_problem_count", "scored_problem_count", "putnam_total_score", "putnam_max_score"):
        value = obj.get(field)
        if value is not None and (not is_int(value) or value < 0):
            issue(issues, "error", path, 1, "summary", field, "must be a nonnegative integer or null")
    for field in ("putnam_total_score_out_of_120", "putnam_total_score_scaled_to_120"):
        value = obj.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
            issue(issues, "error", path, 1, "summary", field, "must be a nonnegative number or null")
    if "putnam_aggregation" in obj and not is_nonempty_string(obj["putnam_aggregation"]):
        issue(issues, "error", path, 1, "summary", "putnam_aggregation", "must be a nonempty string")
    if "score_scale" in obj:
        scale = obj["score_scale"]
        if not isinstance(scale, dict) or scale.get("min") != 0 or scale.get("max") != 10:
            issue(issues, "error", path, 1, "summary", "score_scale", "must be {'min': 0, 'max': 10}")
    if is_int(obj.get("rows")) and is_int(obj.get("completed_scores")) and obj["completed_scores"] > obj["rows"]:
        issue(issues, "error", path, 1, "summary", "completed_scores", "cannot exceed rows")
    for field in ("mean_score", "median_score"):
        value = obj.get(field)
        if value is not None and not isinstance(value, (int, float)):
            issue(issues, "error", path, 1, "summary", field, "must be a number or null")
    counts = obj.get("failure_tag_counts")
    if not isinstance(counts, dict):
        issue(issues, "error", path, 1, "summary", "failure_tag_counts", "must be an object")
    else:
        for key, value in counts.items():
            if not is_nonempty_string(key) or not is_int(value) or value < 0:
                issue(issues, "error", path, 1, "summary", "failure_tag_counts", "must map nonempty strings to nonnegative integers")
    for field in ("status_counts", "score_histogram"):
        if field not in obj:
            continue
        value = obj[field]
        if not isinstance(value, dict):
            issue(issues, "error", path, 1, "summary", field, "must be an object")
            continue
        for key, count in value.items():
            if not is_nonempty_string(key) or not is_int(count) or count < 0:
                issue(issues, "error", path, 1, "summary", field, "must map nonempty strings to nonnegative integers")
    if "per_problem" in obj:
        per_problem = obj["per_problem"]
        if not isinstance(per_problem, list):
            issue(issues, "error", path, 1, "summary", "per_problem", "must be a list")
        else:
            required_problem_fields = {"problem_id", "attempts_scored", "mean_score", "best_score", "median_score"}
            for index, item in enumerate(per_problem):
                if not isinstance(item, dict):
                    issue(issues, "error", path, 1, "summary", f"per_problem[{index}]", "must be an object")
                    continue
                missing = sorted(required_problem_fields - set(item))
                extra = sorted(set(item) - required_problem_fields)
                if missing or extra:
                    issue(issues, "error", path, 1, "summary", f"per_problem[{index}]", f"keys mismatch; missing={missing} extra={extra}")


def print_issues(issues: list[Issue]) -> None:
    for item in issues:
        location = f"{item.path}:{item.line_no}" if item.line_no else str(item.path)
        print(f"{item.severity.upper()} {location} {item.row_id} {item.field}: {item.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict schema validator for Phronesis JSONL artifacts.")
    parser.add_argument("kind", choices=["problem-inputs", "teacher", "sft", "hoped-mi", "eval-prompts", "eval-attempts", "eval-scores", "eval-summary"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", type=Path, default=None, help="Optional parent artifact for cross-file consistency checks.")
    parser.add_argument("--allow-target", action="store_true", help="Allow Putnam 2025 or target-split rows.")
    parser.add_argument("--allow-solution-used", action="store_true", help="Allow solution-guided problem input rows.")
    parser.add_argument("--require-reference-solution", action="store_true", help="Require reference_solution on problem input rows.")
    parser.add_argument("--allow-extra-fields", action="store_true", help="Permit fields outside the strict schema.")
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    issues: list[Issue] = []
    row_count = 0

    if args.kind == "eval-summary":
        validate_eval_summary(args.path, issues)
    else:
        rows = read_jsonl(args.path, issues)
        row_count = len(rows)
        if args.kind == "problem-inputs":
            validate_problem_inputs(args.path, rows, issues, args)
        elif args.kind == "teacher":
            validate_teacher(args.path, rows, issues, args)
        elif args.kind == "sft":
            validate_sft(args.path, rows, issues, args)
        elif args.kind == "hoped-mi":
            validate_hoped_mi(args.path, rows, issues, args)
        elif args.kind == "eval-prompts":
            validate_eval_prompts(args.path, rows, issues, args)
        elif args.kind == "eval-attempts":
            validate_eval_attempts(args.path, rows, issues, args)
        elif args.kind == "eval-scores":
            validate_eval_scores(args.path, rows, issues, args)

    errors = [item for item in issues if item.severity == "error"]
    warnings = [item for item in issues if item.severity == "warning"]
    print(f"validated kind={args.kind} path={args.path} rows={row_count} errors={len(errors)} warnings={len(warnings)}")
    print_issues(issues)

    if errors or (args.fail_on_warnings and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

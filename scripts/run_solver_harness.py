#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_eval_harness import (
    DEFAULT_BASE_URL,
    DEFAULT_JUDGE_PROMPT,
    DEFAULT_SOLVER_PROMPT,
    build_summary,
    extract_final_channel_text,
    is_target_problem,
    make_chat_messages,
    make_solver_messages,
    model_uses_harmony,
    parse_judge_json,
    problem_rows,
    selected_problem,
)
from strict_proof_auditor import normalize_audit_response, score_fields


VALIDATOR = Path(__file__).with_name("validate_artifact_schema.py")

DEFAULT_STRATEGIES: list[dict[str, str]] = [
    {
        "id": "direct",
        "name": "Direct proof",
        "instruction": "Try the most direct rigorous solution. State the main invariant, equation, or construction early.",
    },
    {
        "id": "invariant",
        "name": "Invariant and obstruction search",
        "instruction": "Look for invariants, congruences, monotonic quantities, parity, extremal obstructions, or conserved structures before constructing the proof.",
    },
    {
        "id": "constructive",
        "name": "Constructive proof",
        "instruction": "Prioritize explicit constructions, examples, equality cases, and sharp lower bounds. Prove both attainability and optimality when relevant.",
    },
    {
        "id": "contradiction",
        "name": "Contradiction proof",
        "instruction": "Attempt a contradiction or minimal-counterexample proof. Identify the earliest impossible condition and prove it carefully.",
    },
    {
        "id": "structural",
        "name": "Structural reduction",
        "instruction": "Reduce the problem to a cleaner algebraic, combinatorial, geometric, or analytic structure. Prove the reduction, then solve the reduced problem.",
    },
    {
        "id": "case_split",
        "name": "Case split and edge cases",
        "instruction": "Search for all relevant cases and edge conditions. Make the proof robust against missing boundary cases.",
    },
    {
        "id": "calculation",
        "name": "Calculation audit",
        "instruction": "If computations are involved, derive every transformation explicitly and verify constants, signs, and limiting steps.",
    },
    {
        "id": "alternate",
        "name": "Alternative viewpoint",
        "instruction": "Use a different viewpoint from the obvious one, such as duality, generating functions, coordinates, graph structure, or an equivalent reformulation.",
    },
]


REPAIR_SYSTEM = """You repair Putnam-style proof attempts.

Write a rigorous, self-contained proof of the original problem. Use the critique and repair hint only to fix the proof. Do not mention the judge, critique, repair hint, prior attempt, scoring, or this prompt.

Rules:
- If the prior approach is salvageable, repair it with the smallest necessary mathematical changes.
- If the prior approach is not salvageable, write a clean corrected proof from the right starting point.
- Do not cite obscure theorems or papers.
- Do not skip computations.
- Return only the proof text.
"""


CONSOLIDATE_SYSTEM = """You consolidate proof attempts into one final Putnam-style proof.

You will receive the original problem and several judged candidate proofs. Write the strongest rigorous final proof. Use only correct ideas. Discard false lemmas, wrong constants, invalid cases, and unsupported reductions.

Rules:
- Begin immediately with the final proof. Do not spend the answer planning or comparing candidates.
- Do not mention candidates, judges, scores, critiques, or this prompt.
- The final proof must be self-contained.
- If none of the candidates are valid, write the best corrected proof you can and explicitly avoid uncertain claims.
- Return only the final proof text.
"""


CONCLUSION_SYSTEM = """You extract the final mathematical conclusion from a Putnam-style proof attempt.

Return exactly one JSON object and no other text:
{
  "final_answer": "",
  "conclusion": "",
  "normalized_conclusion": "",
  "confidence": "low"
}

Rules:
- Extract what the proof actually concludes, not what should be true.
- For classification problems, normalized_conclusion should be a compact set description.
- For numerical answers, normalized_conclusion should contain only the expression/value when possible.
- If no stable final conclusion is present, set confidence to low and normalized_conclusion to "unknown".
"""


BREAKER_SYSTEM = """You are an adversarial Putnam proof breaker.

Try to invalidate the submitted proof. Be stricter than a normal grader: look for a false central lemma, hidden assumption, missing boundary case, theorem misuse, or invalid algebra/geometry step. Do not invent objections; if the proof is valid, say so.

Return exactly one JSON object and no other text:
{
  "verdict": "valid",
  "fatal_flaw": "",
  "failure_tags": [],
  "final_answer_status": "unknown",
  "central_lemma_status": "unknown",
  "hidden_assumption_status": "unknown",
  "boundary_case_status": "unknown",
  "theorem_use_status": "unknown",
  "algebra_geometry_status": "unknown",
  "score_cap": 10,
  "confidence": "low",
  "notes": ""
}

Use verdict = "invalid" only when you find a concrete fatal or major flaw. Use verdict = "uncertain" when the proof has suspicious gaps but you cannot identify a precise fatal flaw.
"""


TOURNAMENT_SYSTEM = """You are a strict pairwise Putnam proof judge.

Compare two proof attempts for the same problem. Choose the proof that would receive the higher human Putnam score. Prefer a shorter proof only if mathematical correctness is tied. A proof with a false central lemma must lose to a proof with a valid complete argument, even if the false proof has a correct final answer.

Return exactly one JSON object and no other text:
{
  "winner": "tie",
  "a_score": 0,
  "b_score": 0,
  "reason": "",
  "confidence": "low"
}

winner must be exactly one of "A", "B", or "tie". Scores are integers from 0 to 10.
"""


POLISH_SYSTEM = """You polish an already-valid Putnam proof.

Rewrite the proof for clarity and compactness without changing the mathematical argument or final conclusion. Do not introduce new lemmas, new solution routes, or unsupported claims. If a step is uncertain, preserve the original wording rather than inventing a fix.

Return only the polished proof text.
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def model_is_qwen_thinking(model: str) -> bool:
    normalized = model.lower()
    return "qwen" in normalized and "thinking" in normalized


def split_qwen_thinking_output(
    *,
    model: str,
    text: str,
    response_metadata: dict[str, Any],
    capped: bool,
) -> tuple[str, dict[str, Any]]:
    """Keep Qwen Thinking scratch work out of the judged proof field."""
    if not model_is_qwen_thinking(model):
        return text, response_metadata
    metadata = dict(response_metadata)
    marker = "</think>"
    if marker in text:
        reasoning, proof = text.rsplit(marker, 1)
        existing_reasoning = str(metadata.get("reasoning_content", "") or "")
        reasoning_parts = [part.strip() for part in (existing_reasoning, reasoning) if part and part.strip()]
        combined_reasoning = "\n\n".join(reasoning_parts)
        metadata["reasoning_content"] = combined_reasoning
        metadata["reasoning_content_chars"] = len(combined_reasoning)
        return proof.strip(), metadata
    if capped:
        existing_reasoning = str(metadata.get("reasoning_content", "") or "")
        reasoning_parts = [part.strip() for part in (existing_reasoning, text) if part and part.strip()]
        combined_reasoning = "\n\n".join(reasoning_parts)
        metadata["reasoning_content"] = combined_reasoning
        metadata["reasoning_content_chars"] = len(combined_reasoning)
        return "", metadata
    return text, metadata


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def read_optional_json_object(path: str) -> dict[str, Any]:
    if not path:
        return {}
    answer_path = Path(path)
    if not answer_path.exists():
        raise SystemExit(f"{path}: expected JSON file to exist")
    return read_json(answer_path)


def parse_json_object_arg(value: str, name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name}: invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name}: expected a JSON object")
    return parsed


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_no}: row must be a JSON object")
            yield line_no, obj


def load_jsonl_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line_no, row in iter_jsonl(path):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"{path}:{line_no}: existing row missing nonempty {key}")
        if value in rows:
            raise SystemExit(f"{path}:{line_no}: duplicate {key}={value}")
        rows[value] = row
    return rows


def reset_path(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        path.unlink()


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return parse_judge_json(text)
    except Exception:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("JSON response was not an object")
        return parsed


def extract_json_object_with_reasoning_fallback(
    raw: str,
    response_metadata: dict[str, Any],
    *,
    extract_final: bool,
    model: str,
) -> dict[str, Any]:
    candidates = [extract_final_channel_text(raw) if extract_final or model_uses_harmony(model) else raw, raw]
    reasoning = str(response_metadata.get("reasoning_content", "") or "")
    if reasoning:
        candidates.append(reasoning)
    last_error: Exception | None = None
    for candidate in candidates:
        if not str(candidate).strip():
            continue
        try:
            return extract_json_object(str(candidate))
        except Exception as exc:
            last_error = exc
    raise ValueError("model response did not contain a JSON object") from last_error


def truncate_for_consolidation(proof: str, *, capped: bool) -> str:
    limit = 2600 if capped else 4200
    if len(proof) <= limit:
        return proof
    head_len = int(limit * 0.7)
    tail_len = limit - head_len
    marker = "\n\n[Proof attempt truncated by harness because it was too long or token-capped. Use only reliable ideas from it.]\n\n"
    return proof[:head_len].rstrip() + marker + proof[-tail_len:].lstrip()


def normalize_judge_score(raw_obj: dict[str, Any], *, capped: bool = False) -> dict[str, Any]:
    return normalize_audit_response(raw_obj, capped=capped, strictness="balanced")


def strict_score_fields(score_obj: dict[str, Any]) -> dict[str, Any]:
    return score_fields(score_obj, include_audit=True)


def normalize_conclusion_key(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9+\-*/^=<>.,:{}()[\] ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:240] or "unknown"


def heuristic_conclusion(proof: str) -> dict[str, str]:
    text = proof.strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        answer = boxed[-1].strip()
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        answer = next((line for line in reversed(lines) if any(word in line.lower() for word in ("answer", "therefore", "hence", "thus"))), "")
    key = normalize_conclusion_key(answer)
    return {
        "final_answer": answer,
        "conclusion": answer,
        "normalized_conclusion": key,
        "confidence": "medium" if key != "unknown" else "low",
    }


def answer_key_text(proof: str) -> str:
    conclusion = heuristic_conclusion(proof)
    lines = [line.strip() for line in proof.splitlines() if line.strip()]
    keyword_lines = [
        line
        for line in lines[-24:]
        if any(
            word in line.lower()
            for word in (
                "answer",
                "therefore",
                "hence",
                "thus",
                "conclude",
                "limit",
                "maximum",
                "probability",
                "values of",
            )
        )
    ]
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", proof)
    trailing_lines = lines[-3:]
    fragments = [
        conclusion.get("final_answer", ""),
        conclusion.get("conclusion", ""),
        conclusion.get("normalized_conclusion", ""),
        *boxed[-3:],
        *keyword_lines[-6:],
        *trailing_lines,
    ]
    return "\n".join(
        str(part or "")
        for part in fragments
    )


def expected_answer_gate(
    score: dict[str, Any],
    *,
    problem_id: str,
    proof: str,
    expected_answers: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    spec = expected_answers.get(problem_id)
    if not isinstance(spec, dict):
        return score, None
    answer_text = answer_key_text(proof)
    required_any = [str(pattern) for pattern in spec.get("required_any", []) if str(pattern).strip()]
    forbidden_any = [str(pattern) for pattern in spec.get("forbidden_any", []) if str(pattern).strip()]
    flags = re.IGNORECASE | re.DOTALL
    if forbidden_any and any(re.search(pattern, answer_text, flags) for pattern in forbidden_any):
        out = add_score_cap(score, 2, "forbidden wrong final answer found", "statement_mismatch")
        out["final_answer_status"] = "incorrect"
        out["is_complete"] = False
        return normalize_judge_score(out), "expected_answer_forbidden"
    if required_any and not any(re.search(pattern, answer_text, flags) for pattern in required_any):
        out = dict(score)
        tags = normalized_failure_tags(out.get("failure_tags", []))
        if "answer_key_uncertain" not in tags:
            tags.append("answer_key_uncertain")
        out["failure_tags"] = [tag for tag in tags if tag != "none"] or ["answer_key_uncertain"]
        out["answer_key_status"] = "required_pattern_not_detected"
        out["answer_key_description"] = str(spec.get("description", ""))
        try:
            raw_score = int(out.get("rubric_score", 0) or 0)
        except (TypeError, ValueError):
            raw_score = 0
        if raw_score >= 9:
            out = add_score_cap(out, 8, "expected final answer not confidently detected", "answer_key_uncertain")
        return normalize_judge_score(out), "expected_answer_missing_soft"
    return score, None


def failure_focus(score: dict[str, Any], *, capped: bool = False) -> str:
    tags = {str(tag).strip().lower() for tag in score.get("failure_tags", []) if str(tag).strip()}
    central_status = str(score.get("central_lemma_status", "")).lower()
    theorem_status = str(score.get("theorem_use_status", "")).lower()
    algebra_status = str(score.get("algebra_geometry_status", "")).lower()
    final_status = str(score.get("final_answer_status", "")).lower()
    if capped or "token_cap" in tags:
        return "token_cap_loop"
    if "statement_mismatch" in tags or final_status in {"incorrect", "wrong", "false"}:
        return "wrong_final_answer"
    if "invalid_lemma" in tags or central_status in {"false", "invalid", "fatal", "unsupported"}:
        return "central_lemma"
    if "hidden_assumption" in tags or str(score.get("hidden_assumption_status", "")).lower() in {"fatal", "unsupported"}:
        return "hidden_assumption"
    if "unsupported_theorem" in tags or theorem_status in {"unsupported", "misused", "too_advanced", "invalid"}:
        return "theorem_misuse"
    if "calculation_error" in tags or algebra_status in {"invalid_step", "fatal", "false", "minor_error"}:
        return "calculation_audit"
    if "edge_case" in tags or str(score.get("boundary_case_status", "")).lower() in {"missing", "fatal"}:
        return "boundary_cases"
    return "general_retry"


def retry_instruction_for_focus(focus: str) -> str:
    instructions = {
        "token_cap_loop": "Write a shorter proof. Do not enumerate many failed constructions. State the key invariant or construction, prove it, and stop.",
        "wrong_final_answer": "Start by independently deriving the final answer or classification. Check the result against small cases before writing the proof.",
        "central_lemma": "Identify the central lemma first and prove it before using it. Do not reuse any lemma from the failed attempt unless you can prove it rigorously.",
        "hidden_assumption": "Find and eliminate hidden assumptions. Justify every without-loss, extremal, generic, independence, and divisibility claim.",
        "theorem_misuse": "Avoid obscure or misapplied theorems. Use only standard results and prove any specialized lemma directly.",
        "calculation_audit": "Audit all algebra, geometry, determinants, limits, congruences, signs, constants, and equality cases before concluding.",
        "boundary_cases": "Handle all boundary, equality, endpoint, small-value, degenerate, parity, and zero cases explicitly.",
        "general_retry": "Use a different proof route and make the main mathematical mechanism explicit.",
    }
    return instructions.get(focus, instructions["general_retry"])


def normalized_failure_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.split(r"[,;]+", value)
    else:
        raw = []
    tags: list[str] = []
    for item in raw:
        tag = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        if tag and tag not in tags:
            tags.append(tag)
    return tags


BAD_FINAL_ANSWER_STATUSES = {
    "incorrect",
    "wrong",
    "false",
    "contradicts_problem",
    "statement_mismatch",
}

FALSE_CENTRAL_LEMMA_STATUSES = {
    "false",
    "invalid",
    "fatal",
    "contradicted",
}

WEAK_CENTRAL_LEMMA_STATUSES = {
    "unsupported",
    "unproved",
    "circular",
}


def normalize_status(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")


def score_evidence_text(score: dict[str, Any]) -> str:
    return " ".join(
        str(score.get(key, "") or "")
        for key in ("first_serious_flaw", "critique", "notes", "audit_summary")
    ).lower()


def score_has_wrong_final_answer(score: dict[str, Any]) -> bool:
    tags = set(normalized_failure_tags(score.get("failure_tags", [])))
    if "statement_mismatch" in tags or "wrong_final_answer" in tags:
        return True
    if normalize_status(score.get("final_answer_status")) in BAD_FINAL_ANSWER_STATUSES:
        return True
    evidence = score_evidence_text(score)
    return any(phrase in evidence for phrase in ("wrong final answer", "incorrect final answer", "final answer is wrong"))


def score_has_false_central_lemma(score: dict[str, Any]) -> bool:
    tags = set(normalized_failure_tags(score.get("failure_tags", [])))
    evidence = score_evidence_text(score)
    counterexample_is_concrete = "counterexample" in evidence and not any(
        phrase in evidence
        for phrase in (
            "no counterexample",
            "no concrete counterexample",
            "without a counterexample",
            "does not provide a counterexample",
            "cannot identify a counterexample",
        )
    )
    disproof_markers = (
        "counterexample",
        "contradicts the problem",
        "contradicts the statement",
        "is impossible",
        "fails for",
        "not true when",
        "does not hold",
        "explicitly false",
    )
    has_concrete_disproof = counterexample_is_concrete or any(marker in evidence for marker in disproof_markers if marker != "counterexample")
    central_status = normalize_status(score.get("central_lemma_status"))
    if central_status in {"false", "fatal"} and has_concrete_disproof:
        return True
    if ("false central lemma" in evidence or "false main lemma" in evidence) and has_concrete_disproof:
        return True
    if ("invalid central" in evidence or "invalid main lemma" in evidence) and has_concrete_disproof:
        return True
    return "invalid_lemma" in tags and has_concrete_disproof and any(word in evidence for word in ("central", "main lemma", "key lemma", "core"))


def score_has_central_lemma_issue(score: dict[str, Any]) -> bool:
    tags = set(normalized_failure_tags(score.get("failure_tags", [])))
    central_status = normalize_status(score.get("central_lemma_status"))
    if central_status in FALSE_CENTRAL_LEMMA_STATUSES | WEAK_CENTRAL_LEMMA_STATUSES:
        return True
    evidence = score_evidence_text(score)
    return "invalid_lemma" in tags and any(word in evidence for word in ("central", "main lemma", "key lemma", "core"))


def score_has_unverified_central_lemma(score: dict[str, Any]) -> bool:
    return normalize_status(score.get("central_lemma_status")) in WEAK_CENTRAL_LEMMA_STATUSES


def add_score_cap(score: dict[str, Any], cap: int, reason: str, tag: str) -> dict[str, Any]:
    out = dict(score)
    tags = normalized_failure_tags(out.get("failure_tags", []))
    if tag and tag not in tags:
        tags.append(tag)
    out["failure_tags"] = [tag for tag in tags if tag != "none"] or ["none"]
    try:
        current_score = int(out.get("rubric_score", 0) or 0)
    except (TypeError, ValueError):
        current_score = 0
    try:
        current_cap = int(out.get("score_cap", 10) or 10)
    except (TypeError, ValueError):
        current_cap = 10
    cap = max(0, min(10, int(cap)))
    out["score_cap"] = min(current_cap, cap)
    out["rubric_score"] = min(current_score, cap)
    reasons = list(out.get("score_cap_reasons", []) or [])
    if reason not in reasons:
        reasons.append(reason)
    out["score_cap_reasons"] = reasons
    notes = str(out.get("notes", "") or "")
    cap_note = f"harness_gate={reason}"
    if cap_note not in notes:
        out["notes"] = f"{notes} {cap_note}".strip() if notes else cap_note
    if out["rubric_score"] < 9:
        out["is_complete"] = False
    if out["rubric_score"] <= 2:
        out["is_salvageable"] = False
    if out["rubric_score"] < 10 and not str(out.get("first_serious_flaw", "")).strip():
        out["first_serious_flaw"] = reason
    if out["rubric_score"] < 10 and not str(out.get("critique", "")).strip():
        out["critique"] = str(out.get("first_serious_flaw", reason))
    return normalize_judge_score(out)


def safe_eval_integer_expr(expr: str) -> int | None:
    clean = expr.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*").replace("−", "-")
    clean = clean.replace("{", "").replace("}", "")
    clean = re.sub(r"\s+", "", clean)
    if not re.fullmatch(r"[+\-]?\d+(?:[+\-*][+\-]?\d+){1,12}", clean):
        return None
    try:
        return int(eval(clean, {"__builtins__": {}}, {}))
    except Exception:
        return None


def parse_numeric_literal(value: str) -> float | None:
    clean = value.replace(",", "").strip()
    if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", clean):
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def numeric_comparison_holds(left: float, operator: str, right: float) -> bool:
    op = operator.lower().strip()
    if op in {"<", "less than", "smaller than"}:
        return left < right
    if op in {">", "greater than", "larger than"}:
        return left > right
    if op in {"<=", "\\le", "\\leq", "at most", "no more than"}:
        return left <= right
    if op in {">=", "\\ge", "\\geq", "at least", "no less than"}:
        return left >= right
    return True


def arithmetic_sanity_check(proof: str) -> dict[str, Any]:
    """Catch cheap arithmetic contradictions before a proof can be selected.

    This is deliberately conservative. It only checks explicit scalar
    equalities, simple numeric comparisons, and simple 2D vector
    additions/scalar combinations that appear in the proof text. It is not a
    symbolic proof verifier.
    """

    text = proof.replace("\u2212", "-")
    errors: list[str] = []

    scalar_pattern = re.compile(
        r"(?<![A-Za-z0-9_^+\-*/])(-?\d+(?:\s*(?:\+|-|\*|\\cdot|\\times|×)\s*-?\d+){1,12})\s*=\s*([+\-]?\d+)(?![A-Za-z0-9_])"
    )
    for match in scalar_pattern.finditer(text):
        expr, rhs_text = match.groups()
        value = safe_eval_integer_expr(expr)
        if value is None:
            continue
        rhs = int(rhs_text)
        if value != rhs:
            errors.append(f"{expr.strip()} = {rhs} is false; left side is {value}")
            if len(errors) >= 5:
                break

    comparison_pattern = re.compile(
        r"(?<![A-Za-z0-9_.])([+\-]?\d[\d,]*(?:\.\d+)?)"
        r"(?:\s*,?\s*(?:which|that|this|it|is|still|certainly|clearly|also))*"
        r"\s*(<|>|<=|>=|\\leq?|\\geq?|less than|greater than|smaller than|larger than|at most|at least|no more than|no less than)"
        r"\s*([+\-]?\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    for match in comparison_pattern.finditer(text):
        left_text, operator, right_text = match.groups()
        left = parse_numeric_literal(left_text)
        right = parse_numeric_literal(right_text)
        if left is None or right is None:
            continue
        if not numeric_comparison_holds(left, operator, right):
            errors.append(f"{left_text.strip()} {operator.strip()} {right_text.strip()} is false")
            if len(errors) >= 5:
                break

    vector_add_pattern = re.compile(
        r"\(([+\-]?\d+),\s*([+\-]?\d+)\)\s*\+\s*\(([+\-]?\d+),\s*([+\-]?\d+)\)\s*=\s*\(([+\-]?\d+),\s*([+\-]?\d+)\)"
    )
    for match in vector_add_pattern.finditer(text):
        x1, y1, x2, y2, xr, yr = map(int, match.groups())
        if (x1 + x2, y1 + y2) != (xr, yr):
            errors.append(f"({x1},{y1}) + ({x2},{y2}) = ({xr},{yr}) is false; sum is ({x1 + x2},{y1 + y2})")
            if len(errors) >= 5:
                break

    vector_linear_pattern = re.compile(
        r"([+\-]?\d+)\s*\(([+\-]?\d+),\s*([+\-]?\d+)\)\s*\+\s*([+\-]?\d+)\s*\(([+\-]?\d+),\s*([+\-]?\d+)\)\s*=\s*\(([+\-]?\d+),\s*([+\-]?\d+)\)"
    )
    for match in vector_linear_pattern.finditer(text):
        a, x1, y1, b, x2, y2, xr, yr = map(int, match.groups())
        value = (a * x1 + b * x2, a * y1 + b * y2)
        if value != (xr, yr):
            errors.append(f"{a}({x1},{y1}) + {b}({x2},{y2}) = ({xr},{yr}) is false; sum is {value}")
            if len(errors) >= 5:
                break

    return {
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "errors": errors,
    }


def breaker_has_clean_valid_verdict(breaker: dict[str, Any]) -> bool:
    tags = [tag for tag in normalized_failure_tags(breaker.get("failure_tags", [])) if tag != "none"]
    if tags or str(breaker.get("fatal_flaw", "")).strip():
        return False
    bad_statuses = {
        "incorrect",
        "wrong",
        "false",
        "invalid",
        "fatal",
        "contradicted",
        "contradicts_problem",
        "unsupported",
        "unproved",
        "circular",
        "misused",
        "too_advanced",
        "missing",
        "incomplete",
        "invalid_step",
        "wrong_constant",
        "wrong_sign",
    }
    for key in (
        "final_answer_status",
        "central_lemma_status",
        "hidden_assumption_status",
        "boundary_case_status",
        "theorem_use_status",
        "algebra_geometry_status",
    ):
        status = str(breaker.get(key, "unknown")).strip().lower().replace("-", "_").replace(" ", "_")
        if status in bad_statuses:
            return False
    try:
        return int(breaker.get("score_cap", 10) or 10) >= 10
    except (TypeError, ValueError):
        return True


def breaker_score_cap(breaker: dict[str, Any], verdict: str) -> int:
    default = 10 if verdict == "valid" else 8 if verdict == "uncertain" else 5
    try:
        cap = int(breaker.get("score_cap", default) or default)
    except (TypeError, ValueError):
        cap = default
    cap = max(0, min(10, cap))
    if verdict == "invalid":
        pseudo_score = {
            "failure_tags": breaker.get("failure_tags", []),
            "final_answer_status": breaker.get("final_answer_status", "unknown"),
            "central_lemma_status": breaker.get("central_lemma_status", "unknown"),
            "first_serious_flaw": breaker.get("fatal_flaw", ""),
            "critique": breaker.get("notes", ""),
        }
        if score_has_wrong_final_answer(pseudo_score) or score_has_false_central_lemma(pseudo_score):
            return min(cap, 2)
        return min(max(cap, 5), 5)
    if verdict == "uncertain":
        return min(max(cap, 7), 8)
    return cap


def item_has_clean_valid_breaker(item: dict[str, Any]) -> bool:
    breaker = item.get("breaker")
    return isinstance(breaker, dict) and str(breaker.get("verdict", "")).lower() == "valid" and breaker_has_clean_valid_verdict(breaker)


def breaker_identified_repair_flaw(item: dict[str, Any]) -> bool:
    breaker = item.get("breaker")
    if not isinstance(breaker, dict):
        return False
    if str(breaker.get("status", "")) not in {"completed", ""}:
        return False
    verdict = str(breaker.get("verdict", "")).lower()
    flaw = str(breaker.get("fatal_flaw", "") or "").strip()
    if not flaw or flaw.lower().startswith("proof breaker call failed"):
        return False
    return verdict in {"invalid", "uncertain"}


def score_sort_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
    score = int(item.get("score", {}).get("rubric_score", -1))
    confidence_rank = {"low": 0, "medium": 1, "high": 2}.get(str(item.get("score", {}).get("confidence", "low")), 0)
    capped = 1 if item.get("capped") else 0
    proof_len = len(str(item.get("proof", "")))
    return (score, confidence_rank, -capped, proof_len)


def reasoning_sort_key(item: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int]:
    score = int(item.get("score", {}).get("rubric_score", -1))
    clean_breaker = 1 if item_has_clean_valid_breaker(item) else 0
    wins = int(item.get("tournament_wins", 0))
    ties = int(item.get("tournament_ties", 0))
    cluster_support = int(item.get("cluster_support", 1))
    confidence_rank = {"low": 0, "medium": 1, "high": 2}.get(str(item.get("score", {}).get("confidence", "low")), 0)
    capped = 1 if item.get("capped") else 0
    proof_len = len(str(item.get("proof", "")))
    return (score, clean_breaker, cluster_support, wins, ties, confidence_rank, -capped, proof_len)


def selected_problems(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in problem_rows(Path(args.input)):
        if not selected_problem(problem, args):
            continue
        if is_target_problem(problem) and not args.allow_target:
            raise SystemExit(f"{problem['id']}: refusing target problem without --allow-target")
        rows.append(problem)
        if args.max_problems and len(rows) >= args.max_problems:
            break
    if not rows:
        raise SystemExit("no problems selected")
    return rows


def load_strategies(path: str, attempts_per_problem: int) -> list[dict[str, str]]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"{path}: strategies file must be a JSON list")
        strategies = []
        for idx, item in enumerate(data, 1):
            if not isinstance(item, dict):
                raise SystemExit(f"{path}: strategy {idx} must be an object")
            sid = str(item.get("id", f"strategy_{idx}")).strip()
            name = str(item.get("name", sid)).strip()
            instruction = str(item.get("instruction", "")).strip()
            if not sid or not instruction:
                raise SystemExit(f"{path}: strategy {idx} needs id and instruction")
            strategies.append({"id": sid, "name": name, "instruction": instruction})
    else:
        strategies = list(DEFAULT_STRATEGIES)
    if attempts_per_problem <= len(strategies):
        return strategies[:attempts_per_problem]
    expanded = []
    for index in range(attempts_per_problem):
        base = strategies[index % len(strategies)]
        cycle = index // len(strategies) + 1
        expanded.append(
            {
                "id": f"{base['id']}_v{cycle}",
                "name": f"{base['name']} v{cycle}",
                "instruction": base["instruction"]
                + (
                    "\nUse a genuinely different proof route from earlier attempts. "
                    "Do not repeat the same lemma chain unless you can make it rigorous."
                    if cycle > 1
                    else ""
                ),
            }
        )
    return expanded


def strategy_solver_messages(
    problem: dict[str, Any],
    base_solver_system: str,
    strategy: dict[str, str],
    *,
    model: str,
    chat_format: str,
) -> list[dict[str, str]]:
    instruction = (
        base_solver_system.strip()
        + "\n\nAdditional strategy for this attempt:\n"
        + strategy["instruction"].strip()
        + "\n\nDo not include a proof outline unless it is followed by a complete proof. Return only the proof text."
    )
    user_content = (
        "Your task is to write a proof solution to the following problem.\n\n"
        f"Problem source: {problem['source']}\n"
        f"Problem id: {problem['id']}\n"
        f"Strategy id: {strategy['id']}\n\n"
        f"Problem:\n{problem['problem']}"
    )
    return make_chat_messages(instruction, user_content, model=model, chat_format=chat_format)


def judge_messages(problem: dict[str, Any], proof: str, *, judge_system: str, model: str, chat_format: str) -> list[dict[str, str]]:
    user_content = (
        "Grade this proof attempt.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Proof attempt:\n{proof}"
    )
    return make_chat_messages(judge_system, user_content, model=model, chat_format=chat_format)


def repair_messages(
    problem: dict[str, Any],
    proof: str,
    score: dict[str, Any],
    *,
    model: str,
    chat_format: str,
) -> list[dict[str, str]]:
    user_content = (
        "Repair this proof attempt.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Student attempt:\n{proof}\n\n"
        f"Judge critique:\n{score.get('critique', '')}\n\n"
        f"First serious flaw:\n{score.get('first_serious_flaw', '')}\n\n"
        f"Minimal repair hint:\n{score.get('repair_hint', '')}"
    )
    return make_chat_messages(REPAIR_SYSTEM, user_content, model=model, chat_format=chat_format)


def targeted_retry_messages(
    problem: dict[str, Any],
    failed_proof: str,
    score: dict[str, Any],
    focus: str,
    base_solver_system: str,
    *,
    model: str,
    chat_format: str,
) -> list[dict[str, str]]:
    instruction = (
        base_solver_system.strip()
        + "\n\nTargeted retry instruction:\n"
        + retry_instruction_for_focus(focus)
        + "\n\nReturn only a complete proof. Do not mention the prior attempt, judge, critique, or retry policy."
    )
    user_content = (
        "Write a new proof for the original problem.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"The previous attempt failed with focus={focus}.\n"
        f"First serious flaw:\n{score.get('first_serious_flaw', '')}\n\n"
        f"Critique:\n{score.get('critique', '')}\n\n"
        f"Previous failed proof:\n{failed_proof}"
    )
    return make_chat_messages(instruction, user_content, model=model, chat_format=chat_format)


def conclusion_messages(problem: dict[str, Any], proof: str, *, model: str, chat_format: str) -> list[dict[str, str]]:
    user_content = (
        "Extract the final conclusion from this proof attempt.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Proof attempt:\n{proof}"
    )
    return make_chat_messages(CONCLUSION_SYSTEM, user_content, model=model, chat_format=chat_format)


def breaker_messages(problem: dict[str, Any], proof: str, *, model: str, chat_format: str) -> list[dict[str, str]]:
    user_content = (
        "Try to break this proof.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Proof attempt:\n{proof}"
    )
    return make_chat_messages(BREAKER_SYSTEM, user_content, model=model, chat_format=chat_format)


def tournament_messages(problem: dict[str, Any], a: dict[str, Any], b: dict[str, Any], *, model: str, chat_format: str) -> list[dict[str, str]]:
    user_content = (
        "Compare proof A and proof B.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        "Proof A metadata:\n"
        f"source_id: {a.get('source_id')}\n"
        f"score: {a.get('score', {}).get('rubric_score')}\n"
        f"first_serious_flaw: {a.get('score', {}).get('first_serious_flaw', '')}\n\n"
        f"Proof A:\n{a.get('proof', '')}\n\n"
        "Proof B metadata:\n"
        f"source_id: {b.get('source_id')}\n"
        f"score: {b.get('score', {}).get('rubric_score')}\n"
        f"first_serious_flaw: {b.get('score', {}).get('first_serious_flaw', '')}\n\n"
        f"Proof B:\n{b.get('proof', '')}"
    )
    return make_chat_messages(TOURNAMENT_SYSTEM, user_content, model=model, chat_format=chat_format)


def polish_messages(problem: dict[str, Any], proof: str, *, model: str, chat_format: str) -> list[dict[str, str]]:
    user_content = (
        "Polish this already-valid proof.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Proof:\n{proof}"
    )
    return make_chat_messages(POLISH_SYSTEM, user_content, model=model, chat_format=chat_format)


def consolidate_messages(
    problem: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    model: str,
    chat_format: str,
) -> list[dict[str, str]]:
    blocks = []
    for index, candidate in enumerate(candidates, 1):
        score = candidate.get("score", {})
        proof_text = truncate_for_consolidation(str(candidate.get("proof", "")), capped=bool(candidate.get("capped")))
        blocks.append(
            "\n".join(
                [
                    f"Candidate {index}",
                    f"kind: {candidate.get('kind', '')}",
                    f"score: {score.get('rubric_score')}/10",
                    f"first_serious_flaw: {score.get('first_serious_flaw', '')}",
                    f"critique: {score.get('critique', '')}",
                    f"token_capped: {bool(candidate.get('capped'))}",
                    "proof:",
                    proof_text,
                ]
            )
        )
    user_content = (
        "Consolidate the strongest final proof.\n\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        "Judged candidate proofs:\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    return make_chat_messages(CONSOLIDATE_SYSTEM, user_content, model=model, chat_format=chat_format)


def resolved_role_base_urls(args: argparse.Namespace) -> dict[str, str]:
    base_url = args.base_url
    solver_url = args.solver_base_url or base_url
    judge_url = args.judge_base_url or base_url
    if args.repair_base_url:
        repair_url = args.repair_base_url
    elif args.repair_model == args.solver_model:
        repair_url = solver_url
    elif args.repair_model == args.judge_model:
        repair_url = judge_url
    else:
        repair_url = base_url
    if args.consolidator_base_url:
        consolidator_url = args.consolidator_base_url
    elif args.consolidator_model == args.solver_model:
        consolidator_url = solver_url
    elif args.consolidator_model == args.repair_model:
        consolidator_url = repair_url
    elif args.consolidator_model == args.judge_model:
        consolidator_url = judge_url
    else:
        consolidator_url = base_url
    if args.breaker_base_url:
        breaker_url = args.breaker_base_url
    elif args.breaker_model == args.judge_model:
        breaker_url = judge_url
    elif args.breaker_model == args.repair_model:
        breaker_url = repair_url
    elif args.breaker_model == args.solver_model:
        breaker_url = solver_url
    else:
        breaker_url = base_url
    return {
        "solver": solver_url,
        "judge": judge_url,
        "repair": repair_url,
        "consolidator": consolidator_url,
        "breaker": breaker_url,
    }


class ModelCaller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.semaphore = asyncio.Semaphore(args.max_concurrent)
        self.clients: dict[str, Any] = {}
        self.role_base_urls = resolved_role_base_urls(args)
        self.role_chat_template_kwargs: dict[str, dict[str, Any]] = {
            "solver": parse_json_object_arg(args.solver_chat_template_kwargs, "--solver-chat-template-kwargs"),
            "judge": parse_json_object_arg(args.judge_chat_template_kwargs, "--judge-chat-template-kwargs"),
            "repair": parse_json_object_arg(args.repair_chat_template_kwargs, "--repair-chat-template-kwargs"),
            "consolidator": parse_json_object_arg(args.consolidator_chat_template_kwargs, "--consolidator-chat-template-kwargs"),
            "breaker": parse_json_object_arg(args.breaker_chat_template_kwargs, "--breaker-chat-template-kwargs"),
        }
        self.calls_made = 0
        if args.dry_run:
            return
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise SystemExit(
                "run_solver_harness.py requires openai and httpx. "
                "Install requirements-hoped-mi-collection.txt on the GPU worker."
            ) from exc
        api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        clients_by_url: dict[str, Any] = {}
        for role, base_url in self.role_base_urls.items():
            if base_url not in clients_by_url:
                clients_by_url[base_url] = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key or "EMPTY",
                    timeout=httpx.Timeout(args.timeout_seconds, connect=60.0),
                )
            self.clients[role] = clients_by_url[base_url]

    async def call(
        self,
        *,
        role: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        chat_template_kwargs_override: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], str, float, dict[str, Any]]:
        if self.args.max_total_model_calls > 0 and self.calls_made >= self.args.max_total_model_calls:
            raise RuntimeError("model_call_budget_exhausted")
        self.calls_made += 1
        if self.args.dry_run:
            text = "{}" if json_mode else "DRY RUN proof placeholder."
            return text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "dry_run", 0.0, {}
        client = self.clients.get(role)
        if client is None:
            raise RuntimeError(f"model client is not initialized for role={role}")
        last_error: Exception | None = None
        for retry in range(self.args.retries + 1):
            start = time.monotonic()
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                if stop:
                    kwargs["stop"] = stop
                if model_uses_harmony(model):
                    kwargs["reasoning_effort"] = reasoning_effort or self.args.gpt_oss_reasoning
                chat_template_kwargs = (
                    chat_template_kwargs_override
                    if chat_template_kwargs_override is not None
                    else self.role_chat_template_kwargs.get(role)
                ) or {}
                if chat_template_kwargs:
                    kwargs["extra_body"] = {"chat_template_kwargs": chat_template_kwargs}
                if json_mode and self.args.json_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                async with self.semaphore:
                    response = await client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning_content = str(getattr(message, "reasoning_content", "") or "")
                model_extra = getattr(message, "model_extra", None)
                if not reasoning_content and isinstance(model_extra, dict):
                    reasoning_content = str(model_extra.get("reasoning_content") or "")
                usage = {}
                if getattr(response, "usage", None) is not None:
                    usage_obj = response.usage
                    usage = {
                        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                    }
                metadata = {
                    "content_chars": len(content),
                    "reasoning_content_chars": len(reasoning_content),
                    "reasoning_content": reasoning_content,
                }
                return content, usage, str(choice.finish_reason or ""), round(time.monotonic() - start, 3), metadata
            except Exception as exc:
                last_error = exc
                if retry < self.args.retries:
                    await asyncio.sleep(self.args.retry_sleep_seconds)
        raise RuntimeError(str(last_error))


class SolverHarness:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "candidate_prompts": self.run_dir / "candidate_prompts.jsonl",
            "candidates": self.run_dir / "candidates.jsonl",
            "candidate_scores": self.run_dir / "candidate_scores.jsonl",
            "repairs": self.run_dir / "repairs.jsonl",
            "repair_scores": self.run_dir / "repair_scores.jsonl",
            "consolidations": self.run_dir / "consolidations.jsonl",
            "consolidation_scores": self.run_dir / "consolidation_scores.jsonl",
            "conclusion_extractions": self.run_dir / "conclusion_extractions.jsonl",
            "conclusion_clusters": self.run_dir / "conclusion_clusters.jsonl",
            "arithmetic_checks": self.run_dir / "arithmetic_checks.jsonl",
            "proof_breaks": self.run_dir / "proof_breaks.jsonl",
            "tournament_matches": self.run_dir / "tournament_matches.jsonl",
            "polished_proofs": self.run_dir / "polished_proofs.jsonl",
            "polished_scores": self.run_dir / "polished_scores.jsonl",
            "training_pairs": self.run_dir / "training_pairs.jsonl",
            "calibration_report": self.run_dir / "calibration_report.json",
            "calibration_markdown": self.run_dir / "calibration_report.md",
            "selections": self.run_dir / "selections.jsonl",
            "prompts": self.run_dir / "prompts.jsonl",
            "attempts": self.run_dir / "attempts.jsonl",
            "scores": self.run_dir / "scores.jsonl",
            "summary": self.run_dir / "summary.json",
            "harness_summary": self.run_dir / "harness_summary.json",
            "run_config": self.run_dir / "run_config.json",
            "validation_report": self.run_dir / "validation_report.json",
        }
        if args.overwrite:
            for path in self.paths.values():
                if path.exists():
                    path.unlink()
        self.caller = ModelCaller(args)
        self.base_solver_system = Path(args.solver_prompt).read_text(encoding="utf-8")
        self.base_solver_sha = sha256_text(self.base_solver_system)
        self.judge_system = Path(args.judge_prompt).read_text(encoding="utf-8")
        self.judge_sha = sha256_text(self.judge_system)
        self.strategies = load_strategies(args.strategies, args.attempts_per_problem)
        self.expected_answers = read_optional_json_object(args.expected_answers)
        self.start_time = time.time()
        self.budget_skips: Counter[str] = Counter()

    def write_config(self, status: str) -> None:
        config = {
            "run_id": self.args.run_id,
            "status": status,
            "updated_at": utc_now(),
            "input": self.args.input,
            "run_dir": str(self.run_dir),
            "paths": {key: str(path) for key, path in self.paths.items()},
            "filters": {
                "allow_target": self.args.allow_target,
                "max_problems": self.args.max_problems,
                "problem_id": self.args.problem_id,
                "label": self.args.label,
            },
            "models": {
                "solver": self.args.solver_model,
                "judge": self.args.judge_model,
                "repair": self.args.repair_model,
                "consolidator": self.args.consolidator_model,
                "breaker": self.args.breaker_model,
            },
            "base_urls": {
                "default": self.args.base_url,
                **resolved_role_base_urls(self.args),
            },
            "prompts": {
                "solver_prompt": self.args.solver_prompt,
                "solver_prompt_sha256": self.base_solver_sha,
                "judge_prompt": self.args.judge_prompt,
                "judge_prompt_sha256": self.judge_sha,
            },
            "budgets": {
                "attempts_per_problem": self.args.attempts_per_problem,
                "targeted_retries_per_problem": self.args.targeted_retries_per_problem,
                "repairs_per_problem": self.args.repairs_per_problem,
                "top_candidates_for_consolidation": self.args.top_candidates_for_consolidation,
                "breaker_top_n": self.args.breaker_top_n,
                "tournament_size": self.args.tournament_size,
                "solver_max_tokens": self.args.solver_max_tokens,
                "solver_cap_retry_on_capped": self.args.solver_cap_retry_on_capped,
                "solver_cap_retry_max_tokens": self.args.solver_cap_retry_max_tokens,
                "solver_stop": self.args.solver_stop,
                "judge_max_tokens": self.args.judge_max_tokens,
                "repair_max_tokens": self.args.repair_max_tokens,
                "consolidator_max_tokens": self.args.consolidator_max_tokens,
                "time_limit_minutes": self.args.time_limit_minutes,
                "max_total_model_calls": self.args.max_total_model_calls,
                "estimated_model_calls_per_problem": self.estimated_model_calls_per_problem(),
            },
            "generation": {
                "solver_temperature": self.args.solver_temperature,
                "solver_top_p": self.args.solver_top_p,
                "judge_temperature": self.args.judge_temperature,
                "repair_temperature": self.args.repair_temperature,
                "consolidator_temperature": self.args.consolidator_temperature,
                "chat_format": self.args.chat_format,
                "extract_final_channel": self.args.extract_final_channel,
                "gpt_oss_reasoning": self.args.gpt_oss_reasoning,
                "gpt_oss_solver_reasoning": self.args.gpt_oss_solver_reasoning or self.args.gpt_oss_reasoning,
                "gpt_oss_judge_reasoning": self.args.gpt_oss_judge_reasoning or self.args.gpt_oss_reasoning,
                "gpt_oss_repair_reasoning": self.args.gpt_oss_repair_reasoning or self.args.gpt_oss_reasoning,
                "gpt_oss_consolidator_reasoning": self.args.gpt_oss_consolidator_reasoning or self.args.gpt_oss_reasoning,
                "chat_template_kwargs": {
                    "solver": parse_json_object_arg(self.args.solver_chat_template_kwargs, "--solver-chat-template-kwargs"),
                    "solver_cap_retry": parse_json_object_arg(
                        self.args.solver_cap_retry_chat_template_kwargs,
                        "--solver-cap-retry-chat-template-kwargs",
                    ),
                    "judge": parse_json_object_arg(self.args.judge_chat_template_kwargs, "--judge-chat-template-kwargs"),
                    "repair": parse_json_object_arg(self.args.repair_chat_template_kwargs, "--repair-chat-template-kwargs"),
                    "consolidator": parse_json_object_arg(self.args.consolidator_chat_template_kwargs, "--consolidator-chat-template-kwargs"),
                    "breaker": parse_json_object_arg(self.args.breaker_chat_template_kwargs, "--breaker-chat-template-kwargs"),
                },
            },
            "reasoning_harness": {
                "mode": self.args.harness_mode,
                "candidate_only": self.args.candidate_only,
                "full_reasoning": self.args.full_reasoning,
                "extract_conclusions": self.args.extract_conclusions,
                "adversarial_break": self.args.adversarial_break,
                "tournament": self.args.tournament,
                "polish_final": self.args.polish_final,
                "manual_audit": self.args.manual_audit,
                "enable_arithmetic_sanity_checks": self.args.enable_arithmetic_sanity_checks,
                "conclusion_consensus_gate": self.args.conclusion_consensus_gate,
                "require_breaker_for_repair": self.args.require_breaker_for_repair,
                "require_verified_capped_selection": self.args.require_verified_capped_selection,
                "strict_verified_selection_stages": self.args.strict_verified_selection_stages,
                "require_breaker_for_selection_stages": self.args.require_breaker_for_selection_stages,
                "expected_answers_path": self.args.expected_answers,
                "expected_answers_count": len(self.expected_answers),
            },
            "strategies": self.strategies,
        }
        write_json(self.paths["run_config"], config)

    def final_prompt_rows(self, problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for problem in problems:
            prompt_id = f"{self.args.run_id}::{problem['id']}::attempt_1"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "run_id": self.args.run_id,
                    "problem_id": problem["id"],
                    "source": problem["source"],
                    "year": problem["year"],
                    "label": problem["label"],
                    "attempt_index": 1,
                    "model": self.args.solver_model,
                    "status": "prepared",
                    "created_at": utc_now(),
                    "input_path": self.args.input,
                    "solver_prompt_path": self.args.solver_prompt,
                    "solver_prompt_sha256": self.base_solver_sha,
                    "problem": problem["problem"],
                    "messages": make_solver_messages(
                        problem,
                        self.base_solver_system,
                        model=self.args.solver_model,
                        chat_format=self.args.chat_format,
                    ),
                }
            )
        return rows

    def time_exceeded(self) -> bool:
        return self.args.time_limit_minutes > 0 and (time.time() - self.start_time) >= self.args.time_limit_minutes * 60

    def remaining_model_calls(self) -> int | None:
        if self.args.max_total_model_calls <= 0:
            return None
        return max(0, self.args.max_total_model_calls - self.caller.calls_made)

    def budget_exhausted(self) -> bool:
        remaining = self.remaining_model_calls()
        return remaining is not None and remaining <= 0

    def stop_requested(self) -> bool:
        return self.time_exceeded() or self.budget_exhausted()

    def can_spend(self, stage: str, estimated_calls: int, *, required: bool = False) -> bool:
        if estimated_calls <= 0:
            return True
        remaining = self.remaining_model_calls()
        if remaining is None or remaining >= estimated_calls:
            return True
        self.budget_skips[stage] += 1
        label = "required stage blocked" if required else "optional stage skipped"
        print(f"budget scheduler: {label} for {stage}; need {estimated_calls} calls, remaining {remaining}", flush=True)
        return False

    def estimated_model_calls_per_problem(self) -> dict[str, int]:
        attempts = self.args.attempts_per_problem
        if self.args.candidate_only:
            return {
                "candidate_generation": attempts,
                "total": attempts,
            }
        targeted = self.args.targeted_retries_per_problem
        repairs = self.args.repairs_per_problem
        pool_before_consolidation = attempts + targeted + repairs
        consolidation = 2 if self.args.consolidate else 0
        pool_after_consolidation = pool_before_consolidation + (1 if self.args.consolidate else 0)
        conclusion = pool_after_consolidation if self.args.extract_conclusions else 0
        repair_breakers = min(self.args.repairs_per_problem, attempts + targeted) if self.args.require_breaker_for_repair else 0
        breaker = min(self.args.breaker_top_n, pool_after_consolidation) if self.args.adversarial_break else 0
        consolidation_breaker = 1 if self.args.consolidate and self.args.adversarial_break and self.args.require_breaker_for_selection_stages else 0
        consolidation_conclusion = 1 if self.args.consolidate and self.args.extract_conclusions else 0
        tournament_n = min(self.args.tournament_size, pool_after_consolidation)
        tournament = tournament_n * (tournament_n - 1) // 2 if self.args.tournament else 0
        polish = 2 if self.args.polish_final else 0
        stages = {
            "initial_attempts_and_judges": attempts * 2,
            "targeted_retries_and_judges": targeted * 2,
            "repair_breakers": repair_breakers,
            "repairs_and_judges": repairs * 2,
            "consolidation_and_judge": consolidation,
            "consolidation_breaker": consolidation_breaker,
            "consolidation_conclusion_extraction": consolidation_conclusion,
            "conclusion_extractions": conclusion,
            "adversarial_breaks": breaker,
            "pairwise_tournament": tournament,
            "proof_polish_and_judge": polish,
        }
        stages["total_upper_bound"] = sum(stages.values())
        return stages

    def arithmetic_check_item(self, problem: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        if not self.args.enable_arithmetic_sanity_checks:
            report = {"status": "disabled", "error_count": 0, "errors": []}
            item["arithmetic_check"] = report
            return report
        source_id = str(item.get("source_id", ""))
        existing = load_jsonl_index(self.paths["arithmetic_checks"], "source_id")
        if source_id in existing:
            report = existing[source_id]
            item["arithmetic_check"] = report
            return report
        check = arithmetic_sanity_check(str(item.get("proof", "")))
        report = {
            "source_id": source_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "created_at": utc_now(),
            **check,
        }
        append_jsonl(self.paths["arithmetic_checks"], report)
        item["arithmetic_check"] = report
        return report

    def apply_hard_score_gates(self, problem: dict[str, Any], item: dict[str, Any]) -> None:
        score = normalize_judge_score(item.get("score", {}), capped=bool(item.get("capped")))
        gate_reasons: list[str] = list(item.get("gate_reasons", []) or [])

        score, expected_reason = expected_answer_gate(
            score,
            problem_id=str(problem.get("id", "")),
            proof=str(item.get("proof", "") or ""),
            expected_answers=self.expected_answers,
        )
        if expected_reason:
            gate_reasons.append(expected_reason)

        if score_has_wrong_final_answer(score):
            score = add_score_cap(score, 2, "wrong final answer", "statement_mismatch")
            gate_reasons.append("wrong_final_answer_cap")
        if score_has_false_central_lemma(score):
            score = add_score_cap(score, 2, "false central lemma", "invalid_lemma")
            gate_reasons.append("false_central_lemma_cap")
        elif score_has_central_lemma_issue(score):
            score = add_score_cap(score, 6, "central lemma requires verification", "invalid_lemma")
            gate_reasons.append("central_lemma_review_cap")
        elif score_has_unverified_central_lemma(score):
            score = add_score_cap(score, 7, "unverified central lemma", "invalid_lemma")
            gate_reasons.append("unverified_central_lemma_cap")

        arithmetic = self.arithmetic_check_item(problem, item)
        if arithmetic.get("status") == "failed":
            score = add_score_cap(score, 3, "arithmetic sanity check failed", "calculation_error")
            gate_reasons.append("arithmetic_sanity_cap")

        conclusion = item.get("conclusion")
        if isinstance(conclusion, dict):
            key = normalize_conclusion_key(str(conclusion.get("normalized_conclusion", "") or conclusion.get("conclusion", "")))
            confidence = str(conclusion.get("confidence", "low"))
            item["conclusion_key"] = key
            if key == "unknown" and int(score.get("rubric_score", 0) or 0) >= self.args.accept_score:
                score = add_score_cap(score, 6, "missing or unstable final conclusion", "incomplete_proof")
                gate_reasons.append("missing_conclusion_cap")
            elif confidence == "low" and int(score.get("rubric_score", 0) or 0) >= self.args.accept_score:
                score = add_score_cap(score, 8, "low-confidence final conclusion extraction", "incomplete_proof")
                gate_reasons.append("low_confidence_conclusion_cap")

        if (
            bool(item.get("capped"))
            and self.args.require_verified_capped_selection
            and isinstance(item.get("breaker"), dict)
            and not item_has_clean_valid_breaker(item)
        ):
            score = add_score_cap(score, 4, "token-capped proof lacks independent clean breaker verification", "token_cap")
            gate_reasons.append("unverified_capped_cap")

        item["score"] = score
        item["gate_reasons"] = sorted(set(gate_reasons))

    def apply_conclusion_consensus_gates(self, pool: list[dict[str, Any]]) -> None:
        if not self.args.conclusion_consensus_gate or not self.args.extract_conclusions:
            return
        counts = Counter(str(item.get("conclusion_key", "unknown") or "unknown") for item in pool)
        counts.pop("unknown", None)
        if not counts:
            return
        majority_key, majority_support = counts.most_common(1)[0]
        if majority_support < 2:
            return
        for item in pool:
            key = str(item.get("conclusion_key", "unknown") or "unknown")
            if key == majority_key or item_has_clean_valid_breaker(item):
                continue
            score = item.get("score", {})
            if int(score.get("rubric_score", 0) or 0) >= self.args.accept_score:
                item["score"] = add_score_cap(score, 7, "final conclusion is an unverified minority outlier", "candidate_selection_failure")
                reasons = set(item.get("gate_reasons", []) or [])
                reasons.add("minority_conclusion_cap")
                item["gate_reasons"] = sorted(reasons)

    def selection_gate_reasons(self, item: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        score = item.get("score", {})
        if score_has_wrong_final_answer(score):
            reasons.append("wrong_final_answer")
        if score_has_false_central_lemma(score):
            reasons.append("false_central_lemma")
        arithmetic = item.get("arithmetic_check")
        if isinstance(arithmetic, dict) and arithmetic.get("status") == "failed":
            reasons.append("arithmetic_sanity_failed")
        if bool(item.get("capped")) and self.args.require_verified_capped_selection and not item_has_clean_valid_breaker(item):
            reasons.append("capped_without_clean_breaker")
        if (
            self.args.require_breaker_for_selection_stages
            and self.args.adversarial_break
            and not item_has_clean_valid_breaker(item)
        ):
            reasons.append("missing_clean_breaker_verification")
        return reasons

    def item_passed_strict_verification(self, item: dict[str, Any]) -> bool:
        if self.selection_gate_reasons(item):
            return False
        if (
            self.args.require_breaker_for_selection_stages
            and self.args.adversarial_break
            and not item_has_clean_valid_breaker(item)
        ):
            return False
        return True

    def strict_verified_stage_pool(self, pool: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
        if not self.args.strict_verified_selection_stages:
            return list(pool)
        verified = [item for item in pool if self.item_passed_strict_verification(item)]
        if not verified:
            print(f"{stage}: skipped because no candidate passed strict verification", flush=True)
        return verified

    def select_best_item(self, pool: list[dict[str, Any]]) -> dict[str, Any]:
        for item in pool:
            reasons = self.selection_gate_reasons(item)
            item["selection_gate_reasons"] = reasons
            item["selection_allowed"] = not reasons
        eligible = [item for item in pool if item.get("selection_allowed")]
        if eligible:
            return sorted(eligible, key=reasoning_sort_key, reverse=True)[0]
        fallback = sorted(pool, key=reasoning_sort_key, reverse=True)[0]
        fallback["score"] = add_score_cap(fallback.get("score", {}), 4, "all candidates failed final selection gates", "candidate_selection_failure")
        fallback["selection_forced_fallback"] = True
        return fallback

    async def generate_candidate(self, problem: dict[str, Any], strategy: dict[str, str], attempt_index: int) -> dict[str, Any]:
        candidate_id = f"{self.args.run_id}::{problem['id']}::candidate_{attempt_index}"
        existing = load_jsonl_index(self.paths["candidates"], "candidate_id")
        if candidate_id in existing:
            return existing[candidate_id]
        messages = strategy_solver_messages(
            problem,
            self.base_solver_system,
            strategy,
            model=self.args.solver_model,
            chat_format=self.args.chat_format,
        )
        prompt_row = {
            "candidate_id": candidate_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "strategy_id": strategy["id"],
            "attempt_index": attempt_index,
            "model": self.args.solver_model,
            "created_at": utc_now(),
            "messages": messages,
        }
        append_jsonl(self.paths["candidate_prompts"], prompt_row)
        try:
            proof, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="solver",
                model=self.args.solver_model,
                messages=messages,
                temperature=self.args.solver_temperature,
                max_tokens=self.args.solver_max_tokens,
                top_p=self.args.solver_top_p,
                reasoning_effort=self.args.gpt_oss_solver_reasoning,
                stop=self.args.solver_stop,
            )
            if self.args.extract_final_channel or model_uses_harmony(self.args.solver_model):
                proof = extract_final_channel_text(proof)
            status = "completed"
            error = ""
        except Exception as exc:
            proof = ""
            usage = {}
            finish_reason = "error"
            latency = 0.0
            status = "error"
            error = str(exc)
        capped = finish_reason in {"length", "max_tokens"} or int(usage.get("completion_tokens", 0) or 0) >= self.args.solver_max_tokens
        if status == "completed":
            proof, response_metadata = split_qwen_thinking_output(
                model=self.args.solver_model,
                text=proof,
                response_metadata=response_metadata,
                capped=capped,
            )
        row = {
            "candidate_id": candidate_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "source": problem["source"],
            "year": problem["year"],
            "label": problem["label"],
            "strategy_id": strategy["id"],
            "strategy_name": strategy["name"],
            "attempt_index": attempt_index,
            "model": self.args.solver_model,
            "created_at": utc_now(),
            "status": status,
            "problem": problem["problem"],
            "proof": proof,
            "finish_reason": finish_reason,
            "capped": capped,
            "error": error,
            "latency_seconds": latency,
            "usage": usage,
            "raw_content_chars": response_metadata.get("content_chars", 0) if status == "completed" else 0,
            "reasoning_content_chars": response_metadata.get("reasoning_content_chars", 0) if status == "completed" else 0,
            "reasoning_content": response_metadata.get("reasoning_content", "") if status == "completed" else "",
        }
        append_jsonl(self.paths["candidates"], row)
        print(f"{problem['id']} candidate {attempt_index}/{len(self.strategies)} {strategy['id']}: {status} tokens={usage.get('completion_tokens')} capped={capped}", flush=True)
        return row

    async def generate_targeted_retry(self, problem: dict[str, Any], failed_item: dict[str, Any], retry_index: int) -> dict[str, Any]:
        focus = failure_focus(failed_item.get("score", {}), capped=bool(failed_item.get("capped")))
        candidate_id = f"{self.args.run_id}::{problem['id']}::targeted_retry_{retry_index}_{focus}"
        existing = load_jsonl_index(self.paths["candidates"], "candidate_id")
        if candidate_id in existing:
            return existing[candidate_id]
        messages = targeted_retry_messages(
            problem,
            str(failed_item.get("proof", "")),
            failed_item.get("score", {}),
            focus,
            self.base_solver_system,
            model=self.args.solver_model,
            chat_format=self.args.chat_format,
        )
        prompt_row = {
            "candidate_id": candidate_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "strategy_id": f"targeted_retry_{focus}",
            "attempt_index": len(self.strategies) + retry_index,
            "model": self.args.solver_model,
            "created_at": utc_now(),
            "messages": messages,
            "retry_focus": focus,
            "source_failed_id": failed_item.get("source_id", ""),
        }
        append_jsonl(self.paths["candidate_prompts"], prompt_row)
        use_cap_retry_mode = focus == "token_cap_loop" and self.args.solver_cap_retry_on_capped
        max_tokens = self.args.solver_cap_retry_max_tokens if use_cap_retry_mode else self.args.solver_max_tokens
        chat_template_kwargs_override = (
            parse_json_object_arg(
                self.args.solver_cap_retry_chat_template_kwargs,
                "--solver-cap-retry-chat-template-kwargs",
            )
            if use_cap_retry_mode
            else None
        )
        try:
            proof, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="solver",
                model=self.args.solver_model,
                messages=messages,
                temperature=0.2 if use_cap_retry_mode else max(self.args.solver_temperature * 0.8, 0.2),
                max_tokens=max_tokens,
                top_p=self.args.solver_top_p,
                reasoning_effort=self.args.gpt_oss_solver_reasoning,
                chat_template_kwargs_override=chat_template_kwargs_override,
                stop=self.args.solver_stop,
            )
            if self.args.extract_final_channel or model_uses_harmony(self.args.solver_model):
                proof = extract_final_channel_text(proof)
            status = "completed"
            error = ""
        except Exception as exc:
            proof = ""
            usage = {}
            finish_reason = "error"
            latency = 0.0
            status = "error"
            error = str(exc)
            response_metadata = {}
        capped = finish_reason in {"length", "max_tokens"} or int(usage.get("completion_tokens", 0) or 0) >= max_tokens
        if status == "completed":
            proof, response_metadata = split_qwen_thinking_output(
                model=self.args.solver_model,
                text=proof,
                response_metadata=response_metadata,
                capped=capped,
            )
        row = {
            "candidate_id": candidate_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "source": problem["source"],
            "year": problem["year"],
            "label": problem["label"],
            "strategy_id": f"targeted_retry_{focus}",
            "strategy_name": f"Targeted retry: {focus}",
            "attempt_index": len(self.strategies) + retry_index,
            "model": self.args.solver_model,
            "created_at": utc_now(),
            "status": status,
            "problem": problem["problem"],
            "proof": proof,
            "finish_reason": finish_reason,
            "capped": capped,
            "error": error,
            "latency_seconds": latency,
            "usage": usage,
            "retry_focus": focus,
            "cap_retry_mode": use_cap_retry_mode,
            "cap_retry_max_tokens": max_tokens if use_cap_retry_mode else 0,
            "source_failed_id": failed_item.get("source_id", ""),
            "raw_content_chars": response_metadata.get("content_chars", 0) if status == "completed" else 0,
            "reasoning_content_chars": response_metadata.get("reasoning_content_chars", 0) if status == "completed" else 0,
            "reasoning_content": response_metadata.get("reasoning_content", "") if status == "completed" else "",
        }
        append_jsonl(self.paths["candidates"], row)
        print(f"{problem['id']} targeted retry {retry_index} {focus}: {status} tokens={usage.get('completion_tokens')} capped={capped}", flush=True)
        return row

    async def extract_conclusion(self, problem: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        source_id = str(item.get("source_id", ""))
        existing = load_jsonl_index(self.paths["conclusion_extractions"], "source_id")
        if source_id in existing:
            return existing[source_id]
        proof = str(item.get("proof", ""))
        base = {
            "source_id": source_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "created_at": utc_now(),
        }
        if self.args.dry_run or not proof.strip():
            parsed = heuristic_conclusion(proof)
            row = {**base, "status": "dry_run" if self.args.dry_run else "empty_proof", **parsed, "error": "", "raw_response": "", "latency_seconds": 0.0, "usage": {}}
            append_jsonl(self.paths["conclusion_extractions"], row)
            return row
        try:
            raw, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="judge",
                model=self.args.judge_model,
                messages=conclusion_messages(problem, proof, model=self.args.judge_model, chat_format=self.args.chat_format),
                temperature=0.0,
                max_tokens=self.args.conclusion_max_tokens,
                top_p=1.0,
                json_mode=True,
                reasoning_effort=self.args.gpt_oss_judge_reasoning,
            )
            parsed = extract_json_object_with_reasoning_fallback(
                raw,
                response_metadata,
                extract_final=self.args.extract_final_channel,
                model=self.args.judge_model,
            )
            final_answer = str(parsed.get("final_answer", "") or "")
            conclusion = str(parsed.get("conclusion", final_answer) or final_answer)
            normalized = str(parsed.get("normalized_conclusion", "") or normalize_conclusion_key(conclusion or final_answer))
            confidence = str(parsed.get("confidence", "low"))
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            row = {
                **base,
                "status": "completed",
                "final_answer": final_answer,
                "conclusion": conclusion,
                "normalized_conclusion": normalize_conclusion_key(normalized),
                "confidence": confidence,
                "error": "",
                "raw_response": raw,
                "finish_reason": finish_reason,
                "latency_seconds": latency,
                "usage": usage,
            }
        except Exception as exc:
            fallback = heuristic_conclusion(proof)
            row = {**base, "status": "error", **fallback, "error": str(exc), "raw_response": "", "latency_seconds": 0.0, "usage": {}}
        append_jsonl(self.paths["conclusion_extractions"], row)
        return row

    async def cluster_conclusions(self, problem: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, int]:
        if not self.args.extract_conclusions:
            return {}
        clusters: dict[str, list[dict[str, Any]]] = {}
        for item in pool:
            extraction = await self.extract_conclusion(problem, item)
            key = normalize_conclusion_key(str(extraction.get("normalized_conclusion", "") or extraction.get("conclusion", "")))
            item["conclusion"] = extraction
            item["conclusion_key"] = key
            clusters.setdefault(key, []).append(item)
        support = {key: len(items) for key, items in clusters.items()}
        existing = load_jsonl_index(self.paths["conclusion_clusters"], "cluster_id")
        for index, (key, items) in enumerate(sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])), 1):
            cluster_id = f"{self.args.run_id}::{problem['id']}::cluster_{index}"
            if cluster_id in existing:
                continue
            append_jsonl(
                self.paths["conclusion_clusters"],
                {
                    "cluster_id": cluster_id,
                    "run_id": self.args.run_id,
                    "problem_id": problem["id"],
                    "created_at": utc_now(),
                    "normalized_conclusion": key,
                    "support": len(items),
                    "source_ids": [str(item.get("source_id", "")) for item in items],
                    "scores": [int(item.get("score", {}).get("rubric_score", 0)) for item in items],
                    "best_score": max(int(item.get("score", {}).get("rubric_score", 0)) for item in items),
                },
            )
        for item in pool:
            item["cluster_support"] = support.get(str(item.get("conclusion_key", "")), 1)
        return support

    async def break_proof(self, problem: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        source_id = str(item.get("source_id", ""))
        existing = load_jsonl_index(self.paths["proof_breaks"], "source_id")
        if source_id in existing:
            return existing[source_id]
        proof = str(item.get("proof", ""))
        base = {
            "source_id": source_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "breaker_model": self.args.breaker_model,
            "created_at": utc_now(),
        }
        if self.args.dry_run or not proof.strip():
            row = {
                **base,
                "status": "dry_run" if self.args.dry_run else "empty_proof",
                "verdict": "valid" if proof.strip() else "invalid",
                "fatal_flaw": "" if proof.strip() else "No proof text was produced.",
                "failure_tags": [] if proof.strip() else ["incomplete_proof"],
                "final_answer_status": "unknown",
                "central_lemma_status": "unknown",
                "hidden_assumption_status": "unknown",
                "boundary_case_status": "unknown",
                "theorem_use_status": "unknown",
                "algebra_geometry_status": "unknown",
                "score_cap": 10 if proof.strip() else 0,
                "confidence": "low",
                "notes": "No model call was made.",
                "error": "",
                "raw_response": "",
                "latency_seconds": 0.0,
                "usage": {},
            }
            append_jsonl(self.paths["proof_breaks"], row)
            return row
        try:
            raw, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="breaker",
                model=self.args.breaker_model,
                messages=breaker_messages(problem, proof, model=self.args.breaker_model, chat_format=self.args.chat_format),
                temperature=0.0,
                max_tokens=self.args.breaker_max_tokens,
                top_p=1.0,
                json_mode=True,
                reasoning_effort=self.args.gpt_oss_judge_reasoning,
            )
            parsed = extract_json_object_with_reasoning_fallback(
                raw,
                response_metadata,
                extract_final=self.args.extract_final_channel,
                model=self.args.breaker_model,
            )
            verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
            if verdict not in {"valid", "invalid", "uncertain"}:
                verdict = "uncertain"
            try:
                score_cap = int(parsed.get("score_cap", 10) or 10)
            except (TypeError, ValueError):
                score_cap = 10
            row = {
                **base,
                "status": "completed",
                "verdict": verdict,
                "fatal_flaw": str(parsed.get("fatal_flaw", "")),
                "failure_tags": parsed.get("failure_tags", []),
                "final_answer_status": str(parsed.get("final_answer_status", "unknown")),
                "central_lemma_status": str(parsed.get("central_lemma_status", "unknown")),
                "hidden_assumption_status": str(parsed.get("hidden_assumption_status", "unknown")),
                "boundary_case_status": str(parsed.get("boundary_case_status", "unknown")),
                "theorem_use_status": str(parsed.get("theorem_use_status", "unknown")),
                "algebra_geometry_status": str(parsed.get("algebra_geometry_status", "unknown")),
                "score_cap": max(0, min(10, score_cap)),
                "confidence": str(parsed.get("confidence", "low")),
                "notes": str(parsed.get("notes", "")),
                "error": "",
                "raw_response": raw,
                "finish_reason": finish_reason,
                "latency_seconds": latency,
                "usage": usage,
            }
        except Exception as exc:
            row = {
                **base,
                "status": "error",
                "verdict": "uncertain",
                "fatal_flaw": "Proof breaker call failed.",
                "failure_tags": ["candidate_selection_failure"],
                "final_answer_status": "unknown",
                "central_lemma_status": "unknown",
                "hidden_assumption_status": "unknown",
                "boundary_case_status": "unknown",
                "theorem_use_status": "unknown",
                "algebra_geometry_status": "unknown",
                "score_cap": 7,
                "confidence": "low",
                "notes": "",
                "error": str(exc),
                "raw_response": "",
                "latency_seconds": 0.0,
                "usage": {},
            }
        append_jsonl(self.paths["proof_breaks"], row)
        return row

    def apply_breaker_to_item(self, item: dict[str, Any], breaker: dict[str, Any]) -> None:
        verdict = str(breaker.get("verdict", "uncertain")).lower()
        if verdict == "valid" and breaker_has_clean_valid_verdict(breaker):
            item["breaker"] = breaker
            return
        if verdict == "valid":
            verdict = "uncertain"
        score = dict(item.get("score", {}))
        tags = list(score.get("failure_tags", []))
        for tag in normalized_failure_tags(breaker.get("failure_tags", [])):
            if tag not in tags:
                tags.append(tag)
        cap = breaker_score_cap(breaker, verdict)
        score.update(
            {
                "failure_tags": tags,
                "first_serious_flaw": str(breaker.get("fatal_flaw", "")) or str(score.get("first_serious_flaw", "")),
                "critique": str(breaker.get("notes", "")) or str(score.get("critique", "")),
                "final_answer_status": breaker.get("final_answer_status", score.get("final_answer_status", "unknown")),
                "central_lemma_status": breaker.get("central_lemma_status", score.get("central_lemma_status", "unknown")),
                "hidden_assumption_status": breaker.get("hidden_assumption_status", score.get("hidden_assumption_status", "unknown")),
                "boundary_case_status": breaker.get("boundary_case_status", score.get("boundary_case_status", "unknown")),
                "theorem_use_status": breaker.get("theorem_use_status", score.get("theorem_use_status", "unknown")),
                "algebra_geometry_status": breaker.get("algebra_geometry_status", score.get("algebra_geometry_status", "unknown")),
                "score_cap": min(cap, int(score.get("score_cap", 10) or 10)),
                "audit_summary": f"adversarial_breaker_verdict={verdict}",
            }
        )
        item["score"] = normalize_judge_score(score, capped=bool(item.get("capped")))
        item["breaker"] = breaker

    async def adversarial_break_pool(self, problem: dict[str, Any], pool: list[dict[str, Any]]) -> None:
        if not self.args.adversarial_break:
            return
        top = sorted(pool, key=reasoning_sort_key, reverse=True)[: self.args.breaker_top_n]
        for item in top:
            breaker = await self.break_proof(problem, item)
            self.apply_breaker_to_item(item, breaker)
            self.apply_hard_score_gates(problem, item)

    async def run_tournament(self, problem: dict[str, Any], pool: list[dict[str, Any]]) -> None:
        if not self.args.tournament or len(pool) < 2:
            return
        stage_pool = self.strict_verified_stage_pool(pool, stage="tournament")
        if len(stage_pool) < 2:
            return
        top = sorted(stage_pool, key=reasoning_sort_key, reverse=True)[: self.args.tournament_size]
        index = {str(item.get("source_id", "")): item for item in top}
        for item in top:
            item["tournament_wins"] = int(item.get("tournament_wins", 0))
            item["tournament_losses"] = int(item.get("tournament_losses", 0))
            item["tournament_ties"] = int(item.get("tournament_ties", 0))
        existing = load_jsonl_index(self.paths["tournament_matches"], "match_id")
        for i, a in enumerate(top):
            for j, b in enumerate(top[i + 1 :], i + 1):
                match_id = f"{self.args.run_id}::{problem['id']}::match_{i + 1}_{j + 1}"
                if match_id in existing:
                    row = existing[match_id]
                elif self.args.dry_run:
                    a_score = int(a.get("score", {}).get("rubric_score", 0))
                    b_score = int(b.get("score", {}).get("rubric_score", 0))
                    winner = "A" if a_score > b_score else "B" if b_score > a_score else "tie"
                    row = {
                        "match_id": match_id,
                        "run_id": self.args.run_id,
                        "problem_id": problem["id"],
                        "created_at": utc_now(),
                        "source_a": a["source_id"],
                        "source_b": b["source_id"],
                        "winner": winner,
                        "winner_source_id": a["source_id"] if winner == "A" else b["source_id"] if winner == "B" else "",
                        "a_score": a_score,
                        "b_score": b_score,
                        "reason": "Dry-run deterministic score comparison.",
                        "confidence": "low",
                        "status": "dry_run",
                        "error": "",
                        "raw_response": "",
                        "latency_seconds": 0.0,
                        "usage": {},
                    }
                    append_jsonl(self.paths["tournament_matches"], row)
                else:
                    try:
                        raw, usage, finish_reason, latency, response_metadata = await self.caller.call(
                            role="judge",
                            model=self.args.judge_model,
                            messages=tournament_messages(problem, a, b, model=self.args.judge_model, chat_format=self.args.chat_format),
                            temperature=0.0,
                            max_tokens=self.args.tournament_max_tokens,
                            top_p=1.0,
                            json_mode=True,
                            reasoning_effort=self.args.gpt_oss_judge_reasoning,
                        )
                        parsed = extract_json_object_with_reasoning_fallback(
                            raw,
                            response_metadata,
                            extract_final=self.args.extract_final_channel,
                            model=self.args.judge_model,
                        )
                        winner = str(parsed.get("winner", "tie")).strip()
                        if winner not in {"A", "B", "tie"}:
                            winner = "tie"
                        row = {
                            "match_id": match_id,
                            "run_id": self.args.run_id,
                            "problem_id": problem["id"],
                            "created_at": utc_now(),
                            "source_a": a["source_id"],
                            "source_b": b["source_id"],
                            "winner": winner,
                            "winner_source_id": a["source_id"] if winner == "A" else b["source_id"] if winner == "B" else "",
                            "a_score": max(0, min(10, int(parsed.get("a_score", 0) or 0))),
                            "b_score": max(0, min(10, int(parsed.get("b_score", 0) or 0))),
                            "reason": str(parsed.get("reason", "")),
                            "confidence": str(parsed.get("confidence", "low")),
                            "status": "completed",
                            "error": "",
                            "raw_response": raw,
                            "finish_reason": finish_reason,
                            "latency_seconds": latency,
                            "usage": usage,
                        }
                    except Exception as exc:
                        row = {
                            "match_id": match_id,
                            "run_id": self.args.run_id,
                            "problem_id": problem["id"],
                            "created_at": utc_now(),
                            "source_a": a["source_id"],
                            "source_b": b["source_id"],
                            "winner": "tie",
                            "winner_source_id": "",
                            "a_score": int(a.get("score", {}).get("rubric_score", 0)),
                            "b_score": int(b.get("score", {}).get("rubric_score", 0)),
                            "reason": "Tournament judge call failed.",
                            "confidence": "low",
                            "status": "error",
                            "error": str(exc),
                            "raw_response": "",
                            "latency_seconds": 0.0,
                            "usage": {},
                        }
                    append_jsonl(self.paths["tournament_matches"], row)
                a_item = index.get(str(row.get("source_a")))
                b_item = index.get(str(row.get("source_b")))
                if row.get("winner") == "A" and a_item and b_item:
                    a_item["tournament_wins"] += 1
                    b_item["tournament_losses"] += 1
                elif row.get("winner") == "B" and a_item and b_item:
                    b_item["tournament_wins"] += 1
                    a_item["tournament_losses"] += 1
                elif a_item and b_item:
                    a_item["tournament_ties"] += 1
                    b_item["tournament_ties"] += 1

    async def polish_if_valid(self, problem: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
        if not self.args.polish_final:
            return best
        if int(best.get("score", {}).get("rubric_score", 0)) < self.args.polish_min_score or best.get("capped"):
            return best
        if self.args.adversarial_break:
            breaker = best.get("breaker")
            if not isinstance(breaker, dict) or str(breaker.get("verdict", "")).lower() != "valid":
                return best
        if not self.can_spend("proof_polish_and_judge", 2):
            return best
        polish_id = f"{best['source_id']}::polished"
        existing = load_jsonl_index(self.paths["polished_proofs"], "polish_id")
        if polish_id in existing:
            polished = existing[polish_id]
        else:
            try:
                proof, usage, finish_reason, latency, response_metadata = await self.caller.call(
                    role="consolidator",
                    model=self.args.consolidator_model,
                    messages=polish_messages(problem, str(best.get("proof", "")), model=self.args.consolidator_model, chat_format=self.args.chat_format),
                    temperature=0.0,
                    max_tokens=self.args.consolidator_max_tokens,
                    top_p=1.0,
                    reasoning_effort=self.args.gpt_oss_consolidator_reasoning,
                )
                if self.args.extract_final_channel or model_uses_harmony(self.args.consolidator_model):
                    proof = extract_final_channel_text(proof)
                status = "completed"
                error = ""
            except Exception as exc:
                proof = ""
                usage = {}
                finish_reason = "error"
                latency = 0.0
                response_metadata = {}
                status = "error"
                error = str(exc)
            capped = finish_reason in {"length", "max_tokens"} or int(usage.get("completion_tokens", 0) or 0) >= self.args.consolidator_max_tokens
            if status == "completed":
                proof, response_metadata = split_qwen_thinking_output(
                    model=self.args.consolidator_model,
                    text=proof,
                    response_metadata=response_metadata,
                    capped=capped,
                )
            polished = {
                "polish_id": polish_id,
                "source_id": best["source_id"],
                "run_id": self.args.run_id,
                "problem_id": problem["id"],
                "created_at": utc_now(),
                "status": status,
                "model": self.args.consolidator_model,
                "proof": proof,
                "finish_reason": finish_reason,
                "capped": capped,
                "error": error,
                "latency_seconds": latency,
                "usage": usage,
                "raw_content_chars": response_metadata.get("content_chars", 0) if status == "completed" else 0,
                "reasoning_content_chars": response_metadata.get("reasoning_content_chars", 0) if status == "completed" else 0,
            }
            append_jsonl(self.paths["polished_proofs"], polished)
        if not str(polished.get("proof", "")).strip():
            return best
        polished_score = await self.judge_proof(
            problem,
            source_id=polish_id,
            proof=str(polished.get("proof", "")),
            capped=bool(polished.get("capped")),
            output_path=self.paths["polished_scores"],
            model=self.args.judge_model,
        )
        if int(polished_score.get("rubric_score", 0)) >= int(best.get("score", {}).get("rubric_score", 0)):
            polished_item = {
                "source_id": polish_id,
                "kind": "polished",
                "proof": str(polished.get("proof", "")),
                "capped": bool(polished.get("capped")),
                "score": polished_score,
                "source_row": polished,
                "polished_from": best["source_id"],
            }
            self.apply_hard_score_gates(problem, polished_item)
            if self.selection_gate_reasons(polished_item):
                return best
            return polished_item
        return best

    def write_training_pairs(self, problem: dict[str, Any], pool: list[dict[str, Any]], best: dict[str, Any]) -> None:
        if self.args.harness_mode != "training_data":
            return
        if int(best.get("score", {}).get("rubric_score", 0)) < self.args.accept_score:
            return
        existing = load_jsonl_index(self.paths["training_pairs"], "pair_id")
        rejected = [
            item for item in sorted(pool, key=reasoning_sort_key)
            if item.get("source_id") != best.get("source_id")
            and int(item.get("score", {}).get("rubric_score", 0)) <= self.args.training_pair_max_rejected_score
        ][: self.args.training_pairs_per_problem]
        for index, item in enumerate(rejected, 1):
            pair_id = f"{self.args.run_id}::{problem['id']}::pair_{index}"
            if pair_id in existing:
                continue
            append_jsonl(
                self.paths["training_pairs"],
                {
                    "pair_id": pair_id,
                    "run_id": self.args.run_id,
                    "problem_id": problem["id"],
                    "source": problem["source"],
                    "year": problem["year"],
                    "label": problem["label"],
                    "created_at": utc_now(),
                    "chosen_source_id": best["source_id"],
                    "rejected_source_id": item["source_id"],
                    "chosen_score": strict_score_fields(best.get("score", {})),
                    "rejected_score": strict_score_fields(item.get("score", {})),
                    "problem": problem["problem"],
                    "chosen_proof": best.get("proof", ""),
                    "rejected_proof": item.get("proof", ""),
                    "preference_type": "strict_auditor_reasoning_harness",
                },
            )

    async def judge_proof(
        self,
        problem: dict[str, Any],
        *,
        source_id: str,
        proof: str,
        capped: bool,
        output_path: Path,
        model: str,
    ) -> dict[str, Any]:
        existing = load_jsonl_index(output_path, "source_id")
        if source_id in existing:
            return existing[source_id]
        base = {
            "source_id": source_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "judge_model": model,
            "created_at": utc_now(),
        }
        if self.args.dry_run:
            score_obj = normalize_judge_score(
                {
                    "rubric_score": 0,
                    "failure_tags": ["dry_run"],
                    "first_serious_flaw": "Dry run did not call a judge.",
                    "critique": "Dry run.",
                    "confidence": "low",
                    "notes": "No model call was made.",
                    "is_complete": False,
                    "is_salvageable": False,
                    "repair_hint": "",
                },
                capped=capped,
            )
            row = {**base, "status": "dry_run", **score_obj, "error": "", "raw_judge_response": "", "latency_seconds": 0.0, "usage": {}}
            append_jsonl(output_path, row)
            return row
        if not proof.strip():
            score_obj = normalize_judge_score(
                {
                    "rubric_score": 0,
                    "failure_tags": ["incomplete_proof"],
                    "first_serious_flaw": "No proof text was produced.",
                    "critique": "No proof text was produced.",
                    "confidence": "high",
                    "notes": "",
                    "is_complete": False,
                    "is_salvageable": False,
                    "repair_hint": "",
                },
                capped=capped,
            )
            row = {**base, "status": "completed", **score_obj, "error": "", "raw_judge_response": "", "latency_seconds": 0.0, "usage": {}}
            append_jsonl(output_path, row)
            return row
        try:
            raw, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="judge",
                model=model,
                messages=judge_messages(problem, proof, judge_system=self.judge_system, model=model, chat_format=self.args.chat_format),
                temperature=self.args.judge_temperature,
                max_tokens=self.args.judge_max_tokens,
                top_p=1.0,
                json_mode=True,
                reasoning_effort=self.args.gpt_oss_judge_reasoning,
            )
            parsed = extract_json_object_with_reasoning_fallback(
                raw,
                response_metadata,
                extract_final=self.args.extract_final_channel,
                model=model,
            )
            score_obj = normalize_judge_score(parsed, capped=capped)
            score_obj, _ = expected_answer_gate(
                score_obj,
                problem_id=str(problem.get("id", "")),
                proof=proof,
                expected_answers=self.expected_answers,
            )
            row = {
                **base,
                "status": "completed",
                **score_obj,
                "error": "",
                "raw_judge_response": raw,
                "finish_reason": finish_reason,
                "latency_seconds": latency,
                "usage": usage,
                "raw_content_chars": response_metadata.get("content_chars", 0),
                "reasoning_content_chars": response_metadata.get("reasoning_content_chars", 0),
            }
        except Exception as exc:
            score_obj = normalize_judge_score(
                {
                    "rubric_score": 0,
                    "failure_tags": ["candidate_selection_failure"],
                    "first_serious_flaw": "Judge call failed.",
                    "critique": "",
                    "confidence": "low",
                    "notes": "",
                    "is_complete": False,
                    "is_salvageable": False,
                    "repair_hint": "",
                },
                capped=capped,
            )
            row = {**base, "status": "error", **score_obj, "error": str(exc), "raw_judge_response": "", "latency_seconds": 0.0, "usage": {}}
        append_jsonl(output_path, row)
        return row

    async def repair_candidate(self, problem: dict[str, Any], candidate: dict[str, Any], score: dict[str, Any], repair_index: int) -> dict[str, Any]:
        repair_id = f"{candidate['candidate_id']}::repair_{repair_index}"
        existing = load_jsonl_index(self.paths["repairs"], "repair_id")
        if repair_id in existing:
            return existing[repair_id]
        try:
            proof, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="repair",
                model=self.args.repair_model,
                messages=repair_messages(problem, candidate.get("proof", ""), score, model=self.args.repair_model, chat_format=self.args.chat_format),
                temperature=self.args.repair_temperature,
                max_tokens=self.args.repair_max_tokens,
                top_p=self.args.repair_top_p,
                reasoning_effort=self.args.gpt_oss_repair_reasoning,
            )
            if self.args.extract_final_channel or model_uses_harmony(self.args.repair_model):
                proof = extract_final_channel_text(proof)
            status = "completed"
            error = ""
        except Exception as exc:
            proof = ""
            usage = {}
            finish_reason = "error"
            latency = 0.0
            status = "error"
            error = str(exc)
        capped = finish_reason in {"length", "max_tokens"} or int(usage.get("completion_tokens", 0) or 0) >= self.args.repair_max_tokens
        if status == "completed":
            proof, response_metadata = split_qwen_thinking_output(
                model=self.args.repair_model,
                text=proof,
                response_metadata=response_metadata,
                capped=capped,
            )
        row = {
            "repair_id": repair_id,
            "source_candidate_id": candidate["candidate_id"],
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "source": problem["source"],
            "year": problem["year"],
            "label": problem["label"],
            "repair_index": repair_index,
            "model": self.args.repair_model,
            "created_at": utc_now(),
            "status": status,
            "problem": problem["problem"],
            "proof": proof,
            "finish_reason": finish_reason,
            "capped": capped,
            "error": error,
            "latency_seconds": latency,
            "usage": usage,
            "input_score": strict_score_fields(score),
            "raw_content_chars": response_metadata.get("content_chars", 0) if status == "completed" else 0,
            "reasoning_content_chars": response_metadata.get("reasoning_content_chars", 0) if status == "completed" else 0,
            "reasoning_content": response_metadata.get("reasoning_content", "") if status == "completed" else "",
        }
        append_jsonl(self.paths["repairs"], row)
        print(f"{problem['id']} repair {repair_index}: {status} tokens={usage.get('completion_tokens')} capped={capped}", flush=True)
        return row

    async def consolidate(self, problem: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not self.args.consolidate or not pool:
            return None
        consolidation_id = f"{self.args.run_id}::{problem['id']}::consolidated"
        existing = load_jsonl_index(self.paths["consolidations"], "consolidation_id")
        if consolidation_id in existing:
            return existing[consolidation_id]
        top = sorted(pool, key=score_sort_key, reverse=True)[: self.args.top_candidates_for_consolidation]
        try:
            proof, usage, finish_reason, latency, response_metadata = await self.caller.call(
                role="consolidator",
                model=self.args.consolidator_model,
                messages=consolidate_messages(problem, top, model=self.args.consolidator_model, chat_format=self.args.chat_format),
                temperature=self.args.consolidator_temperature,
                max_tokens=self.args.consolidator_max_tokens,
                top_p=self.args.consolidator_top_p,
                reasoning_effort=self.args.gpt_oss_consolidator_reasoning,
            )
            if self.args.extract_final_channel or model_uses_harmony(self.args.consolidator_model):
                proof = extract_final_channel_text(proof)
            status = "completed"
            error = ""
        except Exception as exc:
            proof = ""
            usage = {}
            finish_reason = "error"
            latency = 0.0
            status = "error"
            error = str(exc)
        capped = finish_reason in {"length", "max_tokens"} or int(usage.get("completion_tokens", 0) or 0) >= self.args.consolidator_max_tokens
        if status == "completed":
            proof, response_metadata = split_qwen_thinking_output(
                model=self.args.consolidator_model,
                text=proof,
                response_metadata=response_metadata,
                capped=capped,
            )
        row = {
            "consolidation_id": consolidation_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "source": problem["source"],
            "year": problem["year"],
            "label": problem["label"],
            "model": self.args.consolidator_model,
            "created_at": utc_now(),
            "status": status,
            "problem": problem["problem"],
            "proof": proof,
            "finish_reason": finish_reason,
            "capped": capped,
            "error": error,
            "latency_seconds": latency,
            "usage": usage,
            "source_ids": [item["source_id"] for item in top],
            "raw_content_chars": response_metadata.get("content_chars", 0) if status == "completed" else 0,
            "reasoning_content_chars": response_metadata.get("reasoning_content_chars", 0) if status == "completed" else 0,
            "reasoning_content": response_metadata.get("reasoning_content", "") if status == "completed" else "",
        }
        append_jsonl(self.paths["consolidations"], row)
        print(f"{problem['id']} consolidation: {status} tokens={usage.get('completion_tokens')} capped={capped}", flush=True)
        return row

    async def run_problem(self, problem: dict[str, Any]) -> None:
        if self.stop_requested():
            return
        selections = load_jsonl_index(self.paths["selections"], "problem_id")
        if problem["id"] in selections:
            return

        candidates: list[dict[str, Any]] = []
        for attempt_index, strategy in enumerate(self.strategies, 1):
            if self.stop_requested():
                break
            initial_stage = "candidate_generation" if self.args.candidate_only else "initial_attempt_and_judge"
            initial_calls = 1 if self.args.candidate_only else 2
            if not self.can_spend(initial_stage, initial_calls, required=True):
                break
            candidate = await self.generate_candidate(problem, strategy, attempt_index)
            if self.args.candidate_only:
                candidates.append(
                    {
                        "source_id": candidate["candidate_id"],
                        "kind": "candidate",
                        "proof": candidate.get("proof", ""),
                        "capped": bool(candidate.get("capped")),
                        "score": {},
                        "source_row": candidate,
                    }
                )
                continue
            score = await self.judge_proof(
                problem,
                source_id=candidate["candidate_id"],
                proof=candidate.get("proof", ""),
                capped=bool(candidate.get("capped")),
                output_path=self.paths["candidate_scores"],
                model=self.args.judge_model,
            )
            candidate_item = {
                "source_id": candidate["candidate_id"],
                "kind": "candidate",
                "proof": candidate.get("proof", ""),
                "capped": bool(candidate.get("capped")),
                "score": score,
                "source_row": candidate,
            }
            self.apply_hard_score_gates(problem, candidate_item)
            candidates.append(candidate_item)
            if int(candidate_item["score"].get("rubric_score", 0)) >= self.args.early_stop_score and not candidate.get("capped"):
                print(f"{problem['id']} early stop at candidate score={candidate_item['score'].get('rubric_score')}", flush=True)
                break

        if self.args.candidate_only:
            print(f"{problem['id']} candidate-only generated {len(candidates)} candidates", flush=True)
            return

        retry_inputs = [
            item
            for item in sorted(candidates, key=score_sort_key, reverse=True)
            if int(item["score"].get("rubric_score", 0)) < self.args.accept_score
        ][: self.args.targeted_retries_per_problem]

        for retry_index, item in enumerate(retry_inputs, 1):
            if self.stop_requested():
                break
            if not self.can_spend("targeted_retry_and_judge", 2):
                break
            retry_candidate = await self.generate_targeted_retry(problem, item, retry_index)
            retry_score = await self.judge_proof(
                problem,
                source_id=retry_candidate["candidate_id"],
                proof=retry_candidate.get("proof", ""),
                capped=bool(retry_candidate.get("capped")),
                output_path=self.paths["candidate_scores"],
                model=self.args.judge_model,
            )
            retry_item = {
                "source_id": retry_candidate["candidate_id"],
                "kind": "targeted_retry",
                "proof": retry_candidate.get("proof", ""),
                "capped": bool(retry_candidate.get("capped")),
                "score": retry_score,
                "source_row": retry_candidate,
            }
            self.apply_hard_score_gates(problem, retry_item)
            candidates.append(retry_item)

        repair_candidates = [
            item
            for item in sorted(candidates, key=score_sort_key, reverse=True)
            if int(item["score"].get("rubric_score", 0)) < self.args.accept_score
            and int(item["score"].get("rubric_score", 0)) >= self.args.min_repair_score
            and not bool(item.get("capped"))
        ]

        if self.args.require_breaker_for_repair:
            checked_repair_candidates: list[dict[str, Any]] = []
            for item in repair_candidates:
                if len(checked_repair_candidates) >= self.args.repairs_per_problem:
                    break
                if not self.can_spend("repair_breaker", 1):
                    break
                breaker = await self.break_proof(problem, item)
                self.apply_breaker_to_item(item, breaker)
                self.apply_hard_score_gates(problem, item)
                if breaker_identified_repair_flaw(item):
                    checked_repair_candidates.append(item)
            repair_inputs = checked_repair_candidates
        else:
            repair_inputs = repair_candidates[: self.args.repairs_per_problem]

        repairs: list[dict[str, Any]] = []
        for repair_index, item in enumerate(repair_inputs, 1):
            if self.stop_requested():
                break
            if not self.can_spend("repair_and_judge", 2):
                break
            repair = await self.repair_candidate(problem, item["source_row"], item["score"], repair_index)
            repair_score = await self.judge_proof(
                problem,
                source_id=repair["repair_id"],
                proof=repair.get("proof", ""),
                capped=bool(repair.get("capped")),
                output_path=self.paths["repair_scores"],
                model=self.args.judge_model,
            )
            repair_item = {
                "source_id": repair["repair_id"],
                "kind": "repair",
                "proof": repair.get("proof", ""),
                "capped": bool(repair.get("capped")),
                "score": repair_score,
                "source_row": repair,
            }
            self.apply_hard_score_gates(problem, repair_item)
            repairs.append(repair_item)

        pool = candidates + repairs
        if not pool:
            best = {
                "source_id": f"{self.args.run_id}::{problem['id']}::empty",
                "kind": "empty",
                "proof": "",
                "capped": False,
                "score": normalize_judge_score({"rubric_score": 0, "failure_tags": ["candidate_selection_failure"], "confidence": "low"}),
                "source_row": {},
            }
        else:
            if self.args.extract_conclusions and self.can_spend("conclusion_extraction", len(pool)):
                await self.cluster_conclusions(problem, pool)
                for item in pool:
                    self.apply_hard_score_gates(problem, item)
                self.apply_conclusion_consensus_gates(pool)
            if self.args.adversarial_break and self.can_spend("adversarial_break", min(self.args.breaker_top_n, len(pool))):
                await self.adversarial_break_pool(problem, pool)
                self.apply_conclusion_consensus_gates(pool)
            for item in pool:
                self.apply_hard_score_gates(problem, item)
            self.apply_conclusion_consensus_gates(pool)

            consolidation_pool = self.strict_verified_stage_pool(pool, stage="consolidation")
            consolidation = None
            if consolidation_pool and self.args.consolidate and self.can_spend("consolidation_and_judge", 2):
                consolidation = await self.consolidate(problem, consolidation_pool)
            if consolidation:
                consolidation_score = await self.judge_proof(
                    problem,
                    source_id=consolidation["consolidation_id"],
                    proof=consolidation.get("proof", ""),
                    capped=bool(consolidation.get("capped")),
                    output_path=self.paths["consolidation_scores"],
                    model=self.args.judge_model,
                )
                consolidation_item = {
                    "source_id": consolidation["consolidation_id"],
                    "kind": "consolidation",
                    "proof": consolidation.get("proof", ""),
                    "capped": bool(consolidation.get("capped")),
                    "score": consolidation_score,
                    "source_row": consolidation,
                }
                self.apply_hard_score_gates(problem, consolidation_item)
                if (
                    self.args.adversarial_break
                    and self.args.require_breaker_for_selection_stages
                    and self.can_spend("consolidation_breaker", 1)
                ):
                    breaker = await self.break_proof(problem, consolidation_item)
                    self.apply_breaker_to_item(consolidation_item, breaker)
                    self.apply_hard_score_gates(problem, consolidation_item)
                if self.args.extract_conclusions and self.can_spend("consolidation_conclusion_extraction", 1):
                    extraction = await self.extract_conclusion(problem, consolidation_item)
                    consolidation_item["conclusion"] = extraction
                    consolidation_item["conclusion_key"] = normalize_conclusion_key(
                        str(extraction.get("normalized_conclusion", "") or extraction.get("conclusion", ""))
                    )
                pool.append(consolidation_item)
                self.apply_conclusion_consensus_gates(pool)

            tournament_n = min(self.args.tournament_size, len(pool))
            tournament_calls = tournament_n * (tournament_n - 1) // 2
            if self.args.tournament and self.can_spend("pairwise_tournament", tournament_calls):
                await self.run_tournament(problem, pool)
            for item in pool:
                self.apply_hard_score_gates(problem, item)
            self.apply_conclusion_consensus_gates(pool)
            best = self.select_best_item(pool)
            best = await self.polish_if_valid(problem, best)
            post_polish_reasons = self.selection_gate_reasons(best)
            best["selection_gate_reasons"] = post_polish_reasons
            best["selection_allowed"] = not post_polish_reasons

        selection = {
            "problem_id": problem["id"],
            "run_id": self.args.run_id,
            "created_at": utc_now(),
            "selected_source_id": best["source_id"],
            "selected_kind": best["kind"],
            "selected_score": strict_score_fields(best["score"]),
            "candidate_count": len(candidates),
            "repair_count": len(repairs),
            "used_consolidation": best["kind"] == "consolidation",
            "used_polish": best["kind"] == "polished",
            "tournament_wins": int(best.get("tournament_wins", 0)),
            "cluster_support": int(best.get("cluster_support", 1)),
            "conclusion_key": str(best.get("conclusion_key", "")),
            "final_answer": str(best.get("conclusion", {}).get("final_answer", "")) if isinstance(best.get("conclusion"), dict) else "",
            "conclusion_confidence": str(best.get("conclusion", {}).get("confidence", "")) if isinstance(best.get("conclusion"), dict) else "",
            "capped": bool(best.get("capped")),
            "selection_allowed": bool(best.get("selection_allowed", True)),
            "selection_gate_reasons": list(best.get("selection_gate_reasons", []) or []),
            "selection_forced_fallback": bool(best.get("selection_forced_fallback", False)),
            "gate_reasons": list(best.get("gate_reasons", []) or []),
            "arithmetic_check": best.get("arithmetic_check", {}),
        }
        append_jsonl(self.paths["selections"], selection)
        self.write_training_pairs(problem, pool, best)
        self.append_final_rows(problem, best)
        print(f"{problem['id']} selected {best['kind']} score={best['score'].get('rubric_score')}", flush=True)

    def append_final_rows(self, problem: dict[str, Any], best: dict[str, Any]) -> None:
        prompt_id = f"{self.args.run_id}::{problem['id']}::attempt_1"
        proof = str(best.get("proof", ""))
        score = normalize_judge_score(best.get("score", {}), capped=bool(best.get("capped")))
        attempt_row = {
            "attempt_id": prompt_id,
            "prompt_id": prompt_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "source": problem["source"],
            "year": problem["year"],
            "label": problem["label"],
            "attempt_index": 1,
            "model": f"{self.args.solver_model}+solver_harness",
            "created_at": utc_now(),
            "problem": problem["problem"],
            "status": "completed" if proof.strip() else "error",
            "proof": proof,
            "error": "" if proof.strip() else "harness selected no proof",
            "latency_seconds": 0.0,
            "usage": {},
        }
        score_row = {
            "attempt_id": prompt_id,
            "run_id": self.args.run_id,
            "problem_id": problem["id"],
            "judge_model": self.args.judge_model,
            "created_at": utc_now(),
            "status": "completed",
            **strict_score_fields(score),
            "error": "",
            "raw_judge_response": "",
            "latency_seconds": 0.0,
            "usage": {},
        }
        append_jsonl(self.paths["attempts"], attempt_row)
        append_jsonl(self.paths["scores"], score_row)

    async def run(self) -> int:
        self.write_config("started")
        problems = selected_problems(self.args)
        if not self.paths["prompts"].exists() or self.args.overwrite:
            write_jsonl(self.paths["prompts"], self.final_prompt_rows(problems))
        for problem in problems:
            if self.stop_requested():
                reason = "time limit" if self.time_exceeded() else "model call budget"
                print(f"{reason} reached; saving partial run", flush=True)
                break
            await self.run_problem(problem)
        if self.args.candidate_only:
            summary = {
                "status": "candidate_only",
                "rows": 0,
                "candidate_rows": sum(1 for _line_no, _row in iter_jsonl(self.paths["candidates"])) if self.paths["candidates"].exists() else 0,
                "scores_path": str(self.paths["scores"]),
            }
            write_json(self.paths["summary"], summary)
            self.write_harness_summary(problems, summary)
            self.write_config("candidate_only_completed")
            return 0
        summary = build_summary(self.paths["scores"])
        write_json(self.paths["summary"], summary)
        self.write_harness_summary(problems, summary)
        self.write_calibration_report(problems)
        validation_rc = self.validate_final_artifacts()
        self.write_config("completed" if validation_rc == 0 else "validation_failed")
        return validation_rc if self.args.fail_on_validation else 0

    def write_harness_summary(self, problems: list[dict[str, Any]], strict_summary: dict[str, Any]) -> None:
        def rows(path: Path) -> list[dict[str, Any]]:
            return [row for _, row in iter_jsonl(path)] if path.exists() else []

        candidate_rows = rows(self.paths["candidates"])
        repair_rows = rows(self.paths["repairs"])
        selection_rows = rows(self.paths["selections"])
        conclusion_rows = rows(self.paths["conclusion_extractions"])
        cluster_rows = rows(self.paths["conclusion_clusters"])
        arithmetic_rows = rows(self.paths["arithmetic_checks"])
        breaker_rows = rows(self.paths["proof_breaks"])
        tournament_rows = rows(self.paths["tournament_matches"])
        polished_rows = rows(self.paths["polished_proofs"])
        training_pair_rows = rows(self.paths["training_pairs"])
        harness_summary = {
            "run_id": self.args.run_id,
            "created_at": utc_now(),
            "harness_mode": self.args.harness_mode,
            "full_reasoning": self.args.full_reasoning,
            "problem_count": len(problems),
            "selected_problem_count": len(selection_rows),
            "candidate_rows": len(candidate_rows),
            "repair_rows": len(repair_rows),
            "conclusion_rows": len(conclusion_rows),
            "conclusion_cluster_rows": len(cluster_rows),
            "arithmetic_check_rows": len(arithmetic_rows),
            "proof_break_rows": len(breaker_rows),
            "tournament_match_rows": len(tournament_rows),
            "polished_rows": len(polished_rows),
            "training_pair_rows": len(training_pair_rows),
            "candidate_cap_count": sum(1 for row in candidate_rows if row.get("capped")),
            "repair_cap_count": sum(1 for row in repair_rows if row.get("capped")),
            "arithmetic_failure_count": sum(1 for row in arithmetic_rows if row.get("status") == "failed"),
            "selection_gate_block_count": sum(1 for row in selection_rows if row.get("selection_gate_reasons")),
            "forced_fallback_selection_count": sum(1 for row in selection_rows if row.get("selection_forced_fallback")),
            "model_calls_made": self.caller.calls_made,
            "remaining_model_calls": self.remaining_model_calls(),
            "budget_skips": dict(self.budget_skips),
            "estimated_model_calls_per_problem": self.estimated_model_calls_per_problem(),
            "strict_summary": strict_summary,
            "selection_scores": {row["problem_id"]: row["selected_score"]["rubric_score"] for row in selection_rows},
            "paths": {key: str(path) for key, path in self.paths.items()},
        }
        write_json(self.paths["harness_summary"], harness_summary)

    def read_manual_audit_scores(self) -> dict[str, int]:
        if not self.args.manual_audit:
            return {}
        path = Path(self.args.manual_audit)
        if not path.exists():
            raise SystemExit(f"{path}: manual audit path does not exist")
        if path.suffix.lower() == ".json":
            data = read_json(path)
            source = data.get("scores", data)
            if not isinstance(source, dict):
                raise SystemExit(f"{path}: expected object or scores object")
            out = {}
            for key, value in source.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    out[str(key)] = max(0, min(10, value))
                elif isinstance(value, dict) and isinstance(value.get("score"), int):
                    out[str(key)] = max(0, min(10, int(value["score"])))
            return out
        text = path.read_text(encoding="utf-8")
        scores: dict[str, int] = {}
        for line in text.splitlines():
            if not line.strip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3 or cells[0].lower() in {"problem", "---"}:
                continue
            label = cells[0]
            candidates = re.findall(r"\b(10|[0-9])\b", cells[2])
            if candidates:
                scores[label] = int(candidates[0])
        return scores

    def write_calibration_report(self, problems: list[dict[str, Any]]) -> None:
        manual_scores = self.read_manual_audit_scores()
        if not manual_scores:
            return
        labels = {problem["id"]: str(problem.get("label", problem["id"])) for problem in problems}
        rows = []
        for _, row in iter_jsonl(self.paths["scores"]):
            label = labels.get(str(row.get("problem_id")), str(row.get("problem_id")))
            manual = manual_scores.get(str(row.get("problem_id")), manual_scores.get(label))
            if manual is None:
                continue
            predicted = row.get("rubric_score")
            if not isinstance(predicted, int) or isinstance(predicted, bool):
                continue
            rows.append(
                {
                    "problem_id": row.get("problem_id"),
                    "label": label,
                    "harness_score": predicted,
                    "manual_score": manual,
                    "delta": predicted - manual,
                    "abs_delta": abs(predicted - manual),
                    "failure_tags": row.get("failure_tags", []),
                }
            )
        summary = {
            "created_at": utc_now(),
            "manual_audit": self.args.manual_audit,
            "rows": rows,
            "count": len(rows),
            "mean_abs_delta": statistics.mean(row["abs_delta"] for row in rows) if rows else None,
            "bias": statistics.mean(row["delta"] for row in rows) if rows else None,
            "exact_match_count": sum(1 for row in rows if row["delta"] == 0),
        }
        write_json(self.paths["calibration_report"], summary)
        markdown_lines = [
            "# Calibration Report",
            "",
            f"- Manual audit: `{self.args.manual_audit}`",
            f"- Compared rows: {summary['count']}",
            f"- Mean absolute delta: {summary['mean_abs_delta']}",
            f"- Bias: {summary['bias']}",
            f"- Exact matches: {summary['exact_match_count']}",
            "",
            "| Problem | Harness | Manual | Delta | Failure tags |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            tags = ", ".join(str(tag) for tag in row.get("failure_tags", []))
            markdown_lines.append(
                f"| {row['label']} | {row['harness_score']} | {row['manual_score']} | {row['delta']} | {tags} |"
            )
        self.paths["calibration_markdown"].write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    def validate_final_artifacts(self) -> int:
        commands = [
            [sys.executable, str(VALIDATOR), "eval-prompts", str(self.paths["prompts"]), "--source", self.args.input],
            [sys.executable, str(VALIDATOR), "eval-attempts", str(self.paths["attempts"]), "--source", str(self.paths["prompts"])],
            [sys.executable, str(VALIDATOR), "eval-scores", str(self.paths["scores"]), "--source", str(self.paths["attempts"])],
            [sys.executable, str(VALIDATOR), "eval-summary", str(self.paths["summary"])],
        ]
        if self.args.allow_target:
            for command in commands[:3]:
                command.append("--allow-target")
        results = []
        for command in commands:
            proc = subprocess.run(command, text=True, capture_output=True)
            if proc.stdout.strip():
                print(proc.stdout.rstrip())
            if proc.stderr.strip():
                print(proc.stderr.rstrip(), file=sys.stderr)
            results.append({"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        report = {"created_at": utc_now(), "ok": all(item["returncode"] == 0 for item in results), "results": results}
        write_json(self.paths["validation_report"], report)
        return 0 if report["ok"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a verifier-guided best-of-N Putnam proof solver harness.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--solver-base-url", default="", help="OpenAI-compatible endpoint for solver and targeted-retry calls. Defaults to --base-url.")
    parser.add_argument("--judge-base-url", default="", help="OpenAI-compatible endpoint for judge, conclusion, and tournament calls. Defaults to --base-url.")
    parser.add_argument("--repair-base-url", default="", help="OpenAI-compatible endpoint for repair calls. Defaults to solver endpoint when repair model equals solver model, otherwise --base-url.")
    parser.add_argument("--consolidator-base-url", default="", help="OpenAI-compatible endpoint for consolidation and polish calls. Defaults to solver endpoint when consolidator model equals solver model, otherwise --base-url.")
    parser.add_argument("--breaker-base-url", default="", help="OpenAI-compatible endpoint for adversarial proof-breaker calls. Defaults to judge endpoint when breaker model equals judge model, otherwise --base-url.")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--solver-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--breaker-model", default="", help="Model used for adversarial proof breaking. Defaults to --judge-model.")
    parser.add_argument("--repair-model", default="")
    parser.add_argument("--consolidator-model", default="")
    parser.add_argument("--solver-prompt", default=str(DEFAULT_SOLVER_PROMPT))
    parser.add_argument("--judge-prompt", default=str(DEFAULT_JUDGE_PROMPT))
    parser.add_argument("--strategies", default="", help="Optional JSON list of strategy objects.")
    parser.add_argument("--expected-answers", default="", help="Optional JSON answer-key gate for closed-answer dev problems. Used only after generation for grading/selection.")
    parser.add_argument("--attempts-per-problem", type=int, default=8)
    parser.add_argument("--targeted-retries-per-problem", type=int, default=None)
    parser.add_argument("--repairs-per-problem", type=int, default=3)
    parser.add_argument("--top-candidates-for-consolidation", type=int, default=4)
    parser.add_argument("--breaker-top-n", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=4)
    parser.add_argument("--accept-score", type=int, default=9)
    parser.add_argument("--early-stop-score", type=int, default=10)
    parser.add_argument("--min-repair-score", type=int, default=2)
    parser.add_argument("--solver-temperature", type=float, default=0.7)
    parser.add_argument("--solver-top-p", type=float, default=0.95)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--repair-temperature", type=float, default=0.3)
    parser.add_argument("--repair-top-p", type=float, default=0.9)
    parser.add_argument("--consolidator-temperature", type=float, default=0.2)
    parser.add_argument("--consolidator-top-p", type=float, default=0.9)
    parser.add_argument("--solver-max-tokens", type=int, default=8192)
    parser.add_argument("--solver-cap-retry-on-capped", action=argparse.BooleanOptionalAction, default=True, help="Use a concise solver retry mode when the prior solver attempt hit the generation cap.")
    parser.add_argument("--solver-cap-retry-max-tokens", type=int, default=4096, help="Max tokens for concise solver retries after a capped thinking attempt.")
    parser.add_argument("--solver-stop", action="append", default=[], help="Stop sequence for solver and solver cap-retry calls. Can be passed multiple times.")
    parser.add_argument("--judge-max-tokens", type=int, default=1800)
    parser.add_argument("--repair-max-tokens", type=int, default=8192)
    parser.add_argument("--consolidator-max-tokens", type=int, default=8192)
    parser.add_argument("--conclusion-max-tokens", type=int, default=600)
    parser.add_argument("--breaker-max-tokens", type=int, default=1200)
    parser.add_argument("--tournament-max-tokens", type=int, default=900)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--time-limit-minutes", type=float, default=0.0)
    parser.add_argument("--max-total-model-calls", type=int, default=0, help="Hard model-call budget. Optional full-reasoning stages skip cleanly when the remaining budget is too small.")
    parser.add_argument("--chat-format", choices=["auto", "default", "gpt-oss"], default="auto")
    parser.add_argument("--extract-final-channel", action="store_true")
    parser.add_argument("--gpt-oss-reasoning", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--gpt-oss-solver-reasoning", choices=["", "low", "medium", "high"], default="", help="Override reasoning effort for solver and targeted-retry calls.")
    parser.add_argument("--gpt-oss-judge-reasoning", choices=["", "low", "medium", "high"], default="", help="Override reasoning effort for judge, conclusion, breaker, and tournament calls.")
    parser.add_argument("--gpt-oss-repair-reasoning", choices=["", "low", "medium", "high"], default="", help="Override reasoning effort for repair calls.")
    parser.add_argument("--gpt-oss-consolidator-reasoning", choices=["", "low", "medium", "high"], default="", help="Override reasoning effort for consolidation and polish calls.")
    parser.add_argument("--solver-chat-template-kwargs", default="", help='JSON object passed as vLLM chat_template_kwargs for solver calls, for example \'{"enable_thinking":false}\'.')
    parser.add_argument("--solver-cap-retry-chat-template-kwargs", default='{"enable_thinking":false}', help="JSON object passed as vLLM chat_template_kwargs for solver retries after token-capped attempts.")
    parser.add_argument("--judge-chat-template-kwargs", default="", help="JSON object passed as vLLM chat_template_kwargs for judge, conclusion, and tournament calls.")
    parser.add_argument("--repair-chat-template-kwargs", default="", help="JSON object passed as vLLM chat_template_kwargs for repair calls.")
    parser.add_argument("--consolidator-chat-template-kwargs", default="", help="JSON object passed as vLLM chat_template_kwargs for consolidation and polish calls.")
    parser.add_argument("--breaker-chat-template-kwargs", default="", help="JSON object passed as vLLM chat_template_kwargs for adversarial breaker calls.")
    parser.add_argument("--json-response-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consolidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-reasoning", action="store_true", help="Enable conclusion clustering, proof breaker, tournament, targeted retries, and validity-gated polish.")
    parser.add_argument("--extract-conclusions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adversarial-break", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tournament", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--polish-final", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-arithmetic-sanity-checks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conclusion-consensus-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-breaker-for-repair", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-verified-capped-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-verified-selection-stages", action=argparse.BooleanOptionalAction, default=None, help="Restrict consolidation and tournament to candidates that pass hard selection gates.")
    parser.add_argument("--require-breaker-for-selection-stages", action=argparse.BooleanOptionalAction, default=None, help="When adversarial breaking is enabled, require a clean breaker verdict before consolidation, tournament, or final selection.")
    parser.add_argument("--polish-min-score", type=int, default=9)
    parser.add_argument("--harness-mode", choices=["eval", "training_data"], default="eval")
    parser.add_argument("--candidate-only", action="store_true", help="Generate solver candidates only, without judge/repair/final selection. Use for staged long-context solver runs.")
    parser.add_argument("--training-pairs-per-problem", type=int, default=3)
    parser.add_argument("--training-pair-max-rejected-score", type=int, default=5)
    parser.add_argument("--manual-audit", default="", help="Optional manual audit JSON/Markdown for calibration.")
    parser.add_argument("--allow-target", action="store_true")
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--problem-id", action="append", default=[])
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-validation", action="store_true")
    args = parser.parse_args()
    if args.attempts_per_problem < 1:
        raise SystemExit("--attempts-per-problem must be >= 1")
    if args.targeted_retries_per_problem is None:
        args.targeted_retries_per_problem = 2 if args.full_reasoning else 0
    if args.full_reasoning:
        if args.extract_conclusions is None:
            args.extract_conclusions = True
        if args.adversarial_break is None:
            args.adversarial_break = True
        if args.tournament is None:
            args.tournament = True
        if args.polish_final is None:
            args.polish_final = True
        if args.conclusion_consensus_gate is None:
            args.conclusion_consensus_gate = True
        if args.require_breaker_for_repair is None:
            args.require_breaker_for_repair = True
        if args.strict_verified_selection_stages is None:
            args.strict_verified_selection_stages = True
        if args.require_breaker_for_selection_stages is None:
            args.require_breaker_for_selection_stages = False
    args.extract_conclusions = bool(args.extract_conclusions) if args.extract_conclusions is not None else False
    args.adversarial_break = bool(args.adversarial_break) if args.adversarial_break is not None else False
    args.tournament = bool(args.tournament) if args.tournament is not None else False
    args.polish_final = bool(args.polish_final) if args.polish_final is not None else False
    args.conclusion_consensus_gate = bool(args.conclusion_consensus_gate) if args.conclusion_consensus_gate is not None else False
    args.require_breaker_for_repair = bool(args.require_breaker_for_repair) if args.require_breaker_for_repair is not None else False
    args.strict_verified_selection_stages = bool(args.strict_verified_selection_stages) if args.strict_verified_selection_stages is not None else False
    args.require_breaker_for_selection_stages = bool(args.require_breaker_for_selection_stages) if args.require_breaker_for_selection_stages is not None else False
    if args.targeted_retries_per_problem < 0:
        raise SystemExit("--targeted-retries-per-problem must be >= 0")
    if args.repairs_per_problem < 0:
        raise SystemExit("--repairs-per-problem must be >= 0")
    if args.top_candidates_for_consolidation < 1:
        raise SystemExit("--top-candidates-for-consolidation must be >= 1")
    if args.breaker_top_n < 1:
        raise SystemExit("--breaker-top-n must be >= 1")
    if args.tournament_size < 2:
        raise SystemExit("--tournament-size must be >= 2")
    if not 0 <= args.polish_min_score <= 10:
        raise SystemExit("--polish-min-score must be between 0 and 10")
    if args.training_pairs_per_problem < 0:
        raise SystemExit("--training-pairs-per-problem must be >= 0")
    if not 0 <= args.training_pair_max_rejected_score <= 10:
        raise SystemExit("--training-pair-max-rejected-score must be between 0 and 10")
    if not args.repair_model:
        args.repair_model = args.solver_model
    if not args.consolidator_model:
        args.consolidator_model = args.solver_model
    if not args.breaker_model:
        args.breaker_model = args.judge_model
    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be >= 1")
    if args.solver_cap_retry_max_tokens < 1:
        raise SystemExit("--solver-cap-retry-max-tokens must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    os.environ["GPT_OSS_REASONING"] = args.gpt_oss_reasoning
    harness = SolverHarness(args)
    return asyncio.run(harness.run())


if __name__ == "__main__":
    raise SystemExit(main())

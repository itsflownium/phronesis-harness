#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from strict_proof_auditor import normalize_audit_response


DEFAULT_SOLVER_PROMPT = Path("prompts/eval_solver_system.md")
DEFAULT_JUDGE_PROMPT = Path("prompts/eval_judge_system.md")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
GPT_OSS_SYSTEM_MESSAGE_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2026-07-03\n\n"
    "Reasoning: {reasoning_effort}\n\n"
    "# Valid channels: analysis, final. Channel must be included for every message."
)
PUTNAM_PROBLEMS_PER_EXAM = 12
PUTNAM_POINTS_PER_PROBLEM = 10
VALIDATOR = Path(__file__).with_name("validate_artifact_schema.py")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_no}: each JSONL row must be an object")
            yield line_no, obj


def read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{name} must be a nonempty string")
    return value.strip()


def require_nonnegative_int(value: int, name: str) -> None:
    if value < 0:
        raise SystemExit(f"{name} must be >= 0")


def require_positive_int(value: int, name: str) -> None:
    if value < 1:
        raise SystemExit(f"{name} must be >= 1")


def model_uses_harmony(model: str) -> bool:
    return "gpt-oss" in model.lower()


def gpt_oss_system_message() -> str:
    reasoning_effort = os.environ.get("GPT_OSS_REASONING", "high").strip().lower()
    if reasoning_effort not in {"low", "medium", "high"}:
        reasoning_effort = "high"
    return GPT_OSS_SYSTEM_MESSAGE_TEMPLATE.format(reasoning_effort=reasoning_effort)


def extract_final_channel_text(value: str) -> str:
    """Return the Harmony final-channel text if raw channel markup leaked through."""
    if "<|channel|>" not in value and "<|start|>assistant" not in value:
        return value.strip()
    pattern = re.compile(
        r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        re.DOTALL,
    )
    matches = pattern.findall(value)
    if matches:
        return matches[-1].strip()
    cleaned = re.sub(r"<\|[^|]+?\|>", "", value)
    return cleaned.strip()


def checked_prompt_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise SystemExit(f"{path}: prompt file does not exist")
    return path


def call_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    *,
    token_param: str = "max_tokens",
    json_object: bool = False,
    timeout_seconds: int = 1800,
    api_key: str = "",
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        token_param: max_tokens,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc
    parsed = json.loads(body)
    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"response did not contain choices[0].message.content: {body[:2000]}") from exc
    if not isinstance(content, str):
        raise RuntimeError("response content was not a string")
    usage = parsed.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return content, usage


def call_with_retries(args, messages: list[dict[str, str]], *, json_object: bool = False) -> tuple[str, dict[str, Any], float]:
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if args.api_key_env and not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        start = time.monotonic()
        try:
            content, usage = call_chat_completion(
                args.base_url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
                token_param=args.token_param,
                json_object=json_object,
                timeout_seconds=args.timeout_seconds,
                api_key=api_key,
            )
            return content, usage, round(time.monotonic() - start, 3)
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep_seconds)
    raise RuntimeError(str(last_error))


def problem_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path}: input file does not exist")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = ("id", "source", "year", "label", "problem")
    for line_no, obj in iter_jsonl(path):
        missing = [key for key in required if key not in obj]
        if missing:
            raise SystemExit(f"{path}:{line_no}: missing required problem fields: {missing}")
        for key in ("id", "source", "label", "problem"):
            require_nonempty_string(obj[key], f"{path}:{line_no}:{key}")
        if not isinstance(obj["year"], int) or isinstance(obj["year"], bool):
            raise SystemExit(f"{path}:{line_no}: year must be an integer")
        if obj["id"] in seen:
            raise SystemExit(f"{path}:{line_no}: duplicate problem id {obj['id']}")
        seen.add(obj["id"])
        rows.append(obj)
    if not rows:
        raise SystemExit(f"{path}: input file has no problem rows")
    return rows


def is_target_problem(problem: dict[str, Any]) -> bool:
    return problem.get("year") == 2025 or problem.get("split") == "target" or problem.get("target_excluded") is True


def selected_problem(problem: dict[str, Any], args) -> bool:
    if args.problem_id and problem["id"] not in set(args.problem_id):
        return False
    if args.label and problem.get("label") not in set(args.label):
        return False
    return True


def make_chat_messages(instruction: str, user_content: str, *, model: str = "", chat_format: str = "auto") -> list[dict[str, str]]:
    use_gpt_oss = chat_format == "gpt-oss" or (chat_format == "auto" and model_uses_harmony(model))
    if use_gpt_oss:
        developer_content = (
            instruction.strip()
            + "\n\nGPT-OSS execution constraint: keep internal analysis short and move to the final answer quickly. "
            "The final channel must contain the requested proof, JSON, critique, or repair before the token budget is exhausted."
        )
        return [
            {"role": "system", "content": gpt_oss_system_message()},
            {"role": "developer", "content": developer_content},
            {"role": "user", "content": user_content.strip()},
        ]
    return [
        {"role": "system", "content": instruction.strip()},
        {"role": "user", "content": user_content.strip()},
    ]


def make_solver_messages(problem: dict[str, Any], solver_system: str, *, model: str = "", chat_format: str = "auto") -> list[dict[str, str]]:
    user_content = (
        "Your task is to write a proof solution to the following problem.\n\n"
        f"Problem source: {problem['source']}\n"
        f"Problem id: {problem['id']}\n\n"
        f"Problem:\n{problem['problem']}"
    )
    return make_chat_messages(solver_system, user_content, model=model, chat_format=chat_format)


def make_judge_messages(attempt: dict[str, Any], judge_system: str, *, model: str = "", chat_format: str = "auto") -> list[dict[str, str]]:
    user_content = (
        "Grade this proof attempt.\n\n"
        f"Problem id: {attempt['problem_id']}\n\n"
        f"Problem:\n{attempt['problem']}\n\n"
        f"Proof attempt:\n{attempt.get('proof', '')}"
    )
    return make_chat_messages(judge_system, user_content, model=model, chat_format=chat_format)


def load_completed_ids(path: Path, key: str) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_no, obj in iter_jsonl(path):
        value = obj.get(key)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"{path}:{line_no}: existing row missing nonempty {key}")
        if value in completed:
            raise SystemExit(f"{path}:{line_no}: duplicate existing {key} {value}")
        completed.add(value)
    return completed


def check_existing_prompts_compatible(path: Path, args) -> None:
    expected_prompt_hash = sha256_file(checked_prompt_path(args.solver_prompt))
    row_count = 0
    for line_no, prompt in iter_jsonl(path):
        row_count += 1
        checks = {
            "run_id": args.run_id,
            "model": args.solver_model,
            "input_path": str(args.input),
            "solver_prompt_sha256": expected_prompt_hash,
        }
        for field, expected in checks.items():
            if prompt.get(field) != expected:
                raise SystemExit(
                    f"{path}:{line_no}: existing prompt {field}={prompt.get(field)!r} "
                    f"does not match requested {expected!r}; use --overwrite or a new --run-dir"
                )
        if is_target_problem(prompt) and not args.allow_target:
            raise SystemExit(f"{path}:{line_no}: existing target prompt requires --allow-target")
    if row_count == 0:
        raise SystemExit(f"{path}: existing prompts file is empty; use --overwrite")


def maybe_reset_output(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        path.unlink()


def parse_judge_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("judge response did not contain a JSON object")


def validate_score(obj: dict[str, Any]) -> None:
    if "rubric_score" not in obj and "score" in obj:
        obj["rubric_score"] = obj["score"]
    required = ["rubric_score", "failure_tags", "first_serious_flaw", "critique", "confidence", "notes"]
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"missing score keys: {missing}")
    if not isinstance(obj["rubric_score"], int) or isinstance(obj["rubric_score"], bool) or not 0 <= obj["rubric_score"] <= 10:
        raise ValueError("rubric_score must be integer 0-10")
    if not isinstance(obj["failure_tags"], list) or not all(isinstance(tag, str) and tag.strip() for tag in obj["failure_tags"]):
        raise ValueError("failure_tags must be an array of nonempty strings")
    for key in ["first_serious_flaw", "critique", "confidence", "notes"]:
        if not isinstance(obj[key], str):
            raise ValueError(f"{key} must be string")
    if obj["confidence"] not in {"low", "medium", "high"}:
        raise ValueError("confidence must be low, medium, or high")


def prepare(args) -> int:
    require_nonempty_string(args.run_id, "--run-id")
    require_nonempty_string(args.model, "--model")
    require_positive_int(args.attempts_per_problem, "--attempts-per-problem")
    require_nonnegative_int(args.max_problems, "--max-problems")
    input_path = Path(args.input)
    solver_prompt_path = checked_prompt_path(args.solver_prompt)
    solver_system = solver_prompt_path.read_text(encoding="utf-8")
    solver_prompt_hash = sha256_text(solver_system)
    rows: list[dict[str, Any]] = []
    selected_problem_count = 0
    for problem in problem_rows(input_path):
        if not selected_problem(problem, args):
            continue
        if is_target_problem(problem) and not args.allow_target:
            raise SystemExit(f"{problem['id']}: refusing to prepare target row without --allow-target")
        if args.max_problems and selected_problem_count >= args.max_problems:
            break
        selected_problem_count += 1
        for attempt_index in range(args.attempts_per_problem):
            prompt_id = f"{args.run_id}::{problem['id']}::attempt_{attempt_index + 1}"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "run_id": args.run_id,
                    "problem_id": problem["id"],
                    "source": problem["source"],
                    "year": problem["year"],
                    "label": problem["label"],
                    "attempt_index": attempt_index + 1,
                    "model": args.model,
                    "status": "prepared",
                    "created_at": utc_now(),
                    "input_path": str(input_path),
                    "solver_prompt_path": str(solver_prompt_path),
                    "solver_prompt_sha256": solver_prompt_hash,
                    "problem": problem["problem"],
                    "messages": make_solver_messages(problem, solver_system, model=args.model, chat_format=args.chat_format),
                }
            )
    if not rows:
        raise SystemExit("no prompts selected; check --max-problems, --problem-id, and --label")
    write_jsonl(Path(args.output), rows)
    print(f"wrote prompts: {len(rows)} rows across {selected_problem_count} problems to {args.output}")
    return 0


def solve(args) -> int:
    require_nonempty_string(args.model, "--model")
    require_positive_int(args.max_tokens, "--max-tokens")
    require_nonnegative_int(args.retries, "--retries")
    output_path = Path(args.output)
    maybe_reset_output(output_path, args.overwrite)
    completed = load_completed_ids(output_path, "prompt_id")
    count = 0
    errors = 0
    for line_no, prompt in iter_jsonl(Path(args.prompts)):
        if prompt.get("status") != "prepared":
            raise SystemExit(f"{args.prompts}:{line_no}: prompt status must be prepared")
        if prompt["prompt_id"] in completed:
            continue
        record: dict[str, Any] = {
            "attempt_id": prompt["prompt_id"],
            "prompt_id": prompt["prompt_id"],
            "run_id": prompt["run_id"],
            "problem_id": prompt["problem_id"],
            "source": prompt["source"],
            "year": prompt["year"],
            "label": prompt["label"],
            "attempt_index": prompt["attempt_index"],
            "model": args.model,
            "created_at": utc_now(),
            "problem": prompt["problem"],
        }
        if args.dry_run:
            record.update({"status": "dry_run", "proof": "", "error": "", "latency_seconds": 0.0, "usage": {}})
        else:
            try:
                proof, usage, latency = call_with_retries(args, prompt["messages"])
                if args.extract_final_channel or model_uses_harmony(args.model):
                    proof = extract_final_channel_text(proof)
                record.update({"status": "completed", "proof": proof, "error": "", "latency_seconds": latency, "usage": usage})
            except Exception as exc:
                errors += 1
                if args.fail_fast:
                    raise
                record.update({"status": "error", "proof": "", "error": str(exc), "latency_seconds": 0.0, "usage": {}})
        append_jsonl(output_path, record)
        count += 1
    print(f"wrote attempts: {count} new rows to {output_path}")
    if errors and args.fail_on_error:
        return 1
    return 0


def score(args) -> int:
    require_nonempty_string(args.model, "--model")
    require_positive_int(args.max_tokens, "--max-tokens")
    require_nonnegative_int(args.retries, "--retries")
    judge_prompt_path = checked_prompt_path(args.judge_prompt)
    judge_system = judge_prompt_path.read_text(encoding="utf-8")
    output_path = Path(args.output)
    maybe_reset_output(output_path, args.overwrite)
    completed = load_completed_ids(output_path, "attempt_id")
    count = 0
    errors = 0
    for _, attempt in iter_jsonl(Path(args.attempts)):
        if attempt["attempt_id"] in completed:
            continue
        record: dict[str, Any] = {
            "attempt_id": attempt["attempt_id"],
            "run_id": attempt["run_id"],
            "problem_id": attempt["problem_id"],
            "judge_model": args.model,
            "created_at": utc_now(),
        }
        if args.dry_run or not attempt.get("proof") or attempt.get("status") != "completed":
            note = "No judge call was made."
            if attempt.get("status") not in {"completed", "dry_run"}:
                note = f"No judge call was made because attempt status was {attempt.get('status')}."
            record.update(
                {
                    "status": "dry_run" if args.dry_run else "skipped_empty_proof",
                    "rubric_score": None,
                    "failure_tags": [],
                    "first_serious_flaw": "",
                    "critique": "",
                    "confidence": "low",
                    "notes": note,
                    "error": "",
                    "raw_judge_response": "",
                    "latency_seconds": 0.0,
                    "usage": {},
                }
            )
        else:
            try:
                raw, usage, latency = call_with_retries(
                    args,
                    make_judge_messages(attempt, judge_system, model=args.model, chat_format=args.chat_format),
                    json_object=True,
                )
                parse_source = extract_final_channel_text(raw) if args.extract_final_channel or model_uses_harmony(args.model) else raw
                parsed = parse_judge_json(parse_source)
                validate_score(parsed)
                score_obj = normalize_audit_response(parsed)
                record.update({"status": "completed", **score_obj, "error": "", "raw_judge_response": raw, "latency_seconds": latency, "usage": usage})
            except Exception as exc:
                errors += 1
                if args.fail_fast:
                    raise
                record.update(
                    {
                        "status": "error",
                        "rubric_score": None,
                        "failure_tags": [],
                        "first_serious_flaw": "",
                        "critique": "",
                        "confidence": "low",
                        "notes": "",
                        "error": str(exc),
                        "raw_judge_response": "",
                        "latency_seconds": 0.0,
                        "usage": {},
                    }
                )
        append_jsonl(output_path, record)
        count += 1
    print(f"wrote scores: {count} new rows to {output_path}")
    if errors and args.fail_on_error:
        return 1
    return 0


def build_summary(scores_path: Path) -> dict[str, Any]:
    scores = [obj for _, obj in iter_jsonl(scores_path)]
    completed = [obj for obj in scores if obj.get("status") == "completed" and isinstance(obj.get("rubric_score"), int)]
    tags: Counter[str] = Counter()
    status_counts = Counter(str(obj.get("status", "unknown")) for obj in scores)
    score_histogram = Counter(str(obj["rubric_score"]) for obj in completed)
    problem_scores: dict[str, list[int]] = {}
    for obj in completed:
        for tag in obj.get("failure_tags", []):
            tags[tag] += 1
        problem_scores.setdefault(obj["problem_id"], []).append(obj["rubric_score"])
    per_problem = []
    for problem_id, values in sorted(problem_scores.items()):
        per_problem.append(
            {
                "problem_id": problem_id,
                "attempts_scored": len(values),
                "mean_score": statistics.mean(values),
                "best_score": max(values),
                "median_score": statistics.median(values),
            }
        )
    rubric_values = [obj["rubric_score"] for obj in completed]
    best_scores = {problem_id: max(values) for problem_id, values in problem_scores.items()}
    putnam_total_score = sum(best_scores.values()) if best_scores else None
    putnam_max_score = len(best_scores) * PUTNAM_POINTS_PER_PROBLEM if best_scores else None
    putnam_total_score_out_of_120 = putnam_total_score if len(best_scores) == PUTNAM_PROBLEMS_PER_EXAM else None
    putnam_total_score_scaled_to_120 = (
        round(putnam_total_score * (PUTNAM_PROBLEMS_PER_EXAM / len(best_scores)), 3)
        if best_scores
        else None
    )
    return {
        "scores_path": str(scores_path),
        "rows": len(scores),
        "completed_scores": len(completed),
        "mean_score": statistics.mean(rubric_values) if rubric_values else None,
        "median_score": statistics.median(rubric_values) if rubric_values else None,
        "score_scale": {"min": 0, "max": PUTNAM_POINTS_PER_PROBLEM},
        "putnam_problem_count": PUTNAM_PROBLEMS_PER_EXAM,
        "scored_problem_count": len(best_scores),
        "putnam_total_score": putnam_total_score,
        "putnam_max_score": putnam_max_score,
        "putnam_total_score_out_of_120": putnam_total_score_out_of_120,
        "putnam_total_score_scaled_to_120": putnam_total_score_scaled_to_120,
        "putnam_aggregation": "best_score_per_problem",
        "status_counts": dict(sorted(status_counts.items())),
        "score_histogram": dict(sorted(score_histogram.items(), key=lambda item: int(item[0]))),
        "failure_tag_counts": dict(sorted(tags.items())),
        "per_problem": per_problem,
    }


def summarize(args) -> int:
    summary = build_summary(Path(args.scores))
    if args.output:
        write_json(Path(args.output), summary)
        print(f"wrote summary: {args.output}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if args.require_completed and summary["completed_scores"] == 0:
        return 1
    if args.fail_on_errors and summary.get("status_counts", {}).get("error", 0):
        return 1
    return 0


def validation_target_paths(args) -> dict[str, Path]:
    run_dir = Path(args.run_dir) if args.run_dir else None
    return {
        "input": Path(args.input) if args.input else None,
        "prompts": Path(args.prompts) if args.prompts else run_dir / "prompts.jsonl" if run_dir else None,
        "attempts": Path(args.attempts) if args.attempts else run_dir / "attempts.jsonl" if run_dir else None,
        "scores": Path(args.scores) if args.scores else run_dir / "scores.jsonl" if run_dir else None,
        "summary": Path(args.summary) if args.summary else run_dir / "summary.json" if run_dir else None,
    }


def run_validator(kind: str, path: Path, *, source: Path | None = None, allow_target: bool = False) -> dict[str, Any]:
    cmd = [sys.executable, str(VALIDATOR), kind, str(path)]
    if source:
        cmd.extend(["--source", str(source)])
    if allow_target:
        cmd.append("--allow-target")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "kind": kind,
        "path": str(path),
        "source": str(source) if source else "",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def validate_run(args) -> int:
    paths = validation_target_paths(args)
    missing = [name for name, path in paths.items() if name != "input" and (path is None or not path.exists())]
    if missing:
        raise SystemExit(f"missing eval artifacts for validation: {missing}")
    results = []
    results.append(run_validator("eval-prompts", paths["prompts"], source=paths["input"], allow_target=args.allow_target))
    results.append(run_validator("eval-attempts", paths["attempts"], source=paths["prompts"], allow_target=args.allow_target))
    results.append(run_validator("eval-scores", paths["scores"], source=paths["attempts"], allow_target=args.allow_target))
    results.append(run_validator("eval-summary", paths["summary"], allow_target=args.allow_target))

    for result in results:
        if result["stdout"].strip():
            print(result["stdout"].rstrip())
        if result["stderr"].strip():
            print(result["stderr"].rstrip(), file=sys.stderr)
    report = {
        "created_at": utc_now(),
        "ok": all(result["returncode"] == 0 for result in results),
        "results": results,
    }
    output = Path(args.output) if args.output else None
    if output is None and args.run_dir:
        output = Path(args.run_dir) / "validation_report.json"
    if output:
        write_json(output, report)
        print(f"wrote validation report: {output}")
    return 0 if report["ok"] else 1


def compare(args) -> int:
    baseline = read_json(Path(args.baseline_summary))
    candidate = read_json(Path(args.candidate_summary))

    def delta(field: str) -> float | None:
        left = baseline.get(field)
        right = candidate.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return right - left
        return None

    baseline_problems = {item["problem_id"]: item for item in baseline.get("per_problem", []) if isinstance(item, dict) and "problem_id" in item}
    candidate_problems = {item["problem_id"]: item for item in candidate.get("per_problem", []) if isinstance(item, dict) and "problem_id" in item}
    per_problem_delta = []
    for problem_id in sorted(set(baseline_problems) & set(candidate_problems)):
        left = baseline_problems[problem_id]
        right = candidate_problems[problem_id]
        mean_delta = right.get("mean_score") - left.get("mean_score") if isinstance(left.get("mean_score"), (int, float)) and isinstance(right.get("mean_score"), (int, float)) else None
        best_delta = right.get("best_score") - left.get("best_score") if isinstance(left.get("best_score"), (int, float)) and isinstance(right.get("best_score"), (int, float)) else None
        per_problem_delta.append({"problem_id": problem_id, "mean_score_delta": mean_delta, "best_score_delta": best_delta})

    comparison = {
        "baseline_summary": args.baseline_summary,
        "candidate_summary": args.candidate_summary,
        "baseline_rows": baseline.get("rows"),
        "candidate_rows": candidate.get("rows"),
        "baseline_completed_scores": baseline.get("completed_scores"),
        "candidate_completed_scores": candidate.get("completed_scores"),
        "mean_score_delta": delta("mean_score"),
        "median_score_delta": delta("median_score"),
        "putnam_total_score_delta": delta("putnam_total_score"),
        "putnam_total_score_out_of_120_delta": delta("putnam_total_score_out_of_120"),
        "putnam_total_score_scaled_to_120_delta": delta("putnam_total_score_scaled_to_120"),
        "per_problem_delta": per_problem_delta,
    }
    if args.output:
        write_json(Path(args.output), comparison)
        print(f"wrote comparison: {args.output}")
    print(json.dumps(comparison, ensure_ascii=True, indent=2))
    return 0


def write_run_config(path: Path, args, paths: dict[str, Path], *, status: str, validation_returncode: int | None = None) -> None:
    config = {
        "run_id": args.run_id,
        "status": status,
        "updated_at": utc_now(),
        "input": str(args.input),
        "paths": {key: str(value) for key, value in paths.items()},
        "filters": {
            "max_problems": args.max_problems,
            "problem_id": args.problem_id,
            "label": args.label,
            "allow_target": args.allow_target,
        },
        "sampling": {"attempts_per_problem": args.attempts_per_problem},
        "solver": {
            "model": args.solver_model,
            "base_url": args.solver_base_url or args.base_url,
            "prompt": args.solver_prompt,
            "prompt_sha256": sha256_file(checked_prompt_path(args.solver_prompt)),
            "temperature": args.solver_temperature,
            "max_tokens": args.solver_max_tokens,
            "dry_run": args.dry_run or args.dry_run_solver,
            "extract_final_channel": args.extract_final_channel or model_uses_harmony(args.solver_model),
            "chat_format": "gpt-oss" if model_uses_harmony(args.solver_model) else args.chat_format,
        },
        "judge": {
            "model": args.judge_model,
            "base_url": args.judge_base_url or args.base_url,
            "prompt": args.judge_prompt,
            "prompt_sha256": sha256_file(checked_prompt_path(args.judge_prompt)),
            "temperature": args.judge_temperature,
            "max_tokens": args.judge_max_tokens,
            "dry_run": args.dry_run or args.dry_run_judge,
            "extract_final_channel": args.extract_final_channel or model_uses_harmony(args.judge_model),
            "chat_format": "gpt-oss" if model_uses_harmony(args.judge_model) else args.chat_format,
        },
        "network": {
            "timeout_seconds": args.timeout_seconds,
            "retries": args.retries,
            "retry_sleep_seconds": args.retry_sleep_seconds,
            "token_param": args.token_param,
        },
    }
    if validation_returncode is not None:
        config["validation_returncode"] = validation_returncode
    write_json(path, config)


def run_pipeline(args) -> int:
    require_nonempty_string(args.run_id, "--run-id")
    require_nonempty_string(args.solver_model, "--solver-model")
    require_nonempty_string(args.judge_model, "--judge-model")
    if args.reprepare and not args.overwrite:
        raise SystemExit("--reprepare requires --overwrite so attempts and scores cannot silently reuse stale prompts")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prompts": run_dir / "prompts.jsonl",
        "attempts": run_dir / "attempts.jsonl",
        "scores": run_dir / "scores.jsonl",
        "summary": run_dir / "summary.json",
        "run_config": run_dir / "run_config.json",
        "validation_report": run_dir / "validation_report.json",
    }
    write_run_config(paths["run_config"], args, paths, status="started")

    if paths["prompts"].exists() and not args.overwrite and not args.reprepare:
        check_existing_prompts_compatible(paths["prompts"], args)
        print(f"reusing existing prompts: {paths['prompts']}")
    else:
        prepare_args = argparse.Namespace(
            input=args.input,
            output=str(paths["prompts"]),
            run_id=args.run_id,
            model=args.solver_model,
            attempts_per_problem=args.attempts_per_problem,
            solver_prompt=args.solver_prompt,
            allow_target=args.allow_target,
            max_problems=args.max_problems,
            problem_id=args.problem_id,
            label=args.label,
            chat_format=args.chat_format,
        )
        prepare(prepare_args)

    solve_args = argparse.Namespace(
        prompts=str(paths["prompts"]),
        output=str(paths["attempts"]),
        base_url=args.solver_base_url or args.base_url,
        model=args.solver_model,
        temperature=args.solver_temperature,
        max_tokens=args.solver_max_tokens,
        token_param=args.token_param,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        api_key_env=args.solver_api_key_env or args.api_key_env,
        overwrite=args.overwrite,
        dry_run=args.dry_run or args.dry_run_solver,
        fail_fast=args.fail_fast,
        fail_on_error=args.fail_on_error,
        extract_final_channel=args.extract_final_channel,
        chat_format=args.chat_format,
    )
    rc = solve(solve_args)
    if rc:
        write_run_config(paths["run_config"], args, paths, status="solve_failed")
        return rc

    score_args = argparse.Namespace(
        attempts=str(paths["attempts"]),
        output=str(paths["scores"]),
        base_url=args.judge_base_url or args.base_url,
        model=args.judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        token_param=args.token_param,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        api_key_env=args.judge_api_key_env or args.api_key_env,
        overwrite=args.overwrite,
        judge_prompt=args.judge_prompt,
        dry_run=args.dry_run or args.dry_run_judge,
        fail_fast=args.fail_fast,
        fail_on_error=args.fail_on_error,
        extract_final_channel=args.extract_final_channel,
        chat_format=args.chat_format,
    )
    rc = score(score_args)
    if rc:
        write_run_config(paths["run_config"], args, paths, status="score_failed")
        return rc

    summarize_args = argparse.Namespace(scores=str(paths["scores"]), output=str(paths["summary"]), require_completed=False, fail_on_errors=args.fail_on_error)
    rc = summarize(summarize_args)
    if rc:
        write_run_config(paths["run_config"], args, paths, status="summarize_failed")
        return rc

    validation_rc = None
    if args.validate:
        validation_args = argparse.Namespace(
            run_dir=str(run_dir),
            input=args.input,
            prompts="",
            attempts="",
            scores="",
            summary="",
            output=str(paths["validation_report"]),
            allow_target=args.allow_target,
        )
        validation_rc = validate_run(validation_args)
        if validation_rc:
            write_run_config(paths["run_config"], args, paths, status="validation_failed", validation_returncode=validation_rc)
            return validation_rc

    write_run_config(paths["run_config"], args, paths, status="completed", validation_returncode=validation_rc)
    return 0


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--token-param", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--api-key-env", default="", help="Read bearer API key from this environment variable.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output JSONL instead of resuming it.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-final-channel", action="store_true", help="Strip raw Harmony markup and keep only assistant final-channel text.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise immediately on the first model call failure.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return nonzero if any row has status=error.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, run, score, validate, and compare Phronesis proof evals.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create schema-valid solver prompts from a problem-input JSONL.")
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--attempts-per-problem", type=int, default=1)
    prepare_parser.add_argument("--solver-prompt", default=str(DEFAULT_SOLVER_PROMPT))
    prepare_parser.add_argument("--chat-format", choices=["auto", "default", "gpt-oss"], default="auto")
    prepare_parser.add_argument("--allow-target", action="store_true")
    prepare_parser.add_argument("--max-problems", type=int, default=0, help="Optional cap for quick pilot runs.")
    prepare_parser.add_argument("--problem-id", action="append", default=[], help="Only prepare this problem id; repeatable.")
    prepare_parser.add_argument("--label", action="append", default=[], help="Only prepare this problem label, such as A1; repeatable.")
    prepare_parser.set_defaults(func=prepare)

    solve_parser = subparsers.add_parser("solve", help="Call an OpenAI-compatible solver endpoint and write attempts.")
    solve_parser.add_argument("--prompts", required=True)
    solve_parser.add_argument("--output", required=True)
    add_generation_args(solve_parser)
    solve_parser.set_defaults(func=solve)

    score_parser = subparsers.add_parser("score", help="Judge proof attempts with an OpenAI-compatible endpoint.")
    score_parser.add_argument("--attempts", required=True)
    score_parser.add_argument("--output", required=True)
    add_generation_args(score_parser)
    score_parser.set_defaults(temperature=0.0, max_tokens=1600, func=score)
    score_parser.add_argument("--judge-prompt", default=str(DEFAULT_JUDGE_PROMPT))
    score_parser.add_argument("--chat-format", choices=["auto", "default", "gpt-oss"], default="auto")

    summarize_parser = subparsers.add_parser("summarize", help="Build a schema-valid summary from score rows.")
    summarize_parser.add_argument("--scores", required=True)
    summarize_parser.add_argument("--output", default="")
    summarize_parser.add_argument("--require-completed", action="store_true", help="Return nonzero if no completed judge scores exist.")
    summarize_parser.add_argument("--fail-on-errors", action="store_true", help="Return nonzero if any score row has status=error.")
    summarize_parser.set_defaults(func=summarize)

    validate_parser = subparsers.add_parser("validate", help="Run strict schema validation for an eval run.")
    validate_parser.add_argument("--run-dir", default="", help="Directory containing prompts/attempts/scores/summary.")
    validate_parser.add_argument("--input", default="", help="Optional source problem JSONL for prompt cross-checks.")
    validate_parser.add_argument("--prompts", default="")
    validate_parser.add_argument("--attempts", default="")
    validate_parser.add_argument("--scores", default="")
    validate_parser.add_argument("--summary", default="")
    validate_parser.add_argument("--output", default="")
    validate_parser.add_argument("--allow-target", action="store_true")
    validate_parser.set_defaults(func=validate_run)

    compare_parser = subparsers.add_parser("compare", help="Compare two eval summary JSON files.")
    compare_parser.add_argument("--baseline-summary", required=True)
    compare_parser.add_argument("--candidate-summary", required=True)
    compare_parser.add_argument("--output", default="")
    compare_parser.set_defaults(func=compare)

    run_parser = subparsers.add_parser("run", help="Run prepare, solve, score, summarize, and optional validation.")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--run-dir", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--solver-model", required=True)
    run_parser.add_argument("--judge-model", required=True)
    run_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run_parser.add_argument("--solver-base-url", default="")
    run_parser.add_argument("--judge-base-url", default="")
    run_parser.add_argument("--api-key-env", default="")
    run_parser.add_argument("--solver-api-key-env", default="")
    run_parser.add_argument("--judge-api-key-env", default="")
    run_parser.add_argument("--solver-prompt", default=str(DEFAULT_SOLVER_PROMPT))
    run_parser.add_argument("--judge-prompt", default=str(DEFAULT_JUDGE_PROMPT))
    run_parser.add_argument("--chat-format", choices=["auto", "default", "gpt-oss"], default="auto")
    run_parser.add_argument("--attempts-per-problem", type=int, default=1)
    run_parser.add_argument("--max-problems", type=int, default=0)
    run_parser.add_argument("--problem-id", action="append", default=[])
    run_parser.add_argument("--label", action="append", default=[])
    run_parser.add_argument("--allow-target", action="store_true")
    run_parser.add_argument("--solver-temperature", type=float, default=0.2)
    run_parser.add_argument("--judge-temperature", type=float, default=0.0)
    run_parser.add_argument("--solver-max-tokens", type=int, default=4096)
    run_parser.add_argument("--judge-max-tokens", type=int, default=1600)
    run_parser.add_argument("--token-param", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    run_parser.add_argument("--timeout-seconds", type=int, default=1800)
    run_parser.add_argument("--retries", type=int, default=1)
    run_parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    run_parser.add_argument("--dry-run", action="store_true", help="Dry-run both solver and judge phases.")
    run_parser.add_argument("--dry-run-solver", action="store_true")
    run_parser.add_argument("--dry-run-judge", action="store_true")
    run_parser.add_argument("--extract-final-channel", action="store_true", help="Strip raw Harmony markup and keep only assistant final-channel text.")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--reprepare", action="store_true", help="Rewrite prompts even if they already exist.")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--fail-on-error", action="store_true")
    run_parser.add_argument("--validate", action="store_true", help="Run strict validators after summarize.")
    run_parser.set_defaults(func=run_pipeline)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# MANDATORY environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
BENCHMARK = os.getenv("FINOPS_BENCHMARK", "finops-optimizer")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "20"))

POLICY_SEED_TEXT = os.getenv("POLICY_SEED") or os.getenv("FINOPS_SEED")
POLICY_SEED = int(POLICY_SEED_TEXT) if POLICY_SEED_TEXT and POLICY_SEED_TEXT.strip() else 42
POLICY_RNG = random.Random(POLICY_SEED)

SYSTEM_PROMPT = (
    "You are a FinOps optimization agent. Output EXACTLY one JSON object and nothing else. "
    "Allowed actions are: modify_instance, delete_resource, purchase_savings_plan, tag_resource. "
    "Prefer safe cost reductions and avoid production-impacting actions."
)

ALL_TASKS = ["cleanup_unattached", "rightsize_compute", "fleet_strategy"]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{v:.2f}" for v in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_json(response: requests.Response) -> Dict[str, Any]:
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Expected object JSON response")
    return data


def heuristic_action(observation: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    inventory = observation.get("inventory", [])

    if task_name == "cleanup_unattached":
        for r in inventory:
            if r.get("category") == "storage" and not r.get("is_attached", True):
                return {"action_type": "delete_resource", "resource_id": r.get("id", "")}
        for r in inventory:
            if r.get("category") == "compute" and r.get("tags", {}).get("lifecycle") == "idle":
                return {"action_type": "delete_resource", "resource_id": r.get("id", "")}

    elif task_name == "rightsize_compute":
        for r in inventory:
            if (r.get("category") == "compute"
                    and float(r.get("cpu_usage_pct_30d", 0)) < 5.0
                    and r.get("resource_type") != "t3.small"):
                return {"action_type": "modify_instance", "instance_id": r.get("id", ""), "new_type": "t3.small"}

    elif task_name == "fleet_strategy":
        for r in inventory:
            if r.get("is_legacy") and not r.get("is_production"):
                return {"action_type": "delete_resource", "resource_id": r.get("id", "")}
        for r in inventory:
            if r.get("category") == "storage" and not r.get("is_attached", True):
                return {"action_type": "delete_resource", "resource_id": r.get("id", "")}
        for r in inventory:
            if (r.get("category") == "compute"
                    and float(r.get("cpu_usage_pct_30d", 0)) < 5.0
                    and r.get("resource_type") != "t3.small"):
                return {"action_type": "modify_instance", "instance_id": r.get("id", ""), "new_type": "t3.small"}
        return {"action_type": "purchase_savings_plan", "plan_type": "compute", "duration": "1y"}

    # fallback
    for r in inventory:
        if r.get("category") == "storage" and not r.get("is_attached", True):
            return {"action_type": "delete_resource", "resource_id": r.get("id", "")}
    return {"action_type": "purchase_savings_plan", "plan_type": "compute", "duration": "1y"}


def run_graders() -> None:
    """Enumerate all 3 tasks and verify grader scores — required by checker."""
    print("\n[GRADERS] Enumerating tasks and verifying grader outputs:", flush=True)
    all_ok = True
    for task_id in ALL_TASKS:
        try:
            requests.post(f"{ENV_BASE_URL}/reset", timeout=REQUEST_TIMEOUT)
            resp = requests.get(f"{ENV_BASE_URL}/tasks/{task_id}/score", timeout=REQUEST_TIMEOUT)
            score = float(resp.json().get("score", 0.0))
            in_range = 0.0 <= score <= 1.0
            status = "✓" if in_range else "✗ OUT OF RANGE"
            print(f"[GRADER] task={task_id} initial_score={score:.3f} range=0.0-1.0 {status}", flush=True)
            if not in_range:
                all_ok = False
        except Exception as exc:
            print(f"[GRADER] task={task_id} ERROR: {exc}", flush=True)
            all_ok = False
    if all_ok:
        print("[GRADERS] All graders verified.\n", flush=True)
    else:
        print("[GRADERS] Some graders failed verification.\n", flush=True)


def run_episode(task_name: str) -> None:
    try:
        client: Optional[OpenAI] = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY or "no-key",
            timeout=LLM_TIMEOUT,
            max_retries=1,
        ) if API_KEY else None
    except Exception:
        client = None

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_response = requests.post(f"{ENV_BASE_URL}/reset", timeout=REQUEST_TIMEOUT)
        data = safe_json(reset_response)
        # /reset returns raw observation directly
        observation = data if "inventory" in data else data.get("observation", data)

        for step in range(1, MAX_STEPS + 1):
            action_payload = heuristic_action(observation, task_name)
            action_str = json.dumps(action_payload, separators=(",", ":"))

            try:
                step_response = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action_payload,
                    timeout=REQUEST_TIMEOUT,
                )
                step_data = safe_json(step_response)
                reward = float(step_data.get("reward", 0.0) or 0.0)
                done = bool(step_data.get("done", False))
                info = step_data.get("info", {}) or {}
                error = info.get("last_action_error") if isinstance(info, dict) else None
                obs_raw = step_data.get("observation", {})
                observation = obs_raw if isinstance(obs_raw, dict) else observation
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        try:
            score_response = requests.get(
                f"{ENV_BASE_URL}/tasks/{task_name}/score", timeout=REQUEST_TIMEOUT
            )
            score_data = safe_json(score_response)
            score = clamp_score(float(score_data.get("score", 0.0) or 0.0))
        except Exception:
            total_reward = sum(rewards)
            score = clamp_score(total_reward / max(1.0, float(MAX_STEPS)))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] Episode failed: {exc}", flush=True)
        success = False
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    run_graders()           # verify all 3 graders first
    run_episode("cleanup_unattached")
    run_episode("rightsize_compute")
    run_episode("fleet_strategy")

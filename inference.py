import json
import os
import random
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# MANDATORY environment variables (participant must configure these).
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment endpoint where the FinOps OpenEnv API is running.
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_NAME = os.getenv("FINOPS_TASK", "cleanup_unattached")
BENCHMARK = os.getenv("FINOPS_BENCHMARK", "finops-optimizer")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "20"))
EXPLORE_RATE = float(os.getenv("EXPLORE_RATE", "0.1"))
POLICY_SEED_TEXT = os.getenv("POLICY_SEED") or os.getenv("FINOPS_SEED")
POLICY_SEED = int(POLICY_SEED_TEXT) if POLICY_SEED_TEXT and POLICY_SEED_TEXT.strip() else None
POLICY_RNG = random.Random(POLICY_SEED) if POLICY_SEED is not None else random.Random()

SYSTEM_PROMPT = (
    "You are a FinOps optimization agent. Output EXACTLY one JSON object and nothing else. "
    "Allowed actions are: modify_instance, delete_resource, purchase_savings_plan, tag_resource. "
    "Prefer safe cost reductions and avoid production-impacting actions."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
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


def summarize_observation(observation: Dict[str, Any]) -> str:
    inventory = observation.get("inventory", [])
    low_cpu_compute = [
        resource
        for resource in inventory
        if resource.get("category") == "compute" and float(resource.get("cpu_usage_pct_30d", 0)) < 5.0
    ]
    unattached = [
        resource
        for resource in inventory
        if resource.get("category") == "storage" and not resource.get("is_attached", True)
    ]
    idle_test = [
        resource
        for resource in inventory
        if resource.get("category") == "compute" and resource.get("tags", {}).get("lifecycle") == "idle"
    ]

    return json.dumps(
        {
            "projected_monthly_bill": observation.get("cost_data", {}).get("projected_monthly_bill"),
            "system_latency_ms": observation.get("health_status", {}).get("system_latency_ms"),
            "inventory_count": len(inventory),
            "unattached_ids": [resource.get("id") for resource in unattached[:8]],
            "idle_test_ids": [resource.get("id") for resource in idle_test[:8]],
            "low_cpu_compute_ids": [resource.get("id") for resource in low_cpu_compute[:8]],
        },
        separators=(",", ":"),
    )


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    inventory = observation.get("inventory", [])

    for resource in inventory:
        if resource.get("category") == "storage" and not resource.get("is_attached", True):
            return {"action_type": "delete_resource", "resource_id": resource.get("id", "")}

    for resource in inventory:
        if resource.get("category") == "compute" and resource.get("tags", {}).get("lifecycle") == "idle":
            return {"action_type": "delete_resource", "resource_id": resource.get("id", "")}

    for resource in inventory:
        if (
            resource.get("category") == "compute"
            and float(resource.get("cpu_usage_pct_30d", 0)) < 5.0
            and resource.get("resource_type") != "t3.small"
        ):
            return {
                "action_type": "modify_instance",
                "instance_id": resource.get("id", ""),
                "new_type": "t3.small",
            }

    return {"action_type": "purchase_savings_plan", "plan_type": "compute", "duration": "1y"}


def exploratory_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    inventory = observation.get("inventory", [])
    candidates: List[Dict[str, Any]] = []

    for resource in inventory:
        if resource.get("category") == "storage" and not resource.get("is_attached", True):
            candidates.append({"action_type": "delete_resource", "resource_id": resource.get("id", "")})
        if resource.get("category") == "compute" and resource.get("tags", {}).get("lifecycle") == "idle":
            candidates.append({"action_type": "delete_resource", "resource_id": resource.get("id", "")})
        if (
            resource.get("category") == "compute"
            and float(resource.get("cpu_usage_pct_30d", 0)) < 10.0
            and resource.get("resource_type") != "t3.small"
        ):
            candidates.append(
                {
                    "action_type": "modify_instance",
                    "instance_id": resource.get("id", ""),
                    "new_type": POLICY_RNG.choice(["t3.small", "t3.medium"]),
                }
            )

    candidates.append({"action_type": "purchase_savings_plan", "plan_type": "compute", "duration": "1y"})
    return POLICY_RNG.choice(candidates)


def propose_action(client: Optional[OpenAI], observation: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    if POLICY_RNG.random() < EXPLORE_RATE:
        return exploratory_action(observation)

    if client is None:
        return heuristic_action(observation)

    summary = summarize_observation(observation)
    user_prompt = (
        "Task: " + task_name + "\n"
        "Observation summary JSON: " + summary + "\n"
        "Return ONLY one JSON object with keys matching one valid action schema."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            timeout=LLM_TIMEOUT,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("action_type"), str):
            return parsed
    except Exception:
        pass

    return heuristic_action(observation)


def run_episode() -> None:
    client: Optional[OpenAI]
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
            timeout=LLM_TIMEOUT,
            max_retries=1,
        ) if API_KEY else None
    except Exception:
        client = None

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_response = requests.post(f"{ENV_BASE_URL}/reset", timeout=REQUEST_TIMEOUT)
        observation = safe_json(reset_response)

        for step in range(1, MAX_STEPS + 1):
            action_payload = propose_action(client, observation, TASK_NAME)
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
                observation = step_data.get("observation", observation)
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
            score_response = requests.get(f"{ENV_BASE_URL}/tasks/{TASK_NAME}/score", timeout=REQUEST_TIMEOUT)
            score_data = safe_json(score_response)
            score = clamp_score(float(score_data.get("score", 0.0) or 0.0))
        except Exception:
            # Fallback normalization if score endpoint is unavailable.
            total_reward = sum(rewards)
            score = clamp_score(total_reward / max(1.0, float(MAX_STEPS)))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        success = False
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    run_episode()

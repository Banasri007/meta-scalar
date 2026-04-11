import json
import os
import random
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Credentials (checker injects API_BASE_URL and API_KEY) ──────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = (
    os.environ.get("API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or "no-key"
)
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_BASE_URL          = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
BENCHMARK             = os.environ.get("FINOPS_BENCHMARK", "finops-optimizer")
MAX_STEPS             = int(os.environ.get("MAX_STEPS", "15"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))
REQUEST_TIMEOUT       = int(os.environ.get("REQUEST_TIMEOUT", "45"))
LLM_TIMEOUT           = int(os.environ.get("LLM_TIMEOUT", "30"))

ALL_TASKS = ["cleanup_unattached", "rightsize_compute", "fleet_strategy"]

# Always create the client — use the injected API_BASE_URL and API_KEY
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=LLM_TIMEOUT, max_retries=1)

SYSTEM_PROMPT = """You are a FinOps cloud cost optimization agent.
You will receive the current cloud inventory and must choose ONE action to reduce costs.

Available action types and their required fields:
1. delete_resource      -> {"action_type": "delete_resource", "resource_id": "<id>"}
2. modify_instance      -> {"action_type": "modify_instance", "instance_id": "<id>", "new_type": "t3.small"}
3. purchase_savings_plan -> {"action_type": "purchase_savings_plan", "plan_type": "compute", "duration": "1y"}
4. tag_resource         -> {"action_type": "tag_resource", "resource_id": "<id>", "tag_key": "env", "tag_value": "optimized"}

Rules:
- Prefer deleting unattached storage volumes (category=storage, is_attached=false)
- Prefer deleting idle compute (tags.lifecycle=idle)
- Downsize compute with cpu_usage_pct_30d < 5% to t3.small
- Never delete is_production=true databases
- Output ONLY valid JSON, no markdown, no explanation"""


# ── Logging ─────────────────────────────────────────────────────────────────

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))

def safe_json(resp: requests.Response) -> Dict[str, Any]:
    resp.raise_for_status()
    d = resp.json()
    return d if isinstance(d, dict) else {}

def summarize_obs(obs: Dict[str, Any]) -> str:
    inv = obs.get("inventory", [])
    bill = obs.get("cost_data", {}).get("projected_monthly_bill", 0)
    latency = obs.get("health_status", {}).get("system_latency_ms", 0)
    unattached = [r for r in inv if r.get("category") == "storage" and not r.get("is_attached", True)]
    idle       = [r for r in inv if r.get("tags", {}).get("lifecycle") == "idle"]
    low_cpu    = [r for r in inv if r.get("category") == "compute"
                  and float(r.get("cpu_usage_pct_30d", 100)) < 5.0
                  and not r.get("is_production")]
    legacy     = [r for r in inv if r.get("is_legacy") and not r.get("is_production")]
    return json.dumps({
        "monthly_bill_usd": bill,
        "latency_ms": latency,
        "unattached_volumes": [{"id": r["id"], "cost": r["monthly_cost"]} for r in unattached[:5]],
        "idle_instances": [{"id": r["id"], "cost": r["monthly_cost"]} for r in idle[:5]],
        "underutilized_compute": [{"id": r["id"], "cpu_pct": r["cpu_usage_pct_30d"], "type": r["resource_type"]} for r in low_cpu[:5]],
        "legacy_resources": [{"id": r["id"], "cost": r["monthly_cost"]} for r in legacy[:3]],
    }, indent=2)

def fallback_action(obs: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Deterministic fallback — used only if LLM call fails."""
    inv = obs.get("inventory", [])
    if task in ("cleanup_unattached", "fleet_strategy"):
        for r in inv:
            if r.get("is_legacy") and not r.get("is_production"):
                return {"action_type": "delete_resource", "resource_id": r["id"]}
        for r in inv:
            if r.get("category") == "storage" and not r.get("is_attached", True):
                return {"action_type": "delete_resource", "resource_id": r["id"]}
        for r in inv:
            if r.get("tags", {}).get("lifecycle") == "idle":
                return {"action_type": "delete_resource", "resource_id": r["id"]}
    for r in inv:
        if r.get("category") == "compute" and float(r.get("cpu_usage_pct_30d", 100)) < 5.0 \
                and r.get("resource_type") != "t3.small" and not r.get("is_production"):
            return {"action_type": "modify_instance", "instance_id": r["id"], "new_type": "t3.small"}
    return {"action_type": "purchase_savings_plan", "plan_type": "compute", "duration": "1y"}


def llm_action(obs: Dict[str, Any], task: str, history: List[str]) -> Dict[str, Any]:
    """Call LLM via the injected API_BASE_URL/API_KEY — always attempted."""
    summary = summarize_obs(obs)
    history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(history[-5:])) or "  none"
    user_msg = f"Task: {task}\n\nCurrent state:\n{summary}\n\nPrevious actions:\n{history_str}\n\nChoose the next action. Output ONLY JSON."
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("action_type"), str):
            return parsed
    except Exception as exc:
        print(f"[LLM_ERROR] {exc}", flush=True)
    return fallback_action(obs, task)


# ── Grader verification ───────────────────────────────────────────────────────

def run_graders() -> None:
    """Enumerate all 3 tasks and verify grader scores — required by checker."""
    print("\n[GRADERS] Enumerating tasks and verifying grader outputs:", flush=True)
    all_ok = True
    for task_id in ALL_TASKS:
        try:
            requests.post(f"{ENV_BASE_URL}/reset",
                          json={}, timeout=REQUEST_TIMEOUT)
            resp = requests.get(f"{ENV_BASE_URL}/tasks/{task_id}/score",
                                timeout=REQUEST_TIMEOUT)
            score = float(resp.json().get("score", 0.0))
            ok = 0.0 <= score <= 1.0
            mark = "✓" if ok else "✗ OUT OF RANGE"
            print(f"[GRADER] task={task_id} initial_score={score:.3f} range=0.0-1.0 {mark}", flush=True)
            if not ok:
                all_ok = False
        except Exception as exc:
            print(f"[GRADER] task={task_id} ERROR: {exc}", flush=True)
            all_ok = False
    status = "All graders verified." if all_ok else "Some graders failed."
    print(f"[GRADERS] {status}\n", flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    history: List[str] = []

    log_start(task_name)

    try:
        # Reset environment
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={}, timeout=REQUEST_TIMEOUT)
        data = safe_json(reset_resp)
        # /reset returns raw Observation: {inventory, cost_data, health_status, status_message}
        obs = data if "inventory" in data else data.get("observation", data)

        for step in range(1, MAX_STEPS + 1):
            # Always call LLM (uses injected API_BASE_URL + API_KEY)
            action = llm_action(obs, task_name, history)
            action_str = json.dumps(action, separators=(",", ":"))
            history.append(action_str)

            try:
                step_resp = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=REQUEST_TIMEOUT,
                )
                step_data = safe_json(step_resp)
                reward  = float(step_data.get("reward", 0.0) or 0.0)
                done    = bool(step_data.get("done", False))
                info    = step_data.get("info", {}) or {}
                error   = info.get("last_action_error") if isinstance(info, dict) else None
                obs_raw = step_data.get("observation", {})
                if isinstance(obs_raw, dict) and obs_raw:
                    obs = obs_raw
            except Exception as exc:
                reward, done, error = 0.0, True, str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get final score from grader endpoint
        try:
            score_resp = requests.get(
                f"{ENV_BASE_URL}/tasks/{task_name}/score",
                timeout=REQUEST_TIMEOUT,
            )
            score = clamp(float(safe_json(score_resp).get("score", 0.0) or 0.0))
        except Exception:
            score = clamp(sum(rewards) / max(1.0, float(MAX_STEPS)))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_graders()
    run_episode("cleanup_unattached")
    run_episode("rightsize_compute")
    run_episode("fleet_strategy")

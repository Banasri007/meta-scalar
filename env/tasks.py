from dataclasses import dataclass
from typing import Dict, List

from .engine import FinOpsEngine


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    name: str
    difficulty: str
    description: str


TASKS: List[TaskDefinition] = [
    TaskDefinition(
        id="cleanup_unattached",
        name="The Garbage Collector",
        difficulty="easy",
        description="Delete 5 unattached volumes and 2 idle test instances.",
    ),
    TaskDefinition(
        id="rightsize_compute",
        name="The Right-Sizer",
        difficulty="medium",
        description="Resize underutilized VMs while keeping latency under 200ms.",
    ),
    TaskDefinition(
        id="fleet_strategy",
        name="The Budget Strategist",
        difficulty="hard",
        description="Balance deletes, rightsizing, and savings plans for strong ROI.",
    ),
]

# Validator requires scores strictly in (0.0, 1.0) — never exactly 0 or 1.
_SCORE_MIN = 0.05
_SCORE_MAX = 0.95


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _bound(score: float) -> float:
    """Clamp raw score into (_SCORE_MIN, _SCORE_MAX) so it is never exactly 0 or 1."""
    return round(_clip(score, _SCORE_MIN, _SCORE_MAX), 4)


def score_cleanup_unattached(env: FinOpsEngine) -> float:
    remaining_unattached = {
        resource.id
        for resource in env.resources
        if resource.category == "storage" and not resource.is_attached
    }
    remaining_idle_test = {
        resource.id
        for resource in env.resources
        if resource.category == "compute" and resource.tags.get("lifecycle") == "idle"
    }

    removed_volume_cost = sum(
        env.baseline_cost_by_id.get(resource_id, 0.0)
        for resource_id in env.unattached_volume_ids
        if resource_id not in remaining_unattached
    )
    removed_idle_test_count = sum(
        1 for resource_id in env.idle_test_instance_ids if resource_id not in remaining_idle_test
    )

    volume_score = (
        removed_volume_cost / env.initial_unattached_volume_cost
        if env.initial_unattached_volume_cost
        else 0.0
    )
    test_instance_score = removed_idle_test_count / max(1, len(env.idle_test_instance_ids))
    raw = (0.7 * volume_score) + (0.3 * test_instance_score)
    return _bound(raw)


def score_rightsize_compute(env: FinOpsEngine) -> float:
    candidates = [
        resource
        for resource in env.resources
        if resource.id in env.underutilized_vm_ids and resource.category == "compute"
    ]
    candidate_ids = {resource.id for resource in candidates}
    deleted_candidates = [
        resource_id for resource_id in env.underutilized_vm_ids if resource_id not in {r.id for r in env.resources}
    ]

    theoretical_cost = 0.0
    actual_cost = 0.0
    for resource_id in env.underutilized_vm_ids:
        theoretical_cost += 25.0
        if resource_id in candidate_ids:
            current_resource = next(r for r in candidates if r.id == resource_id)
            actual_cost += current_resource.monthly_cost
        else:
            actual_cost += 0.0

    theoretical_max_savings = max(
        1.0,
        sum(env.baseline_cost_by_id.get(rid, 0.0) for rid in env.underutilized_vm_ids) - theoretical_cost,
    )
    actual_savings = sum(env.baseline_cost_by_id.get(rid, 0.0) for rid in env.underutilized_vm_ids) - actual_cost

    score = actual_savings / theoretical_max_savings
    if env.system_latency_ms >= 200.0:
        score *= 0.2
    return _bound(score)


def score_fleet_strategy(env: FinOpsEngine) -> float:
    cost_reduction = (env.initial_bill - env.get_effective_bill()) / max(1.0, env.initial_bill)
    cost_component = _clip(cost_reduction / 0.4)

    no_downtime_bonus = 0.5 if env.downtime_events == 0 else 0.0

    savings_plan_discounts = env.get_total_bill() - env.get_effective_bill()
    roi = (env.initial_bill - env.get_effective_bill()) / max(1.0, savings_plan_discounts)
    roi_bonus = 0.2 if roi > 0.3 else 0.0

    raw_score = cost_component + no_downtime_bonus + roi_bonus
    return _bound(raw_score / 1.7)


def get_task_score(env: FinOpsEngine, task_id: str) -> float:
    if task_id == "cleanup_unattached":
        return score_cleanup_unattached(env)
    if task_id == "rightsize_compute":
        return score_rightsize_compute(env)
    if task_id == "fleet_strategy":
        return score_fleet_strategy(env)
    raise KeyError(f"Unknown task id: {task_id}")


def list_tasks() -> List[Dict[str, str]]:
    return [task.__dict__ for task in TASKS]

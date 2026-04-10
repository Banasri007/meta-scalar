import os
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import (
    Action,
    CloudResource,
    CostData,
    DeleteResourceAction,
    HealthStatus,
    ModifyInstanceAction,
    Observation,
    PurchaseSavingsPlanAction,
    Reward,
    TagResourceAction,
)


@dataclass
class SavingsPlan:
    plan_type: str
    duration: str
    discount_rate: float
    coverage_ratio: float


class FinOpsEngine:
    INSTANCE_CATALOG: Dict[str, Dict[str, float]] = {
        "m5.xlarge": {"monthly_cost": 150.0, "capacity": 1.0},
        "m5.large": {"monthly_cost": 90.0, "capacity": 0.65},
        "t3.medium": {"monthly_cost": 45.0, "capacity": 0.4},
        "t3.small": {"monthly_cost": 25.0, "capacity": 0.25},
        "t3.micro": {"monthly_cost": 10.0, "capacity": 0.1},
    }

    def __init__(self) -> None:
        seed_text = os.getenv("FINOPS_SEED", "42")
        self.seed: Optional[int] = int(seed_text) if seed_text and seed_text.strip() else 42
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.resources: List[CloudResource] = []
        self.system_latency_ms: float = 80.0
        self.initial_bill: float = 0.0
        self.base_latency_ms: float = 80.0
        self.step_count: int = 0
        self.max_steps: int = 35
        self.throttling_events: int = 0
        self.downtime_events: int = 0
        self.critical_failures: int = 0
        self.change_pressure: float = 0.0  # Rises with each infra change
        self.savings_plans: List[SavingsPlan] = []
        self.baseline_cost_by_id: Dict[str, float] = {}
        self.underutilized_vm_ids: List[str] = []
        self.unattached_volume_ids: List[str] = []
        self.idle_test_instance_ids: List[str] = []
        self.initial_unattached_volume_cost: float = 0.0
        self.initial_idle_test_cost: float = 0.0
        self.reset_counter: int = 0
        self.reset()

    def _make_id(self, prefix: str, length: int = 8) -> str:
        if self.seed is None:
            return f"{prefix}-{uuid.uuid4().hex[:length]}"
        hex_chars = "0123456789abcdef"
        suffix = "".join(self.rng.choice(hex_chars) for _ in range(length))
        return f"{prefix}-{suffix}"

    def reset(self) -> Observation:
        self.reset_counter += 1
        self.resources = []
        self.step_count = 0
        self.throttling_events = 0
        self.downtime_events = 0
        self.critical_failures = 0
        self.change_pressure = 0.0
        self.savings_plans = []

        # Medium task candidates: 10 over-provisioned VMs (<5% CPU).
        for idx in range(10):
            self.resources.append(
                CloudResource(
                    id=self._make_id("i", 8),
                    category="compute",
                    resource_type="m5.xlarge",
                    monthly_cost=150.0,
                    cpu_usage_pct_30d=self.rng.uniform(1.2, 4.8),
                    memory_usage_pct_30d=self.rng.uniform(8.0, 22.0),
                    network_io_mbps_30d=self.rng.uniform(3.0, 20.0),
                    is_attached=True,
                    is_production=idx < 6,
                    tags={"env": "prod" if idx < 6 else "staging"},
                )
            )

        # Easy task candidates: 5 unattached volumes.
        for _ in range(5):
            self.resources.append(
                CloudResource(
                    id=self._make_id("vol", 8),
                    category="storage",
                    resource_type="gp3",
                    monthly_cost=22.0,
                    cpu_usage_pct_30d=0.0,
                    memory_usage_pct_30d=0.0,
                    network_io_mbps_30d=0.0,
                    is_attached=False,
                    is_production=False,
                    tags={"env": "orphan"},
                )
            )

        # Easy task candidates: 2 idle test instances.
        for _ in range(2):
            self.resources.append(
                CloudResource(
                    id=self._make_id("i-test", 6),
                    category="compute",
                    resource_type="m5.large",
                    monthly_cost=90.0,
                    cpu_usage_pct_30d=self.rng.uniform(0.3, 1.2),
                    memory_usage_pct_30d=self.rng.uniform(2.0, 8.0),
                    network_io_mbps_30d=self.rng.uniform(0.1, 2.0),
                    is_attached=True,
                    is_production=False,
                    tags={"env": "test", "lifecycle": "idle"},
                )
            )

        # Hard task mixed fleet: one legacy compute and one legacy DB.
        self.resources.append(
            CloudResource(
                id=self._make_id("legacy", 6),
                category="compute",
                resource_type="m5.xlarge",
                monthly_cost=160.0,
                cpu_usage_pct_30d=6.0,
                memory_usage_pct_30d=18.0,
                network_io_mbps_30d=4.0,
                is_attached=True,
                is_production=False,
                is_legacy=True,
                tags={"env": "legacy"},
            )
        )
        self.resources.append(
            CloudResource(
                id=self._make_id("db-legacy", 6),
                category="database",
                resource_type="db.r6g.large",
                monthly_cost=320.0,
                cpu_usage_pct_30d=9.0,
                memory_usage_pct_30d=22.0,
                network_io_mbps_30d=10.0,
                is_attached=True,
                is_production=False,
                is_legacy=True,
                tags={"env": "legacy"},
            )
        )

        # Critical production database that must not be deleted.
        self.resources.append(
            CloudResource(
                id=self._make_id("db-prod", 6),
                category="database",
                resource_type="db.r6g.xlarge",
                monthly_cost=520.0,
                cpu_usage_pct_30d=42.0,
                memory_usage_pct_30d=58.0,
                network_io_mbps_30d=55.0,
                is_attached=True,
                is_production=True,
                tags={"env": "prod", "service": "payments"},
            )
        )

        self.base_latency_ms = 85.0
        self.system_latency_ms = self.base_latency_ms
        self.initial_bill = self.get_total_bill()
        self.baseline_cost_by_id = {resource.id: resource.monthly_cost for resource in self.resources}
        self.underutilized_vm_ids = [
            resource.id
            for resource in self.resources
            if resource.category == "compute" and resource.cpu_usage_pct_30d < 5.0
        ]
        self.unattached_volume_ids = [
            resource.id
            for resource in self.resources
            if resource.category == "storage" and not resource.is_attached
        ]
        self.idle_test_instance_ids = [
            resource.id
            for resource in self.resources
            if resource.category == "compute" and resource.tags.get("lifecycle") == "idle"
        ]
        self.initial_unattached_volume_cost = sum(
            self.baseline_cost_by_id.get(resource_id, 0.0) for resource_id in self.unattached_volume_ids
        )
        self.initial_idle_test_cost = sum(
            self.baseline_cost_by_id.get(resource_id, 0.0) for resource_id in self.idle_test_instance_ids
        )
        return self.get_observation("Environment reset. Optimization opportunities loaded.")

    def state(self) -> Observation:
        """Return the current observable environment state without mutation."""
        return self.get_observation("Current state requested.")

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        self.step_count += 1
        previous_bill = self.get_effective_bill()
        action_reward = 0.0
        status_message = "No-op."
        info: Dict = {}

        if isinstance(action, DeleteResourceAction):
            action_reward, status_message = self._handle_delete(action)
            self.change_pressure = min(5.0, self.change_pressure + 0.4)
        elif isinstance(action, ModifyInstanceAction):
            action_reward, status_message = self._handle_modify(action)
            self.change_pressure = min(5.0, self.change_pressure + 0.8)
        elif isinstance(action, PurchaseSavingsPlanAction):
            action_reward, status_message = self._handle_savings_plan(action)
        elif isinstance(action, TagResourceAction):
            action_reward, status_message = self._handle_tag(action)

        self._recalculate_latency()
        current_bill = self.get_effective_bill()
        bill_delta = previous_bill - current_bill

        # Progress signal: positive when projected bill decreases.
        bill_change_reward = bill_delta / 200.0
        reward_total = action_reward + bill_change_reward
        reward = Reward(
            total=round(reward_total, 4),
            action_reward=round(action_reward, 4),
            bill_change_reward=round(bill_change_reward, 4),
        )

        done = self.step_count >= self.max_steps or current_bill <= self.initial_bill * 0.55
        info["monthly_bill_delta"] = round(bill_delta, 2)
        info["effective_monthly_bill"] = round(current_bill, 2)
        info["system_latency_ms"] = round(self.system_latency_ms, 2)
        info["throttling_events"] = self.throttling_events
        info["downtime_events"] = self.downtime_events

        return self.get_observation(status_message), reward, done, info

    def _handle_delete(self, action: DeleteResourceAction) -> Tuple[float, str]:
        target = self._find_resource(action.resource_id)
        if target is None:
            return -0.1, f"Resource {action.resource_id} not found."

        if target.category == "database" and target.is_production:
            self.critical_failures += 1
            self.downtime_events += 1
            self.system_latency_ms += 80.0  # Immediate latency spike
            return -1.0, "Critical failure: attempted deletion of production database blocked."

        reward = 0.0
        if target.category == "storage" and not target.is_attached:
            reward += 0.1
        if target.category == "compute" and target.tags.get("lifecycle") == "idle":
            reward += 0.1
        if target.is_production and target.category == "compute":
            # Deleting production compute triggers throttling
            self.throttling_events += 1
            self.system_latency_ms += 30.0
            reward -= 0.2

        self.resources = [resource for resource in self.resources if resource.id != target.id]
        return reward, f"Deleted resource {target.id}."

    def _handle_modify(self, action: ModifyInstanceAction) -> Tuple[float, str]:
        target = self._find_resource(action.instance_id)
        if target is None:
            return -0.1, f"Instance {action.instance_id} not found."
        if target.category != "compute":
            return -0.2, f"Resource {action.instance_id} is not a compute instance."
        if action.new_type not in self.INSTANCE_CATALOG:
            return -0.1, f"Unsupported instance type {action.new_type}."

        current_profile = self.INSTANCE_CATALOG.get(target.resource_type)
        new_profile = self.INSTANCE_CATALOG[action.new_type]
        if current_profile is None:
            return -0.1, f"Current type {target.resource_type} is not resizable in this simulator."

        old_type = target.resource_type
        old_cost = target.monthly_cost
        old_capacity = current_profile["capacity"]
        new_capacity = new_profile["capacity"]
        expected_cpu = target.cpu_usage_pct_30d * (old_capacity / new_capacity)

        target.resource_type = action.new_type
        target.monthly_cost = new_profile["monthly_cost"]
        target.cpu_usage_pct_30d = min(100.0, expected_cpu)
        target.memory_usage_pct_30d = min(100.0, target.memory_usage_pct_30d * (old_capacity / new_capacity))

        reward = max(0.0, (old_cost - target.monthly_cost) / 250.0)

        # Penalty if throttling occurs after resize.
        if expected_cpu >= 100.0:
            self.throttling_events += 1
            self.system_latency_ms += 25.0
            reward -= 0.5
        elif expected_cpu >= 85.0 and target.is_production:
            # Risky resize of production instance
            self.throttling_events += 1
            self.system_latency_ms += 15.0
            reward -= 0.2

        self._apply_resize_noise(target)

        return reward, f"Modified {target.id} from {old_type} to {action.new_type}."

    def _apply_resize_noise(self, modified_target: CloudResource) -> None:
        # Small stochastic drift after infra changes to mimic real workload variance.
        for resource in self.resources:
            if resource.category != "compute":
                continue
            cpu_drift = self.rng.uniform(-2.0, 3.0)
            mem_drift = self.rng.uniform(-1.5, 2.0)
            resource.cpu_usage_pct_30d = max(0.0, min(100.0, resource.cpu_usage_pct_30d + cpu_drift))
            resource.memory_usage_pct_30d = max(0.0, min(100.0, resource.memory_usage_pct_30d + mem_drift))

        # Occasional spike on production compute after a resize action.
        if self.rng.random() < 0.2:
            prod_compute = [r for r in self.resources if r.category == "compute" and r.is_production]
            if prod_compute:
                spike_target = self.rng.choice(prod_compute)
                spike = self.rng.uniform(12.0, 25.0)
                spike_target.cpu_usage_pct_30d = min(100.0, spike_target.cpu_usage_pct_30d + spike)

    def _handle_savings_plan(self, action: PurchaseSavingsPlanAction) -> Tuple[float, str]:
        if any(plan.plan_type == action.plan_type and plan.duration == action.duration for plan in self.savings_plans):
            return -0.05, "Savings plan already purchased for this type and duration."

        discount_rate = 0.2 if action.duration == "1y" else 0.35
        plan = SavingsPlan(
            plan_type=action.plan_type,
            duration=action.duration,
            discount_rate=discount_rate,
            coverage_ratio=0.6,
        )
        self.savings_plans.append(plan)
        return 0.05, f"Purchased {action.duration} savings plan for {action.plan_type}."

    def _handle_tag(self, action: TagResourceAction) -> Tuple[float, str]:
        target = self._find_resource(action.resource_id)
        if target is None:
            return -0.05, f"Resource {action.resource_id} not found."

        is_new_tag = action.tag_key not in target.tags
        target.tags[action.tag_key] = action.tag_value
        return (0.02 if is_new_tag else 0.0), f"Tagged {target.id} with {action.tag_key}={action.tag_value}."

    def _recalculate_latency(self) -> None:
        over_utilized_production = [
            resource
            for resource in self.resources
            if resource.category == "compute" and resource.is_production and resource.cpu_usage_pct_30d > 80.0
        ]
        severe_over_utilized = [resource for resource in over_utilized_production if resource.cpu_usage_pct_30d >= 100.0]

        latency_from_risk = len(over_utilized_production) * 28.0 + len(severe_over_utilized) * 45.0
        latency_from_churn = self.change_pressure * 8.0  # Churn from repeated infra changes
        latency_from_throttle = self.throttling_events * 12.0

        computed = self.base_latency_ms + latency_from_risk + latency_from_churn + latency_from_throttle

        # Clamp latency and apply decay — system stabilizes over time if no bad actions
        self.system_latency_ms = min(computed, 800.0)
        self.change_pressure = max(0.0, self.change_pressure - 0.1)  # Pressure decays slowly

        # High latency auto-triggers throttling
        if self.system_latency_ms > 400.0 and self.rng.random() < 0.3:
            self.throttling_events += 1

        # Very high latency triggers downtime
        if self.system_latency_ms > 600.0 and self.rng.random() < 0.2:
            self.downtime_events += 1

    def _find_resource(self, resource_id: str) -> Optional[CloudResource]:
        return next((resource for resource in self.resources if resource.id == resource_id), None)

    def get_total_bill(self) -> float:
        return sum(resource.monthly_cost for resource in self.resources)

    def get_effective_bill(self) -> float:
        gross_bill = self.get_total_bill()
        total_discount = 0.0

        for plan in self.savings_plans:
            eligible_spend = sum(
                resource.monthly_cost
                for resource in self.resources
                if resource.category == plan.plan_type and resource.is_production
            )
            total_discount += eligible_spend * plan.discount_rate * plan.coverage_ratio

        return max(0.0, gross_bill - total_discount)

    def get_observation(self, message: str) -> Observation:
        return Observation(
            inventory=self.resources,
            cost_data=CostData(
                daily_burn_rate=round(self.get_effective_bill() / 30.0, 2),
                projected_monthly_bill=round(self.get_effective_bill(), 2),
            ),
            health_status=HealthStatus(
                system_latency_ms=round(self.system_latency_ms, 2),
                throttling_events=self.throttling_events,
                downtime_events=self.downtime_events,
            ),
            status_message=message,
        )

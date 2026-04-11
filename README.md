---
title: FinOps Cloud Optimizer
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---



# FinOps Agent Environment (OpenEnv)

This repository provides a simulated cloud cost optimization environment where an agent must reduce spend without harming production reliability.

## What Is Implemented

- Typed action space using Pydantic discriminated unions:
  - `modify_instance`
  - `delete_resource`
  - `purchase_savings_plan`
  - `tag_resource`
- Rich observation space:
  - `inventory` (VMs, DBs, Storage)
  - 30-day metrics (`cpu_usage_pct_30d`, `memory_usage_pct_30d`, `network_io_mbps_30d`)
  - `cost_data` (`daily_burn_rate`, `projected_monthly_bill`)
  - `health_status` (`system_latency_ms`, throttling and downtime counters)
- Reward shaping:
  - `+0.1` for successful idle-resource cleanup actions
  - `-0.5` for throttling after unsafe downsizing
  - `-1.0` for attempted production database deletion
  - Progress signal from bill delta each step
- Built-in graders for three tasks:
  - `cleanup_unattached`
  - `rightsize_compute`
  - `fleet_strategy`

## Project Structure

- `main.py`: FastAPI OpenEnv entrypoint
- `env/models.py`: action/observation schemas
- `env/engine.py`: simulator state + step logic
- `env/tasks.py`: task definitions and scoring
- `baseline_inference.py`: heuristic baseline runner
- `openenv.yaml`: environment metadata
- `Dockerfile`: container build for HF Spaces

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Baseline Rollout

Start the server first, then in another terminal:

```bash
python baseline_inference.py
```

The script performs a deterministic policy:
1. Delete unattached volumes and idle test instances.
2. Right-size low-utilization compute.
3. Purchase a 1-year compute savings plan.
4. Print task scores and final cost/latency.

## Useful API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /tasks/{task_id}/score`

## Suggested Next Iterations

1. Add action history + audit log to observations.
2. Introduce stochastic demand spikes and rollback actions.
3. Add task-specific episode caps and per-task resets.
4. Add unit tests for scoring and safety constraints.

"""
P2A Trainer — subclasses verl's FullyAsyncTrainer to inject advantage reshaping.

After verl's compute_advantage() produces uniform A_seq per trajectory,
this trainer modifies advantages per-span based on whether the agent's
read actions intersect the precomputed bonus map call graph.

Integration point: override _fit_compute_advantage() — no verl fork needed.
"""

import os

import numpy as np
import ray
import torch

from p2a.core import (
    BonusMapStore,
    compute_p2a_multiplier,
    match_reads_to_callgraph,
    parse_read_actions,
    parse_read_actions_from_tool_calls,
)


def apply_p2a_reshape(batch, bonus_map_store, m_max=3.0, tracking_mode="view_and_bash"):
    """Apply P2A multiplicative advantage reshape to a batch.

    For each trajectory in the batch:
    1. Look up the instance's bonus map
    2. Decode the response tokens to text
    3. Parse read actions from the text
    4. Match reads against call graph nodes to get min distance
    5. Compute multiplier m(d)^sign(A)
    6. Multiply the advantage tensor for this trajectory

    Note: In Uni-Agent's fully-async trainer, each trajectory is one row
    in the batch (trajectory-row, not step-row). The entire trajectory's
    tokens are in one response_ids/advantages row.

    Args:
        batch: verl DataProto with batch["advantages"], batch["returns"],
               batch["response_mask"], and non_tensor_batch metadata.
        bonus_map_store: BonusMapStore instance.
        m_max: Maximum multiplier hyperparameter.
        tracking_mode: "view_only" or "view_and_bash".

    Returns:
        Modified batch with reshaped advantages and returns.
    """
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    response_mask = batch.batch["response_mask"]

    n_trajectories = advantages.shape[0]
    n_reshaped = 0
    total_multiplier = 0.0

    for i in range(n_trajectories):
        instance_id = _get_instance_id(batch, i)
        if instance_id is None:
            continue

        bonus_map = bonus_map_store.get(instance_id)
        if bonus_map is None:
            continue

        mask = response_mask[i].bool()
        masked_adv = advantages[i][mask]
        if masked_adv.numel() == 0:
            continue

        a_sign = int(torch.sign(masked_adv.mean()).item())
        step_traces = _get_p2a_step_traces(batch, i)
        if step_traces:
            row_reshaped, row_multiplier_sum = _reshape_step_spans(
                advantages=advantages,
                returns=returns,
                response_mask=response_mask,
                row_idx=i,
                step_traces=step_traces,
                bonus_map=bonus_map,
                m_max=m_max,
                advantage_sign=a_sign,
                tracking_mode=tracking_mode,
            )
            n_reshaped += row_reshaped
            total_multiplier += row_multiplier_sum
            continue

        reads = _extract_reads_for_trajectory(batch, i, tracking_mode)
        if not reads:
            continue

        distance = match_reads_to_callgraph(reads, bonus_map)
        if distance < 0:
            continue

        multiplier = compute_p2a_multiplier(distance, m_max, a_sign)
        advantages[i][mask] = advantages[i][mask] * multiplier
        returns[i][mask] = returns[i][mask] * multiplier

        n_reshaped += 1
        total_multiplier += multiplier

    batch.batch["advantages"] = advantages
    batch.batch["returns"] = returns

    metrics = {
        "p2a/n_reshaped": n_reshaped,
        "p2a/n_total": n_trajectories,
        "p2a/reshape_rate": n_reshaped / max(n_trajectories, 1),
        "p2a/avg_multiplier": total_multiplier / max(n_reshaped, 1),
    }
    return batch, metrics


def _get_instance_id(batch, idx):
    """Extract instance_id from batch metadata for trajectory idx."""
    if "instance_id" in batch.non_tensor_batch:
        return _as_python_scalar(batch.non_tensor_batch["instance_id"][idx])
    if "uid" in batch.non_tensor_batch:
        uid = _as_python_scalar(batch.non_tensor_batch["uid"][idx])
        if isinstance(uid, str) and "__" in uid:
            return uid
    return None


def _as_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _extract_reads_for_trajectory(batch, idx, tracking_mode):
    """Extract read actions for a trajectory, trying two sources.

    Path A (preferred): Use structured p2a_step_traces from Codex's
    rollout-side instrumentation (UNI_AGENT_P2A_TRACE=1).
    Path B (fallback): Decode response_text and regex-parse.
    """
    step_traces = _get_p2a_step_traces(batch, idx)
    if step_traces:
        all_reads = []
        for trace in step_traces:
            tool_calls = trace.get("tool_calls", [])
            if tool_calls:
                all_reads.extend(
                    parse_read_actions_from_tool_calls(tool_calls)
                )
        return all_reads

    response_text = _get_response_text(batch, idx)
    if response_text:
        return parse_read_actions(response_text, tracking_mode=tracking_mode)

    return []


def _get_response_text(batch, idx):
    """Decode response tokens back to text for trajectory idx."""
    if "response_text" in batch.non_tensor_batch:
        value = _as_python_scalar(batch.non_tensor_batch["response_text"][idx])
        return value if isinstance(value, str) else None
    return None


def _get_p2a_step_traces(batch, idx):
    """Extract P2A step traces recorded by Codex's rollout-side instrumentation.

    The rollout records p2a_step_traces in extra_fields when
    UNI_AGENT_P2A_TRACE=1. Each trace has tool_calls with structured
    function name + arguments.
    """
    if "p2a_step_traces" in batch.non_tensor_batch:
        traces = _as_python_scalar(batch.non_tensor_batch["p2a_step_traces"][idx])
        return traces if traces else None
    if "extra_fields" in batch.non_tensor_batch:
        ef = _as_python_scalar(batch.non_tensor_batch["extra_fields"][idx])
        if isinstance(ef, dict) and "p2a_step_traces" in ef:
            return ef["p2a_step_traces"]
    return None


def _reshape_step_spans(
    *,
    advantages,
    returns,
    response_mask,
    row_idx,
    step_traces,
    bonus_map,
    m_max,
    advantage_sign,
    tracking_mode,
):
    row_reshaped = 0
    row_multiplier_sum = 0.0
    row_len = int(response_mask[row_idx].shape[0])

    for trace in step_traces:
        if not isinstance(trace, dict):
            continue
        reads = []
        tool_calls = trace.get("tool_calls") or []
        if tool_calls:
            reads.extend(parse_read_actions_from_tool_calls(tool_calls))
        response_text = trace.get("response_text")
        if response_text:
            reads.extend(parse_read_actions(response_text, tracking_mode=tracking_mode))
        if not reads:
            continue

        distance = match_reads_to_callgraph(reads, bonus_map)
        if distance < 0:
            continue

        start = max(int(trace.get("response_start", 0)), 0)
        end = min(int(trace.get("response_end", row_len)), row_len)
        if end <= start:
            continue

        span_mask = response_mask[row_idx].bool().clone()
        span_mask[:start] = False
        span_mask[end:] = False
        if not span_mask.any():
            continue

        multiplier = compute_p2a_multiplier(distance, m_max, advantage_sign)
        advantages[row_idx][span_mask] = advantages[row_idx][span_mask] * multiplier
        returns[row_idx][span_mask] = returns[row_idx][span_mask] * multiplier
        row_reshaped += 1
        row_multiplier_sum += multiplier

    return row_reshaped, row_multiplier_sum


def _unwrap_ray_actor_class(cls):
    metadata = getattr(cls, "__ray_metadata__", None)
    return getattr(metadata, "modified_class", cls)


def create_p2a_trainer_cls(base_trainer_cls):
    """Dynamically create a P2A trainer class that subclasses the given base.

    The upstream FullyAsyncTrainer is @ray.remote decorated. We unwrap its
    original Python class first, subclass that, then decorate the P2A subclass.

    Usage:
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
        P2ATrainer = create_p2a_trainer_cls(FullyAsyncTrainer)
        trainer = P2ATrainer.remote(config=..., ...)
    """
    base_cls = _unwrap_ray_actor_class(base_trainer_cls)

    class P2AFullyAsyncTrainer(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            bonus_map_dir = os.environ.get("P2A_BONUS_MAP_DIR", "")
            self._p2a_bonus_map_store = BonusMapStore(bonus_map_dir) if bonus_map_dir else None
            self._p2a_m_max = float(os.environ.get("P2A_M_MAX", "3.0"))
            self._p2a_tracking_mode = os.environ.get("P2A_TRACKING_MODE", "view_and_bash")
            self._p2a_enabled = self._p2a_bonus_map_store is not None
            if self._p2a_enabled:
                print(
                    "[P2A] Enabled. "
                    f"bonus_map_dir={bonus_map_dir}, "
                    f"m_max={self._p2a_m_max}, "
                    f"tracking={self._p2a_tracking_mode}"
                )
            else:
                print("[P2A] Disabled (P2A_BONUS_MAP_DIR not set). Running vanilla training.")

        def _fit_compute_advantage(self, batch):
            batch = super()._fit_compute_advantage(batch)

            if self._p2a_enabled:
                batch, p2a_metrics = apply_p2a_reshape(
                    batch,
                    self._p2a_bonus_map_store,
                    m_max=self._p2a_m_max,
                    tracking_mode=self._p2a_tracking_mode,
                )
                self.metrics.update(p2a_metrics)

            return batch

    return ray.remote(num_cpus=10)(P2AFullyAsyncTrainer)

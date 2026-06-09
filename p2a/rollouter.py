"""P2A rollouter adapters for live validation process metrics."""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ray

from p2a.validation_metrics import compute_validation_p2a_metrics, validation_records_from_batch


def _unwrap_ray_actor_class(cls):
    metadata = getattr(cls, "__ray_metadata__", None)
    return getattr(metadata, "modified_class", cls)


def create_p2a_rollouter_cls(base_rollouter_cls):
    """Create a FullyAsyncRollouter subclass that logs val-p2a metrics."""
    base_cls = _unwrap_ray_actor_class(base_rollouter_cls)

    class P2AFullyAsyncRollouter(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._p2a_eval_bonus_map_dir = os.environ.get("P2A_EVAL_BONUS_MAP_DIR", "")
            self._p2a_eval_near_threshold = float(os.environ.get("P2A_EVAL_NEAR_THRESHOLD", "0.5"))
            self._p2a_eval_tracking_mode = os.environ.get("P2A_TRACKING_MODE", "view_and_bash")
            self._p2a_eval_m_max = float(os.environ.get("P2A_M_MAX", "3.0"))
            self._p2a_eval_details_dir = os.environ.get("P2A_EVAL_DETAILS_DIR", "")
            if self._p2a_eval_bonus_map_dir:
                print(
                    "[P2A Eval] Enabled validation process metrics. "
                    f"bonus_map_dir={self._p2a_eval_bonus_map_dir}, "
                    f"near_threshold={self._p2a_eval_near_threshold}, "
                    f"tracking={self._p2a_eval_tracking_mode}"
                )
            else:
                print("[P2A Eval] Disabled (P2A_EVAL_BONUS_MAP_DIR not set).")

        def _p2a_details_out(self) -> str | None:
            if not self._p2a_eval_details_dir:
                return None
            step = getattr(self, "global_steps", 0)
            return str(Path(self._p2a_eval_details_dir) / f"validation_step_{step}.jsonl")

        def _validate(self, merged: bool = False):
            if merged or not self._p2a_eval_bonus_map_dir:
                return super()._validate(merged=merged)
            return self._validate_with_p2a_metrics()

        def _validate_with_p2a_metrics(self):
            from verl import DataProto
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            from verl.trainer.ppo.reward import extract_reward

            data_source_lst = []
            reward_extra_infos_dict: dict[str, list] = defaultdict(list)
            p2a_records: list[dict[str, Any]] = []

            sample_inputs = []
            sample_outputs = []
            sample_gts = []
            sample_scores = []
            sample_turns = []
            sample_uids = []

            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )

                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                    interleave=True,
                )

                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
                ]
                sample_gts.extend(ground_truths)

                test_gen_batch = self._get_gen_batch(test_batch)
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

                if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                    self.checkpoint_manager.sleep_replicas()
                    batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                    test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                    self.checkpoint_manager.update_weights(self.global_steps)

                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                print("validation generation end")

                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True

                input_ids = test_batch.batch["prompts"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])

                reward_tensor, reward_extra_info = extract_reward(test_batch)

                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                p2a_records.extend(
                    validation_records_from_batch(
                        test_batch,
                        output_texts=output_texts,
                        scores=scores,
                    )
                )

                reward_extra_infos_dict["reward"].extend(scores)
                for key, values in reward_extra_info.items():
                    if key not in reward_extra_infos_dict:
                        reward_extra_infos_dict[key] = []
                    if isinstance(values, np.ndarray):
                        reward_extra_infos_dict[key].extend(values.tolist())
                    else:
                        reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

                if "__num_turns__" in test_batch.non_tensor_batch:
                    sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    gts=sample_gts,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                )

            for key_info, lst in reward_extra_infos_dict.items():
                assert len(lst) == 0 or len(lst) == len(sample_scores), (
                    f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                )

            data_sources = np.concatenate(data_source_lst, axis=0)
            metric_dict = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
            try:
                p2a_metrics, _details = compute_validation_p2a_metrics(
                    p2a_records,
                    bonus_map_dir=self._p2a_eval_bonus_map_dir,
                    tracking_mode=self._p2a_eval_tracking_mode,
                    near_threshold=self._p2a_eval_near_threshold,
                    m_max=self._p2a_eval_m_max,
                    details_out=self._p2a_details_out(),
                )
                metric_dict.update(p2a_metrics)
            except Exception as exc:
                print(f"[P2A Eval] validation metric scoring failed: {exc}")
                metric_dict["val-p2a/error"] = 1.0
            return metric_dict

    return ray.remote(num_cpus=10)(P2AFullyAsyncRollouter)

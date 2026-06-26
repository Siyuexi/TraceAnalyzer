"""
P2A training entry point — drop-in replacement for verl's fully_async_main.

Usage:
    python3 -m p2a.main \
      'hydra.searchpath=[pkg://verl.trainer.config]' ...

The ONLY difference from the upstream entry point: _create_trainer() uses
P2AFullyAsyncTrainer instead of FullyAsyncTrainer, and _create_rollouter()
uses P2AFullyAsyncRollouter for optional validation process metrics.

When P2A_BONUS_MAP_DIR is not set, the trainer runs in vanilla mode
(no advantage reshaping), so this entry point works for baseline too.
"""

import hydra
import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner as _BaseTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

from p2a.rollouter import create_p2a_rollouter_cls
from p2a.trainer import create_p2a_trainer_cls

P2AFullyAsyncTrainer = create_p2a_trainer_cls(FullyAsyncTrainer)
P2AFullyAsyncRollouter = create_p2a_rollouter_cls(FullyAsyncRollouter)
_BaseTaskRunnerClass = getattr(getattr(_BaseTaskRunner, "__ray_metadata__", None), "modified_class", _BaseTaskRunner)
_CONFIG_PATH = "../uni-agent/verl/verl/experimental/fully_async_policy/config"
_CONFIG_NAME = "fully_async_ppo_megatron_trainer"


@ray.remote(num_cpus=1)
class P2ATaskRunner(_BaseTaskRunnerClass):
    def _create_rollouter(self, config) -> None:
        print("[P2A MAIN] Creating P2AFullyAsyncRollouter...")
        rollouter = P2AFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[P2A MAIN] P2AFullyAsyncRollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[P2A MAIN] Creating P2AFullyAsyncTrainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = P2AFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[P2A MAIN] P2AFullyAsyncTrainer created and initialized successfully")


@hydra.main(config_path=_CONFIG_PATH, config_name=_CONFIG_NAME, version_base=None)
def main(config):
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    from p2a.datasets import assert_training_data_sources_allowed

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    assert config.async_training.use_trainer_do_validate is False, "use_trainer_do_validate is not ready to use."

    from verl.trainer.ppo.utils import need_reward_model

    if need_reward_model(config) and config.async_training.use_trainer_do_validate:
        raise NotImplementedError(
            "use_trainer_do_validate with GenRM/DisRM is not yet supported."
        )

    from time import time

    start_time = time()
    assert_training_data_sources_allowed(config.data.train_files)
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=P2ATaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

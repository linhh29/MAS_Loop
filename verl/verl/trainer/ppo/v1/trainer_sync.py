# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # A one-step replay phase can launch actor and rollout from the exact same
        # checkpoint.  In that narrowly-scoped case the initial full-model PCCL
        # transfer is redundant and may be skipped explicitly by the launcher.
        if _env_flag("MAS_LOOP_SKIP_INITIAL_ROLLOUT_WEIGHT_SYNC"):
            logger.warning(
                "Skipping initial actor->rollout weight sync because the launcher "
                "guarantees identical actor and rollout checkpoints."
            )
            return
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        # Strict MAS joint training runs exactly one optimizer step per process.
        # If no further rollout can occur in this process, syncing the just-trained
        # weights into its soon-to-be-destroyed vLLM instance has no consumer.
        if _env_flag("MAS_LOOP_SKIP_POST_STEP_ROLLOUT_WEIGHT_SYNC"):
            logger.warning(
                "Skipping post-step actor->rollout weight sync for a terminal "
                "single-step replay phase; the actor checkpoint remains canonical."
            )
            return
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # Sleep integrated replicas first, then wait for the external MAS vLLM
        # phase to finish its own acknowledged sleep before optimizer state is
        # loaded.  This is MAS-only and leaves ordinary VERL training unchanged.
        self.checkpoint_manager.sleep_replicas()
        from mas_loop_rl.vllm_arbiter import wait_for_external_vllm_idle_enabled, wait_until_idle

        if wait_for_external_vllm_idle_enabled():
            wait_until_idle()

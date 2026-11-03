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
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000
MAS_LOOP_GENERATE_ADMISSION_ENV = "MAS_LOOP_RAY_GENERATE_MAX_OUTSTANDING"
MAS_LOOP_LB_METRICS_DIR_ENV = "MAS_LOOP_RAY_LB_METRICS_DIR"
MAS_LOOP_LB_METRICS_INTERVAL_ENV = "MAS_LOOP_RAY_LB_METRICS_INTERVAL_SEC"
MAS_LOOP_LB_CONTROL_CONCURRENCY = 256


def get_generate_admission_limit() -> int:
    """Return the optional process-wide MAS-Loop Ray generate admission cap.

    ``max_num_seqs`` controls vLLM scheduler/KV capacity.  It must not also be
    used as the number of Ray RPCs that may be delivered to one async actor at
    once: a 12,288-trajectory MAS-Loop batch can otherwise flood the actor
    mailbox before vLLM has admitted the requests.  A value of zero (or an
    unset variable) preserves upstream behavior.
    """

    value = os.getenv(MAS_LOOP_GENERATE_ADMISSION_ENV, "").strip()
    if not value:
        return 0
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(f"{MAS_LOOP_GENERATE_ADMISSION_ENV} must be an integer, got {value!r}") from exc
    if limit < 0:
        raise ValueError(f"{MAS_LOOP_GENERATE_ADMISSION_ENV} must be >= 0, got {limit}")
    return limit


class _GlobalRequestLoadBalancerCore:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        max_outstanding: int = 0,
    ):
        if not servers:
            raise ValueError("servers must be non-empty")
        if max_outstanding < 0:
            raise ValueError(f"max_outstanding must be >= 0, got {max_outstanding}")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._max_outstanding = max_outstanding
        self._admission_condition = asyncio.Condition()
        self._admission_waiters = 0
        self._admission_peak = 0

    def _total_inflight(self) -> int:
        return sum(self._inflight_requests.values())

    @ray.method(concurrency_group="admission")
    async def acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        When ``max_outstanding`` is non-zero this method is also the one global
        admission gate shared by every AgentLoopWorker.  Waiting happens before
        returning the vLLM actor handle, so queued work remains on this tiny
        coordinator actor instead of delivering thousands of large generate
        calls to the vLLM Ray actor.

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        async with self._admission_condition:
            while self._max_outstanding and self._total_inflight() >= self._max_outstanding:
                self._admission_waiters += 1
                try:
                    await self._admission_condition.wait()
                finally:
                    self._admission_waiters -= 1

            # Try sticky session first.
            if request_id in self._request_id_to_server:
                server_id = self._request_id_to_server[request_id]
                if server_id in self._inflight_requests:
                    self._inflight_requests[server_id] += 1
                    self._admission_peak = max(self._admission_peak, self._total_inflight())
                    return server_id, self._servers[server_id]
                # Server was removed, clear stale cache entry and re-select.
                del self._request_id_to_server[request_id]

            if not self._inflight_requests:
                raise RuntimeError("No available servers in load balancer")

            server_id = min(self._inflight_requests, key=self._inflight_requests.get)
            self._request_id_to_server[request_id] = server_id
            self._inflight_requests[server_id] += 1
            self._admission_peak = max(self._admission_peak, self._total_inflight())
            return server_id, self._servers[server_id]

    @ray.method(concurrency_group="control")
    async def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        async with self._admission_condition:
            if server_id not in self._inflight_requests:
                return
            if self._inflight_requests[server_id] > 0:
                self._inflight_requests[server_id] -= 1
                # One release creates one slot. Avoid a thundering herd.
                self._admission_condition.notify(1)

    @ray.method(concurrency_group="control")
    async def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        async with self._admission_condition:
            for sid, handle in servers.items():
                self._inflight_requests[sid] = 0
                self._servers[sid] = handle
            self._admission_condition.notify_all()
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    @ray.method(concurrency_group="control")
    async def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        async with self._admission_condition:
            for sid in server_ids:
                self._inflight_requests.pop(sid, None)
                self._servers.pop(sid, None)
            self._admission_condition.notify_all()
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    @ray.method(concurrency_group="control")
    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    @ray.method(concurrency_group="control")
    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    @ray.method(concurrency_group="control")
    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
            "max_outstanding": self._max_outstanding,
            "admission_waiters": self._admission_waiters,
            "admission_peak": self._admission_peak,
        }


def make_global_request_load_balancer_actor(admission_concurrency: int = 1000):
    """Create a Ray actor class with concurrency groups fixed at definition time.

    Ray 2.49 rejects ``concurrency_groups`` passed through ``ActorClass.options``;
    they must be supplied when the actor class is defined.  Keeping this in a
    factory lets MAS-Loop size the admission group from the runtime cap while
    preserving the public default actor class used by upstream callers/tests.
    """
    if admission_concurrency <= 0:
        raise ValueError(f"admission_concurrency must be > 0, got {admission_concurrency}")
    return ray.remote(
        concurrency_groups={
            "admission": admission_concurrency,
            "control": MAS_LOOP_LB_CONTROL_CONCURRENCY,
        }
    )(_GlobalRequestLoadBalancerCore)


# Public upstream-compatible actor class. MAS-Loop's manager uses the factory
# above when it needs a larger runtime admission group.
GlobalRequestLoadBalancer = make_global_request_load_balancer_actor()


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        server_handles: Optional[list[ray.actor.ActorHandle]] = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry. Optional; subclasses that
                manage server routing externally can pass None.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        # Acquire the model-wide sleep lease before bounded Ray admission.
        # These handles let the first lease holder wake the engine without
        # first consuming one of the scarce generate RPC slots.
        self._server_handles = list(server_handles or [])
        self._lb_metrics_last_sample = 0.0
        self._lb_metrics_sampling = False

    async def _maybe_write_load_balancer_metrics(self, *, acquire_wait_sec: float | None = None) -> None:
        """Periodically expose the Ray admission gate without adding OSSFS I/O.

        Every AgentLoop process overwrites one tiny JSON file in the run-local
        ``/tmp`` control directory. Sampling is best-effort and rate limited
        before awaiting the control RPC, so concurrent generations cannot
        create a telemetry thundering herd.
        """
        directory = os.getenv(MAS_LOOP_LB_METRICS_DIR_ENV, "").strip()
        if not directory or self._load_balancer is None or self._lb_metrics_sampling:
            return
        try:
            interval = max(1.0, float(os.getenv(MAS_LOOP_LB_METRICS_INTERVAL_ENV, "5")))
        except ValueError:
            interval = 5.0
        now = time.monotonic()
        if now - self._lb_metrics_last_sample < interval:
            return
        self._lb_metrics_last_sample = now
        self._lb_metrics_sampling = True
        try:
            status = await self._load_balancer.get_status.remote()
            payload = {
                "time": time.time(),
                "pid": os.getpid(),
                "acquire_wait_sec": acquire_wait_sec,
                **dict(status),
            }
            path = Path(directory) / f"ray-lb-{os.getpid()}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            # Telemetry must never change rollout correctness or availability.
            logger.debug("failed to sample MAS-Loop Ray admission state", exc_info=True)
        finally:
            self._lb_metrics_sampling = False

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        return await self._load_balancer.acquire_server.remote(request_id=request_id)

    async def _release_server(self, server_id: str) -> None:
        """Return one bounded Ray slot before dropping the model sleep lease.

        Fire-and-forget release after dropping the sleep lease allowed the
        coordinator to switch models before the next Orchestra request was
        admitted, producing coarse 4096-request waves.
        """
        await self._load_balancer.release_server.remote(server_id=server_id)

    def _mas_loop_exclusive_enabled(self) -> bool:
        return os.getenv("MAS_LOOP_EXCLUSIVE_VLLM", "0") in {"1", "true", "True"}

    async def _mas_loop_actor_acquire(self) -> bool:
        if not self._mas_loop_exclusive_enabled():
            return False
        # Lazy import keeps upstream Verl behavior unchanged outside this
        # repository's explicitly enabled MAS-Loop jobs.
        from mas_loop_rl import vllm_arbiter

        owner = os.getenv("MAS_LOOP_ACTOR_OWNER", "verl-actor-rollout")
        max_active = int(
            os.getenv(
                "MAS_LOOP_ACTOR_VLLM_MAX_ACTIVE",
                os.getenv("MAS_LOOP_VLLM_MAX_ACTIVE", "1024"),
            )
        )
        # Native async phase RPC is essential here.  Blocking acquires above
        # max_active must not consume the default executor threads needed by
        # completed requests to release their slots.
        lock, first = await vllm_arbiter.async_acquire_transition(owner, max_active=max_active)
        if first:
            try:
                if not self._server_handles:
                    raise RuntimeError("MAS-Loop exclusive rollout requires registered server handles")
                sleeping = await asyncio.gather(
                    *[server.is_sleeping.remote() for server in self._server_handles]
                )
                await asyncio.gather(
                    *[
                        server.wake_up.remote()
                        for server, is_sleeping in zip(self._server_handles, sleeping, strict=True)
                        if bool(is_sleeping)
                    ]
                )
                await vllm_arbiter.async_commit_first_acquire(lock, owner)
            except Exception as exc:
                await vllm_arbiter.async_abort_first_acquire(lock, owner, exc)
                raise
        return True

    async def _mas_loop_actor_release(self) -> None:
        if not self._mas_loop_exclusive_enabled():
            return
        from mas_loop_rl import vllm_arbiter

        owner = os.getenv("MAS_LOOP_ACTOR_OWNER", "verl-actor-rollout")
        lock, _, should_sleep = await vllm_arbiter.async_acquire_release_transition(owner)
        sleep_error = None
        if should_sleep:
            try:
                states = await asyncio.gather(
                    *[server.is_sleeping.remote() for server in self._server_handles]
                )
                # Request-level switching must preserve the current rollout
                # weights. HYBRID server.sleep() uses destructive level 2.
                await asyncio.gather(
                    *[
                        server.sleep_preserve_weights.remote()
                        for server, is_sleeping in zip(self._server_handles, states, strict=True)
                        if not bool(is_sleeping)
                    ]
                )
                final_states = await asyncio.gather(
                    *[server.is_sleeping.remote() for server in self._server_handles]
                )
                if not all(bool(value) for value in final_states):
                    raise RuntimeError("integrated actor vLLM did not enter sleep mode")
            except Exception as exc:
                sleep_error = exc
                logger.exception("MAS-Loop actor vLLM sleep failed")
        await vllm_arbiter.async_finish_release(
            lock, owner, slept=bool(should_sleep and sleep_error is None), error=sleep_error
        )

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        mas_loop_owned = False
        server_id = None
        try:
            # Model lease first, bounded Ray admission second. Same-model
            # requests can queue cheaply at the Unix coordinator and keep one
            # wake phase alive, while only the configured safe number of large
            # generate RPCs reaches Ray/vLLM.
            mas_loop_owned = await self._mas_loop_actor_acquire()
            acquire_started = time.monotonic()
            server_id, server = await self._acquire_server(request_id)
            await self._maybe_write_load_balancer_metrics(
                acquire_wait_sec=max(0.0, time.monotonic() - acquire_started)
            )
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            # Cancellation must not leak either admission or phase ownership.
            # Defer propagation until both releases have completed.
            cancelled: asyncio.CancelledError | None = None
            if server_id is not None:
                release_task = asyncio.create_task(self._release_server(server_id))
                try:
                    await asyncio.shield(release_task)
                except asyncio.CancelledError as exc:
                    cancelled = exc
                    await asyncio.shield(release_task)
            if mas_loop_owned:
                phase_release_task = asyncio.create_task(self._mas_loop_actor_release())
                try:
                    await asyncio.shield(phase_release_task)
                except asyncio.CancelledError as exc:
                    cancelled = cancelled or exc
                    await asyncio.shield(phase_release_task)
            if cancelled is not None:
                raise cancelled


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort"):
                break

            await asyncio.sleep(1)

        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        start_rank: int = 0,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = None):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to ``self.start_rank``
                so standalone replicas can avoid Ray named-actor collisions with hybrid
                replicas (which start at 0) when both coexist (e.g. separate async).
        """
        if start_rank is None:
            start_rank = self.start_rank
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        # Flush immediately: short MAS-Loop rollouts can finish before buffered Ray stdout is emitted,
        # which prevents the external KV-cache monitor from discovering /metrics in time.
        print(f"LLMServerManager: {self.server_addresses}", flush=True)

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    async def _init_global_load_balancer(self) -> None:
        max_outstanding = get_generate_admission_limit()
        # Blocked acquire_server coroutines plus release/control RPCs must all
        # be schedulable, otherwise an actor-level concurrency cap equal to the
        # admission limit can deadlock at full occupancy.
        admission_concurrency = max_outstanding if max_outstanding else 1000
        concurrency_groups = {
            "admission": admission_concurrency,
            # A dedicated group guarantees that release RPCs can run while
            # every admission coroutine is waiting at a full gate.
            "control": MAS_LOOP_LB_CONTROL_CONCURRENCY,
        }
        load_balancer_actor = make_global_request_load_balancer_actor(admission_concurrency)
        self.global_load_balancer = load_balancer_actor.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            max_outstanding=max_outstanding,
        )
        if max_outstanding:
            print(
                "LLMServerManager: MAS-Loop Ray generate admission "
                f"max_outstanding={max_outstanding} concurrency_groups={concurrency_groups}",
                flush=True,
            )

    def get_client(self, client_cls=LLMServerClient, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate (default: ``LLMServerClient``).
                Pass ``FullyAsyncLLMServerClient`` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        return client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            server_handles=self.server_handles,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])

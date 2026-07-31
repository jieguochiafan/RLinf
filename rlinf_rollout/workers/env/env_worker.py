# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async environment worker.

Ported from the training repo's ``rlinf/workers/env/{env_worker,async_env_worker}.py``,
keeping only the async/service path. Removed trainer couplings:

* ``calculate_adv_and_returns`` / ``use_training_pipeline`` micro-batch packing are
  gone. Advantage/return computation is trainer-side; the only hook left is the
  optional :class:`~rlinf_rollout.postprocess.TrajectoryPostprocessor`.
* ``compute_bootstrap_rewards`` -> :class:`~rlinf_rollout.postprocess.BootstrapRewardShaper`.
* ``actor_channel.put(trajectory)`` and ``get_actor_split_num()`` (which read the
  actor's world size) -> :class:`~rlinf_rollout.api.v1.TrajectorySink` whose
  :class:`~rlinf_rollout.api.v1.ConsumerSpec` declares the partitioning.
* ``cfg.actor.model`` -> ``cfg.policy.model``; ``cfg.algorithm.*`` /
  ``cfg.runner.*`` -> ``cfg.rollout.*`` (see :mod:`rlinf_rollout.config.rollout`).
"""

import asyncio
import gc
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf_rollout.api.v1 import ConsumerSpec, PartitionAxis, TrajectorySink
from rlinf_rollout.config import RolloutConfig
from rlinf_rollout.data.convert import trajectory_to_api
from rlinf_rollout.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedLerobotRolloutResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf_rollout.envs import get_env_cls
from rlinf_rollout.envs.action_utils import prepare_actions
from rlinf_rollout.envs.utils import get_env_attr
from rlinf_rollout.envs.wrappers import RecordVideo
from rlinf_rollout.postprocess import (
    BootstrapRewardShaper,
    TrajectoryPostprocessor,
    load_trajectory_postprocessor,
)
from rlinf_rollout.scheduler import Channel, Cluster, Worker
from rlinf_rollout.sinks import ChannelTrajectorySink, NullTrajectorySink
from rlinf_rollout.utils.data_iter_utils import split_list
from rlinf_rollout.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    update_nested_cfg,
)
from rlinf_rollout.utils.placement import HybridComponentPlacement
from rlinf_rollout.workers.env.history_manager import HistoryManager

__all__ = ["AsyncEnvWorker"]


class AsyncEnvWorker(Worker):
    """Steps vectorized envs against a rollout worker and emits trajectories.

    The worker owns ``stage_num`` vectorized envs (pipeline stages). It runs as a
    resident coroutine: :meth:`interact` loops forever, publishing metrics after
    every pass and handing finished trajectories to a
    :class:`~rlinf_rollout.api.v1.TrajectorySink`.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.rollout_cfg = RolloutConfig.from_dictconfig(cfg)
        self.should_stop = False

        self.env_list: list[Any] = []
        self.eval_env_list: list[Any] = []
        self.last_obs_list: list[Any] = []
        self.last_intervened_info_list: list[Any] = []
        self._prefetched_train_bootstrap: Optional[list[EnvOutput]] = None
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = bool(cfg.rollout.collect_transitions)
        self.collect_prev_infos = bool(cfg.rollout.collect_prev_infos)
        self.stage_num = self.rollout_cfg.stage_num
        self.enable_rlt = self.rollout_cfg.plugin_enabled("rlt")
        self.enable_online_lerobot = self.rollout_cfg.plugin_enabled("online_lerobot")

        reward_cfg = OmegaConf.select(cfg, "reward", default=None) or {}
        self.reward_mode = reward_cfg.get("reward_mode", "per_step")
        self.history_reward_assign = reward_cfg.get("history_reward_assign", False)
        self.use_reward_model = reward_cfg.get("use_reward_model", False)
        self.use_realworld_reward = reward_cfg.get("standalone_realworld", False)
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        self.reward_weight = float(reward_cfg.get("reward_weight", 1.0))
        self.env_reward_weight = float(reward_cfg.get("env_reward_weight", 0.0))

        self.model_cfg = self.rollout_cfg.policy_model_cfg
        self.enable_train = self.rollout_cfg.enable_train
        self.enable_eval = self.rollout_cfg.enable_eval
        self.rollout_epoch = self.rollout_cfg.rollout_epoch
        self.eval_rollout_epoch = self.rollout_cfg.eval_rollout_epoch

        train_env_cfg = OmegaConf.select(cfg, "env.train", default=None)
        eval_env_cfg = OmegaConf.select(cfg, "env.eval", default=None)
        self.train_enable_offload = (
            bool(train_env_cfg.get("enable_offload", False))
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            bool(eval_env_cfg.get("enable_offload", False))
            if eval_env_cfg is not None
            else False
        )
        assert not (self.train_enable_offload or self.eval_enable_offload), (
            "env.*.enable_offload is not supported by AsyncEnvWorker."
        )

        if self.enable_train:
            self.train_num_envs_per_stage = (
                self.rollout_cfg.total_num_train_envs
                // self._world_size
                // self.stage_num
            )
            self.train_batch_size = self.rollout_cfg.train_batch_size
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.rollout_cfg.total_num_eval_envs
                // self._world_size
                // self.stage_num
            )
            self.eval_batch_size = self.rollout_cfg.eval_batch_size
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]

        self.n_train_chunk_steps = self.rollout_cfg.n_train_chunk_steps
        self.n_eval_chunk_steps = self.rollout_cfg.n_eval_chunk_steps

        self.bootstrap_shaper = BootstrapRewardShaper.from_config(cfg)
        self.trajectory_postprocessor: Optional[TrajectoryPostprocessor] = (
            load_trajectory_postprocessor(
                OmegaConf.select(
                    cfg, "rollout.postprocess.trajectory_postprocessor", default=None
                )
            )
        )
        self._trajectory_sink: Optional[TrajectorySink] = None

        self.env_decoupled_mode = self.rollout_cfg.decoupled
        if self.env_decoupled_mode:
            self.batch_router: dict[str, list] = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "env world size must be >= rollout world size in decoupled mode."
            )

        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self._interact_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------- trajectory sink

    @property
    def trajectory_sink(self) -> TrajectorySink:
        """The sink trajectories are published to (``NullTrajectorySink`` if unset)."""
        if self._trajectory_sink is None:
            self._trajectory_sink = NullTrajectorySink()
        return self._trajectory_sink

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        """Install the destination for collected trajectories.

        Args:
            sink: Any :class:`~rlinf_rollout.api.v1.TrajectorySink`. Its
                :class:`~rlinf_rollout.api.v1.ConsumerSpec` decides how many shards
                each trajectory is split into.
        """
        self._trajectory_sink = sink

    def build_channel_sink(self, channel: Channel) -> ChannelTrajectorySink:
        """Build a channel-backed sink using ``sink.num_shards`` from the config."""
        num_shards = self.rollout_cfg.sink_num_shards or 1
        spec = ConsumerSpec(
            num_partitions=num_shards,
            axis=PartitionAxis.ENV if num_shards > 1 else PartitionAxis.NONE,
        )
        return ChannelTrajectorySink(channel, spec)

    @property
    def num_sink_shards(self) -> int:
        """Number of partitions the consumer declared."""
        return self.trajectory_sink.consumer_spec.num_partitions

    # ------------------------------------------------------------------ setup

    def _prepare_rollout_results(self, rollout_results: list | None = None) -> list:
        if self.enable_online_lerobot and rollout_results is not None:
            for stage_rollout in rollout_results:
                stage_rollout.rewards.clear()
            return rollout_results

        max_episode_length = self.cfg.env.train.max_episode_steps
        if self.enable_online_lerobot:
            only_success = bool(
                OmegaConf.select(
                    self.cfg,
                    "rollout.plugins.dagger.online_lerobot.only_success",
                    default=False,
                )
            )
            return [
                EmbodiedLerobotRolloutResult(
                    max_episode_length=max_episode_length,
                    num_envs=self.train_num_envs_per_stage,
                    only_success=only_success,
                    num_action_chunks=self.rollout_cfg.num_action_chunks,
                    action_dim=self.rollout_cfg.action_dim,
                )
                for _ in range(self.stage_num)
            ]
        return [
            EmbodiedRolloutResult(max_episode_length=max_episode_length)
            for _ in range(self.stage_num)
        ]

    def init_worker(self):
        """Create the vectorized envs and their wrappers."""
        # Barrier ensuring every env finished its import-time setup (needed by
        # RealWorld envs that spin up ROS nodes).
        self.broadcast(True, groups=[(self._group_name, list(range(self._world_size)))])

        self.update_env_cfg()

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        if self.enable_train and self.reward_mode == "history_buffer":
            self.train_history_managers = [
                HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                for _ in range(self.stage_num)
            ]
            self.history_lengths = [{} for _ in range(self.stage_num)]

        self._init_env()

    def update_env_cfg(self):
        """Apply per-rank ``override_cfgs`` and realworld reward wiring."""
        for name in ("train", "eval"):
            if name == "train" and not self.enable_train:
                continue
            if name == "eval" and not self.enable_eval:
                continue
            env_cfg = self.cfg.env[name]
            override_cfgs = env_cfg.get("override_cfgs", None)
            if override_cfgs is not None:
                assert len(override_cfgs) > self._rank, (
                    f"{len(override_cfgs)=} > {self._rank=}"
                )
                general_override_cfg = OmegaConf.to_container(
                    env_cfg.get("override_cfg", {}), resolve=True
                )
                rank_override_cfg = OmegaConf.to_container(
                    override_cfgs[self._rank], resolve=True
                ).copy()
                base_cfg: dict[str, Any] = {}
                base_cfg = update_nested_cfg(base_cfg, general_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, rank_override_cfg)
                setattr(env_cfg, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(env_cfg)

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []
        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            data_collection = env_cfg.get("data_collection", None)
            if data_collection is not None and getattr(
                data_collection, "enabled", False
            ):
                from rlinf_rollout.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(data_collection, "export_format", "pickle"),
                    robot_type=getattr(data_collection, "robot_type", "panda"),
                    fps=getattr(data_collection, "fps", 10),
                    only_success=getattr(data_collection, "only_success", False),
                    finalize_interval=getattr(
                        data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train and self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))

    # ---------------------------------------------------------- env stepping

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: Any, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """Execute one action chunk in the training env of ``stage_id``."""
        exec_actions = prepare_actions(
            raw_chunk_actions=chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.rollout_cfg.num_action_chunks,
            action_dim=self.rollout_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
            env_cfg=self.cfg.env.train,
        )
        if isinstance(chunk_actions, dict):
            chunk_actions["actions"] = exec_actions
        else:
            chunk_actions = exec_actions
        env_info: dict[str, Any] = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        extracted_obs = obs_list[-1] if obs_list else None
        infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any() and "final_info" in infos:
            final_info = infos["final_info"]
            for key in final_info["episode"]:
                env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = infos.get("intervene_action")
        intervene_flags = infos.get("intervene_flag")
        rlt_switch_flags = infos.get("rlt_switch_flags")
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            rlt_switch_flags=rlt_switch_flags,
        )
        chunk_step_payload = {
            "chunk_actions": exec_actions,
            "obs_list": obs_list,
            "terminations": chunk_terminations,
            "truncations": chunk_truncations,
            "infos_list": infos_list,
        }
        return env_output, env_info, chunk_step_payload

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """Execute one action chunk in the eval env of ``stage_id``."""
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.rollout_cfg.num_action_chunks,
            action_dim=self.rollout_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            env_cfg=self.cfg.env.eval,
        )
        env_info: dict[str, Any] = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        extracted_obs = obs_list[-1] if obs_list else None
        infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones.any(dim=1)
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            env_infos=infos if isinstance(infos, dict) else None,
            rlt_switch_flags=infos.get("rlt_switch_flags"),
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Defaults to the last rollout observation of each env, then overrides envs
        that terminated earlier in the chunk with the ``final_observation``
        captured at that substep.
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)
        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue
                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer the batch dim of a routed shard.

        Channels may carry a ``RolloutResult``, a reward tensor or plain eval
        actions into the same recv, so probe rather than assume dataclass fields.
        """
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return int(data.shape[0])
        if isinstance(data, RolloutResult):
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(data, field_name, None)
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
            forward_inputs = getattr(data, "forward_inputs", None)
            if forward_inputs:
                first_tensor = next(iter(forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return int(first_tensor.shape[0])
            raise ValueError("Cannot infer batch size from rollout result.")
        from rlinf_rollout.scheduler import infer_batch_size

        return infer_batch_size(data)

    # --------------------------------------------------------- reward model

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        """Round-trip observations through the external reward worker group."""
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None

        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None and getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
            reward_input["dones"] = dones

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)

        if last_run:
            reward_input["last_run"] = torch.ones(
                (self.train_num_envs_per_stage, 1), dtype=torch.bool
            )

        self.send_to(
            group_name=self.cfg.reward.group_name,
            channel=send_channel,
            data=reward_input,
            tag="train_reward_obs",
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        )
        reward_output = self.recv_from(
            group_name=self.cfg.reward.group_name,
            channel=recv_channel,
            tag="train_reward_obs",
            batch_size=self.train_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        return {
            key: clone_nested_to_cpu(env_infos[key])
            for key in self.env_infos_reward_keys
            if key in env_infos
        }

    def _scatter_terminal_reward_output(
        self, env_output: EnvOutput, reward_output: torch.Tensor
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def assign_history_reward(self, stage_id: int, reward_model_output: torch.Tensor):
        """Back-propagate a history-buffer reward over the steps it summarizes."""
        reward_assign_lengths = [
            min(
                history_buffer_length[env_id]
                for history_buffer_length in self.history_lengths[stage_id].values()
            )
            for env_id in range(self.train_num_envs_per_stage)
        ]
        rollout_rewards = self.rollout_results[stage_id].rewards
        rollout_rewards_length = len(rollout_rewards)
        reward_assign_lengths = [
            min(length, rollout_rewards_length) for length in reward_assign_lengths
        ]
        if not any(reward_assign_lengths):
            return
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        for env_id, reward_assign_length in enumerate(reward_assign_lengths):
            for reward_assign_step in range(2, reward_assign_length + 1):
                rollout_rewards[-reward_assign_step][env_id] += reward[env_id]

    # ---------------------------------------------------------- rollout input

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        """Produce the first observation batch of a rollout epoch."""

        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.rollout_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                if self.enable_online_lerobot:
                    rollout_results = getattr(self, "rollout_results", None)
                    if rollout_results is not None:
                        rollout_results[stage_id].reset_episode_buffers()
                dones = get_zero_dones()
                env_outputs.append(
                    EnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        terminations=dones.clone(),
                        truncations=dones.clone(),
                        final_obs=infos.get("final_observation"),
                        env_infos=infos if isinstance(infos, dict) else None,
                        intervene_actions=None,
                        intervene_flags=None,
                    )
                )
        else:
            dones = get_zero_dones()
            for stage_id in range(self.stage_num):
                env_outputs.append(
                    EnvOutput(
                        obs=self.last_obs_list[stage_id],
                        rewards=None,
                        dones=dones,
                        terminations=dones.clone(),
                        truncations=dones.clone(),
                        intervene_actions=self.last_intervened_info_list[stage_id][0],
                        intervene_flags=self.last_intervened_info_list[stage_id][1],
                    )
                )
        return env_outputs

    def _build_rollout_input_data(self, env_batch: dict[str, Any]) -> dict[str, Any]:
        data = {"obs": env_batch["obs"], "final_obs": env_batch["final_obs"]}
        if self.enable_rlt:
            data["rlt_switch_flags"] = env_batch.get("rlt_switch_flags", None)
            data["intervene_flags"] = env_batch.get("intervene_flags", None)
        return data

    def _send_rollout_input(
        self, rollout_channel: Channel, env_output: EnvOutput, stage_id: int, mode: str
    ) -> None:
        self.send_to(
            group_name=self.rollout_cfg.rollout_group_name,
            channel=rollout_channel,
            data=self._build_rollout_input_data(env_output.to_dict()),
            mode=mode,
            tag="rollout_results",
            route_key=stage_id if not self.env_decoupled_mode else None,
            decoupled_mode=self.env_decoupled_mode,
        )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        for stage_id in range(self.stage_num):
            self._send_rollout_input(
                rollout_channel, env_outputs[stage_id], stage_id, "train"
            )
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def _recv_rollout_result(
        self, input_channel: Channel, stage_id: int
    ) -> RolloutResult:
        return self.recv_from(
            group_name=self.rollout_cfg.rollout_group_name,
            channel=input_channel,
            tag="train_rollout_results",
            route_key=stage_id if not self.env_decoupled_mode else None,
            batch_size=self.train_batch_size,
            merge_fn=RolloutResult.merge_rollout_results,
            infer_batch_size_fn=self._infer_rollout_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any]
    ):
        """Accumulate one step's env info into the metric buffers."""
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        """Remember the trailing observation for the next auto-reset epoch."""
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    def finish_rollout(self, mode: str = "train"):
        """Flush videos and advance the env's reset-state cursor."""
        env_list = self.env_list if mode == "train" else self.eval_env_list
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        for env in env_list:
            if env_cfg.video_cfg.save_video:
                flush_video = get_env_attr(env, "flush_video")
                if callable(flush_video):
                    flush_video()
            if mode == "train" or not env_cfg.auto_reset:
                env.update_reset_state_ids()

    # --------------------------------------------------------- sink publishing

    @Worker.timer("env/publish_trajectories")
    async def publish_trajectories(self, rollout_result: EmbodiedRolloutResult) -> None:
        """Split one stage's buffer per the consumer spec and push it to the sink.

        Replaces the training repo's ``send_rollout_trajectories``, which split by
        the actor's world size and put internal dataclasses straight onto a channel.
        """
        sink = self.trajectory_sink
        spec = sink.consumer_spec
        if spec.partition_sizes:
            trajectories: list[Trajectory] = (
                rollout_result.to_splited_trajectories_by_sizes(
                    list(spec.partition_sizes)
                )
            )
        else:
            trajectories = rollout_result.to_splited_trajectories(spec.num_partitions)
        rollout_result.clear()

        producer = str(self.model_cfg.model_type)
        for partition, trajectory in enumerate(trajectories):
            if self.trajectory_postprocessor is not None:
                trajectory = self.trajectory_postprocessor.process(trajectory)
            payload = trajectory_to_api(
                trajectory,
                producer=producer,
                required_keys=tuple(spec.required_keys),
            )
            await sink.put(payload, partition=partition)
        del trajectories
        gc.collect()

    @Worker.timer("env/publish_lerobot_episodes")
    async def publish_lerobot_episodes(self, episodes: list[list[dict]]) -> None:
        """Push completed LeRobot-format episodes to the sink."""
        if not episodes:
            return
        num_shards = self.num_sink_shards
        chunks = (
            [episodes]
            if num_shards <= 1
            else split_list(episodes, num_shards, enforce_divisible_batch=False)
        )
        sink = self.trajectory_sink
        for partition, chunk in enumerate(chunks):
            if not chunk:
                continue
            await sink.put(chunk, partition=partition)

    # ------------------------------------------------------------- interaction

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results = self._prepare_rollout_results(
            getattr(self, "rollout_results", None)
        )
        env_metrics: dict[str, list] = defaultdict(list)
        rlt_pending_obs: list[dict[str, Any] | None] = [None] * self.stage_num

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(rollout_channel)

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    await asyncio.sleep(0)
                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    self._apply_intervened_actions(stage_id, env_output)

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self._collect_reward_model_output(
                            env_output,
                            reward_channel,
                            input_channel,
                            stage_id,
                            env_metrics,
                        )

                    rollout_result = self._recv_rollout_result(input_channel, stage_id)
                    self._append_chunk_step(
                        stage_id, env_output, rollout_result, reward_model_output
                    )
                    if rollout_result.intervene_flags is not None:
                        self.rollout_results[
                            stage_id
                        ].mark_last_step_with_intervene_flags(
                            rollout_result.intervene_flags
                        )
                    if self.enable_rlt and self.collect_transitions:
                        from rlinf_rollout.plugins.rlt.transition import (
                            update_rlt_transitions,
                        )

                        update_rlt_transitions(
                            stage_id,
                            rlt_pending_obs,
                            self.rollout_results,
                            rollout_result,
                            cache_current=True,
                        )

                    env_output, env_info, chunk_step_payload = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    stage_rollout = self.rollout_results[stage_id]
                    if isinstance(stage_rollout, EmbodiedLerobotRolloutResult):
                        stage_rollout.append_chunk_episode_data(
                            rollout_result=rollout_result, **chunk_step_payload
                        )
                    self._send_rollout_input(
                        rollout_channel, env_output, stage_id, "train"
                    )
                    if self.collect_transitions and not self.enable_rlt:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        stage_rollout.append_transitions(curr_obs, next_obs)

                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)

            # Terminal round: record the epoch's last observation without stepping.
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                self._apply_intervened_actions(stage_id, env_output)

                reward_model_output = None
                if reward_channel is not None:
                    reward_model_output = self._collect_reward_model_output(
                        env_output,
                        reward_channel,
                        input_channel,
                        stage_id,
                        env_metrics,
                        last_run=epoch == self.rollout_epoch - 1,
                    )

                rollout_result = self._recv_rollout_result(input_channel, stage_id)
                self._append_chunk_step(
                    stage_id, env_output, rollout_result, reward_model_output
                )
                if self.enable_rlt and self.collect_transitions:
                    from rlinf_rollout.plugins.rlt.transition import (
                        update_rlt_transitions,
                    )

                    update_rlt_transitions(
                        stage_id,
                        rlt_pending_obs,
                        self.rollout_results,
                        rollout_result,
                        cache_current=False,
                    )

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if self.enable_online_lerobot:
            for stage_id in range(self.stage_num):
                await self.publish_lerobot_episodes(
                    self.rollout_results[stage_id].drain_episodes()
                )
        else:
            for stage_id in range(self.stage_num):
                await self.publish_trajectories(self.rollout_results[stage_id])

        return {
            key: torch.cat(value, dim=0).contiguous().cpu()
            for key, value in env_metrics.items()
        }

    def _apply_intervened_actions(self, stage_id: int, env_output: EnvOutput) -> None:
        if env_output.intervene_actions is None:
            return
        self.rollout_results[stage_id].update_last_actions(
            env_output.intervene_actions, env_output.intervene_flags
        )

    def _collect_reward_model_output(
        self,
        env_output: EnvOutput,
        reward_channel: Channel,
        input_channel: Channel,
        stage_id: int,
        env_metrics: dict[str, list],
        *,
        last_run: bool = False,
    ) -> torch.Tensor | None:
        reward_model_output = self.get_reward_model_output(
            env_output,
            send_channel=reward_channel,
            recv_channel=input_channel,
            stage_id=stage_id,
            last_run=last_run,
        )
        if reward_model_output is not None:
            env_metrics["reward_model_output"].append(
                reward_model_output.detach().float().reshape(-1).cpu()
            )
        return reward_model_output

    def _append_chunk_step(
        self,
        stage_id: int,
        env_output: EnvOutput,
        rollout_result: RolloutResult,
        reward_model_output: torch.Tensor | None,
    ) -> None:
        rewards = self.bootstrap_shaper.shape(
            rewards=env_output.rewards,
            dones=env_output.dones,
            truncations=env_output.truncations,
            bootstrap_values=rollout_result.bootstrap_values,
            reward_model_output=reward_model_output,
        )
        self.rollout_results[stage_id].append_step_result(
            ChunkStepResult(
                actions=rollout_result.forward_inputs.get("action", None),
                prev_logprobs=(
                    rollout_result.prev_logprobs if self.collect_prev_infos else None
                ),
                prev_values=(
                    rollout_result.prev_values if self.collect_prev_infos else None
                ),
                forward_inputs=rollout_result.forward_inputs,
                versions=rollout_result.versions,
                dones=env_output.dones,
                truncations=env_output.truncations,
                terminations=env_output.terminations,
                rewards=rewards,
            )
        )
        if (
            self.reward_mode == "history_buffer"
            and self.history_reward_assign
            and reward_model_output is not None
        ):
            self.assign_history_reward(stage_id, reward_model_output)

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        trajectory_channel: Channel | None,
        metric_channel: Channel,
    ):
        """Resident collection loop; returns only when :meth:`stop` cancels it."""
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        if trajectory_channel is not None and self._trajectory_sink is None:
            self.set_trajectory_sink(self.build_channel_sink(trajectory_channel))

        self._interact_task = asyncio.create_task(
            self._interact(
                input_channel, rollout_channel, reward_channel, metric_channel
            )
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        metric_channel: Channel,
    ):
        while True:
            env_metrics = await self._run_interact_once(
                input_channel, rollout_channel, reward_channel
            )
            metric_channel.put(
                {
                    "rank": self._rank,
                    "env": {f"env/{k}": v for k, v in env_metrics.items()},
                    "time": {
                        f"time/env/{k}": v
                        for k, v in self.pop_execution_times().items()
                    },
                },
                async_op=True,
            )

    async def stop(self):
        """Cancel the resident collection loop and release the sink."""
        self.should_stop = True
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()
        if self._trajectory_sink is not None:
            await self._trajectory_sink.close()

    # ------------------------------------------------------------- evaluation

    @Worker.timer("evaluate")
    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        """Run ``eval_rollout_epoch`` evaluation epochs and return env metrics."""
        eval_metrics: dict[str, list] = defaultdict(list)
        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos.get("final_observation"),
                        env_infos=infos if isinstance(infos, dict) else None,
                    )
                    self._send_rollout_input(
                        rollout_channel, env_output, stage_id, "eval"
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    rollout_results = self.recv_from(
                        group_name=self.rollout_cfg.rollout_group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.eval_batch_size,
                        infer_batch_size_fn=self._infer_rollout_batch_size
                        if self.env_decoupled_mode
                        else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    raw_chunk_actions = getattr(
                        rollout_results, "actions", rollout_results
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )
                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    is_last_step = eval_step == self.n_eval_chunk_steps - 1
                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and is_last_step
                        ):
                            continue
                    elif is_last_step:
                        continue
                    self._send_rollout_input(
                        rollout_channel, env_output, stage_id, "eval"
                    )

            self.finish_rollout(mode="eval")

        return {
            key: torch.cat(value, dim=0).contiguous().cpu()
            for key, value in eval_metrics.items()
        }

# 异步训练执行指令说明

本文档用于集中保存 RLinf 异步训练、资源 profile 和轨迹收集策略相关的运行指令。
每条指令都应说明它解决的场景、可直接复制的命令、关键变量含义、重要 Hydra
覆盖项，以及运行产物写到哪里。这样后续只需要调整轨迹数、资源放置、batch size
或 profile 参数，就能复用同一类命令，而不是重新从日志里还原 shell 片段。

新增相似指令时，建议按以下结构记录：

- `用途`: 说明这条命令验证或采集什么，例如吞吐、CPU/GPU 利用率、轨迹等待策略。
- `推荐指令`: 给出优先使用的 wrapper 或入口脚本命令。
- `特殊变量`: 解释 shell 环境变量如何影响输出目录、采样间隔、profile 范围等。
- `关键覆盖项`: 解释 Hydra override 对资源放置、轨迹收集、replay buffer、训练 batch
  的影响。
- `输出位置`: 列出日志、trace、图表、统计结果等关键产物路径。
- `语义说明`: 对容易混淆的参数关系做解释，例如轨迹数阈值和训练 batch size 的区别。

## 每 16 条轨迹训练一次并采集资源利用率

### 用途

这是一个异步训练资源利用率 profile 示例。它把 actor 训练触发阈值设置为
16 条轨迹：每轮训练前，actor 等待 replay buffer 收到足够轨迹，然后再执行一次
actor training。该命令还会采集 torch profiler、CPU sampler，并在训练结束后生成
聚合后的资源利用率数据和曲线图。

### 推荐指令

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_16traj_2gpu_$(date +%Y%m%d_%H%M%S) \
TRAJECTORIES_PER_TRAIN=16 \
PROFILE_ACTIVE_STEPS=25 \
CPU_INTERVAL=0.1 \
bash tools/run_async_gipo_resource_profile.sh \
  cluster.component_placement.actor=5 \
  cluster.component_placement.rollout=4 \
  cluster.component_placement.env=4-5 \
  env.train.total_num_envs=64 \
  actor.micro_batch_size=64 \
  actor.global_batch_size=256 \
  rollout.pipeline_stage_num=1
```

### 特殊变量

- `RUN_DIR`: 本次运行的输出目录。训练日志、trajectory timestamp、torch trace、
  CPU 采样和聚合后的资源利用率图都会写到这里。
- `TRAJECTORIES_PER_TRAIN`: 每次 actor training 前要收集到 replay buffer 的轨迹数。
  wrapper 会把它同步覆盖到：
  `env.train.v17_continuous_collect.target_trajectories`、
  `algorithm.replay_buffer.min_buffer_size`、
  `algorithm.replay_buffer.cache_size`、
  `algorithm.replay_buffer.sample_window_size` 和
  `actor.recv_drain_max_trajectories`。
- `PROFILE_ACTIVE_STEPS`: torch profiler 的 active step 数。值越大，trace 覆盖越长，
  但 trace 文件也更大。
- `CPU_INTERVAL`: CPU sampler 的采样间隔，单位秒。`0.1` 表示 100ms 采样一次。

### 关键覆盖项

- `cluster.component_placement.actor=5`: actor/training worker 放在 GPU 5。
- `cluster.component_placement.rollout=4`: rollout/generation worker 放在 GPU 4。
- `cluster.component_placement.env=4-5`: env worker 绑定到 GPU 4 和 GPU 5 对应资源域。
- `env.train.total_num_envs=64`: 训练环境数量。提高该值通常会加速轨迹产生，
  也会增加 CPU 和 env 侧压力。
- `actor.micro_batch_size=64`: actor 训练 micro batch size。
- `actor.global_batch_size=256`: actor 训练 global batch size。单 actor rank 时，
  该值就是一次 train global batch 的样本数。
- `rollout.pipeline_stage_num=1`: rollout pipeline stage 数。

### 输出位置

主要输出位于 `${RUN_DIR}`：

- `train.log`: 主训练日志。
- `train.exit`: wrapper 保留的训练退出码。
- `rollout_trajectory_timestamps/`: actor 收到的轨迹时间戳。
- `resource_profile/cpu/`: CPU sampler 原始输出。
- `resource_profile/torch/`: torch profiler 原始 trace。
- `resource_profile/derived/`: 聚合后的 1s 粒度 CPU/GPU/worker profile。
- `resource_profile/resource_utilization.png`: 三面板资源利用率图。
- `resource_profile/resource_utilization.pdf`: PDF 版本资源利用率图。

### 等效底层训练指令

如果只想运行训练，不需要 wrapper 的 CPU sampler、torch trace 聚合和自动绘图，
可以直接运行下面的 `train_async.py` 指令。它显式写出“每 16 条轨迹训练一次”
需要同步设置的 Hydra 覆盖项。

```bash
cd /data1/miliang/RLinf
source /data1/miliang/RLinf/libero_openpi/bin/activate

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH:-}
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export RLINF_TRAINING_EVAL_LOCAL_RAY=1

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_train_16traj_2gpu_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=4,5,6,7 python examples/embodiment/train_async.py \
  --config-path /data1/miliang/RLinf/examples/embodiment/config \
  --config-name libero_spatial_async_gipo_openpi_pi05_verify \
  runner.max_epochs=8 \
  runner.max_steps=8 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.logger_backends=[] \
  runner.logger.log_path="${RUN_DIR}" \
  cluster.component_placement.actor=5 \
  cluster.component_placement.rollout=4 \
  cluster.component_placement.env=4-5 \
  env.train.total_num_envs=64 \
  env.train.v17_continuous_collect.target_trajectories=16 \
  algorithm.replay_buffer.min_buffer_size=16 \
  algorithm.replay_buffer.cache_size=16 \
  algorithm.replay_buffer.sample_window_size=16 \
  +actor.recv_drain_max_trajectories=16 \
  actor.micro_batch_size=64 \
  actor.global_batch_size=256 \
  rollout.pipeline_stage_num=1
```

### 语义说明

“每 16 条轨迹训练一次”不是由 `actor.global_batch_size` 单独控制，而是由以下
收集和 replay buffer 参数共同控制：

- `env.train.v17_continuous_collect.target_trajectories=16`: rollout/env 这一轮连续
  收集 16 条轨迹。
- `algorithm.replay_buffer.min_buffer_size=16`: actor 训练前等待 replay buffer 至少有
  16 条轨迹，这是训练启动的直接门槛。
- `algorithm.replay_buffer.cache_size=16`: replay buffer cache 容量。
- `algorithm.replay_buffer.sample_window_size=16`: replay buffer 采样窗口。
- `+actor.recv_drain_max_trajectories=16`: actor 每轮最多从接收队列 drain 16 条轨迹。

`actor.global_batch_size=256` 控制 actor 训练 batch size；`actor.micro_batch_size=64`
表示单 actor rank 下每个 global batch 会拆成 4 个 micro-batch 做梯度累积。

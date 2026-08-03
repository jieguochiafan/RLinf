# RPent Architecture Base

> 基于本地仓库快照（2026-08-01）的实现级分析。所有代码引用均可在对应文件中核对。
>
> 本文回答两个问题：(1) 项目整体框架是什么；(2) 是否存在 env worker / rollout worker / Ray 基础设施。

---

## 0. 结论先行

**仓库中没有 Ray，也没有 env worker / rollout worker。** RPent 不是 RL 训练框架，而是**单 episode、LLM-in-the-loop 的推理编排框架**：并行/分布式那一层被有意剥掉，取而代之的是「三个本地子进程 + 轻量 RPC」。

证据：

- 全仓 grep `import ray|ray.init|@ray.remote|ray.remote` → **0 个文件命中**
- `worker` 一词仅出现在线程命名（`rpent/planner/codex.py`）、注释，以及一处 `worker_info=None` 参数
- `pyproject.toml` 的依赖列表中没有 `ray`

---

## 1. 项目框架

### 1.1 分层结构

代码按职责分成「框架侧」与「环境侧」两棵树。环境侧不属于安装包（`pyproject.toml` 的 `[tool.setuptools.packages.find]` 只 include `rpent*`），靠把 repo root 注入 `sys.path` 后 import。

```text
rpent/                框架侧（与具体机器人无关）
  cli/main.py           唯一入口（console script: rpent），只做装配
  cli/tui.py            交互模式的终端读取与首轮 prompt 解析
  planner/              LLM agent runtime：api_loop / claude_code / codex + base
  planner/utils/        http_mcp_server.py —— 把 Toolkit 暴露给外部 SDK runtime
  tools/toolkit.py      工具容器基类（schema+handler 注册、执行、结果转多模态 block）
  tools/common.py       跨环境共享的 file/IO 工具
  envs/                 EnvSpec / RunConfig / PromptBundle + 按名 importlib 加载
  utils/                rpc / http_rpc / socket_rpc / daemon / config / egl / logging
                        + vla_client / sam3_client / resources / templates
  dashboard/            本地 FastAPI + SSE 监控与启动器（可选）
  context/              prompt 工具与共享 prompt 段落

robots/               环境侧（可插拔，无需注册代码）
  libero/               env_server / env_client / vla_server / sam3_server
                        tools.py（LiberoPrimitives, ~77KB）/ toolkit.py
                        prompt_bundle.py / prompts/ / guides/
  (robocasa/ franka/ so101/)   规划中

scripts/              安装与代理脚本（LIBERO PRO/PLUS、codex proxy）
docs/                 Sphinx 文档（source-en / source-zh 双语）
```

### 1.2 扩展点契约

环境**不需要注册表**。`rpent/envs/base.py` 直接 `importlib.import_module(f"robots.{name}")`，只要求该包暴露两个工厂：

```python
# robots/<name>/__init__.py
def get_env_spec() -> EnvSpec: ...
def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None) -> Toolkit: ...
```

`EnvSpec`（`rpent/envs/env_spec.py`，frozen dataclass）携带环境身份、`PromptBundle`，以及三个 runner hook：

| Hook | 签名 | 职责 |
|------|------|------|
| `add_cli_args` | `(parser, use_dashboard) -> None` | 环境把自己的 CLI flag 注册到共享 parser；`use_dashboard=True` 时把必填项改为选填，留给 dashboard 填 |
| `parse_config` | `(args) -> RunConfig` | 校验并派生 `recipe_tag` / `output_dir` / `prompt_vars` / `dashboard_state` / `task_desc` |
| `init_runtime` | `(args, output_dir) -> (list[ProcessDaemon], dict)` | 拉起或 attach 服务进程，返回 daemon 列表与 `primitives_kwargs` |

这三个 hook 是让 `main.py` **完全环境无关**的关键：CLI 参数由环境自己注册，运行时进程由环境自己拉起。`main.py` 里不 import 任何环境专属类。

> `rpent/cli/main.py` 顶部有显式约定：不得从其它 `rpent` 模块反向 import `rpent.cli`，否则会形成 import 环。

### 1.3 一次 run 的时序

`rpent/cli/main.py` 只做装配，顺序如下：

1. **两阶段 argparse**：先 `parse_known_args()` 拿到 `--env` / `--dashboard`
2. `env_spec.add_cli_args(parser, use_dashboard=...)` — 环境注册自己的 flag
3. `parser.parse_args()` 完整校验，保留 argparse 原生 usage / 报错
4. 若 `--dashboard`：先起 launcher，**阻塞等用户点 Run**，再把提交的配置覆盖回 `args`
5. `env_spec.parse_config(args)` → `RunConfig`（dashboard 模式下在这里补校验必填项）
6. `init_output_dir()` 建目录、配置 `run.log`
7. `ensure_resources(env_name)` 从 HF dataset `RLinf/RPent-memory` 同步 memory（`HF_HUB_OFFLINE=1` 跳过；失败只 warn，不阻断）
8. `build_planner(...)` + 从 `PromptBundle` 渲染 system / user prompt
9. **`env_spec.init_runtime(args, output_dir)`** — 服务进程在这里起来
10. `get_toolkit(env_name, primitives_kwargs=..., video_path=..., dashboard=...)`
11. `planner.solve(...)` 跑工具调用循环
12. `finally`：`write_recipe(recipe_tag)` → `toolkit.close()`（落 `episode.mp4`）→ 逐个 `daemon.stop()`
13. 写 `<output_dir>/transcript_<recipe_tag>.json`（图片 payload 已被 `_strip_images` 剔除）

### 1.4 三条正交的解耦轴

| 轴 | 切换方式 | 上层是否受影响 |
|----|---------|--------------|
| **Planner** | `--planner api\|claude_code\|codex` | 工具与 prompt 完全不变 |
| **Env** | 新建 `robots/<name>/` 并实现 env-client 协议 | planner / tools 不变 |
| **Transport** | `http` / `socket`（两个 Server 类接口同形，drop-in 替换） | planner 不变 |

Planner 侧统一使用 MCP 命名空间 `mcp__rpent__<tool>`（`rpent/planner/base.py` 的 `add_mcp_prefix` / `strip_mcp_prefix`）；`claude_code` 与 `codex` 通过 `rpent/planner/utils/http_mcp_server.py` 把**同一个 Toolkit** 暴露给外部 SDK runtime —— 这是三种 planner 能在同一 benchmark 上公平对比的前提。

`build_planner` 的差异点：

- `api`：pydantic-ai `infer_model`，需带 provider 前缀（`anthropic:` / `openai:` / `openai-chat:`）；API key 始终从 provider 自己的环境变量读，`--base-url` 仅覆盖 base URL
- `claude_code`：Claude Agent SDK 子进程，带 `--max-budget-usd`（默认 `MAX_BUDGET_USD` 或 10）
- `codex`：Codex SDK 子进程，超时取 `CODEX_TIMEOUT_S` → `CELL_TIMEOUT_S` → 1200

两个子进程 planner 都会把 `get_memory_dir(env_name)` 加入 `extra_dirs`，让 agent 能读写跨 run 持久化的 memory。

---

## 2. 有没有 env worker / rollout worker / Ray

### 2.1 没有 Ray：进程模型是自研的 `ProcessDaemon`

`rpent/utils/daemon.py` 里约 60 行的 `ProcessDaemon` 就是全部的「基础设施」：

| 能力 | 实现 | 替代了 Ray 的什么 |
|------|------|-----------------|
| 端口分配 | `pick_free_port()` 在 `127.0.0.1` 上 bind 0 抢一个空闲端口 | 服务发现 |
| 进程拉起 | `subprocess.Popen(cmd, stdin=PIPE, stdout=logfile, stderr=STDOUT, cwd=repo_root)` | actor 创建 |
| **父死子亡** | 子进程 `watch_parent_death()` 起线程阻塞 `sys.stdin.buffer.read()`；父进程一挂管道 EOF 立即返回 → 触发 `shutdown_event` | actor 生命周期 / 孤儿回收 |
| 就绪探测 | `wait_for_ready()` 轮询 `healthz`，同时 `daemon.poll()` 检查子进程是否已崩溃 → **快速失败**而非傻等 300s | 健康检查 |
| 清理 | `stop()`：`terminate()` → 15s 后 `kill()` | actor 销毁 |

`pick_free_port` 的注释坦白承认「socket 关闭到子进程 bind 之间有小竞态，在 127.0.0.1 上实践中接近零」—— 这类取舍贯穿全套基础设施：够用、可读、无外部依赖。

### 2.2 三个 RPC 服务：单例 facade，不是 worker

`robots/libero/__init__.py::_init_runtime` 是唯一的服务装配点。对三个服务走**完全相同**的「有 endpoint 就 attach，没有就 spawn」逻辑：

| 服务 | 进程内容 | 客户端 | CLI flag |
|------|---------|-------|---------|
| `robots/libero/env_server.py` | `rlinf.envs.libero.libero_env.LiberoEnv`，**`num_envs=1`** | `LiberoEnvClient` | `--env-endpoint` |
| `robots/libero/vla_server.py` | Pi0.5（`rlinf.models.embodiment.openpi.get_model`），构造时加载一次并常驻 | `VLAClient` | `--vla-endpoint` |
| `robots/libero/sam3_server.py` | SAM 3.0 分割 | `Sam3Client` | `--sam3-endpoint` |

三者都继承 `RpcFacade`（`rpent/utils/rpc.py`）：基类持有 `shutdown` / `healthz`、transport 绑定、parent-watch、清理；子类只实现 `_dispatch`。

**关键限制**：`RpcFacade.serve` 里 dispatch 包了一把全局 `threading.Lock`。服务器虽然是 `ThreadingHTTPServer` / `ThreadingTCPServer`，但业务调用被**串行化** —— 因为底下的 MuJoCo env 和 GPU 模型都是单实例、非并发安全的。这从设计上就排除了「多 worker 并行 rollout」。

进程隔离的实际动因很具体：**agent 进程完全不 import torch / mujoco**。`env_server.py` 顶部甚至有

```python
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, \
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"
```

所有返回值一律经 `_to_numpy_tree()` 把 torch tensor 转成 CPU numpy 才过线。

### 2.3 与 RLinf 的关系：只借用叶子类，绕过整个 Ray 编排层

RPent 通过 `pyproject.toml` 的 `rpent[rlinf]` extra 依赖 RLinf，并在 `env_server.py` / `vla_server.py` 里把 RLinf checkout 插入 `sys.path`（`RPENT_RLINF_ROOT` / `RLINF_REPO_PATH`，缺省 `../rlinf`）。但**只 import 两个叶子类**：`LiberoEnv` 和 `openpi.get_model`。RLinf 自身的 Ray cluster / placement group / rollout worker 那一整套完全没被使用。

最直接的证据是 `make_env()` 里这行构造：

```python
LiberoEnv(cfg=cfg, num_envs=1, seed_offset=0,
          total_num_processes=1, worker_info=None)
```

`total_num_processes=1`、`worker_info=None` —— 这些参数正是 RLinf 分布式 worker 用来做 shard 划分的，这里被显式置空。同时 `LiberoEnvFacade` 硬编码 `self._env_idx = 0`，所有 obs 都 `_strip()` 掉 leading env 维度，让客户端「不必用 num_envs 思考」（注释原话）。

`build_env_cfg()` 里其余配置也都指向单 episode 评测：`auto_reset=False`、`is_eval=True`、`group_size=1`、`use_fixed_reset_state_ids=True`、`camera_depths=True`（为了 `back_project` 从深度反投影到世界坐标）。

### 2.4 「rollout」的对应物在哪

RL 里的 rollout worker，在这里退化成 `LiberoPrimitives`（`robots/libero/tools.py`）中的同步闭环，跑在 **agent 进程的主线程**上：

```python
def _vlm_chunk(self, instruction):
    self._last_obs["task_descriptions"] = instruction     # 覆盖 prompt
    actions, _ = self.model.predict_action_batch(...)     # RPC → vla_server, [chunk=5, 7]
    chunk_obs, _r, _t, _tr, _i = self.env.chunk_step(actions)
    # ↑ 单次 RPC → env_server，per-step 循环在 server 侧执行
    self.set_obs(obs)
```

在此之上套 `for c in range(max_chunks)` 就构成两个 VLA 原语：

| 原语 | 循环上限 | 成功判据 |
|------|---------|---------|
| `pi0_pick` | `max_chunks=24` | 启发式：eef 下降 ≥ 10cm（`descent_done`）→ 从最低点回升 ≥ `lift_thresh`（默认 0.05）→ 且夹爪开度 < `gripper_closed_thresh`（默认 0.06）；或 LIBERO 官方 `terminated` |
| `pi0_doubled` | `max_chunks=20` | 只认 LIBERO 官方 `terminated`（用于旋钮、开关、短推等非抓取接触技能） |

`pi0_pick` 的判据刻意区分了「下降到底部」和「抓取后回升」：只看 `peak_z - min_z` 会在下降到最低点时误触发，所以引入 `post_min_peak_z`（观测到新的最低点后重置）。

`move_to` / `rotate_wrist` / `rotate_pitch` / `move_pose` / `release` / `set_gripper` 则是**脚本化 OSC delta 控制**，走 `env.step` 单步，不经 VLA。`move_to` 里还有一条踩坑记录：提取世界 yaw 必须用 `atan2(R[1,0], R[0,0])`，不能用 `as_euler('zyx')[0]` —— 后者在 gripper 朝下（`R[2,2] ≈ -1`）时返回 `-world_yaw`，会静默翻转旋转方向。

两个值得注意的工程细节：

1. **chunk 打包**：整个 action chunk 一次 RPC 送出，per-step 循环留在 env server 侧，把 5 次往返压成 1 次。只在录像时才用 `return_all_frames=True` 取回全部中间帧。
2. **端点元数据校验**：`LiberoEnvClient.__init__` 会调 `env.get_env_meta()` 与自己的 `expected_meta`（suite / task / seed / max_episode_steps）做**严格相等断言**，防止 attach 到残留的旧 env_server。

### 2.5 传输层两套实现

| | `rpent/utils/socket_rpc.py` | `rpent/utils/http_rpc.py`（默认） |
|---|---|---|
| 编码 | pickle，4 字节大端长度前缀 | JSON；numpy 编码为 `{"__ndarray__": b64, "dtype", "shape"}` |
| 连接 | one-request-per-connection | HTTP `POST /call` |
| 错误 | 响应体 `ok=False` + traceback | **永远返回 HTTP 200**，失败信息在 body 的 `ok=False` 里 |
| 备注 | 注释坦白说明「同主机同用户，所以用 pickle 而非更防御性的编码」 | 用 base64 原始字节而非 `tolist()`，避免把每个元素字符串化 |

两个 Server 类接口同形（`server_address` / `serve_forever` / `shutdown` / `server_close`），在 `RpcFacade.serve` 里按 `transport` 参数二选一。

超时是**按方法名硬编码的表**（`robots/libero/env_client.py::_TIMEOUT_S`）：

| 方法 | 超时 |
|------|------|
| `env.reset` | 120s |
| `env.step` | 60s |
| `env.chunk_step` | 120s |
| `env.render_camera` | 120s |
| 其他（default） | 30s |

### 2.6 GPU 分配：手工传参而非调度器

没有 Ray 的资源调度，`--cuda-device` 直接透传给三个 server 的命令行，且**各自处理方式不同**：

- `vla_server` / `sam3_server`：正常设 `CUDA_VISIBLE_DEVICES`
- `env_server`：**故意不设**，反而主动 `pop` 掉它

原因写在 `env_server.py::main()` 的长注释里：robosuite 在 import 时会断言 `MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`（子串检查），这个断言假设 EGL 序号等于 CUDA ordinal，在多卡机器上会误崩；而该断言以 `CUDA_VISIBLE_DEVICES != ""` 为门。于是清空它，改成分别 pin 两个后端：

```python
configure_egl_device(args.cuda_device)   # → MUJOCO_EGL_DEVICE_ID
torch.cuda.set_device(args.cuda_device)  # → torch 默认设备
```

清空 `CUDA_VISIBLE_DEVICES` 还有一个副作用是刻意利用的：multiprocessing spawn 出的 render worker 继承环境变量，因此也会跳过那条断言。

---

## 3. 完整数据流

一次工具调用穿过的全部层次（以 `pi0_pick` 为例）：

```text
LLM (Claude / GPT / …)
  │  tool_use: mcp__rpent__pi0_pick
  ▼
Planner (api_loop / claude_code / codex)
  │  toolkit.execute_tool("pi0_pick", {...})
  ▼
LiberoToolkit._step  ── 记录 step_idx、切片存 action clip、dump_state
  │  getattr(self._primitives, "pi0_pick")(**kwargs)
  ▼
LiberoPrimitives.pi0_pick  ── for c in range(max_chunks): _vlm_chunk()
  │        ├── VLAClient.predict_action_batch  ──RPC──▶ vla_server (Pi0.5)
  │        └── LiberoEnvClient.chunk_step      ──RPC──▶ env_server (LiberoEnv)
  ▼
ToolResult  ── view_driver_state(step_idx) 组装文本 + PNG base64 image block
  │
  ▼
回到 LLM 作为下一轮上下文（同时推给 dashboard SSE）
```

结束条件：LLM 调用 `finish`（`success` / `failure` / `stuck`），或触达 `--max-turns` / `--max-episode-steps`。

`ToolResult`（`rpent/tools/toolkit.py`）负责最后一层格式转换：把结果 dict 里的 `_image_bytes` / `_image_cam_bytes` / `_image_wrist_bytes` 拆出来编成 Anthropic 形状的 image block，文本部分裁到 `MAX_TEXT_BYTES_IN_RESULT = 60000` 字节。`execute_tool` 捕获所有异常并转成 `{"error", "traceback"}` 返回给 LLM，而不是让 run 崩掉 —— 这是 agent 能自行恢复的前提。

---

## 4. 如果要加分布式，缺什么

当前架构为横向扩展留好了缝：

**已具备**

- 三个 `--*-endpoint` 参数支持 attach 到任意已存在的远程服务（`[protocol://]host:port`，`protocol=http|socket`）
- `RpcFacade` 能 bind `0.0.0.0`（代码里已处理 `0.0.0.0` → `127.0.0.1` 的回显转换）
- agent 进程极轻（不含 torch / mujoco），横向复制成本低

**缺失**

- 没有多 run 的 scheduler / 资源池 / placement 策略
- 没有跨 episode 的批处理（`num_envs=1` 且 dispatch 被全局锁串行化）
- 没有失败重试、断点续跑

要跑 N 个任务的 benchmark，目前只有两条路：

1. 在外面用 shell 并行起 N 个 `rpent` 进程，每个自带 3 个子进程 —— 简单，但模型会被重复加载 N 次
2. **预先在多卡上常驻 `vla_server` / `sam3_server` 池，让轻量 agent 进程通过 `--vla-endpoint` / `--sam3-endpoint` 复用它们** —— 这是这套设计明显倾向的路径，因为模型加载是最贵的一步

若选方案 2，需注意 `RpcFacade` 的全局锁会让共享的 vla_server 变成串行瓶颈。真要并发，需要去掉那把锁并确认 Pi0.5 前向的线程安全性，或者干脆按 GPU 起多个 server 实例、在外面做轮询分发。

---

## 附：关键文件索引

| 关注点 | 文件 |
|--------|------|
| 入口与装配 | `rpent/cli/main.py` |
| 环境加载与契约 | `rpent/envs/base.py`, `rpent/envs/env_spec.py` |
| 进程生命周期 | `rpent/utils/daemon.py` |
| RPC 基类与就绪探测 | `rpent/utils/rpc.py` |
| 传输实现 | `rpent/utils/http_rpc.py`, `rpent/utils/socket_rpc.py` |
| 工具容器 | `rpent/tools/toolkit.py` |
| Planner 构造 | `rpent/planner/base.py` |
| 服务装配（LIBERO） | `robots/libero/__init__.py::_init_runtime` |
| 环境服务 | `robots/libero/env_server.py`, `robots/libero/env_client.py` |
| VLA 服务 | `robots/libero/vla_server.py`, `rpent/utils/vla_client.py` |
| 原语实现 | `robots/libero/tools.py::LiberoPrimitives` |
| GPU / EGL 处理 | `rpent/utils/egl.py`, `env_server.py::main()` |

上游 Sphinx 文档中的对应章节：`docs/source-en/rst_source/development/architecture.rst`（System Internals）与 `interfaces.rst`。

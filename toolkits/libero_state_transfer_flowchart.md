# LIBERO State Transfer Rendering Flowchart and Timings

Test setup unless otherwise noted:

- LIBERO suite: `libero_spatial`
- Task id / trial id / seed: `0 / 0 / 0`
- Task: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- Cameras: `agentview`, `robot0_eye_in_hand`
- Resolution: `256x256`
- Source actions: seeded random actions, `action_seed=2026`
- Transfer interval: every 10 source `env.step(action)` calls
- State transfer payload: `flat_state = env.get_sim_state()`

```mermaid
flowchart LR
    subgraph A["源仿真器 Env A"]
        A0["初始化 Env A<br/>seed=0<br/>同一 BDDL / task / trial<br/>相机: agentview + wrist<br/>分辨率: 256x256"]
        A1["reset()<br/>set_init_state(init_state)"]
        A2["循环执行 env.step(action)<br/>--random-action 随机采样<br/>action_seed=2026 可复现<br/>真实 NVIDIA EGL<br/>source rollout mean: 20.35 ms<br/>benchmark mean: 12.79 ms"]
        A3{"step_idx % 10 == 0 ?"}
        A4["读取并保存 flat_state<br/>flat_state = env_a.get_sim_state()<br/>保存到 flat_states/flat_state_step_*.npy"]
        A5["源侧强制重渲染对照图<br/>regenerate_obs_from_state(flat_state)"]
        A6["源图像<br/>agentview_image<br/>robot0_eye_in_hand_image"]
    end

    subgraph B["第二个仿真器 Env B"]
        B0["初始化 Env B<br/>seed=0<br/>同一 BDDL / task / trial<br/>同一相机和分辨率"]
        B1["reset 前重新 seed(0)<br/>reset()<br/>set_init_state(init_state)<br/>确保静态模型一致"]
        B2["不执行 env.step(action)"]
        B3["接收 flat_state"]
        B4["写入状态并同步派生量<br/>set_state_from_flattened(flat_state)<br/>sim.forward()<br/>forward only mean: 0.0587 ms"]
        B5["强制更新 observables 并渲染<br/>_update_observables(force=True)<br/>内部调用 sim.render(...)<br/>真实 NVIDIA EGL mean: 2.09 ms"]
        B6["渲染图像<br/>agentview_image<br/>robot0_eye_in_hand_image"]
    end

    subgraph C["比较结果"]
        C1["逐像素比较两路图像"]
        C2["真实 NVIDIA EGL:<br/>状态复现成功<br/>个别像素存在 GPU 渲染微小差异<br/>max_abs_diff <= 2"]
    end

    A0 --> A1 --> A2 --> A3
    A3 -- "否" --> A2
    A3 -- "是，每 10 步" --> A4
    A4 -- "传递 flat_state<br/>float64 (92,)<br/>736 bytes" --> B3
    B0 --> B1 --> B2
    B3 --> B4 --> B5 --> B6
    A4 --> A5 --> A6
    A6 --> C1
    B6 --> C1 --> C2
```

## Data Sizes

| Data | Type / shape | Size | Transferred |
|---|---:|---:|---|
| `flat_state` array | `float64 (92,)` | `736 bytes` | Yes, every 10 steps |
| saved `flat_state_*.npy` file | NumPy `.npy` | `864 bytes` each | Saved every 10 steps |
| `agentview_image` | `uint8 (256, 256, 3)` | `196,608 bytes` | No, only for comparison |
| `robot0_eye_in_hand_image` | `uint8 (256, 256, 3)` | `196,608 bytes` | No, only for comparison |
| `body_pos` | `float64 (38, 3)` | `912 bytes` | No, if seed/reset sequence is identical |
| `body_quat` | `float64 (38, 4)` | `1,216 bytes` | No, if seed/reset sequence is identical |

## Timing Costs: Real NVIDIA EGL

This is the relevant non-sandbox result. The command was run outside the tool
sandbox so the Python process could access `/dev/nvidia*` and `/dev/dri/*`.
OpenGL reported:

| Field | Value |
|---|---|
| `GL_VENDOR` | `NVIDIA Corporation` |
| `GL_RENDERER` | `NVIDIA A800-SXM4-80GB/PCIe/SSE2` |
| `GL_VERSION` | `4.6.0 NVIDIA 580.65.06` |

Command environment:

```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=0
```

Output:

`/tmp/rlinf_libero_state_transfer_egl_real_gpu/libero_state_transfer_validation.json`

| Operation | Mean | Median | P95 | Notes |
|---|---:|---:|---:|---|
| Source rollout `env.step(action)` over 50 steps | `20.346 ms` | `19.877 ms` | `21.454 ms` | Includes normal source stepping and observation rendering; first step was slower at `37.0 ms` |
| Benchmark `env.step(action)` over 100 iters | `12.786 ms` | `12.679 ms` | `13.898 ms` | High-level LIBERO step with random action from a restored state |
| Env B `flat_state -> render` | `2.090 ms` | `2.094 ms` | `2.133 ms` | `set_state + sim.forward + _post_process + _update_observables(force=True) + _get_observations`; includes two camera renders |
| `env.sim.forward()` | `0.0587 ms` | `0.0584 ms` | `0.0596 ms` | State synchronization only; no physical time advance and no rendering |
| `env.sim.step()` | `0.0748 ms` | `0.0581 ms` | `0.0606 ms` | Raw MuJoCo step only; mean includes a small outlier |

Real GPU EGL speedups compared with sandbox llvmpipe EGL:

| Operation | Sandbox llvmpipe EGL | Real NVIDIA EGL | Speedup |
|---|---:|---:|---:|
| Env B `flat_state -> render` | `162.798 ms` | `2.090 ms` | `77.9x` |
| Benchmark `env.step(action)` | `179.559 ms` | `12.786 ms` | `14.0x` |
| Source rollout `env.step(action)` | `183.086 ms` | `20.346 ms` | `9.0x` |

Image comparison under real NVIDIA EGL:

| Camera | Equal syncs | Max abs diff | Mean abs diff notes |
|---|---:|---:|---|
| `agentview_image` | `3 / 5` | `1` | Differences only at tiny pixel-level scale |
| `robot0_eye_in_hand_image` | `4 / 5` | `2` | Differences only at tiny pixel-level scale |

The GPU result is not always bit-exact, but the differences are extremely small
and consistent with GPU rasterization/readback nondeterminism rather than state
transfer failure.

## Timing Costs: Sandbox Mesa llvmpipe

The same command run inside the tool sandbox used `MUJOCO_GL=egl`, but the
Python process could not see `/dev/nvidia*` or `/dev/dri/*`. OpenGL reported:

| Field | Value |
|---|---|
| `GL_VENDOR` | `Mesa` |
| `GL_RENDERER` | `llvmpipe (LLVM 15.0.7, 256 bits)` |
| `GL_VERSION` | `4.5 (Compatibility Profile) Mesa 23.2.1-1ubuntu3.1~22.04.3` |

This means sandbox EGL was CPU software rendering, not NVIDIA GPU rendering.

| Operation | Mean | Median | P95 | Notes |
|---|---:|---:|---:|---|
| Source rollout `env.step(action)` over 50 steps | `183.086 ms` | `183.081 ms` | `187.237 ms` | `/tmp/rlinf_libero_state_transfer_egl_retest` |
| Benchmark `env.step(action)` over 100 iters | `179.559 ms` | `178.796 ms` | `184.086 ms` | High-level LIBERO step |
| Env B `flat_state -> render` | `162.798 ms` | `163.912 ms` | `169.748 ms` | Includes two camera renders |
| `env.sim.forward()` | `0.0616 ms` | `0.0613 ms` | `0.0626 ms` | State synchronization only |
| `env.sim.step()` | `0.0618 ms` | `0.0617 ms` | `0.0623 ms` | Raw MuJoCo step only |

Sandbox OSMesa was also Mesa llvmpipe and therefore similar:

| Operation | Sandbox EGL llvmpipe | Sandbox OSMesa llvmpipe |
|---|---:|---:|
| Source rollout `env.step(action)` mean | `183.086 ms` | `182.521 ms` |
| Benchmark `env.step(action)` mean | `179.559 ms` | `178.413 ms` |
| Env B `flat_state -> render` mean | `162.798 ms` | `161.639 ms` |

## Backend Diagnosis

Why sandbox EGL and OSMesa were equally slow:

- `MUJOCO_GL=egl` selected `EGLGLContext`, but the renderer was still Mesa
  `llvmpipe`.
- `MUJOCO_GL=osmesa` selected `OSMesaGLContext`, also backed by Mesa
  `llvmpipe`.
- In the sandbox, `/dev/nvidia*` and `/dev/dri/*` were not visible to Python.
- Forcing NVIDIA EGL with
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
  returned `eglQueryDevicesEXT() == 0` in the sandbox.
- Running the same check outside the sandbox exposed `/dev/nvidia0..7` and
  `/dev/dri/renderD128..135`, and EGL reported `NVIDIA A800-SXM4-80GB`.

## Interpretation

The main cost is rendering observation images, not MuJoCo state synchronization
or raw physics stepping:

- `sim.forward()` is about `0.06 ms`.
- Raw `sim.step()` is also about `0.06 ms` median.
- Software llvmpipe rendering makes two 256x256 cameras cost about `163 ms`.
- Real NVIDIA EGL makes the same two-camera rerender cost about `2.1 ms`.

## Conclusion

With the same BDDL, trial, camera configuration, resolution, and reset random
sequence, Env A only needs to send `flat_state` to Env B. The transferred data is
`736 bytes` per synchronization, and Env B can rerender the corresponding images
without running `env.step(action)`.

In the true NVIDIA EGL environment, Env B needs about `2.09 ms` to generate the
two 256x256 camera images from `flat_state`. In the sandbox llvmpipe environment,
the same operation takes about `163 ms`, which is why EGL and OSMesa looked
similar before GPU device access was enabled.

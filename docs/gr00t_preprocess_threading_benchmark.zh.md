# GR00T preprocessing 多线程 A/B 测试报告

## 1. 目的

本测试评估 LIBERO + GR00T N1.5 rollout preprocessing 中，将 NumPy 图片转换为
PIL 图片的步骤交给持久 `ThreadPoolExecutor` 是否可以进一步降低延迟，并进一步
测试完全绕过 PIL、直接向 Eagle processor 传入 NumPy/Tensor 图片。

测试中的优化路径还包含：

- 按 `(task description, image count)` 缓存 chat template；
- 跳过输入已经是内存图片时多余的 `process_vision_info()` 遍历；
- 批量构造 state padding 和 state mask；
- 保留 Eagle processor 最终的 batched image processing 和 tokenization。

该优化仅安装在 A/B 测试模型实例上，尚未接入生产 rollout。

## 2. 测试环境

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB，GPU 1 |
| 模型 | `/data1/gaobowen/model/RLinf-Gr00t-SFT-Object` |
| Python 环境 | `/data1/gaobowen/RLinf/.venv-libero-gr00t` |
| 代码 | `/data1/gaobowen/RLinf_mmm` |
| batch size | 12 |
| 每个样本图片数 | 2 |
| 每批图片数 | 24 |
| warmup | 5 次 |
| measure | 20 次/配置 |
| PIL worker 数 | 1、2、4、8、16 |

每个配置先使用相同 observation 对原始和优化 preprocessing 输出做逐字段精确比较，
包括 tensor dtype、shape 和全部元素。只有完全一致才进入计时。

## 3. 结果

| PIL workers | 原始平均 | 优化平均 | 优化 P50 | 优化 P95 | 报告 speedup | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 48.59 ms | **44.12 ms** | 43.12 ms | 50.24 ms | 1.101x | 是 |
| 2 | 48.66 ms | **42.98 ms** | **41.67 ms** | **49.87 ms** | **1.132x** | 是 |
| 4 | 49.22 ms | 47.58 ms | 46.27 ms | 56.21 ms | 1.034x | 是 |
| 8 | 175.68 ms | 139.04 ms | 51.19 ms | 606.44 ms | 1.263x | 是 |
| 16 | 131.32 ms | 190.27 ms | 48.80 ms | 825.31 ms | 0.690x | 是 |

绕过 PIL 的结果：

| Eagle 图片输入 | 原始平均 | 优化平均 | 优化 P50 | 优化 P95 | speedup | 输出一致 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PIL，单线程 | 48.59 ms | 44.12 ms | 43.12 ms | 50.24 ms | 1.101x | 是 |
| NumPy，直接输入 | 48.60 ms | 26.00 ms | 24.80 ms | 31.99 ms | 1.869x | 是 |
| Tensor，零拷贝视图 | 51.13 ms | 23.18 ms | 22.57 ms | 27.14 ms | 2.206x | 是 |
| Tensor，整批 image processor | 48.68 ms | 21.38 ms | 20.53 ms | 26.47 ms | 2.277x | 是 |
| Tensor，整批处理 + token cache | 48.64 ms | **17.23 ms** | **16.73 ms** | **20.99 ms** | **2.823x** | 是 |
| 纯 Tensor transform + 上述优化 | 49.03 ms | **15.25 ms** | **15.72 ms** | **16.81 ms** | **3.215x** | 是 |

原始路径和优化路径在所有线程数下均逐元素完全一致。

### 3.1 1–4 线程的稳定区间

- 1 线程优化版相对原始路径平均降低约 9.2%；
- 2 线程平均降低约 11.7%，是本轮最优配置；
- 2 线程相对 1 线程优化版仅进一步降低约 2.6%；
- 4 线程平均延迟反而比 1 线程优化版高约 7.8%，P95 也升至 56.21 ms。

这说明 NumPy -> PIL 转换可获得的并行收益很小。每批只有 24 张已经 resize 到
224×224 左右的图片，线程调度、Future 管理和结果收集很快超过单张转换的计算收益。

### 3.2 8–16 线程受到严重系统抖动

8 和 16 线程测试期间，原始路径本身的 P50 仍约为 51–52 ms，但 P95 分别达到
706 ms 和 555 ms，说明机器当时存在明显 CPU 调度或资源竞争。因而这两行的平均
speedup 不能与 1–4 线程直接比较，也不能将 8 线程表面的 1.263x 视为有效收益。

高线程数仍暴露了明确的尾延迟风险：

- 8 线程优化路径 P95 为 606 ms；
- 16 线程优化路径 P95 为 825 ms；
- 16 线程平均延迟比同轮原始路径高约 44.9%。

在真实训练中会同时运行多个 rollout worker。如果每个 worker 再创建 8–16 个线程，
CPU oversubscription 和上下文切换会比这个单 worker 测试更严重。

### 3.3 绕过 PIL 是主要收益来源

Eagle processor 的公开接口支持 PIL、NumPy 和 PyTorch Tensor。原始 GR00T 路径先将
已经位于内存中的 NumPy 图片逐张包装为 PIL，随后 Eagle processor 又将它们转换为
Tensor。直接传 NumPy 后平均延迟从约 48.60 ms 降至 26.00 ms；通过
`torch.from_numpy()` 创建零拷贝 Tensor 视图后进一步降至 23.18 ms。

两条路径的全部 preprocessing 输出均与原始 PIL 路径逐元素完全一致，包括
`eagle_pixel_values`、`eagle_image_sizes`、token、attention mask、state 和 state mask。
因此对于当前固定的 uint8 RGB LIBERO observation，不经过 PIL 不会改变模型输入。

继续将24张图片从24次独立 image-processor 调用合并为一次，平均延迟进一步降至
21.38 ms。LIBERO 同一批中的 instruction 和图片 shape 固定时，展开视觉占位符后的文本
也不变；按完整展开文本缓存 token 和 attention mask 后，平均延迟降至17.23 ms。缓存键
包含展开后的完整文本，因此 instruction、图片数或 tile 数变化时会自动 miss 并重新
tokenize。

### 3.4 纯 Tensor crop/resize

原始路径会执行 env Torch -> NumPy -> float Tensor -> crop/resize -> uint8 NumPy ->
Tensor。测试版直接在原始 env Tensor 上完成相同的 center crop、bilinear resize 和
uint8量化，然后送入整批 Eagle processor。全部最终输出逐元素一致，平均 preprocessing
降至15.25 ms，相对同轮原始路径约3.21x。

### 3.5 H2D共享传输与pinned memory

原始 `prepare_input()` 分别构造 backbone input 和 action-head input，再分别递归执行
`.to(device)`；当前两个 `BatchFeature` 都包含完整输入，因此相同CPU tensor会产生两套
GPU copy。测试将输入只传输一次，再让两个消费者共享GPU tensor。

| H2D方案 | 平均 | P50 | P95 | 输出一致 |
| --- | ---: | ---: | ---: | --- |
| 当前两次 `prepare_input` copy | 2.388 ms | 2.338 ms | 2.458 ms | 是 |
| 单次共享 pageable copy | 1.141 ms | 1.137 ms | 1.150 ms | 是 |
| 预分配 pinned buffer + async copy stream | **1.011 ms** | **0.704 ms** | **1.031 ms** | 是 |

单次共享传输已经消除约1.25 ms重复copy。Pinned结果假设输入已经位于可复用的pinned
buffer中，不包含每一步临时调用`pin_memory()`的分配和复制成本；生产实现必须复用固定
buffer，不能每批新建pinned tensor。当前表格在每次copy后同步以测量真实完成延迟，尚未
计入与GPU compute重叠后可能获得的额外收益。

## 4. 结论与建议

1. 纯Tensor transform、整批 image processor 和静态 token cache 的组合是本轮最佳
   preprocessing方案：平均15.25 ms，约3.21x；它不需要额外工作线程。
2. chat-template cache、跳过冗余 vision-info 遍历和批量 state/mask 也属于有效优化。
3. 两个持久 PIL worker 相对单线程 PIL 优化仅多获得约 2.6%，收益较小；在多 rollout worker 的生产配置中
   是否仍有收益需要重新验证。
4. 不建议使用 4 个以上 PIL worker。4 线程已经退化，8–16 线程出现严重尾延迟和
   CPU oversubscription 风险。
5. 在生产接入前，应在真实 rollout worker 数量和 CPU affinity 下重复测试 1、2 线程，
   至少运行数百次，报告多个独立 run 的中位数和置信区间。
6. 下一项更有价值的优化是将下一batch的整个CPU preprocessing与当前batch的GPU
   inference重叠，而不是继续增加图片转换线程。
7. H2D应改为单次传输后由backbone/action head共享；如使用pinned memory，应维护固定
   双缓冲并配合独立copy stream，避免每批pin/unpin。

当前建议生产实现优先采用 Tensor 直传，不引入 PIL 线程池。接入时应仅对已确认的
uint8 RGB LIBERO observation 启用，并保留原始 transform 回退开关。其他 environment、
图片 dtype、channel layout 和 augmentation 配置需要分别验证，不能直接沿用本结论。

## 5. 是否影响最终输出

### 5.1 当前A/B测试没有改变模型输入

每一种优化都先对相同的batch size 12 LIBERO observation运行原始和优化preprocessing，
再对以下字段进行逐元素精确比较，而不是使用浮点容差：

- `eagle_pixel_values`；
- `eagle_image_sizes`；
- `eagle_input_ids`；
- `eagle_attention_mask`；
- `state`和`state_mask`；
- `embodiment_id`。

纯Tensor transform、整批image processor、token cache和H2D共享传输均得到：

```text
outputs_exactly_equal: true
```

H2D测试还分别比较了backbone和action-head收到的GPU tensor，dtype、shape和所有元素
也与当前`prepare_input()`完全相同。因此在当前测试条件下，优化没有改变VLM和action
head的输入。

给定完全相同的模型参数、输入、CUDA执行模式和随机数状态，模型的action、previous
logprob和value应保持相同。优化没有修改denoise step、采样分布、noise、模型算子或
action postprocessing。

### 5.2 仍需完成完整generation验证

目前优化只安装在独立A/B测试路径，尚未接入生产`predict_action_batch()`；当前验证覆盖
preprocessing输出和H2D结果，尚未直接比较完整generation返回的：

- env action；
- `prev_logprobs`；
- `prev_values`；
- replay buffer中的全部`forward_inputs`；
- 多个连续generation下的随机数序列。

生产接入时必须固定相同observation和随机种子，对这些结果进行逐字段比较。Diffusion
sampling使用随机数，因此不能先运行一次baseline、再继续使用已经推进的同一个generator
运行optimized；两条路径必须恢复完全相同的CPU/CUDA RNG state后再比较。

### 5.3 适用边界和潜在风险

当前“输出完全一致”的结论只覆盖本次LIBERO rollout条件：uint8 RGB、两个视角、固定
图片shape、eval transform和GR00T N1.5。以下情况需要重新验证：

- ManiSkill、IsaacLab或其他environment；
- 非uint8、灰度、RGBA、不同channel layout或动态图片尺寸；
- 启用随机crop、color jitter等training augmentation；
- 每个样本的图片数量或动态tile数发生变化；
- instruction在episode内或batch间变化；
- Eagle processor或Transformers版本变化。

Token cache使用展开后的完整文本作为key，instruction、图片数或tile结构变化时会cache
miss并重新tokenize。生产实现仍应限制cache容量，避免任务文本不断变化造成无界增长。

共享H2D tensor要求backbone和action head只读输入；如果任一消费者执行in-place修改，
共享会改变另一个消费者看到的值。接入前应增加测试并在必要时只对确认只读的字段共享。

综合判断：当前实验没有发现任何输出变化，优化在已测试输入上是数值等价的；但在完成
完整generation和生产rollout一致性测试前，不能宣称训练结果已经得到端到端保证。

## 6. 复现

测试脚本：
[`toolkits/rollout_eval/profiling/gr00t_preprocess_ab.py`](../toolkits/rollout_eval/profiling/gr00t_preprocess_ab.py)

示例命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
EMBODIED_PATH=/data1/gaobowen/RLinf_mmm/examples/embodiment \
PYTHONPATH=/data1/gaobowen/RLinf_mmm \
/data1/gaobowen/RLinf/.venv-libero-gr00t/bin/python \
  -m toolkits.rollout_eval.profiling.gr00t_preprocess_ab \
  --batch-size 12 \
  --warmup 5 \
  --iterations 20 \
  --pil-workers 1 \
  --image-input-mode tensor_batched \
  --pure-tensor-transforms \
  --benchmark-h2d \
  --override actor.model.model_path=/data1/gaobowen/model/RLinf-Gr00t-SFT-Object \
  --override rollout.model.model_path=/data1/gaobowen/model/RLinf-Gr00t-SFT-Object \
  --output /tmp/gr00t-preprocess-ab-thread2.json
```

原始结果文件：

- `/tmp/gr00t-preprocess-ab.json`
- `/tmp/gr00t-preprocess-ab-thread2.json`
- `/tmp/gr00t-preprocess-ab-thread4.json`
- `/tmp/gr00t-preprocess-ab-thread8.json`
- `/tmp/gr00t-preprocess-ab-thread16.json`
- `/tmp/gr00t-preprocess-ab-numpy.json`
- `/tmp/gr00t-preprocess-ab-tensor.json`
- `/tmp/gr00t-preprocess-ab-tensor-batched.json`
- `/tmp/gr00t-preprocess-ab-tensor-batched-token-cache.json`
- `/tmp/gr00t-preprocess-ab-pure-tensor-h2d.json`

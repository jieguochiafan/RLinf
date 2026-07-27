# MPS/CUDA 初始化验证命令

用于验证当前用户进程是否能通过 `/tmp/nvidia-mps` 正常初始化 CUDA。

```bash
/bin/bash -lc 'source /data1/miliang/RLinf/libero_openpi/bin/activate && CUDA_VISIBLE_DEVICES=4 timeout 60s python -c '"'"'import torch; print("after import torch", flush=True); print("is_available", torch.cuda.is_available(), flush=True); print("device_count", torch.cuda.device_count(), flush=True); x=torch.ones((1,), device="cuda"); torch.cuda.synchronize(); print("cuda ok", x.item(), flush=True)'"'"''
```

正常结果应在 60 秒内打印到：

```text
cuda ok 1.0
```

如果只打印到 `after import torch` 后卡住或超时，说明当前 MPS/CUDA 初始化仍不可用。

基于RoboCasa评测平台的强化学习训练
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档提供了在RLinf框架中使用RoboCasa环境进行强化学习训练任务的全面指南。
RoboCasa Kitchen专注于厨房环境中的操作任务，具有多样化的厨房布局、物体和操作任务。
RoboCasa Kitchen将真实的厨房环境与多样化的操作挑战相结合，使其成为开发可泛化机器人策略的理想基准。

主要目标是训练能够执行以下任务的视觉-语言-动作模型:

1. **视觉理解**: 处理来自多个摄像头视角的RGB图像。
2. **语言理解**: 解释自然语言任务指令。
3. **操作技能**: 执行复杂的厨房任务，如拾取-放置、开关门和电器控制。

环境
----

**RoboCasa环境**

- **环境**: RoboCasa Kitchen厨房仿真环境(基于robosuite构建)
- **机器人**: Panda机械臂带移动底座(PandaOmron)，配备夹爪
- **观测**: 多视角RGB图像(机器人视角+腕部相机) + 本体感知状态
- **动作空间**: 12维连续动作

  - 3D机械臂位置增量
  - 3D机械臂旋转增量
  - 1D夹爪控制 (开/关)
  - 4D底座控制
  - 1D模式选择（控制底座/机械臂）

**任务类别**

RoboCasa Kitchen提供了涵盖多个类别的24个原子任务（不包含需要底座移动的NavigateKitchen原子任务）:

*门操作任务*:

- ``OpenSingleDoor``: 打开柜门或微波炉门
- ``CloseSingleDoor``: 关闭柜门或微波炉门
- ``OpenDoubleDoor``: 打开双开门柜子
- ``CloseDoubleDoor``: 关闭双开门柜子
- ``OpenDrawer``: 打开抽屉
- ``CloseDrawer``: 关闭抽屉

*拾取和放置任务*:

- ``PnPCounterToCab``: 从柜台拾取并放置到柜子中
- ``PnPCabToCounter``: 从柜子拾取并放置到柜台上
- ``PnPCounterToSink``: 从柜台拾取并放置到水槽中
- ``PnPSinkToCounter``: 从水槽拾取并放置到柜台上
- ``PnPCounterToStove``: 从柜台拾取并放置到炉灶上
- ``PnPStoveToCounter``: 从炉灶拾取并放置到柜台上
- ``PnPCounterToMicrowave``: 从柜台拾取并放置到微波炉中
- ``PnPMicrowaveToCounter``: 从微波炉拾取并放置到柜台上

*电器控制任务*:

- ``TurnOnMicrowave``: 打开微波炉
- ``TurnOffMicrowave``: 关闭微波炉
- ``TurnOnSinkFaucet``: 打开水龙头
- ``TurnOffSinkFaucet``: 关闭水龙头
- ``TurnSinkSpout``: 旋转水槽喷嘴
- ``TurnOnStove``: 打开炉灶
- ``TurnOffStove``: 关闭炉灶

*咖啡制作任务*:

- ``CoffeeSetupMug``: 放置咖啡杯
- ``CoffeeServeMug``: 将咖啡倒入杯中
- ``CoffeePressButton``: 按下咖啡机按钮

**观测结构**

- **主相机图像** (``base_image``): 机器人左侧视角 (224×224 RGB)
- **腕部相机图像** (``wrist_image``): 末端执行器视角相机 (224×224 RGB)
- **腕部相机图像** (``extra_view_image``): 机器人右侧视角 (224×224 RGB, 默认不包含。)
- **本体感知状态** (``state``): 25维向量，包含:
  - ``[0:3]`` 末端执行器位置 (x, y, z)
  - ``[3:7]`` 末端执行器四元数 (w, x, y, z)
  - ``[7:9]`` 夹爪关节位置
  - ``[9:11]`` 夹爪关节速度
  - ``[11:14]`` 末端执行器相对于底座的位置 (x, y, z)
  - ``[14:18]`` 末端执行器相对于底座的四元数 (w, x, y, z)
  - ``[18:21]`` 底座位置 (x, y, z)
  - ``[21:25]`` 底座四元数 (w, x, y, z)

**数据结构**

- **图像**: 主相机RGB张量 ``[batch_size, 3, 224, 224]`` 和腕部相机 ``[batch_size, 3, 224, 224]``。也可以包含右侧相机RGB张量 ``[batch_size, 3, 224, 224]``。
- **状态**: 本体感知状态张量 ``[batch_size, 25]``。 
- **任务描述**: 自然语言指令
- **动作**: 12维连续动作
- **奖励**: 基于任务完成的稀疏奖励

算法
----

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用GAE(广义优势估计)进行优势估计

   - 带比率限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

2. **GRPO (组相对策略优化)**

   - 对于每个状态/提示，策略生成 *G* 个独立动作

   - 通过减去组的平均奖励来计算每个动作的优势

依赖安装
--------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-robocasa
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-robocasa

**选项 2：自建环境**

通过运行以下命令在您的环境中直接安装依赖：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env robocasa
   source .venv/bin/activate

数据集下载
-----------------

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets   # 注意: 需要下载的资源大约有5GB

模型下载
--------------

.. code-block:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-RoboCasa

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-RoboCasa --local-dir RLinf-Pi0-RoboCasa

Pi0.5 与 RoboCasa365 兼容支持
-----------------------------

RLinf 同时支持原始 RoboCasa 的 25 维状态，以及 Pi0.5 RoboCasa checkpoint
使用的标准 16 维状态。环境通过 ``state_space`` 选择输出形式。Pi0.5 示例配置
使用 ``16d``，原有 Pi0 配置的默认行为保持不变。

设置转换后的 checkpoint 路径并启动 Pi0.5 训练：

.. code-block:: bash

   export EMBODIED_PATH=$PWD/examples/embodiment
   export ROBOCASA_MODEL_PATH=/path/to/model/pi05_robocasa
   bash examples/embodiment/run_embodiment.sh \
      robocasa_closedrawer_ppo_openpi_pi05

OpenPI 数据配置 ``pi05_robocasa_human`` 对应公开 RoboCasa LeRobot 格式，
``pi05_robocasa365_closedrawer`` 对应统一的 RoboCasa365 格式。已经压缩为
16 维的状态会直接使用；25 维状态会自动选择字段并按 checkpoint 需要的顺序排列。

如需对 RoboCasa365 做监督微调，设置 ``ROBOCASA_SFT_DATA`` 后，使用
``examples/sft/config/robocasa_sft_openpi_pi05.yaml`` 和
``examples/sft/train_vla_sft.py`` 启动。

还可以绕过 RLinf 环境 runner，使用 RoboCasa 官方环境工厂做独立评估：

.. code-block:: bash

   python toolkits/standalone_eval_scripts/robocasa/native_openpi_eval.py \
      --model-path "$ROBOCASA_MODEL_PATH" \
      --output-dir /tmp/robocasa-native-eval

如果环境子进程启动失败，可以设置 ``ROBOCASA_WORKER_LOG_DIR``，为每个环境进程
保存独立诊断文件。子进程 traceback 也会回传给父 worker，避免只看到无信息的管道异常。

低开销 Reset 随机化
--------------------

启用 ``reset_optimization_enabled: true`` 后，RoboCasa 会复用已经编译的
MuJoCo model 和渲染上下文。``reset_randomization`` 仍然可以在每次 reset 时
改变抽屉状态和物品位置，也可以轮换当前模型中已经加载的纹理及物品视觉 mesh：

.. code-block:: yaml

   reset_randomization:
     enabled: true
     drawer_open_range: [0.65, 1.0]
     resample_object_placements: true
     randomize_preloaded_textures: true
     shuffle_object_visuals: true
     material_color_jitter: 0.15

该路径不会在 reset 时加载新资产或重新编译 XML。纹理随机化仅在当前场景已经
加载的纹理之间轮换；物品视觉轮换会改变各物品槽位显示的预加载 mesh，但保留
原始碰撞几何。因此它适用于成功条件不依赖物品交互的 ``CloseDrawer`` 等任务。
对于要求视觉和碰撞几何严格一致的操作任务，应关闭 ``shuffle_object_visuals``。

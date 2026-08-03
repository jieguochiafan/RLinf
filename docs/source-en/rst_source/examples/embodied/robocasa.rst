RL with RoboCasa Benchmark
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide for reinforcement learning training tasks using the RoboCasa environment in the RLinf framework.
RoboCasa Kitchen focuses on manipulation tasks in kitchen environments, featuring diverse kitchen layouts, objects, and manipulation tasks.
RoboCasa Kitchen combines realistic kitchen environments with diverse manipulation challenges, making it an ideal benchmark for developing generalizable robotic policies.

The main goal is to train vision-language-action models capable of performing the following tasks:

1. **Visual Understanding**: Process RGB images from multiple camera viewpoints.
2. **Language Understanding**: Interpret natural language task instructions.
3. **Manipulation Skills**: Execute complex kitchen tasks such as pick-and-place, opening/closing doors, and appliance control.

Environment
-----------

**RoboCasa Simulation Platform**

- **Environment**: RoboCasa Kitchen simulation environment (built on robosuite)
- **Robot**: Panda manipulator with mobile base (PandaOmron), equipped with gripper
- **Observation**: Multi-view RGB images (robot view + wrist camera) + proprioceptive state
- **Action Space**: 12-dimensional continuous actions

  - 3D arm position delta
  - 3D arm rotation delta
  - 1D gripper control (open/close)
  - 4D base control
  - 1D mode selection (control base or arm)

**Task Categories**

RoboCasa Kitchen provides 24 atomic tasks covering multiple categories (excluding NavigateKitchen atomic task that requires base movement):

*Door Manipulation Tasks*:

- ``OpenSingleDoor``: Open cabinet or microwave door
- ``CloseSingleDoor``: Close cabinet or microwave door
- ``OpenDoubleDoor``: Open double cabinet doors
- ``CloseDoubleDoor``: Close double cabinet doors
- ``OpenDrawer``: Open drawer
- ``CloseDrawer``: Close drawer

*Pick and Place Tasks*:

- ``PnPCounterToCab``: Pick from counter and place into cabinet
- ``PnPCabToCounter``: Pick from cabinet and place on counter
- ``PnPCounterToSink``: Pick from counter and place in sink
- ``PnPSinkToCounter``: Pick from sink and place on counter
- ``PnPCounterToStove``: Pick from counter and place on stove
- ``PnPStoveToCounter``: Pick from stove and place on counter
- ``PnPCounterToMicrowave``: Pick from counter and place in microwave
- ``PnPMicrowaveToCounter``: Pick from microwave and place on counter

*Appliance Control Tasks*:

- ``TurnOnMicrowave``: Turn on microwave
- ``TurnOffMicrowave``: Turn off microwave
- ``TurnOnSinkFaucet``: Turn on sink faucet
- ``TurnOffSinkFaucet``: Turn off sink faucet
- ``TurnSinkSpout``: Turn sink spout
- ``TurnOnStove``: Turn on stove
- ``TurnOffStove``: Turn off stove

*Coffee Making Tasks*:

- ``CoffeeSetupMug``: Setup coffee mug
- ``CoffeeServeMug``: Serve coffee into mug
- ``CoffeePressButton``: Press coffee machine button

**Observation Structure**

- **Base Camera Image** (``base_image``): Robot left view (224×224 RGB)
- **Wrist Camera Image** (``wrist_image``): End-effector view camera (224×224 RGB)
- **Wrist Camera Image** (``extra_view_image``): Robot right view (224×224 RGB, not included by default.)
- **Proprioceptive State** (``state``): 25-dimensional vector containing:
  - ``[0:3]`` End-effector position (x, y, z)
  - ``[3:7]`` End-effector quaternion (w, x, y, z)
  - ``[7:9]`` Gripper joint position
  - ``[9:11]`` Gripper joint velocities
  - ``[11:14]`` End-effector position relative to base (x, y, z)
  - ``[14:18]`` End-effector quaternion relative to base (w, x, y, z)
  - ``[18:21]`` Base position (x, y, z)
  - ``[21:25]`` Base quaternion (w, x, y, z)

**Data Structure**

- **Images**: Left camera RGB tensor ``[batch_size, 3, 224, 224]`` and wrist camera ``[batch_size, 3, 224, 224]``. Right camera RGB tensor ``[batch_size, 3, 224, 224]`` can also be included.
- **State**: Proprioceptive state tensor ``[batch_size, 25]``. 
- **Task Description**: Natural language instructions
- **Actions**: 12-dimensional continuous actions
- **Reward**: Sparse reward based on task completion

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group's mean reward.

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-robocasa
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-robocasa

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env robocasa
   source .venv/bin/activate

Dataset Download
-----------------

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 5GB

Model Download
--------------

.. code-block:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-RoboCasa

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-RoboCasa --local-dir RLinf-Pi0-RoboCasa

Pi0.5 and RoboCasa365 Compatibility
-----------------------------------

RLinf supports both the original 25-dimensional RoboCasa state and the
canonical 16-dimensional state used by Pi0.5 RoboCasa checkpoints. The
``state_space`` field controls which representation the environment emits.
The Pi0.5 example selects ``16d`` without changing the existing Pi0 example.

Set the converted checkpoint path and launch the Pi0.5 configuration with:

.. code-block:: bash

   export EMBODIED_PATH=$PWD/examples/embodiment
   export ROBOCASA_MODEL_PATH=/path/to/model/pi05_robocasa
   bash examples/embodiment/run_embodiment.sh \
      robocasa_closedrawer_ppo_openpi_pi05

OpenPI data configuration ``pi05_robocasa_human`` reads the public RoboCasa
LeRobot schema, while ``pi05_robocasa365_closedrawer`` reads the unified
RoboCasa365 schema. Compact 16-dimensional state arrays are accepted directly;
25-dimensional arrays are selected and reordered automatically.

For RoboCasa365 supervised fine-tuning, set ``ROBOCASA_SFT_DATA`` and use
``examples/sft/config/robocasa_sft_openpi_pi05.yaml`` with
``examples/sft/train_vla_sft.py``.

An independent evaluator is available for checking the policy against
RoboCasa's official environment factory, without the RLinf environment runner:

.. code-block:: bash

   python toolkits/standalone_eval_scripts/robocasa/native_openpi_eval.py \
      --model-path "$ROBOCASA_MODEL_PATH" \
      --output-dir /tmp/robocasa-native-eval

For subprocess startup failures, set ``ROBOCASA_WORKER_LOG_DIR`` to write one
diagnostic file per environment process. Child tracebacks are also propagated
to the parent worker instead of appearing as a silent pipe failure.

Low-cost Reset Randomization
----------------------------

With ``reset_optimization_enabled: true``, RoboCasa keeps the compiled MuJoCo
model and render context. ``reset_randomization`` can still vary the drawer
state and object placements on every reset. It can also permute textures and
object visual meshes that are already present in the compiled model:

.. code-block:: yaml

   reset_randomization:
     enabled: true
     drawer_open_range: [0.65, 1.0]
     resample_object_placements: true
     randomize_preloaded_textures: true
     shuffle_object_visuals: true
     material_color_jitter: 0.15

This path never loads a new asset or recompiles XML during reset. Texture
permutation is limited to textures already loaded by the current scene. Object
visual shuffling changes which preloaded mesh is shown at each object slot but
keeps the original collision geometry, so it is intended for tasks such as
``CloseDrawer`` whose success condition does not depend on object interaction.
Disable ``shuffle_object_visuals`` for manipulation tasks where exact visual and
collision geometry alignment is required.

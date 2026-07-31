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

"""Model-type registry for the standalone rollout system.

Vendored from the training repo's ``rlinf/config.py``, keeping only the pieces the
rollout system needs: the open ``SupportedModel`` registry, the set of embodied
model types, and precision parsing. None of the training-side config builders
(Megatron/FSDP/optimizer/algorithm validation) are carried over.
"""

import dataclasses
from typing import ClassVar, Optional, Union

import torch

__all__ = [
    "EMBODIED_MODEL",
    "SupportedModel",
    "torch_dtype_from_precision",
]


@dataclasses.dataclass(frozen=True)
class SupportedModel:
    """An open enum of model types.

    New model types are added through :meth:`register` (also called by
    ``rlinf_rollout.models.register_model``), so out-of-tree policies can join
    without patching this module.
    """

    value: str

    models: ClassVar[dict[str, "SupportedModel"]] = {}

    @classmethod
    def register(cls, value: str, force: bool = False) -> "SupportedModel":
        """Register ``value`` as a known model type and return its singleton."""
        if not value:
            raise ValueError("model_type must be a non-empty string.")
        if value in cls.models:
            if not force:
                raise ValueError(
                    f"Model type `{value}` is already registered. "
                    "Set force=True to override it."
                )
        else:
            cls.models[value] = cls.__private_create__(value)
        return cls.models[value]

    @classmethod
    def get(cls, value: str) -> "SupportedModel":
        """Look up a registered model type, raising if it is unknown."""
        if value not in cls.models:
            supported_models = sorted(cls.models)
            raise NotImplementedError(
                f"Model Type: {value} not supported. Supported models: {supported_models}"
            )
        return cls.models[value]

    def __new__(cls, value: str):
        return cls.get(value)

    @classmethod
    def __private_create__(cls, value: str) -> "SupportedModel":
        obj = object.__new__(cls)
        object.__setattr__(obj, "value", value)
        return obj


SupportedModel.OPENVLA = SupportedModel.register("openvla", force=True)
SupportedModel.OPENVLA_OFT = SupportedModel.register("openvla_oft", force=True)
SupportedModel.OPENPI = SupportedModel.register("openpi", force=True)
SupportedModel.OPENPI_PYTORCH = SupportedModel.register("openpi_pytorch", force=True)
SupportedModel.STARVLA = SupportedModel.register("starvla", force=True)
SupportedModel.MLP_POLICY = SupportedModel.register("mlp_policy", force=True)
SupportedModel.RLT_MLP_POLICY = SupportedModel.register("rlt_mlp_policy", force=True)
SupportedModel.GR00T = SupportedModel.register("gr00t", force=True)
SupportedModel.GR00T_N1D6 = SupportedModel.register("gr00t_n1d6", force=True)
SupportedModel.GR00T_N1D7 = SupportedModel.register("gr00t_n1d7", force=True)
SupportedModel.DEXBOTIC_PI = SupportedModel.register("dexbotic_pi", force=True)
SupportedModel.DEXBOTIC_DM0 = SupportedModel.register("dexbotic_dm0", force=True)
SupportedModel.DREAMZERO = SupportedModel.register("dreamzero", force=True)
SupportedModel.CNN_POLICY = SupportedModel.register("cnn_policy", force=True)
SupportedModel.FLOW_POLICY = SupportedModel.register("flow_policy", force=True)
SupportedModel.CMA_POLICY = SupportedModel.register("cma", force=True)
SupportedModel.LINGBOTVLA = SupportedModel.register("lingbotvla", force=True)
SupportedModel.ABOT_M0 = SupportedModel.register("abot_m0", force=True)
SupportedModel.RESNET_REWARD = SupportedModel.register("resnet", force=True)
SupportedModel.CFG_MODEL = SupportedModel.register("cfg_model", force=True)
SupportedModel.RECAP_VALUE_MODEL = SupportedModel.register(
    "recap_value_model", force=True
)
SupportedModel.STEAM_VALUE_MODEL = SupportedModel.register(
    "steam_value_model", force=True
)

EMBODIED_MODEL: set[SupportedModel] = {
    SupportedModel.OPENVLA,
    SupportedModel.OPENVLA_OFT,
    SupportedModel.OPENPI,
    SupportedModel.OPENPI_PYTORCH,
    SupportedModel.STARVLA,
    SupportedModel.MLP_POLICY,
    SupportedModel.RLT_MLP_POLICY,
    SupportedModel.GR00T,
    SupportedModel.GR00T_N1D6,
    SupportedModel.GR00T_N1D7,
    SupportedModel.DEXBOTIC_PI,
    SupportedModel.DEXBOTIC_DM0,
    SupportedModel.DREAMZERO,
    SupportedModel.CNN_POLICY,
    SupportedModel.FLOW_POLICY,
    SupportedModel.CMA_POLICY,
    SupportedModel.LINGBOTVLA,
    SupportedModel.ABOT_M0,
    SupportedModel.RESNET_REWARD,
    SupportedModel.CFG_MODEL,
    SupportedModel.RECAP_VALUE_MODEL,
    SupportedModel.STEAM_VALUE_MODEL,
}


def torch_dtype_from_precision(
    precision: Union[int, str, None],
) -> Optional[torch.dtype]:
    """Map a config ``precision`` value onto a ``torch.dtype``."""
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None, "null"]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )

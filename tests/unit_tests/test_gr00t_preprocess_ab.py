# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

from toolkits.rollout_eval.profiling.gr00t_preprocess_ab import (
    assert_nested_equal,
    benchmark,
    parse_args,
    summarize_ms,
)


def test_summarize_ms_interpolates_percentiles() -> None:
    assert summarize_ms([1.0, 2.0, 3.0, 4.0]) == {
        "avg_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
    }


def test_assert_nested_equal_accepts_exact_outputs() -> None:
    output = {
        "tensor": torch.tensor([[1, 2]]),
        "array": np.array([3.0], dtype=np.float32),
    }
    copied = {"tensor": output["tensor"].clone(), "array": output["array"].copy()}
    assert_nested_equal(output, copied)


def test_assert_nested_equal_reports_tensor_difference() -> None:
    with pytest.raises(AssertionError, match="tensor elements differ"):
        assert_nested_equal(torch.tensor([1]), torch.tensor([2]))


def test_benchmark_checks_outputs_and_reports_speedup() -> None:
    result = benchmark(
        lambda: {"value": torch.tensor([1])},
        lambda: {"value": torch.tensor([1])},
        warmup=0,
        iterations=2,
    )
    assert result["outputs_exactly_equal"] is True
    assert result["speedup"] > 0


def test_parse_args_uses_rollout_batch_size() -> None:
    args = parse_args([])
    assert args.config_name == "libero_object_ppo_gr00t"
    assert args.batch_size == 12
    assert args.image_input_mode == "pil"

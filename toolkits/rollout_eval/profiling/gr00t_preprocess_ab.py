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

"""A/B benchmark for a faster GR00T N1.5 rollout preprocessor.

The optimized path is deliberately installed only on the model instance used by
this benchmark.  It avoids repeated chat-template rendering and the generic
``process_vision_info`` traversal, while preserving the final batched Eagle
processor call.  No production code is changed by this module.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(values: list[float]) -> dict[str, float]:
    """Summarize non-empty latency samples in milliseconds."""
    if not values:
        raise ValueError("latency samples must not be empty")
    return {
        "avg_ms": float(mean(values)),
        "p50_ms": float(_percentile(values, 0.50)),
        "p95_ms": float(_percentile(values, 0.95)),
    }


def assert_nested_equal(baseline: Any, optimized: Any, path: str = "output") -> None:
    """Assert exact equality for nested GR00T preprocessing outputs."""
    import torch

    if isinstance(baseline, dict):
        if not isinstance(optimized, dict):
            raise AssertionError(f"{path}: type differs")
        if baseline.keys() != optimized.keys():
            raise AssertionError(
                f"{path}: keys differ: {baseline.keys()} != {optimized.keys()}"
            )
        for key in baseline:
            assert_nested_equal(baseline[key], optimized[key], f"{path}.{key}")
        return
    if torch.is_tensor(baseline):
        if not torch.is_tensor(optimized):
            raise AssertionError(f"{path}: tensor type differs")
        if baseline.dtype != optimized.dtype or baseline.shape != optimized.shape:
            raise AssertionError(
                f"{path}: tensor metadata differs: "
                f"{baseline.dtype}/{tuple(baseline.shape)} != "
                f"{optimized.dtype}/{tuple(optimized.shape)}"
            )
        if not torch.equal(baseline.cpu(), optimized.cpu()):
            difference = (baseline.cpu() != optimized.cpu()).sum().item()
            raise AssertionError(f"{path}: {difference} tensor elements differ")
        return
    if isinstance(baseline, np.ndarray):
        if not isinstance(optimized, np.ndarray) or not np.array_equal(
            baseline, optimized
        ):
            raise AssertionError(f"{path}: numpy values differ")
        return
    if baseline != optimized:
        raise AssertionError(f"{path}: {baseline!r} != {optimized!r}")


def make_optimized_apply_batch(
    transform: Any,
    *,
    pil_executor: Executor | None = None,
    image_input_mode: str = "pil",
) -> Callable[[dict, int], dict]:
    """Build an inference-only replacement for ``GR00TTransform.apply_batch``."""
    import torch
    from PIL import Image

    prompt_cache: dict[tuple[str, int], str] = {}
    token_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def process_tensor_batch(
        text_list: list[str], image_inputs: list[Any], images_per_sample: int
    ) -> dict[str, Any]:
        processor = transform.eagle_processor
        processor_kwargs_type = processor.__call__.__globals__[
            "Eagle2_5_VLProcessorKwargs"
        ]
        output_kwargs = processor._merge_kwargs(
            processor_kwargs_type,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            return_tensors="pt",
            padding=True,
        )
        image_outputs = processor.image_processor(
            images=image_inputs,
            videos=None,
            **output_kwargs["images_kwargs"],
        )

        image_kwargs = output_kwargs["images_kwargs"]
        min_tiles = image_kwargs.get(
            "min_dynamic_tiles", processor.image_processor.min_dynamic_tiles
        )
        max_tiles = image_kwargs.get(
            "max_dynamic_tiles", processor.image_processor.max_dynamic_tiles
        )
        use_thumbnail = image_kwargs.get(
            "use_thumbnail", processor.image_processor.use_thumbnail
        )
        tile_size = processor.image_processor.size.get("height", 448)
        tile_counts = [
            processor.get_number_tiles_based_on_image_size(
                (int(image.shape[-3]), int(image.shape[-2])),
                min_tiles,
                max_tiles,
                use_thumbnail,
                tile_size,
            )
            for image in image_inputs
        ]

        expanded_text = []
        placeholder_pattern = re.compile(rf"<{processor.image_placeholder}-(\d+)>")
        for sample_index, text in enumerate(text_list):
            sample_offset = sample_index * images_per_sample

            def replace_placeholder(match: re.Match[str]) -> str:
                image_index = int(match.group(1)) - 1
                tile_count = tile_counts[sample_offset + image_index]
                image_tokens = (
                    processor.image_token * tile_count * processor.tokens_per_tile
                )
                return (
                    f"<image {image_index + 1}>{processor.image_start_token}"
                    f"{image_tokens}{processor.image_end_token}"
                )

            expanded_text.append(placeholder_pattern.sub(replace_placeholder, text))

        token_key = tuple(expanded_text)
        if token_key not in token_cache:
            token_cache[token_key] = dict(
                processor.tokenizer(
                    expanded_text,
                    **output_kwargs["text_kwargs"],
                )
            )
        text_outputs = token_cache[token_key]
        return {**text_outputs, **image_outputs}

    def render_prompt(language: str, images: list[Any]) -> str:
        cache_key = (language, len(images))
        if cache_key not in prompt_cache:
            content = [{"type": "image", "image": image} for image in images]
            content = content + [{"type": "text", "text": language}]
            # Pixel values do not participate in rendering the chat template.
            prompt_cache[cache_key] = transform.eagle_processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_cache[cache_key]

    def optimized_apply_batch(data: dict, batch_size: int) -> dict:
        if transform.training:
            raise RuntimeError(
                "The optimized A/B preprocessor is inference-only; call eval() first"
            )

        videos = data["video"]
        states = data.get("state")
        language_values = (
            data[transform._language_key]
            if transform._language_key is not None
            else [transform.default_instruction] * batch_size
        )
        images_per_sample = int(videos.shape[1] * videos.shape[2])
        if torch.is_tensor(videos):
            flat_images = videos.reshape(-1, *videos.shape[-3:]).to(torch.uint8)
        else:
            flat_images = videos.reshape(-1, *videos.shape[-3:]).astype(
                np.uint8, copy=False
            )

        def to_rgb_pil(image: np.ndarray) -> Any:
            return Image.fromarray(image).convert("RGB")

        if image_input_mode == "numpy":
            image_inputs = list(flat_images)
        elif image_input_mode in ("tensor", "tensor_batched"):
            image_tensor = (
                flat_images
                if torch.is_tensor(flat_images)
                else torch.from_numpy(flat_images)
            )
            image_inputs = list(image_tensor)
        elif pil_executor is None:
            image_inputs = [to_rgb_pil(image) for image in flat_images]
        else:
            image_inputs = list(pil_executor.map(to_rgb_pil, flat_images))

        text_list = []
        for index in range(batch_size):
            start = index * images_per_sample
            pil_images = image_inputs[start : start + images_per_sample]
            language = language_values[index]
            if isinstance(language, (list, np.ndarray)):
                language = language[0]
            text_list.append(render_prompt(str(language), pil_images))

        if image_input_mode == "tensor_batched":
            eagle_inputs = process_tensor_batch(
                text_list,
                image_inputs,
                images_per_sample,
            )
        else:
            eagle_inputs = transform.eagle_processor(
                text=text_list,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
            )
        output = {f"eagle_{key}": value for key, value in eagle_inputs.items()}

        if states is None:
            state = np.zeros(
                (batch_size, transform.state_horizon, transform.max_state_dim)
            )
            state_mask = np.zeros_like(state, dtype=bool)
        else:
            if torch.is_tensor(states):
                states = states.detach().cpu().numpy()
            state_dims = min(states.shape[-1], transform.max_state_dim)
            state = np.zeros(
                (batch_size, transform.state_horizon, transform.max_state_dim),
                dtype=states.dtype,
            )
            state[..., :state_dims] = states[..., :state_dims]
            state_mask = np.zeros_like(state, dtype=bool)
            state_mask[..., :state_dims] = True
        output["state"] = torch.from_numpy(state)
        output["state_mask"] = torch.from_numpy(state_mask)
        output["embodiment_id"] = torch.full(
            (batch_size,), transform.get_embodiment_tag(), dtype=torch.int64
        )
        return output

    return optimized_apply_batch


def benchmark(
    baseline: Callable[[], Any],
    optimized: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Verify both paths and measure their CPU wall latency."""
    baseline_output = baseline()
    optimized_output = optimized()
    assert_nested_equal(baseline_output, optimized_output)

    for _ in range(warmup):
        baseline()
        optimized()

    baseline_ms = []
    optimized_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        baseline()
        baseline_ms.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        optimized()
        optimized_ms.append((time.perf_counter() - start) * 1000.0)

    baseline_summary = summarize_ms(baseline_ms)
    optimized_summary = summarize_ms(optimized_ms)
    return {
        "outputs_exactly_equal": True,
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "speedup": baseline_summary["avg_ms"] / optimized_summary["avg_ms"],
    }


def benchmark_h2d(
    model: Any, normalized_input: dict[str, Any], iterations: int
) -> dict:
    """Compare current duplicated H2D with shared and pinned single-copy paths."""
    import torch

    cpu_tensors = {
        key: value for key, value in normalized_input.items() if torch.is_tensor(value)
    }

    def cast_and_copy(value: torch.Tensor, *, non_blocking: bool) -> torch.Tensor:
        dtype = model.action_head.dtype if torch.is_floating_point(value) else None
        return value.to(model.device, dtype=dtype, non_blocking=non_blocking)

    baseline_backbone, baseline_action = model.prepare_input(normalized_input)
    torch.cuda.synchronize()
    shared = {
        key: cast_and_copy(value, non_blocking=False)
        for key, value in cpu_tensors.items()
    }
    torch.cuda.synchronize()
    for key, value in shared.items():
        if not torch.equal(value, baseline_backbone[key]):
            raise AssertionError(f"H2D backbone value differs for {key}")
        if not torch.equal(value, baseline_action[key]):
            raise AssertionError(f"H2D action-head value differs for {key}")

    pinned = {key: value.pin_memory() for key, value in cpu_tensors.items()}
    copy_stream = torch.cuda.Stream()

    def measure(operation: Callable[[], Any]) -> dict[str, float]:
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
        return summarize_ms(samples)

    def shared_pageable_copy() -> dict[str, torch.Tensor]:
        return {
            key: cast_and_copy(value, non_blocking=False)
            for key, value in cpu_tensors.items()
        }

    def shared_pinned_copy() -> dict[str, torch.Tensor]:
        with torch.cuda.stream(copy_stream):
            copied = {
                key: cast_and_copy(value, non_blocking=True)
                for key, value in pinned.items()
            }
        torch.cuda.current_stream().wait_stream(copy_stream)
        return copied

    return {
        "outputs_exactly_equal": True,
        "current_prepare_input": measure(lambda: model.prepare_input(normalized_input)),
        "shared_pageable_copy": measure(shared_pageable_copy),
        "shared_pinned_async_copy": measure(shared_pinned_copy),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="examples/embodiment/config")
    parser.add_argument("--config-name", default="libero_object_ppo_gr00t")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument(
        "--task-description",
        default="pick up the object and place it in the target container",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--pil-workers",
        type=int,
        default=4,
        help="Persistent worker threads for NumPy-to-PIL conversion; 1 disables it",
    )
    parser.add_argument(
        "--image-input-mode",
        choices=("pil", "numpy", "tensor", "tensor_batched"),
        default="pil",
        help="Input representation passed to the Eagle image processor",
    )
    parser.add_argument(
        "--pure-tensor-transforms",
        action="store_true",
        help="Keep LIBERO crop/resize and state preparation in torch",
    )
    parser.add_argument(
        "--benchmark-h2d",
        action="store_true",
        help="Also compare current, shared, and pinned H2D transfer paths",
    )
    parser.add_argument("--output", default="/tmp/gr00t-preprocess-ab.json")
    args = parser.parse_args(argv)
    if (
        args.batch_size <= 0
        or args.iterations <= 0
        or args.warmup < 0
        or args.pil_workers <= 0
    ):
        parser.error("batch size/iterations must be positive and warmup non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    from toolkits.rollout_eval.profiling.gr00t_generation_profile import (
        _build_model,
        _load_cfg,
        _make_env_obs,
    )

    args = parse_args(argv)
    cfg = _load_cfg(args)
    model = _build_model(cfg)
    env_obs = _make_env_obs(args)

    env_obs["states"] = env_obs["states"].to("cpu").float()
    observations = model.obs_convert_fn(env_obs)
    obs_copy = {
        key: value if isinstance(value, np.ndarray) else np.array(value)
        for key, value in observations.items()
    }

    final_transform = model._modality_transform.transforms[-1]
    original_apply_batch = final_transform.apply_batch
    preceding_transforms = model._modality_transform.transforms[:-1]

    def run_with(apply_batch: Callable[[dict, int], dict]) -> Any:
        transformed = {key: value.copy() for key, value in obs_copy.items()}
        for transform in preceding_transforms:
            transformed = transform(transformed)
        is_batched, batch_size = final_transform.check_keys_and_batch_size(transformed)
        if not is_batched:
            raise AssertionError("The A/B benchmark requires batched observations")
        return apply_batch(transformed, batch_size)

    def run_pure_tensor(apply_batch: Callable[[dict, int], dict]) -> Any:
        import torch

        crop = next(
            item
            for item in preceding_transforms
            if item.__class__.__name__ == "VideoCrop"
        )
        resize = next(
            item
            for item in preceding_transforms
            if item.__class__.__name__ == "VideoResize"
        )
        state_transform = next(
            item
            for item in preceding_transforms
            if item.__class__.__name__ == "StateActionTransform"
            and all(key.startswith("state.") for key in item.apply_to)
        )
        concat = next(
            item
            for item in preceding_transforms
            if item.__class__.__name__ == "ConcatTransform"
        )

        batch_size = int(env_obs["main_images"].shape[0])
        views = torch.stack((env_obs["main_images"], env_obs["wrist_images"]), dim=0)
        # [V, B, H, W, C] -> [V*B, C, H, W], matching VideoTransform.
        views = views.reshape(-1, *views.shape[-3:]).permute(0, 3, 1, 2)
        views = views.to(torch.float32).div(255.0)
        views = crop.eval_transform(views)
        views = resize.eval_transform(views)
        views = views.permute(0, 2, 3, 1).mul(255).to(torch.uint8)
        views = views.reshape(2, batch_size, 1, *views.shape[-3:]).permute(
            1, 2, 0, 3, 4, 5
        )

        states = env_obs["states"].unsqueeze(1)
        state_data = {
            "state.x": states[..., 0:1],
            "state.y": states[..., 1:2],
            "state.z": states[..., 2:3],
            "state.roll": states[..., 3:4],
            "state.pitch": states[..., 4:5],
            "state.yaw": states[..., 5:6],
            "state.gripper": states[..., 6:],
        }
        state_data = state_transform(state_data)
        state = torch.cat(
            [state_data[key] for key in concat.state_concat_order], dim=-1
        )
        transformed = {
            "video": views,
            "state": state,
            "annotation.human.action.task_description": env_obs["task_descriptions"],
        }
        return apply_batch(transformed, batch_size)

    executor = (
        ThreadPoolExecutor(max_workers=args.pil_workers)
        if args.pil_workers > 1
        else None
    )
    try:
        optimized_apply_batch = make_optimized_apply_batch(
            final_transform,
            pil_executor=executor,
            image_input_mode=args.image_input_mode,
        )
        result = benchmark(
            lambda: run_with(original_apply_batch),
            lambda: (
                run_pure_tensor(optimized_apply_batch)
                if args.pure_tensor_transforms
                else run_with(optimized_apply_batch)
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
    finally:
        if executor is not None:
            executor.shutdown()
    result["pil_workers"] = args.pil_workers
    result["image_input_mode"] = args.image_input_mode
    result["pure_tensor_transforms"] = args.pure_tensor_transforms
    if args.benchmark_h2d:
        optimized_output = (
            run_pure_tensor(optimized_apply_batch)
            if args.pure_tensor_transforms
            else run_with(optimized_apply_batch)
        )
        result["h2d"] = benchmark_h2d(model, optimized_output, args.iterations)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

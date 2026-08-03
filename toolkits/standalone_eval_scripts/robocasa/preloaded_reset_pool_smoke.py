#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke-test soft RoboCasa resets with preloaded objects and textures.

The test creates exactly one CloseDrawer environment. All candidate objects,
including their collision meshes, and all candidate textures are compiled into
one MuJoCo model during initialization. Subsequent resets only move one object
into the drawer, park the others outside the scene, and switch material texture
IDs. No model rebuild or texture upload is performed between captured frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

POOL_OBJECTS = (
    ("apple", "objaverse/apple/apple_14/model.xml", 1.0),
    ("corn", "objaverse/corn/corn_0/model.xml", 0.9),
    ("mug", "objaverse/mug/mug_2/model.xml", 0.9),
    (
        "bottled_water",
        "objaverse/bottled_water/bottled_water_14/model.xml",
        0.65,
    ),
)
TEXTURE_GEOM_PATTERNS = ("wall_", "floor_", "counter_", "cab_", "drawer_")


def parse_args() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-resets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--texture-size", type=int, default=512)
    return parser.parse_args()


def make_texture_pool(output_dir: Path, size: int) -> list[Path]:
    """Create visibly distinct procedural textures for the preload pool."""
    texture_dir = output_dir / "preloaded_textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    palettes = (
        ((185, 70, 55), (244, 197, 113)),
        ((42, 110, 135), (170, 220, 214)),
        ((77, 126, 75), (215, 224, 150)),
        ((106, 71, 145), (226, 174, 219)),
    )
    yy, xx = np.mgrid[:size, :size]
    texture_paths = []
    for index, (dark, light) in enumerate(palettes):
        if index == 0:
            mask = ((xx // 48 + yy // 48) % 2).astype(np.float32)
        elif index == 1:
            mask = ((xx + yy) % 96 < 48).astype(np.float32)
        elif index == 2:
            mask = ((xx // 24) % 3 == 0).astype(np.float32)
        else:
            radius = np.sqrt((xx % 96 - 48) ** 2 + (yy % 96 - 48) ** 2)
            mask = (radius < 25).astype(np.float32)
        image = (
            np.asarray(dark)[None, None, :] * mask[..., None]
            + np.asarray(light)[None, None, :] * (1.0 - mask[..., None])
        ).astype(np.uint8)
        texture_path = texture_dir / f"texture_{index:02d}.png"
        Image.fromarray(image).save(texture_path)
        texture_paths.append(texture_path.resolve())
    return texture_paths


def _model_identity(env: Any) -> int:
    """Return the identity of the underlying compiled MuJoCo model."""
    model = env.sim.model
    return id(getattr(model, "_model", model))


def _look_at_xyaxes(camera_pos: np.ndarray, target: np.ndarray) -> str:
    """Return MuJoCo camera xyaxes oriented from camera_pos toward target."""
    backward = camera_pos - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    values = np.concatenate([right, up])
    return " ".join(f"{value:.9g}" for value in values)


def _write_contact_sheet(frames: list[Path], labels: list[str], output: Path) -> None:
    """Write a labelled contact sheet for quick visual comparison."""
    images = [Image.open(frame).convert("RGB") for frame in frames]
    columns = min(2, len(images))
    rows = math.ceil(len(images) / columns)
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images) + 28
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y + 28))
        draw.text((x + 6, y + 7), label, fill="black")
    sheet.save(output)
    for image in images:
        image.close()


def _build_env_class(*, include_debug_camera: bool = True):
    """Build the subclass after RoboCasa is available in the selected venv."""
    from lxml import etree as et
    from robocasa.environments.kitchen.kitchen import FixtureType
    from robocasa.environments.kitchen.single_stage.kitchen_drawer import CloseDrawer
    from robocasa.models import assets_root

    class PreloadedPoolCloseDrawer(CloseDrawer):
        """CloseDrawer with one compiled pool of physical objects and textures."""

        def __init__(
            self,
            *args: Any,
            texture_paths: list[Path],
            object_specs: tuple[tuple[str, str, float], ...],
            **kwargs: Any,
        ) -> None:
            self._pool_texture_paths = texture_paths
            self._pool_object_specs = object_specs
            self._pool_reset_index = -1
            self._pool_drawer_pose = None
            self._pool_material_ids = None
            self._pool_texture_target_geoms = None
            self.active_pool_index = 0
            super().__init__(*args, **kwargs)

        def _get_obj_cfgs(self) -> list[dict[str, Any]]:
            drawer_placement = {
                "fixture": self.drawer,
                "size": (0.30, 0.30),
                "pos": (None, -0.75),
                "offset": (0, -self.drawer.size[1] * 0.55),
            }
            counter_placement = {
                "fixture": self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                "sample_region_kwargs": {"ref": self.drawer},
                "size": (1.0, 0.50),
                "pos": (None, -1.0),
                "offset": (0.0, 0.10),
            }
            configs = []
            for index, (_, relative_path, scale) in enumerate(
                self._pool_object_specs
            ):
                configs.append(
                    {
                        "name": f"pool_obj_{index}",
                        "obj_groups": os.path.join(
                            assets_root,
                            "objects",
                            relative_path,
                        ),
                        "graspable": True,
                        "object_scale": scale,
                        "placement": (
                            drawer_placement if index == 0 else counter_placement
                        ),
                    }
                )
            return configs

        def edit_model_xml(self, xml_str: str) -> str:
            xml_str = super().edit_model_xml(xml_str)
            root = et.fromstring(xml_str)
            asset = root.find("asset")
            if asset is None:
                raise RuntimeError("RoboCasa XML has no <asset> element")
            for index, texture_path in enumerate(self._pool_texture_paths):
                et.SubElement(
                    asset,
                    "texture",
                    name=f"rlinf_pool_texture_{index}",
                    type="2d",
                    file=os.fspath(texture_path),
                )
                et.SubElement(
                    asset,
                    "material",
                    name=f"rlinf_pool_material_{index}",
                    texture=f"rlinf_pool_texture_{index}",
                    texrepeat="3 3",
                    texuniform="true",
                    specular="0.1",
                    shininess="0.1",
                )

            if include_debug_camera:
                drawer_pos = np.asarray(self.object_placements["pool_obj_0"][0])
                camera_pos = drawer_pos + np.array([-0.08, -0.72, 0.32])
                worldbody = root.find("worldbody")
                if worldbody is None:
                    raise RuntimeError("RoboCasa XML has no <worldbody> element")
                et.SubElement(
                    worldbody,
                    "camera",
                    name="rlinf_pool_closeup",
                    mode="fixed",
                    pos=" ".join(f"{value:.9g}" for value in camera_pos),
                    xyaxes=_look_at_xyaxes(camera_pos, drawer_pos),
                    fovy="42",
                )
            return et.tostring(root, encoding="unicode")

        def _capture_drawer_pose(self) -> None:
            if self._pool_drawer_pose is not None:
                return
            placement = self.object_placements["pool_obj_0"]
            self._pool_drawer_pose = (
                np.array(placement[0], copy=True),
                np.array(placement[1], copy=True),
            )

        def _select_physical_object(self, index: int) -> None:
            self._capture_drawer_pose()
            drawer_pos, drawer_quat = self._pool_drawer_pose
            for candidate_index in range(len(self._pool_object_specs)):
                name = f"pool_obj_{candidate_index}"
                _, _, obj = self.object_placements[name]
                if candidate_index == index:
                    pos = np.array(drawer_pos, copy=True)
                    quat = np.array(drawer_quat, copy=True)
                else:
                    pos = np.array([candidate_index * 0.5, 0.0, -5.0])
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                self.object_placements[name] = (pos, quat, obj)

        def _initialize_texture_switch(self) -> None:
            if self._pool_material_ids is not None:
                return
            import mujoco

            model = self.sim.model
            raw_model = getattr(model, "_model", model)
            material_ids = []
            for index in range(len(self._pool_texture_paths)):
                material_id = mujoco.mj_name2id(
                    raw_model,
                    mujoco.mjtObj.mjOBJ_MATERIAL,
                    f"rlinf_pool_material_{index}",
                )
                if material_id < 0:
                    raise RuntimeError(f"Preloaded material {index} was not compiled")
                material_ids.append(material_id)

            target_geom_ids = []
            for geom_id, geom_name in enumerate(model.geom_names):
                if any(
                    pattern in geom_name.lower() for pattern in TEXTURE_GEOM_PATTERNS
                ) and int(model.geom_matid[geom_id]) >= 0:
                    target_geom_ids.append(geom_id)
            if not target_geom_ids:
                raise RuntimeError("No fixture texture geometry was found")
            self._pool_material_ids = material_ids
            self._pool_texture_target_geoms = np.asarray(
                target_geom_ids,
                dtype=np.int64,
            )

        def _select_texture(self, index: int) -> None:
            self._initialize_texture_switch()
            material_id = self._pool_material_ids[index]
            self.sim.model.geom_matid[self._pool_texture_target_geoms] = material_id
            self.sim.forward()

        def _reset_internal(self) -> None:
            self._pool_reset_index += 1
            self.active_pool_index = self._pool_reset_index % len(
                self._pool_object_specs
            )
            self._select_physical_object(self.active_pool_index)
            super()._reset_internal()
            self._select_texture(self.active_pool_index % len(self._pool_texture_paths))

        def active_object_metadata(self) -> dict[str, Any]:
            """Describe and locate the currently active physical object."""
            index = self.active_pool_index
            name = f"pool_obj_{index}"
            obj = self.objects[name]
            cfg = next(cfg for cfg in self.object_cfgs if cfg["name"] == name)
            qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]))
            return {
                "pool_index": index,
                "requested_category": self._pool_object_specs[index][0],
                "sampled_mjcf_path": cfg["info"]["mjcf_path"],
                "qpos": qpos.tolist(),
            }

    return PreloadedPoolCloseDrawer


def create_env(texture_paths: list[Path], seed: int, image_size: int):
    """Create the single environment and compile all pool assets once."""
    from robosuite.controllers import load_composite_controller_config

    env_class = _build_env_class()
    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )
    return env_class(
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=["robot0_agentview_left", "rlinf_pool_closeup"],
        camera_widths=image_size,
        camera_heights=image_size,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        hard_reset=False,
        translucent_robot=False,
        texture_paths=texture_paths,
        object_specs=POOL_OBJECTS,
    )


def main() -> None:
    """Run resets, save initial-state frames, and report timings."""
    args = parse_args()
    if args.num_resets < 1:
        raise ValueError("--num-resets must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texture_paths = make_texture_pool(args.output_dir, args.texture_size)

    initialize_start = time.perf_counter()
    env = create_env(texture_paths, args.seed, args.image_size)
    initialize_seconds = time.perf_counter() - initialize_start
    initial_model_identity = _model_identity(env)

    records = []
    frame_paths = []
    labels = []
    try:
        for reset_index in range(args.num_resets):
            reset_start = time.perf_counter()
            obs = env.reset()
            reset_seconds = time.perf_counter() - reset_start
            frame = np.ascontiguousarray(obs["rlinf_pool_closeup_image"][::-1])
            frame_path = args.output_dir / f"reset_{reset_index:02d}.png"
            Image.fromarray(frame).save(frame_path)
            agent_frame = np.ascontiguousarray(
                obs["robot0_agentview_left_image"][::-1]
            )
            agent_frame_path = (
                args.output_dir / f"reset_{reset_index:02d}_agentview.png"
            )
            Image.fromarray(agent_frame).save(agent_frame_path)

            object_metadata = env.active_object_metadata()
            same_model = _model_identity(env) == initial_model_identity
            record = {
                "reset_index": reset_index,
                "reset_seconds": reset_seconds,
                "same_compiled_model": same_model,
                "texture_index": object_metadata["pool_index"] % len(texture_paths),
                **object_metadata,
                "frame": frame_path.name,
                "agentview_frame": agent_frame_path.name,
            }
            records.append(record)
            frame_paths.append(frame_path)
            labels.append(
                f"reset {reset_index}: {record['requested_category']} / "
                f"texture {record['texture_index']} / {reset_seconds:.3f}s"
            )
            print(json.dumps(record, ensure_ascii=False))
    finally:
        env.close()

    metadata = {
        "seed": args.seed,
        "single_environment": True,
        "hard_reset": False,
        "initialization_seconds": initialize_seconds,
        "compiled_model_identity": initial_model_identity,
        "object_pool_size": len(POOL_OBJECTS),
        "texture_pool_size": len(texture_paths),
        "records": records,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_contact_sheet(
        frame_paths,
        labels,
        args.output_dir / "reset_contact_sheet.png",
    )
    print(f"Initialization: {initialize_seconds:.3f}s")
    print(f"Frames and metadata: {args.output_dir}")


if __name__ == "__main__":
    main()

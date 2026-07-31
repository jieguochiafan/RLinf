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

"""``rollout-serve``: the daemon entry point of the standalone rollout system.

Usage::

    rollout-serve --config configs/embodied_eval.yaml
    rollout-serve --config configs/llm_generate.yaml --set rollout.group_size=4
    rollout-serve --config configs/llm_generate.yaml --dry-run

``--dry-run`` validates the config and prints the launch plan (worker groups,
placement, endpoints, task source) without starting Ray, which makes it a cheap
pre-flight check in CI and on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from rlinf_rollout.config import RolloutConfigError
from rlinf_rollout.serve.service import RolloutService, load_service_config

__all__ = ["build_parser", "main", "run"]


def build_parser() -> argparse.ArgumentParser:
    """Build the ``rollout-serve`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rollout-serve",
        description=(
            "Serve a standalone RLinf rollout system (embodied env+policy or LLM "
            "engines). Trainers connect with rlinf_rollout.client.RolloutClient."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the service YAML config (see rlinf_rollout/serve/configs).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key, e.g. --set rollout.batch_size=8. Repeatable.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Service name, overriding rollout.serve.name (default: 'rollout').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the launch plan without starting Ray.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully resolved config before serving.",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        cfg = load_service_config(args.config, args.overrides)
        service = RolloutService(cfg, service_name=args.name)
    except (RolloutConfigError, NotImplementedError) as exc:
        print(f"rollout-serve: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg))

    if args.dry_run:
        print(json.dumps(service.plan().to_dict(), indent=2, default=str))
        return 0

    service.run()
    return 0


def main() -> None:
    """Console-script entry point; exits with :func:`run`'s code."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()

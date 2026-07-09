#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rlinf.utils.rollout_profile import summarize_rollout_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    args = parser.parse_args()
    outputs = summarize_rollout_profile(args.profile_dir)
    print(outputs.summary_path)
    print(outputs.timeline_path)
    print(outputs.report_path)


if __name__ == "__main__":
    main()

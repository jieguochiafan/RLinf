#!/usr/bin/env python3
"""Prefix stdin lines with wall-clock timestamps."""

from __future__ import annotations

import datetime as dt
import sys


def main() -> None:
    for line in sys.stdin:
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        sys.stdout.write(f"[{now}] {line}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

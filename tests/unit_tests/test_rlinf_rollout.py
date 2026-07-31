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

"""Run the rollout system's own test suite from the main repo's CI.

The real tests live in ``rlinf_rollout/tests`` so that they travel with the
package when it is exported via ``git subtree split``. This shim keeps them
covered by the main repo's ``tests/unit_tests`` job.
"""

from pathlib import Path

import pytest

ROLLOUT_TESTS = Path(__file__).resolve().parents[2] / "rlinf_rollout" / "tests"


def test_rlinf_rollout_test_suite():
    assert ROLLOUT_TESTS.is_dir(), f"missing test directory: {ROLLOUT_TESTS}"
    exit_code = pytest.main(["-q", "-p", "no:cacheprovider", str(ROLLOUT_TESTS)])
    assert exit_code == 0, f"rlinf_rollout test suite failed (exit code {exit_code})"

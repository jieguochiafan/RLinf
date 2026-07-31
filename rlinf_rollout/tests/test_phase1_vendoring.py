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

"""Guard the Phase 1 vendored infrastructure (scheduler, utils, data, weight sync).

These tests are static wherever possible so they run without Ray or a GPU: they
check that the vendored tree is self-contained, that every internal import
resolves, and that packaging metadata matches the directory layout. The import
smoke test needs the real runtime dependencies and skips when they are missing.
"""

import ast
import importlib
import sys
import tomllib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_ROOT.parent
PACKAGE_NAME = PACKAGE_ROOT.name

MIN_RAY_VERSION = "2.47.0"

# Modules that are imported by the vendored code but only arrive in a later
# phase. Both sites are guarded (``TYPE_CHECKING`` / lazy function-level import)
# and carry a ``TODO(agent)`` marker.
FORWARD_REFERENCES = {
    "rlinf_rollout.envs.realworld.common.camera.lumos_camera",  # Phase 2
    "rlinf_rollout.workers.rollout.sglang.sglang_worker",  # Phase 3
}

# The Phase 1 deliverable: the modules that must exist after vendoring.
VENDORED_SUBPACKAGES = (
    "scheduler",
    "scheduler.channel",
    "scheduler.cluster",
    "scheduler.collective",
    "scheduler.dynamic_scheduler",
    "scheduler.hardware",
    "scheduler.hardware.accelerators",
    "scheduler.hardware.robots",
    "scheduler.manager",
    "scheduler.placement",
    "scheduler.worker",
    "utils",
    "data",
    "weight_sync",
)

VENDORED_MODULES = (
    # utils minimal set (logging/timers come in through the dependency closure)
    "utils.placement",
    "utils.data_iter_utils",
    "utils.distributed",
    "utils.metric_utils",
    "utils.nested_dict_process",
    "utils.data_process",
    "utils.http_client",
    "utils.utils",
    "utils.logging",
    "utils.timers",
    # internal data representation (converted to api/v1 at the boundary)
    "data.io_struct",
    "data.embodied_io_struct",
    "data.utils",
    # weight sync backends
    "weight_sync.base",
    "weight_sync.bucket_syncer",
    "weight_sync.compressor",
    "weight_sync.patch_syncer",
)


# Directories that may appear inside the package root but are not source:
# bytecode caches and the artifacts of a local `python -m build` / editable install.
NON_SOURCE_DIRS = {"__pycache__", "build", "dist", ".venv", ".pytest_cache"}


def _is_build_artifact(path: Path) -> bool:
    parts = path.relative_to(PACKAGE_ROOT).parts
    return any(part in NON_SOURCE_DIRS or part.endswith(".egg-info") for part in parts)


def _python_sources() -> list[Path]:
    return [path for path in PACKAGE_ROOT.rglob("*.py") if not _is_build_artifact(path)]


def _module_exists(dotted: str) -> bool:
    """Whether ``dotted`` maps to a module or package inside the vendored tree."""
    base = PACKAGE_PARENT / Path(*dotted.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _internal_imports() -> list[tuple[Path, int, str]]:
    """Collect every ``rlinf_rollout.*`` module referenced by an import statement."""
    found = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".")[0] == PACKAGE_NAME:
                    found.append((path, node.lineno, name))
    return found


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subpackage", VENDORED_SUBPACKAGES)
def test_vendored_subpackage_exists(subpackage):
    assert _module_exists(f"{PACKAGE_NAME}.{subpackage}"), (
        f"missing vendored subpackage: {subpackage}"
    )


@pytest.mark.parametrize("module", VENDORED_MODULES)
def test_vendored_module_exists(module):
    assert _module_exists(f"{PACKAGE_NAME}.{module}"), (
        f"missing vendored module: {module}"
    )


def test_weight_syncer_is_reachable_under_its_new_name():
    """The syncer moved out of ``hybrid_engines`` into a top-level package."""
    source = (PACKAGE_ROOT / "weight_sync" / "__init__.py").read_text(encoding="utf-8")
    assert "WeightSyncer" in source
    assert not (PACKAGE_ROOT / "hybrid_engines").exists()


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------


def test_every_internal_import_resolves():
    unresolved = [
        f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {module}"
        for path, lineno, module in _internal_imports()
        if not _module_exists(module)
        and not _module_exists(module.rsplit(".", 1)[0])
        and module not in FORWARD_REFERENCES
        and module.rsplit(".", 1)[0] not in FORWARD_REFERENCES
    ]
    assert not unresolved, "imports pointing at non-existent modules:\n" + "\n".join(
        unresolved
    )


def test_forward_references_are_documented_and_guarded():
    """Imports of not-yet-vendored modules must be lazy and marked with a TODO."""
    for path, lineno, module in _internal_imports():
        if module not in FORWARD_REFERENCES:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        raw = lines[lineno - 1]
        assert raw.startswith(" "), (
            f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: forward reference to "
            f"{module} must be lazy (indented), not a module-level import"
        )
        context = "\n".join(lines[max(0, lineno - 6) : lineno])
        assert "TODO(agent)" in context, (
            f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: forward reference to "
            f"{module} needs a TODO(agent) note naming the phase that vendors it"
        )


def test_vendored_code_keeps_no_stale_training_package_paths():
    """The bulk rewrite must not leave ``rlinf.`` module paths behind."""
    offenders = []
    for path in _python_sources():
        # The test modules spell out `rlinf.` literals as part of these checks.
        if path.parent == PACKAGE_ROOT / "tests":
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "rlinf." in line.replace("rlinf_rollout.", ""):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {line}")
    assert not offenders, "stale training-package paths:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# Packaging metadata
# ---------------------------------------------------------------------------


def test_pyproject_packages_match_the_directory_tree():
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = set(pyproject["tool"]["setuptools"]["packages"])
    on_disk = {
        PACKAGE_NAME
        if init.parent == PACKAGE_ROOT
        else f"{PACKAGE_NAME}." + ".".join(init.parent.relative_to(PACKAGE_ROOT).parts)
        for init in PACKAGE_ROOT.rglob("__init__.py")
        if not _is_build_artifact(init)
    }
    on_disk.discard(f"{PACKAGE_NAME}.tests")  # tests are not shipped in the wheel
    assert declared == on_disk, (
        "[tool.setuptools].packages is out of sync with the tree; "
        f"missing={sorted(on_disk - declared)} stale={sorted(declared - on_disk)}"
    )


# ---------------------------------------------------------------------------
# Runtime smoke test (needs the real dependencies)
# ---------------------------------------------------------------------------


def _runtime_available() -> str:
    try:
        from importlib.metadata import version

        from packaging import version as vs
    except ImportError as exc:  # pragma: no cover - packaging ships with pip
        return str(exc)
    try:
        if vs.parse(version("ray")) < vs.parse(MIN_RAY_VERSION):
            return f"ray>={MIN_RAY_VERSION} required, found {version('ray')}"
    except Exception as exc:
        return f"ray not installed ({exc})"
    try:
        import torch  # noqa: F401
        from torch.distributed.tensor import DTensor  # noqa: F401
    except ImportError as exc:
        return f"torch>=2.5 with DTensor required ({exc})"
    return ""


@pytest.mark.parametrize("module", VENDORED_SUBPACKAGES + VENDORED_MODULES)
def test_vendored_module_imports(module):
    reason = _runtime_available()
    if reason:
        pytest.skip(f"runtime dependency unavailable: {reason}")
    dotted = f"{PACKAGE_NAME}.{module}"
    importlib.import_module(dotted)
    assert dotted in sys.modules


@pytest.fixture
def cluster_cls():
    reason = _runtime_available()
    if reason:
        pytest.skip(f"runtime dependency unavailable: {reason}")
    from rlinf_rollout.scheduler.cluster.cluster import Cluster

    return Cluster


@pytest.mark.parametrize("value", [None, "0", "false", "off", "no"])
def test_ray_code_sync_is_opt_in(cluster_cls, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RLINF_CODE_WORKING_DIR", raising=False)
    else:
        monkeypatch.setenv("RLINF_CODE_WORKING_DIR", value)
    assert cluster_cls._prepare_ray_code_sync_runtime_env_fragment() == (None, ())


@pytest.mark.parametrize("mode", ["auto", "package_dir", "checkout_root"])
def test_ray_code_sync_ships_the_rollout_package(cluster_cls, monkeypatch, mode):
    """All accepted spellings must resolve to the vendored package directory.

    ``auto`` covers both layouts: the in-tree checkout and the standalone repo
    produced by ``git subtree split``, where ``pyproject.toml`` sits inside the
    package directory itself.
    """
    value = {
        "auto": "auto",
        "package_dir": str(PACKAGE_ROOT),
        "checkout_root": str(PACKAGE_PARENT),
    }[mode]
    monkeypatch.setenv("RLINF_CODE_WORKING_DIR", value)
    fragment, strip_roots = cluster_cls._prepare_ray_code_sync_runtime_env_fragment()
    assert fragment == {"py_modules": [str(PACKAGE_ROOT)]}
    assert str(PACKAGE_ROOT) in strip_roots and str(PACKAGE_PARENT) in strip_roots


def test_ray_code_sync_rejects_the_training_package(cluster_cls, monkeypatch):
    training_pkg = PACKAGE_PARENT / "rlinf"
    if not training_pkg.is_dir():
        pytest.skip("training-side rlinf package is not checked out next to us")
    monkeypatch.setenv("RLINF_CODE_WORKING_DIR", str(training_pkg))
    with pytest.raises(RuntimeError, match=PACKAGE_NAME):
        cluster_cls._prepare_ray_code_sync_runtime_env_fragment()


def test_ray_code_sync_rejects_relative_paths(cluster_cls, monkeypatch):
    monkeypatch.setenv("RLINF_CODE_WORKING_DIR", "relative/path")
    with pytest.raises(RuntimeError, match="absolute path"):
        cluster_cls._prepare_ray_code_sync_runtime_env_fragment()

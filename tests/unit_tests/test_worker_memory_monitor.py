from __future__ import annotations

from tools.worker_memory_monitor import (
    parse_gpu_index_uuid_map,
    parse_nvidia_compute_apps,
    parse_status_rss_mib,
)


def test_parse_status_rss_mib_reads_vmrss_kib() -> None:
    status = """Name:\tpython
VmPeak:\t  123456 kB
VmRSS:\t   204800 kB
"""

    assert parse_status_rss_mib(status) == 200.0


def test_parse_nvidia_compute_apps_handles_units_and_uuid() -> None:
    text = """1234, GPU-abc, 4096 MiB
5678, GPU-def, 1024
"""

    rows = parse_nvidia_compute_apps(text)

    assert rows[1234].gpu_uuid == "GPU-abc"
    assert rows[1234].used_memory_mib == 4096.0
    assert rows[5678].used_memory_mib == 1024.0


def test_parse_gpu_index_uuid_map_maps_uuid_to_index() -> None:
    text = """0, GPU-abc
1, GPU-def
"""

    assert parse_gpu_index_uuid_map(text) == {"GPU-abc": "0", "GPU-def": "1"}

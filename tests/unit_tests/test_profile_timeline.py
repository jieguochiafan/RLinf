# Copyright 2025 The RLinf Authors.
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

import json
from asyncio import run

from rlinf.utils.profile_timeline import TimelineRecorder, timeline_span


def test_disabled_timeline_does_not_create_output(tmp_path):
    recorder = TimelineRecorder(
        tmp_path,
        component="env",
        rank=0,
        enabled=False,
    )

    with recorder.span("env.step"):
        pass

    assert list(tmp_path.iterdir()) == []


def test_timeline_records_complete_span(tmp_path):
    recorder = TimelineRecorder(
        tmp_path,
        component="rollout",
        rank=2,
        enabled=True,
    )

    with recorder.span("rollout.inference", chunk_step=3):
        pass
    recorder.close()

    process_dir = next(tmp_path.iterdir())
    event = json.loads((process_dir / "events.jsonl").read_text().strip())
    assert event["name"] == "rollout.inference"
    assert event["component"] == "rollout"
    assert event["rank"] == 2
    assert event["args"]["chunk_step"] == 3
    assert event["duration_ns"] >= 0
    assert event["end_wall_ns"] >= event["start_wall_ns"]
    assert (process_dir / "time_anchor.json").exists()


def test_timeline_decorator_records_worker_context(tmp_path):
    class Worker:
        def __init__(self):
            self.version = 4
            self.timeline = TimelineRecorder(
                tmp_path,
                component="actor",
                rank=0,
                enabled=True,
            )

        @timeline_span("actor.train", include_args=("batch_id",))
        async def train(self, batch_id):
            return batch_id

    worker = Worker()
    assert run(worker.train(7)) == 7
    worker.timeline.close()

    process_dir = next(tmp_path.iterdir())
    event = json.loads((process_dir / "events.jsonl").read_text().strip())
    assert event["name"] == "actor.train"
    assert event["args"] == {"batch_id": 7, "policy_version": 4}

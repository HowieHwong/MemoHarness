from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from memoharness.harbor.loop import (
    _TeeTextStream,
    _fallback_harbor_status,
    _harbor_result_is_terminal,
    _observed_shard_terminal_tasks,
)


def _job_result(*, finished: bool, running: int, completed: int) -> dict:
    return {
        "finished_at": "2026-07-26T12:00:00Z" if finished else None,
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": completed,
            "n_errored_trials": 0,
            "n_running_trials": running,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
        },
    }


class HarborExitWatchdogTests(unittest.TestCase):
    def test_unfinished_top_level_result_is_not_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            payload = _job_result(finished=False, running=1, completed=0)
            (job_dir / "result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            log_path = job_dir / "harbor.console.log"
            log_path.write_text("", encoding="utf-8")

            self.assertFalse(_harbor_result_is_terminal(payload))
            self.assertEqual(
                _fallback_harbor_status(job_dir, log_path)["stage"],
                "running harbor",
            )

    def test_finalized_top_level_result_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            payload = _job_result(finished=True, running=0, completed=1)
            (job_dir / "result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            log_path = job_dir / "harbor.console.log"
            log_path.write_text("", encoding="utf-8")

            self.assertTrue(_harbor_result_is_terminal(payload))
            self.assertEqual(
                _fallback_harbor_status(job_dir, log_path)["stage"],
                "completed",
            )

    def test_partial_trial_result_does_not_trigger_completion_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            trial_dir = job_dir / "train-fasttext__trial"
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "train-fasttext",
                        "finished_at": None,
                        "verifier_result": None,
                        "exception_info": None,
                    }
                ),
                encoding="utf-8",
            )

            terminal, exceptions = _observed_shard_terminal_tasks(
                job_dir,
                ["train-fasttext"],
            )
            self.assertEqual(terminal, set())
            self.assertEqual(exceptions, set())

    def test_final_reward_triggers_completion_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            trial_dir = job_dir / "train-fasttext__trial"
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "train-fasttext",
                        "finished_at": "2026-07-26T12:00:00Z",
                        "verifier_result": {"rewards": {"reward": 1.0}},
                        "exception_info": None,
                    }
                ),
                encoding="utf-8",
            )

            terminal, exceptions = _observed_shard_terminal_tasks(
                job_dir,
                ["train-fasttext"],
            )
            self.assertEqual(terminal, {"train-fasttext"})
            self.assertEqual(exceptions, set())

    def test_renderer_output_is_not_mirrored_to_console_log(self):
        primary = io.StringIO()
        mirror = io.StringIO()
        stream = _TeeTextStream(primary, mirror, mirror_path=Path("console.log"))

        stream.write_from_renderer("dynamic progress")
        self.assertEqual(primary.getvalue(), "dynamic progress")
        self.assertEqual(mirror.getvalue(), "")

        stream.write("normal log")
        self.assertTrue(primary.getvalue().endswith("normal log"))
        self.assertEqual(mirror.getvalue(), "normal log")


if __name__ == "__main__":
    unittest.main()

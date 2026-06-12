"""Shared base for failure-inject verify integration tests."""

import os
import re
import unittest
from typing import ClassVar

from typer.testing import CliRunner

from nika.cli.main import app
from nika.service.mcp_server.mcp_session_context import SESSION_ID_ENV, get_lab_name
from nika.utils.session_store import SessionStore


class FailureInjectVerifyTestCase(unittest.TestCase):
    """Start a Kathara lab per test; bind all operations to session_id."""

    SCENARIO: ClassVar[str]
    ENV_RUN_ARGS: ClassVar[list[str]] = []

    runner: CliRunner
    session_id: str
    _prev_nika_session_id: str | None

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id = self._start_env()
        self._prev_nika_session_id = os.environ.get(SESSION_ID_ENV)
        os.environ[SESSION_ID_ENV] = self.session_id
        self._assert_session_ready()

    def tearDown(self) -> None:
        if getattr(self, "session_id", None):
            self.runner.invoke(app, ["env", "stop", "--session-id", self.session_id])
        if getattr(self, "_prev_nika_session_id", None) is None:
            os.environ.pop(SESSION_ID_ENV, None)
        else:
            os.environ[SESSION_ID_ENV] = self._prev_nika_session_id

    def _start_env(self) -> str:
        args = ["env", "run", self.SCENARIO, *self.ENV_RUN_ARGS]
        result = self.runner.invoke(app, args)
        if result.exit_code != 0:
            raise RuntimeError(f"nika env run failed:\n{result.output}")
        match = re.search(r"session_id=(\S+)", result.output.strip())
        if match is None:
            raise RuntimeError(f"session_id not found in env run output:\n{result.output}")
        return match.group(1)

    def _session_row(self) -> dict:
        return SessionStore().get_session(self.session_id)

    def _assert_session_ready(self) -> None:
        row = self._session_row()
        self.assertEqual(row["session_id"], self.session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], self.SCENARIO)
        self.assertIsNotNone(row.get("lab_name"), "lab_name must be set after env run")
        self.assertIn(self.SCENARIO, row["lab_name"])
        self.assertRegex(
            self.session_id,
            r"^\d{8}-\d{6}-[0-9a-f]{6}$",
            "session_id does not match expected YYYYMMDD-HHMMSS-{6hex} format",
        )
        self.assertEqual(get_lab_name(), row["lab_name"])

    def _scenario_kwargs(self) -> dict:
        row = self._session_row()
        kwargs = dict(row.get("scenario_params") or {})
        if row.get("lab_name"):
            kwargs["lab_name"] = row["lab_name"]
        if row.get("scenario_topo_size") is not None:
            kwargs["topo_size"] = row["scenario_topo_size"]
        return kwargs

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, **self._scenario_kwargs())

    @property
    def lab_name(self) -> str:
        """Lab name bound to the current session (resolved via session_id)."""
        return self._session_row()["lab_name"]

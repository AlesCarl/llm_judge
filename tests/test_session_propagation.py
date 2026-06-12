"""Session propagation integration test.

Verifies that the session ID is correctly threaded through every stage of the
pipeline:

  Stage 1 — ``nika env run``         : session created in SessionStore
  Stage 2 — ``nika failure inject``  : session enriched (task_description,
                                        session_dir, ground_truth)
  Stage 3 — mcp_session_context      : in-process resolution of NIKA_SESSION_ID
                                        → lab_name / session_dir
  Stage 4 — diagnosis MCP tools      : MCPServerConfig injects NIKA_SESSION_ID
                                        into kathara subprocess; tools execute
                                        in the correct lab
  Stage 5 — submit MCP tool          : task_mcp_server reads NIKA_SESSION_ID,
                                        writes submission.json to the correct
                                        session_dir

Run via:  uv run python -m unittest tests/test_session_propagation.py -v
"""

import asyncio
import json
import os
import re
import unittest
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from typer.testing import CliRunner

from agent.utils.mcp_servers import MCPServerConfig
from nika.cli.main import app
from nika.service.mcp_server.mcp_session_context import (
    get_lab_name,
    get_session_dir,
    require_session_id,
)
from nika.utils.session_store import SessionStore

SCENARIO = "simple_bgp"
PROBLEM = "link_down"


def _tool_text_list(result: object) -> list[str]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return [result]
    if not isinstance(result, list):
        return [str(result)]
    return [str(item["text"]) if isinstance(item, dict) and "text" in item else str(item) for item in result]


class SessionPropagationTest(unittest.TestCase):
    """Ordered integration steps that verify session ID flows through all stages."""

    runner: CliRunner
    session_id: str | None = None
    session_dir: Path | None = None
    env_destroyed: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.session_id and not cls.env_destroyed:
            cls.runner.invoke(app, ["env", "stop", "--session-id", cls.session_id])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _invoke_ok(self, args: list[str]) -> str:
        result = self.runner.invoke(app, args)
        self.assertEqual(result.exit_code, 0, result.output)
        return result.output

    def _load_json(self, filename: str) -> dict:
        assert self.session_dir is not None
        return json.loads((self.session_dir / filename).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Stage 1 — session creation
    # ------------------------------------------------------------------

    def test_step_01_env_creates_session_in_store(self) -> None:
        """env run creates a session and persists it correctly in SessionStore."""
        run_output = self._invoke_ok(["env", "run", SCENARIO])

        match = re.search(r"session_id=(\S+)", run_output.strip())
        self.assertIsNotNone(match, f"session_id missing from env run output:\n{run_output}")
        session_id = match.group(1)
        type(self).session_id = session_id

        # session_id format: YYYYMMDD-HHMMSS-{6hex}
        self.assertRegex(
            session_id,
            r"^\d{8}-\d{6}-[0-9a-f]{6}$",
            "session_id does not match expected YYYYMMDD-HHMMSS-{6hex} format",
        )

        row = SessionStore().get_session(session_id)

        # session document fields
        self.assertEqual(row["session_id"], session_id)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["scenario_name"], SCENARIO)
        self.assertIsNotNone(row.get("lab_name"), "lab_name must be set after env run")
        # lab_name is bound to this session (should contain scenario name)
        self.assertIn(SCENARIO, row["lab_name"])

    # ------------------------------------------------------------------
    # Stage 2 — failure injection enriches session
    # ------------------------------------------------------------------

    def test_step_02_failure_inject_enriches_session(self) -> None:
        """failure inject populates task_description, session_dir, and ground_truth."""
        self.assertIsNotNone(self.session_id)

        self._invoke_ok(
            [
                "failure",
                "inject",
                PROBLEM,
                "--session-id",
                self.session_id,
                "--set",
                "host_name=pc1",
                "--set",
                "intf_name=eth0",
            ]
        )

        row = SessionStore().get_session(self.session_id)

        # problem metadata stored on session
        self.assertIn(PROBLEM, row.get("problem_names", []))
        self.assertEqual(row.get("root_cause_name"), PROBLEM)

        # task_description must be non-empty and carry lab/network info
        task_description = row.get("task_description", "")
        self.assertTrue(len(task_description) > 0, "task_description should be non-empty")

        # session_dir must include the session_id so artifacts are scoped
        session_dir_str = row.get("session_dir", "")
        self.assertIn(self.session_id, session_dir_str, "session_dir must contain session_id")
        self.assertIn(PROBLEM, session_dir_str, "session_dir must contain root_cause_name")

        session_dir = Path(session_dir_str)
        type(self).session_dir = session_dir

        # ground_truth.json must exist and be valid
        gt_path = session_dir / "ground_truth.json"
        self.assertTrue(gt_path.exists(), "ground_truth.json must exist after failure inject")
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        self.assertTrue(ground_truth.get("is_anomaly"))
        self.assertIn(PROBLEM, ground_truth.get("root_cause_name", []))

    # ------------------------------------------------------------------
    # Stage 3 — mcp_session_context in-process unit test
    # ------------------------------------------------------------------

    def test_step_03_mcp_session_context_resolves_from_env_var(self) -> None:
        """mcp_session_context functions correctly read NIKA_SESSION_ID from env."""
        self.assertIsNotNone(self.session_id)

        row = SessionStore().get_session(self.session_id)
        expected_lab_name = row["lab_name"]
        expected_session_dir = row["session_dir"]

        prev = os.environ.get("NIKA_SESSION_ID")
        try:
            os.environ["NIKA_SESSION_ID"] = self.session_id

            # require_session_id returns the env var value exactly
            resolved_id = require_session_id()
            self.assertEqual(resolved_id, self.session_id)

            # get_lab_name resolves to the lab stored in session
            resolved_lab = get_lab_name()
            self.assertEqual(resolved_lab, expected_lab_name)

            # get_session_dir resolves to the session_dir stored in session
            resolved_dir = get_session_dir()
            self.assertEqual(resolved_dir, expected_session_dir)

        finally:
            if prev is None:
                os.environ.pop("NIKA_SESSION_ID", None)
            else:
                os.environ["NIKA_SESSION_ID"] = prev

    # ------------------------------------------------------------------
    # Stage 4 — diagnosis MCP tools execute in the correct lab
    # ------------------------------------------------------------------

    def test_step_04_diagnosis_mcp_tools_use_session_lab(self) -> None:
        """Diagnosis MCP tools spawned with NIKA_SESSION_ID execute against the correct lab."""
        self.assertIsNotNone(self.session_id)

        # MCPServerConfig must embed the correct session ID in the subprocess env
        mcp_config = MCPServerConfig(session_id=self.session_id)
        server_env = mcp_config._server_env()
        self.assertEqual(
            server_env["NIKA_SESSION_ID"],
            self.session_id,
            "MCPServerConfig must set NIKA_SESSION_ID to the current session_id",
        )

        config = mcp_config.load_config(if_submit=False)

        # Restrict to the base server we need for this test
        diagnosis_config = {k: v for k, v in config.items() if k == "kathara_base_mcp_server"}

        async def _run() -> dict:
            client = MultiServerMCPClient(connections=diagnosis_config)
            tools = {t.name: t for t in await client.get_tools()}

            # get_reachability — no arguments; returns connectivity matrix
            self.assertIn("get_reachability", tools, "get_reachability tool must be available")
            reach_result = await tools["get_reachability"].ainvoke({})
            reach_str = str(reach_result)
            self.assertTrue(len(reach_str) > 0, "get_reachability must return non-empty output")

            # get_host_net_config — verify host network config is accessible for the injected lab
            self.assertIn("get_host_net_config", tools, "get_host_net_config tool must be available")
            host_config_result = await tools["get_host_net_config"].ainvoke({"host_name": "pc1"})
            host_config_str = str(host_config_result)
            self.assertTrue(len(host_config_str) > 0, "get_host_net_config must return non-empty output")

            # exec_shell — execute a command on a host in the lab
            self.assertIn("exec_shell", tools, "exec_shell tool must be available")
            exec_result = await tools["exec_shell"].ainvoke({"host_name": "pc1", "command": "hostname"})
            exec_str = str(exec_result)
            self.assertTrue(len(exec_str) > 0, "exec_shell must return non-empty output")

            return {
                "reachability": reach_str,
                "host_net_config": host_config_str,
                "exec_shell": exec_str,
            }

        results = asyncio.run(_run())

        # Results should come from the lab named in the session, so they must
        # not contain error messages about unknown session or missing lab.
        for key, output in results.items():
            self.assertNotIn(
                "NIKA_SESSION_ID is not set",
                output,
                f"{key}: NIKA_SESSION_ID was not propagated to MCP subprocess",
            )
            self.assertNotIn(
                "Session",
                output[:50] if "not running" in output else "",
                f"{key}: session was not found in subprocess",
            )

    # ------------------------------------------------------------------
    # Stage 5 — submit via MCP writes to the correct session_dir
    # ------------------------------------------------------------------

    def test_step_05_submit_via_mcp_writes_to_correct_session_dir(self) -> None:
        """task_mcp_server.submit() resolves NIKA_SESSION_ID → session_dir and writes submission.json."""
        self.assertIsNotNone(self.session_id)
        self.assertIsNotNone(self.session_dir)

        config = MCPServerConfig(session_id=self.session_id).load_config(if_submit=True)

        async def _run() -> str:
            client = MultiServerMCPClient(connections=config)
            tools = {t.name: t for t in await client.get_tools()}

            self.assertIn("list_avail_problems", tools)
            self.assertIn("submit", tools)

            # list_avail_problems returns the available root cause names
            avail_raw = await tools["list_avail_problems"].ainvoke({})
            avail = _tool_text_list(avail_raw)
            self.assertTrue(len(avail) > 0, "list_avail_problems must return at least one entry")
            self.assertIn(PROBLEM, avail, f"{PROBLEM} must be among available problems")

            # submit using a known root cause from the available list
            submit_result = await tools["submit"].ainvoke(
                {
                    "is_anomaly": True,
                    "faulty_devices": ["pc1"],
                    "root_cause_name": [PROBLEM],
                }
            )
            return str(submit_result)

        result_str = asyncio.run(_run())
        self.assertIn("success", result_str.lower(), f"submit tool should report success; got: {result_str}")

        # submission.json must appear in the session-scoped directory
        submission_path = self.session_dir / "submission.json"
        self.assertTrue(
            submission_path.exists(),
            f"submission.json not found at {submission_path}",
        )

        # the path must encode the session_id — confirms correct session binding
        self.assertIn(
            self.session_id,
            str(submission_path),
            "submission.json path must contain session_id",
        )

        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        self.assertIn("is_anomaly", submission)
        self.assertIn("faulty_devices", submission)
        self.assertIn("root_cause_name", submission)
        self.assertTrue(submission["is_anomaly"])
        self.assertIn("pc1", submission["faulty_devices"])
        self.assertIn(PROBLEM, submission["root_cause_name"])


if __name__ == "__main__":
    unittest.main()

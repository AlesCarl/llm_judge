"""CLI command handlers: argument wiring and error mapping."""

import unittest
from unittest.mock import patch

import typer

from nika.cli.commands.failure import failure_inject


class FailureCliTest(unittest.TestCase):
    def test_failure_inject_maps_workflow_errors_to_badparameter(self) -> None:
        with patch(
            "nika.workflows.failure_inject.inject_failure",
            side_effect=ValueError("need --session-id"),
        ):
            with self.assertRaises(typer.BadParameter) as exc_info:
                failure_inject(problems=["link_down"])
        self.assertIn("need --session-id", str(exc_info.exception))

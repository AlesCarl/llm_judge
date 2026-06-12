"""Unit tests for Kathara Docker container resolution."""

import unittest
from unittest.mock import MagicMock, patch

from nika.service.kathara.docker_utils import get_machine_container


class GetMachineContainerTest(unittest.TestCase):
    def test_raises_when_machine_not_found(self) -> None:
        with patch("nika.service.kathara.docker_utils.Kathara") as mock_kathara:
            mock_kathara.get_instance.return_value.get_machine_stats.return_value = iter([None])
            with self.assertRaisesRegex(ValueError, "No container found"):
                get_machine_container(lab_name="simple_bgp__tag", host_name="pc1")

    def test_returns_container_from_machine_stats(self) -> None:
        container = MagicMock(name="container")
        stats = MagicMock(machine_api_object=container)
        with patch("nika.service.kathara.docker_utils.Kathara") as mock_kathara:
            mock_kathara.get_instance.return_value.get_machine_stats.return_value = iter([stats])
            resolved = get_machine_container(lab_name="simple_bgp__tag", host_name="pc1")
        self.assertIs(resolved, container)
        mock_kathara.get_instance.return_value.get_machine_stats.assert_called_once_with(
            machine_name="pc1",
            lab_name="simple_bgp__tag",
        )


if __name__ == "__main__":
    unittest.main()

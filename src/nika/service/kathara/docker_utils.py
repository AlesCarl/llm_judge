"""Resolve Docker containers for Kathara lab machines."""

from __future__ import annotations

from docker.models.containers import Container
from Kathara.manager.Kathara import Kathara


def get_machine_container(*, lab_name: str, host_name: str) -> Container:
    """Return the Docker container for ``host_name`` inside ``lab_name``."""
    stats = next(
        Kathara.get_instance().get_machine_stats(machine_name=host_name, lab_name=lab_name),
        None,
    )
    if stats is None:
        raise ValueError(f"No container found for host {host_name!r} in lab {lab_name!r}.")
    return stats.machine_api_object

"""Runtime fixes for the Kathara dependency."""

from __future__ import annotations

import logging
from typing import Optional

_logger = logging.getLogger(__name__)


def patch_kathara_file_conversion() -> None:
    """Patch Kathara's text conversion helper to close file handles."""
    from Kathara import utils

    if getattr(utils.convert_win_2_linux, "_nika_closes_files", False):
        return

    def convert_win_2_linux(filename: str, write: bool = False) -> Optional[bytes]:
        if not utils.is_binary(filename):
            try:
                with open(filename, mode="r", encoding="utf-8-sig") as file_obj:
                    file_content = (
                        file_obj.read().replace("\n\r", "\n").replace("\r\n", "\n")
                    )
                if not write:
                    return file_content.encode("utf-8")
                with open(
                    filename, mode="w", encoding="utf-8", newline="\n"
                ) as file_obj_write:
                    file_obj_write.write(file_content)
                return None
            except Exception:
                pass

        if not write:
            with open(filename, mode="rb") as file_obj:
                return file_obj.read()
        return None

    convert_win_2_linux._nika_closes_files = True  # type: ignore[attr-defined]
    utils.convert_win_2_linux = convert_win_2_linux


def docker_engine_reachable() -> bool:
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        return False
    return True


def allow_privileged_without_root(*, is_admin: bool | None = None) -> bool:
    """Return True when NIKA should bypass Kathara's host-root privileged gate."""
    if is_admin is None:
        from Kathara import utils

        is_admin = utils.is_admin()
    if is_admin:
        return False
    return docker_engine_reachable()


def patch_kathara_privileged_without_root() -> None:
    """Allow privileged Kathara devices without host root when Docker is usable.

    Kathara refuses ``privileged=true`` unless ``os.getuid() == 0``, even though
    members of the ``docker`` group can create privileged containers via the
    Docker API. Patching this gate lets k3s scenarios run under a normal user
    for batch benchmarks.
    """
    from Kathara import utils
    from Kathara.manager.docker.DockerMachine import DockerMachine

    if getattr(DockerMachine.create, "_nika_priv_without_root", False):
        return

    original_create = DockerMachine.create
    original_is_admin = utils.is_admin

    def create(self, machine):  # type: ignore[no-untyped-def]
        if machine.is_privileged() and allow_privileged_without_root(
            is_admin=original_is_admin()
        ):
            _logger.debug(
                "Allowing privileged Kathara device %r without host root "
                "(Docker engine reachable)",
                machine.name,
            )

            def _is_admin_allow() -> bool:
                return True

            utils.is_admin = _is_admin_allow
            try:
                return original_create(self, machine)
            finally:
                utils.is_admin = original_is_admin
        return original_create(self, machine)

    create._nika_priv_without_root = True  # type: ignore[attr-defined]
    DockerMachine.create = create  # type: ignore[method-assign]


patch_kathara_file_conversion()
patch_kathara_privileged_without_root()

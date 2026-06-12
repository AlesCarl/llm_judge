import os


class MCPServerConfig:
    def __init__(self, session_id: str):
        base_dir = os.getenv("BASE_DIR")
        if not base_dir:
            raise ValueError("BASE_DIR must be set before starting MCP servers.")
        if not session_id:
            raise ValueError("session_id is required to start MCP servers.")
        self.mcp_server_dir = os.path.join(base_dir, "src/nika/service/mcp_server")
        self.session_id = session_id

    def _server_env(self) -> dict[str, str]:
        return {
            "NIKA_SESSION_ID": self.session_id,
            "BASE_DIR": os.getenv("BASE_DIR", ""),
        }

    def load_config(self, if_submit: bool = False) -> dict:
        if if_submit:
            config = {
                "task_mcp_server": {
                    "command": "python3",
                    "args": [f"{self.mcp_server_dir}/task_mcp_server.py"],
                    "transport": "stdio",
                },
            }
        else:
            config = {
                "kathara_base_mcp_server": {
                    "command": "python3",
                    "args": [f"{self.mcp_server_dir}/kathara_base_mcp_server.py"],
                    "transport": "stdio",
                },
                "kathara_frr_mcp_server": {
                    "command": "python3",
                    "args": [f"{self.mcp_server_dir}/kathara_frr_mcp_server.py"],
                    "transport": "stdio",
                },
                "kathara_bmv2_mcp_server": {
                    "command": "python3",
                    "args": [f"{self.mcp_server_dir}/kathara_bmv2_mcp_server.py"],
                    "transport": "stdio",
                },
                "kathara_telemetry_mcp_server": {
                    "command": "python3",
                    "args": [f"{self.mcp_server_dir}/kathara_telemetry_mcp_server.py"],
                    "transport": "stdio",
                },
            }

        for server in config.values():
            server["env"] = self._server_env()
        return config

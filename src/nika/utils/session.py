import json
import os
from datetime import datetime
from typing import Any

from nika.config import BASE_DIR, RESULTS_DIR
from nika.utils.session_store import SessionStore

class Session:
    def __init__(self) -> None:
        self.store = SessionStore()
        self.start_time = None
        self.end_time = None

    def init_session(
        self,
        *,
        session_id: str,
        scenario_name: str,
        lab_name: str,
        scenario_topo_size: str | None,
        scenario_params: dict | None = None,
    ) -> None:
        self.session_id = session_id
        self.scenario_name = scenario_name
        self.lab_name = lab_name
        self.scenario_topo_size = scenario_topo_size
        self.scenario_params = scenario_params or {}
        os.makedirs(f"{BASE_DIR}/runtime", exist_ok=True)
        self.store.create_session(
            {
                "session_id": self.session_id,
                "lab_name": self.lab_name,
                "scenario_name": self.scenario_name,
                "scenario_topo_size": self.scenario_topo_size,
                "scenario_params_json": self.scenario_params,
                "status": "running",
            }
        )

    def load_running_session(self, session_id: str | None = None):
        session_meta = (
            self.store.get_session(session_id)
            if session_id is not None
            else self.store.get_unique_running_session()
        )
        if session_meta.get("status") != "running":
            raise ValueError(f"Session '{session_meta.get('session_id')}' is not running.")
        for key, value in session_meta.items():
            if key.endswith("_json"):
                continue
            setattr(self, key, value)
        return self

    def _write_session(self) -> str:
        if not hasattr(self, "session_id"):
            raise ValueError("Session ID is not set.")
        payload = dict(self.__dict__)
        payload.pop("store", None)
        payload.pop("problem_names_json", None)
        payload.pop("scenario_params_json", None)
        payload.pop("eval_metrics_json", None)
        payload.pop("llm_judge_json", None)
        payload.pop("eval_summary_json", None)
        if "problem_names" in payload:
            payload["problem_names_json"] = payload.pop("problem_names")
        if "scenario_params" in payload:
            payload["scenario_params_json"] = payload.pop("scenario_params")
        if "eval_metrics" in payload:
            payload["eval_metrics_json"] = payload.pop("eval_metrics")
        if "llm_judge" in payload:
            payload["llm_judge_json"] = payload.pop("llm_judge")
        if "eval_summary" in payload:
            payload["eval_summary_json"] = payload.pop("eval_summary")
        allowed_columns = {
            "lab_name",
            "scenario_name",
            "scenario_topo_size",
            "status",
            "problem_names_json",
            "root_cause_name",
            "task_description",
            "scenario_params_json",
            "agent_type",
            "llm_backend",
            "model",
            "start_time",
            "end_time",
            "eval_metrics_json",
            "llm_judge_json",
            "eval_summary_json",
            "session_dir",
        }
        payload = {k: v for k, v in payload.items() if k in allowed_columns}
        self.store.update_session(self.session_id, payload)
        return self.session_id

    def update_session(self, key: str, value: Any):
        setattr(self, key, value)
        if hasattr(self, "problem_names") and hasattr(self, "session_id"):
            if len(self.problem_names) > 1:
                self.root_cause_name = "multiple_faults"
            else:
                self.root_cause_name = self.problem_names[0]
            self.session_dir = f"{RESULTS_DIR}/{self.root_cause_name}/{self.session_id}"
        self._write_session()

    def write_gt(self, gt: dict[str, Any]):
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.session_dir + "/ground_truth.json", "w") as f:
            f.write(json.dumps(gt, indent=4))

    def clear_session(self):
        if not hasattr(self, "session_id"):
            raise ValueError("Session ID is not set.")
        self.store.update_session(self.session_id, {"status": "finished"})

    def start_session(self):
        self.start_time = datetime.now().timestamp()
        self._write_session()

    def end_session(self):
        self.end_time = datetime.now().timestamp()
        self._write_session()

    def __str__(self) -> str:
        payload = dict(self.__dict__)
        payload.pop("store", None)
        return str(payload)


if __name__ == "__main__":
    session = Session()
    session.load_running_session()
    print(session)

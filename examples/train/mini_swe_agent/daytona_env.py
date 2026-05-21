"""Daytona sandbox environment for mini-swe-agent.

Implements the mini-swe-agent ``Environment`` protocol using the Daytona
SDK. Each environment instance creates a sandbox from the SWE-bench
Docker image and executes commands via ``sandbox.process.exec()``.

Requires:
    - ``daytona-sdk`` package installed
    - ``DAYTONA_API_KEY`` and ``DAYTONA_BASE_URL`` environment variables set
      (or defaults to local Daytona instance)
"""

import json
import logging
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from minisweagent.exceptions import Submitted

logger = logging.getLogger(__name__)


@dataclass
class DaytonaEnvironmentConfig:
    """Configuration for the Daytona sandbox environment."""

    image: str = ""
    cwd: str = "/testbed"
    timeout: int = 180
    pull_timeout: int = 1200
    env: Dict[str, str] = field(default_factory=dict)
    base_url: str = ""
    api_key: str = ""
    target: str = "us"
    metrics_dir: str = ""
    """Directory to write per-sandbox execution metrics. Empty = no metrics."""


class _AttrDict(dict):
    """Dict subclass that supports attribute access (for Jinja2 templates)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None


class DaytonaEnvironment:
    """Execute commands in a Daytona sandbox.

    Drop-in replacement for mini-swe-agent's ``DockerEnvironment``,
    using the Daytona SDK to create and manage sandboxes.
    """

    def __init__(self, *, logger: Optional[logging.Logger] = None, **kwargs):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = DaytonaEnvironmentConfig(**{
            k: v for k, v in kwargs.items() if hasattr(DaytonaEnvironmentConfig, k)
        })
        self._sandbox = None
        self._daytona = None
        self._metrics: List[Dict[str, Any]] = []
        self._create_time: float = 0
        self._start_sandbox()

    def _start_sandbox(self):
        """Create a Daytona sandbox from the configured image.

        Uses a generous timeout for sandbox creation (10 min) to handle
        initial Docker image pulls while preventing indefinite hangs.
        """
        from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxFromImageParams

        config = DaytonaConfig(
            api_key=self.config.api_key or os.environ.get("DAYTONA_API_KEY", ""),
            api_url=self.config.base_url or os.environ.get("DAYTONA_BASE_URL", "http://localhost:3000/api"),
            target=self.config.target or os.environ.get("DAYTONA_TARGET", "us"),
        )
        self._daytona = Daytona(config=config)

        sandbox_name = f"swe-{uuid.uuid4().hex[:12]}"
        self.logger.debug(f"Creating Daytona sandbox {sandbox_name} from image {self.config.image}")

        t0 = time.time()
        params = CreateSandboxFromImageParams(
            image=self.config.image,
            language="python",
            name=sandbox_name,
        )
        self._sandbox = self._daytona.create(params, timeout=self.config.pull_timeout)
        self._create_time = time.time() - t0
        self.logger.info(f"Created Daytona sandbox: {self._sandbox.id} ({self._create_time:.1f}s)")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the Daytona sandbox.

        After each successful execution, checks if the output starts with
        ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` (the mini-swe-agent
        submission marker).  If so, raises :class:`Submitted` so the agent
        loop terminates and the diff is captured as the submission — matching
        the behavior of the local Docker and Singularity environments.
        """
        command = action.get("command", "")
        cwd = cwd or self.config.cwd

        env_vars = dict(self.config.env)

        t0 = time.time()
        try:
            response = self._sandbox.process.exec(
                command=command,
                cwd=cwd,
                env=env_vars if env_vars else None,
                timeout=timeout or self.config.timeout,
            )
            elapsed = time.time() - t0
            result = _AttrDict({
                "output": response.result,
                "returncode": response.exit_code,
            })
            self._metrics.append({
                "step": len(self._metrics),
                "elapsed_s": round(elapsed, 3),
                "cmd_len": len(command),
                "output_len": len(response.result),
                "exit_code": response.exit_code,
            })
            self._check_finished(result)
            return result
        except Submitted:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            self.logger.error(f"Daytona exec error: {e}")
            self._metrics.append({
                "step": len(self._metrics),
                "elapsed_s": round(elapsed, 3),
                "cmd_len": len(command),
                "output_len": 0,
                "exit_code": -1,
                "error": str(e),
            })
            return _AttrDict({
                "output": str(e),
                "returncode": 1,
                "exception_info": str(e),
            })

    def _check_finished(self, output: dict):
        """Raise :class:`Submitted` if the output contains the submission marker.

        Matches the logic in ``minisweagent.environments.local.LocalEnvironment``.
        """
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        """Return template variables for the agent."""
        config_dict = {
            "image": self.config.image,
            "cwd": self.config.cwd,
            "timeout": self.config.timeout,
        }
        return {**config_dict, **platform.uname()._asdict(), **kwargs}

    def serialize(self) -> dict:
        """Serialize environment info for trajectory saving."""
        return {
            "info": {
                "config": {
                    "environment": {
                        "image": self.config.image,
                        "cwd": self.config.cwd,
                        "sandbox_id": self._sandbox.id if self._sandbox else None,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def close(self):
        """Delete the Daytona sandbox and save metrics."""
        # Save metrics if configured
        if self.config.metrics_dir and self._metrics:
            metrics_path = Path(self.config.metrics_dir)
            metrics_path.mkdir(parents=True, exist_ok=True)
            sandbox_id = self._sandbox.id if self._sandbox else "unknown"
            metrics_file = metrics_path / f"sandbox_{sandbox_id}.json"
            metrics_data = {
                "sandbox_id": sandbox_id,
                "image": self.config.image,
                "create_time_s": round(self._create_time, 3),
                "total_exec_time_s": round(sum(m["elapsed_s"] for m in self._metrics), 3),
                "num_steps": len(self._metrics),
                "steps": self._metrics,
            }
            try:
                with open(metrics_file, "w") as f:
                    json.dump(metrics_data, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to save metrics: {e}")

        if self._sandbox and self._daytona:
            try:
                self._daytona.delete(self._sandbox)
                self.logger.info(f"Deleted Daytona sandbox: {self._sandbox.id}")
            except Exception as e:
                self.logger.warning(f"Failed to delete sandbox: {e}")

    def __del__(self):
        self.close()

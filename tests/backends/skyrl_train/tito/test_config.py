"""Tests for TITO TITOConfig."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.config import TITOConfig


class TestTITOConfig:
    """Tests for TITOConfig dataclass."""

    def test_defaults(self):
        cfg = TITOConfig()
        assert cfg.enabled is False
        assert cfg.tool_call_parser == "hermes"
        assert cfg.log_path is None
        assert cfg.prefix_check is True
        assert cfg.port == 0
        assert cfg.use_renderer is False
        assert cfg.renderer_name == "auto"

    def test_custom_values(self):
        cfg = TITOConfig(
            enabled=True,
            tool_call_parser="none",
            log_path="/tmp/tito",
            prefix_check=False,
            port=8080,
            use_renderer=True,
            renderer_name="qwen3",
        )
        assert cfg.enabled is True
        assert cfg.tool_call_parser == "none"
        assert cfg.log_path == "/tmp/tito"
        assert cfg.prefix_check is False
        assert cfg.port == 8080
        assert cfg.use_renderer is True
        assert cfg.renderer_name == "qwen3"

    def test_nested_in_inference_engine_config(self):
        """TITOConfig should be constructable via build_nested_dataclass."""
        from skyrl.train.config.config import InferenceEngineConfig

        ie_cfg = InferenceEngineConfig()
        assert isinstance(ie_cfg.tito, TITOConfig)
        assert ie_cfg.tito.enabled is False

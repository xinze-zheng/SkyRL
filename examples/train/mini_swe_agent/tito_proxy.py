"""Backwards-compatible import wrapper.

The TITO proxy has moved to ``skyrl.backends.skyrl_train.inference_servers.tito``.
This module re-exports the key classes for any code that imports from the old path.
"""

from skyrl.backends.skyrl_train.inference_servers.tito.config import TITOConfig  # noqa: F401
from skyrl.backends.skyrl_train.inference_servers.tito.proxy import TITOProxyActor as TITOProxy  # noqa: F401

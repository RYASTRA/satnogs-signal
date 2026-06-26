"""Read API tokens from the environment under the names this project uses.

Centralizing the lookup means tools pick up tokens explicitly (and pass them to
the relevant client/push call) instead of relying on a library auto-detecting a
specific variable name — so a `.env` using `huggingface_access_token` or
`satnogs_network_api_key` just works without shell-mapping.
"""
from __future__ import annotations

import os
from typing import Optional

# Accept several spellings; first non-empty match wins (most specific first).
_HF_TOKEN_VARS = ("huggingface_access_token", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token")
_SATNOGS_TOKEN_VARS = ("satnogs_network_api_key", "SATNOGS_NETWORK_API_KEY", "SATNOGS_API_TOKEN")


def _first_env(names) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def hf_token() -> Optional[str]:
    """Hugging Face Hub write token, or None if unset."""
    return _first_env(_HF_TOKEN_VARS)


def satnogs_token() -> Optional[str]:
    """SatNOGS Network API token, or None if unset."""
    return _first_env(_SATNOGS_TOKEN_VARS)

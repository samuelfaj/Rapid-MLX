"""Metal 4 / Neural Accelerator runtime detection.

Pure observability. Does NOT enable kernels yet.
"""
from __future__ import annotations

import functools
import logging
import os
import platform

import mlx.core as mx

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def device_name() -> str:
    try:
        return mx.device_info().get("device_name", "")
    except Exception:
        return ""


@functools.lru_cache(maxsize=1)
def macos_version() -> tuple[int, int]:
    parts = platform.mac_ver()[0].split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (0, 0)


@functools.lru_cache(maxsize=1)
def metal4_available() -> bool:
    if os.environ.get("LIGHTNING_DISABLE_METAL4") == "1":
        return False
    major, minor = macos_version()
    if (major, minor) < (26, 2):
        return False
    return True  # MLX 0.30+ enables NAX internally when supported


@functools.lru_cache(maxsize=1)
def m5_nax_hint() -> bool:
    if not metal4_available():
        return False
    return "M5" in device_name()


def log_capabilities() -> None:
    logger.info(
        "metal4=%s m5_nax=%s device=%s macos=%s",
        metal4_available(),
        m5_nax_hint(),
        device_name(),
        macos_version(),
    )

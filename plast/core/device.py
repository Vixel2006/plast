from __future__ import annotations

from typing import Any

import plast._plast_cpp_core as cpp_bindings

_DEFAULT_DEVICE = None


def set_default_device(device: "Device"):
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def get_default_device() -> "Device":
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None:
        _DEFAULT_DEVICE = Device("cpu")  # Default to CPU if not set
    return _DEFAULT_DEVICE


class Device:
    def __init__(self, type: str, index: int = 0):
        if not isinstance(type, str):
            raise TypeError("Device type must be a string.")
        if not isinstance(index, int) or index < 0:
            raise ValueError("Device index must be a non-negative integer.")

        self.type = type.lower()
        self.index = index

    @property
    def cpp_device_type(self) -> Any:
        if self.type == "cpu":
            return cpp_bindings.DeviceType.CPU
        elif self.type == "cuda":
            return cpp_bindings.DeviceType.CUDA
        else:
            raise ValueError(f"Unknown device type: {self.type}")

    def __repr__(self) -> str:
        if self.index == 0:
            return f"{self.type}"
        return f"{self.type}:{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return NotImplemented
        return self.type == other.type and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.type, self.index))

    @staticmethod
    def parse(device_str: str) -> "Device":
        parts = device_str.lower().split(":")
        device_type = parts[0]
        device_index = 0
        if len(parts) > 1:
            try:
                device_index = int(parts[1])
            except ValueError:
                raise ValueError(
                    f"Invalid device string format: {device_str}. Index must be an integer."
                )
        return Device(device_type, device_index)

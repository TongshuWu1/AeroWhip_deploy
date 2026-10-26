"""Windows XInput reader; no drone connection and no background thread."""

import ctypes
from ctypes import wintypes
import math
import sys

from aerowhip.physics.controller_interface import ControlInput


class _Pad(ctypes.Structure):
    _fields_ = [("buttons", wintypes.WORD), ("lt", ctypes.c_ubyte),
                ("rt", ctypes.c_ubyte), ("lx", ctypes.c_short),
                ("ly", ctypes.c_short), ("rx", ctypes.c_short),
                ("ry", ctypes.c_short)]


class _State(ctypes.Structure):
    _fields_ = [("packet", wintypes.DWORD), ("pad", _Pad)]


def _stick(x: int, y: int, deadzone: float = 0.20) -> tuple[float, float]:
    """Radial dead zone, rescaled continuously, with bounded diagonal input."""
    x = x / (32768 if x < 0 else 32767)
    y = y / (32768 if y < 0 else 32767)
    radius = math.hypot(x, y)
    if radius <= deadzone:
        return 0.0, 0.0
    scale = (min(radius, 1.0) - deadzone) / ((1.0 - deadzone) * radius)
    return x * scale, y * scale


class XboxGamepad:
    def __init__(self) -> None:
        self._get_state = None
        if sys.platform == "win32":
            for name in ("xinput1_4.dll", "xinput9_1_0.dll"):
                try:
                    self._dll = ctypes.WinDLL(name)
                    self._get_state = self._dll.XInputGetState
                    self._get_state.argtypes = [wintypes.DWORD, ctypes.POINTER(_State)]
                    self._get_state.restype = wintypes.DWORD
                    break
                except OSError:
                    continue
        self._index = None
        self._armed = False
        self._buttons = 0
        self.status = "Xbox disconnected"

    def disarm(self) -> None:
        self._armed = False
        if self._index is not None:
            self.status = "Release controls"

    def poll(self) -> ControlInput:
        if self._get_state is None:
            self.status = "Xbox unavailable"
            return ControlInput()
        state = _State()
        indices = range(4) if self._index is None else (self._index,)
        for index in indices:
            if self._get_state(index, ctypes.byref(state)) == 0:
                self._index = index
                break
        else:
            self._index = None
            self._armed = False
            self._buttons = 0
            self.status = "Xbox disconnected"
            return ControlInput()
        lx, ly = _stick(state.pad.lx, state.pad.ly)
        rx, ry = _stick(state.pad.rx, state.pad.ry)
        buttons = state.pad.buttons
        pressed = buttons & ~self._buttons
        self._buttons = buttons
        if not self._armed:
            self._armed = not any((lx, ly, rx, ry, buttons))
            self.status = ("Xbox connected" if self._armed else "Release controls")
            return ControlInput()
        self.status = "Xbox connected"
        return ControlInput(ly, lx, ry, rx, bool(pressed & 0x1000),
                            bool(pressed & 0x2000))

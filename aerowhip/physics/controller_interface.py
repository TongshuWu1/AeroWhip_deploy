"""Local controller boundary: normalized world XYZ/yaw input, not motor commands.

One controller connects over localhost TCP. Each newline-delimited JSON packet
contains axes (four values in [-1, 1]), toggle/reset booleans, and a monotonic
timestamp. Only fresh packets are applied; silence/disconnect returns neutral
input so the existing reference generator brakes to a hold. No physics lives
in this transport, and neither endpoint imports a GUI or a gamepad backend.
"""

from dataclasses import dataclass
import json
import math
import socket
from time import monotonic


DEFAULT_PORT = 47653
TIMEOUT_S = 0.35


@dataclass(frozen=True)
class ControlInput:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    toggle: bool = False
    reset: bool = False


class ControllerReceiver:
    def __init__(self, port=DEFAULT_PORT, *, running=lambda: False):
        self.listener = socket.socket()
        try:
            self.listener.bind(("127.0.0.1", port))
            self.listener.listen(1)
            self.listener.setblocking(False)
        except Exception:
            self.listener.close()
            raise
        self.running = running
        self.peer = None
        self.buffer = b""
        self.latest = ControlInput()
        self.last_time = 0.0
        self.armed = False

    def _disconnect(self):
        if self.peer is not None:
            self.peer.close()
        self.peer = None
        self.buffer = b""
        self.latest = ControlInput()
        self.armed = False

    def poll(self):
        now = monotonic()
        if self.peer is None:
            try:
                self.peer, _ = self.listener.accept()
                self.peer.setblocking(False)
                self.last_time = now
            except BlockingIOError:
                return ControlInput()
        toggle = reset = False
        try:
            # Bound work per viewer tick even if a client misbehaves.
            for _ in range(8):
                try:
                    data = self.peer.recv(4096)
                except BlockingIOError:
                    break
                if not data:
                    self._disconnect()
                    return ControlInput()
                self.buffer += data
                if len(self.buffer) > 16384:
                    raise ValueError("Controller packet too large")
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    packet = json.loads(line)
                    axes = packet["axes"]
                    stamp = packet["time"]
                    if (len(axes) != 4 or any(type(v) not in (int, float) or not math.isfinite(v)
                                             or abs(v) > 1 for v in axes)
                            or type(stamp) not in (int, float) or not math.isfinite(stamp)
                            or type(packet["toggle"]) is not bool or type(packet["reset"]) is not bool):
                        raise ValueError("Invalid controller packet")
                    if not 0 <= monotonic() - stamp <= TIMEOUT_S:
                        continue
                    self.last_time = now
                    if not self.armed:
                        self.armed = not any(axes) and not packet["toggle"] and not packet["reset"]
                        continue
                    self.latest = ControlInput(*axes)
                    toggle ^= packet["toggle"]
                    reset |= packet["reset"]
            if now - self.last_time > TIMEOUT_S:
                self._disconnect()
                return ControlInput()
            reply = json.dumps({"running": bool(self.running()), "armed": self.armed}) + "\n"
            self.peer.sendall(reply.encode())
        except (OSError, ValueError, TypeError, KeyError):
            self._disconnect()
            return ControlInput()
        v = self.latest
        return ControlInput(v.x, v.y, v.z, v.yaw, toggle, reset)

    def close(self):
        self._disconnect()
        self.listener.close()


class ControllerClient:
    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.peer = None
        self.buffer = b""
        self.retry_at = 0.0
        self.last_reply = 0.0
        self.running = False
        self.armed = False

    def send(self, value: ControlInput):
        now = monotonic()
        if self.peer is None:
            if now < self.retry_at:
                return False
            self.retry_at = now + 1.0
            try:
                # Loopback only, with a short bound during startup/reconnect.
                self.peer = socket.create_connection(("127.0.0.1", self.port), timeout=0.02)
                self.peer.setblocking(False)
                self.last_reply = now
            except OSError:
                self.close()
                return False
        try:
            packet = {"axes": [max(-1.0, min(1.0, v)) for v in (value.x, value.y, value.z, value.yaw)],
                      "toggle": value.toggle, "reset": value.reset, "time": now}
            self.peer.sendall((json.dumps(packet) + "\n").encode())
            for _ in range(8):
                try:
                    data = self.peer.recv(4096)
                except BlockingIOError:
                    break
                if not data:
                    raise ConnectionError("Simulator disconnected")
                self.buffer += data
                if len(self.buffer) > 16384:
                    raise ValueError("Simulator reply too large")
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    reply = json.loads(line)
                    self.running = bool(reply["running"])
                    self.armed = bool(reply["armed"])
                    self.last_reply = now
            if now - self.last_reply > TIMEOUT_S:
                raise ConnectionError("Simulator stopped responding")
        except (OSError, ValueError, TypeError, KeyError):
            self.close()
            return False
        return True

    def close(self):
        if self.peer is not None:
            self.peer.close()
        self.peer = None
        self.buffer = b""
        self.armed = False

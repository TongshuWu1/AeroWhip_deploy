"""Native pixel sizing for the standalone VTK window on Windows."""

import ctypes
from ctypes import wintypes
import sys


def enable_native_resolution() -> None:
    """Call before creating a window, avoiding Windows bitmap enlargement."""
    if sys.platform != "win32":
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
    # A host such as an IDE may already have established process awareness.
    if not user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
        error = ctypes.get_last_error()
        if error != 5:  # ERROR_ACCESS_DENIED: process mode already set.
            raise ctypes.WinError(error)
    user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    if not user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4)):
        raise ctypes.WinError(ctypes.get_last_error())


def configure_native_window(plotter, *, start_maximized: bool) -> None:
    """Place on the active monitor and maximize with normal window controls.

    Off-screen renders keep their explicit pixel dimensions and 96 DPI.
    """
    plotter.render_window.SetDPI(96)
    if sys.platform != "win32" or plotter.off_screen:
        return

    class MonitorInfo(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("monitor", wintypes.RECT),
                    ("work", wintypes.RECT), ("flags", wintypes.DWORD)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    user32.GetDpiForWindow.restype = wintypes.UINT
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL

    foreground = user32.GetForegroundWindow()
    monitor = user32.MonitorFromWindow(foreground, 2)  # Nearest monitor.
    info = MonitorInfo(size=ctypes.sizeof(MonitorInfo))
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    plotter.render_window.SetPosition(info.work.left + 32, info.work.top + 32)
    available = (info.work.right - info.work.left, info.work.bottom - info.work.top)
    plotter.window_size = tuple(min(size, int(space * 0.85))
                                for size, space in zip(plotter.window_size, available))

    first_render = True

    def update_window(current_plotter):
        nonlocal first_render
        # VTK exposes HWND as a SWIG pointer string, not a Python integer.
        pointer = current_plotter.render_window.GetGenericWindowId()
        if not pointer:
            return
        hwnd = wintypes.HWND(int(pointer.split("_")[1], 16))
        dpi = user32.GetDpiForWindow(hwnd)
        if dpi and current_plotter.render_window.GetDPI() != dpi:
            current_plotter.render_window.SetDPI(dpi)
        if first_render:
            first_render = False  # ShowWindow may trigger a nested resize/render.
            if start_maximized:
                user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE; retains title bar/taskbar.

    plotter.add_on_render_callback(update_window)

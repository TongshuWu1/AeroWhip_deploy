"""Virtual sticks and keyboard UI owned by the standalone gamepad controller."""

import math
import tkinter as tk
from tkinter import ttk

from simulation.controller_interface import ControlInput


class VirtualGamepad:
    MOVEMENT_KEYS = {"w", "a", "s", "d", "up", "down", "left", "right"}

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gamepad")
        self.root.resizable(False, False)
        self.closed = False
        self.keys = set()
        self.sticks = [[0.0, 0.0], [0.0, 0.0]]
        self._toggle = self._reset = False
        self.source = tk.StringVar(value="virtual")
        self._mode = "virtual"
        self._canvases = []
        self._knobs = []
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        sources = ttk.Frame(frame)
        sources.pack(fill="x")
        for label, value in (("Virtual", "virtual"), ("Xbox", "xbox")):
            ttk.Radiobutton(sources, text=label, variable=self.source, value=value,
                            command=self.clear).pack(side="left", padx=(0, 12))
        pads = ttk.Frame(frame)
        pads.pack(pady=12)
        for index, title in enumerate(("Move", "Height / Yaw")):
            column = ttk.Frame(pads)
            column.grid(row=0, column=index, padx=4)
            ttk.Label(column, text=title, font=("Segoe UI", 10, "bold")).pack()
            canvas = tk.Canvas(column, width=168, height=168, bg="#eef2f5",
                               highlightthickness=0, takefocus=True)
            canvas.pack(pady=6)
            canvas.create_oval(24, 24, 144, 144, outline="#9daebb", width=2)
            canvas.create_line(24, 84, 144, 84, fill="#c6d1da")
            canvas.create_line(84, 24, 84, 144, fill="#c6d1da")
            for x, y, label in ((84, 12, "+X" if index == 0 else "+Z"),
                                (84, 156, "-X" if index == 0 else "-Z"),
                                (12, 84, "-Y" if index == 0 else "-"),
                                (156, 84, "+Y" if index == 0 else "+")):
                canvas.create_text(x, y, text=label, fill="#455969")
            self._knobs.append(canvas.create_oval(71, 71, 97, 97, fill="#25838b", outline=""))
            self._canvases.append(canvas)
            canvas.bind("<ButtonPress-1>", lambda e, i=index: self._drag(i, e))
            canvas.bind("<B1-Motion>", lambda e, i=index: self._drag(i, e))
            canvas.bind("<ButtonRelease-1>", lambda e, i=index: self._release(i))
            canvas.bind("<Leave>", lambda e, i=index: self._release(i))
            ttk.Label(column, text="W A S D" if index == 0 else "Arrow keys").pack()
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=14)
        self.run_button = ttk.Button(buttons, text="Run", command=self._queue_toggle)
        self.run_button.pack(side="left", expand=True, fill="x", padx=(0, 8))
        ttk.Button(buttons, text="Reset", command=self._queue_reset).pack(side="left", expand=True, fill="x")
        self.status = tk.StringVar(value="Waiting for simulation")
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")
        self.root.bind("<KeyPress>", self._key_down)
        self.root.bind("<KeyRelease>", self._key_up)
        self.root.bind("<FocusOut>", lambda e: self.root.after_idle(self._check_focus))
        # Handle motion/action keys before Tk button/radio class bindings, so
        # Space cannot toggle twice and arrow keys cannot switch input modes.
        def prioritize_keys(widget):
            for child in widget.winfo_children():
                tags = child.bindtags()
                child.bindtags((str(self.root),) + tuple(t for t in tags if t != str(self.root)))
                prioritize_keys(child)
        prioritize_keys(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()

    def _queue_toggle(self):
        self._toggle = True

    def _queue_reset(self):
        self.clear()
        self._reset = True

    def _key_down(self, event):
        key = event.keysym.lower()
        if key in self.MOVEMENT_KEYS | {"space", "t"}:
            if key not in self.keys:
                if key == "space":
                    self._queue_toggle()
                elif key == "t":
                    self._queue_reset()
            self.keys.add(key)
            return "break"

    def _key_up(self, event):
        self.keys.discard(event.keysym.lower())
        return "break"

    def _drag(self, index, event):
        if self.source.get() != "virtual":
            return
        self._canvases[index].focus_set()
        x, y = (event.x - 84) / 60, (84 - event.y) / 60
        scale = max(1.0, math.hypot(x, y))
        self.sticks[index] = [x / scale, y / scale]
        self._draw(index)

    def _draw(self, index):
        x, y = self.sticks[index]
        cx, cy = 84 + x * 60, 84 - y * 60
        self._canvases[index].coords(self._knobs[index], cx-13, cy-13, cx+13, cy+13)

    def _release(self, index):
        self.sticks[index] = [0.0, 0.0]
        if not self.closed:
            self._draw(index)

    def clear(self):
        # Keep held action keys latched until KeyRelease (including auto-repeat).
        self.keys.difference_update(self.MOVEMENT_KEYS)
        self._toggle = self._reset = False
        for index in range(2):
            self._release(index)

    def _check_focus(self):
        if self.closed:
            return
        focus = self.root.focus_displayof()
        if focus is None or focus.winfo_toplevel() != self.root:
            self.clear()
            self.keys.clear()

    def poll(self) -> ControlInput:
        if self.closed:
            return ControlInput()
        self._check_focus()
        self._mode = self.source.get()
        k = self.keys
        lx, ly = self.sticks[0]
        rx, ry = self.sticks[1]
        # Keyboard and mouse share the same bounded downstream reference path.
        value = ControlInput(ly + ("w" in k) - ("s" in k),
                             lx + ("d" in k) - ("a" in k),
                             ry + ("up" in k) - ("down" in k),
                             rx + ("right" in k) - ("left" in k),
                             self._toggle, self._reset)
        self._toggle = self._reset = False
        return value

    @property
    def mode(self):
        return self._mode

    def close(self):
        if not self.closed:
            self.clear()
            self.closed = True
            self.root.destroy()

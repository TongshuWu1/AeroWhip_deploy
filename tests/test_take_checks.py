"""Clock-alignment estimate and take check on synthetic Motive/logger files."""
import csv
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import unittest

import numpy as np

from helpers import project_temporary
from aerowhip.takes import check_take, estimate_clock_alignment
from aerowhip.takes.take_data import load_take

OFFSET_S = 3.2173


def drone_path(t):
    """Hover with a sharp vertical bump at 2 s and a slow sway."""
    bump = .15*np.exp(-((t-2.)/.12)**2)
    return np.column_stack([.1*np.sin(.7*t), .05*np.cos(.4*t), 1.+bump])


def write_motive(path, *, frames=600, up="Z", drop_frame=None, missing_marker=None, stationary=False):
    t = np.arange(frames)/100.
    p = np.full((frames, 3), [0., 0., 1.]) if stationary else drone_path(t)
    keep = np.ones(frames, bool)
    if drop_frame is not None:
        keep[drop_frame] = False
    labels, kinds, axes, types = ["", "Name"], ["", ""], ["Frame", "Time (Seconds)"], ["", "Type"]
    columns = []
    for c, q in zip("XYZW", (0., 0., 0., 1.)):
        labels.append("cf_2"); kinds.append("Rotation"); axes.append(c); types.append("Rigid Body")
        columns.append(np.full(frames, q))
    for k, c in enumerate("XYZ"):
        labels.append("cf_2"); kinds.append("Position"); axes.append(c); types.append("Rigid Body")
        columns.append(p[:, k])
    for i in range(1, 11):
        hang = np.array([0., 0., -.1*i]) if up == "Z" else np.array([0., -.1*i, 0.])
        marker = p + hang
        if missing_marker == i:
            marker[100:150] = np.nan
        for k, c in enumerate("XYZ"):
            labels.append(f"cable:c{i}"); kinds.append("Position"); axes.append(c); types.append("Marker")
            columns.append(marker[:, k])
    header = ["Format Version", "1.23", "Capture Frame Rate", "100.000000", "Export Frame Rate", "100.000000",
              "Rotation Type", "Quaternion", "Length Units", "Meters", "Coordinate Space", "Global"]
    with path.open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerows([header, [], types, labels, ["", "ID"] + ["1"]*(len(labels)-2), kinds, axes])
        for i in np.flatnonzero(keep):
            out.writerow([i, f"{t[i]:.6f}"] + ["" if np.isnan(c[i]) else f"{c[i]:.6f}" for c in columns])


def write_logger(path, *, stationary=False):
    rng = np.random.default_rng(0)
    motive_t = np.arange(-1., 7., .01)
    p = np.full((len(motive_t), 3), [0., 0., 1.]) if stationary else drone_path(motive_t)
    receipt = motive_t + OFFSET_S + rng.uniform(0, .004, len(motive_t))  # 0-4 ms receipt latency
    with path.open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["time_s", "x", "y", "z", "pose_valid"])
        for ti, pi in zip(receipt, p):
            out.writerow([f"{ti:.6f}", *(f"{v:.6f}" for v in pi), 1])


def quiet(function, *args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return function(*args)


class TakeCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = project_temporary()
        self.take = Path(self.directory.name)/"batch/flight_take/T_001.csv"
        self.take.parent.mkdir(parents=True)

    def tearDown(self):
        self.directory.cleanup()

    def test_alignment_recovers_known_offset_and_is_applied(self):
        write_motive(self.take)
        write_logger(self.take.with_name("experiment_T_001.csv"))
        self.assertEqual(quiet(estimate_clock_alignment.main, [str(self.take)]), 0)
        saved = json.loads((self.take.parent.parent/"time_alignment.json").read_text())["T_001"]
        # The estimate includes receipt latency, here uniform 0-4 ms with a 2 ms mean.
        self.assertAlmostEqual(saved["offset_s"], OFFSET_S+.002, delta=.001)
        self.assertFalse(saved["clock_verified"])
        self.assertTrue(load_take(self.take).alignment_applied)
        self.assertEqual(quiet(estimate_clock_alignment.main, [str(self.take)]), 1)  # no silent overwrite

    def test_alignment_refuses_stationary_take(self):
        write_motive(self.take, stationary=True)
        write_logger(self.take.with_name("experiment_T_001.csv"), stationary=True)
        self.assertEqual(quiet(estimate_clock_alignment.main, [str(self.take)]), 1)
        self.assertFalse((self.take.parent.parent/"time_alignment.json").exists())

    def test_clean_take_is_usable(self):
        write_motive(self.take)
        report = check_take.check(self.take, min_window_s=1.8)
        self.assertEqual(report["blocking"], [])
        self.assertEqual(report["up_axis"]["dominant"], "-Z")
        self.assertEqual(report["timing"]["frame_jumps"], 0)

    def test_gap_dropout_and_up_axis_are_reported(self):
        write_motive(self.take, drop_frame=300, missing_marker=4)
        report = check_take.check(self.take, min_window_s=1.8)
        self.assertEqual(report["timing"]["frame_jumps"], 1)
        self.assertAlmostEqual(report["completeness"]["cable:c4"]["longest_dropout_s"], .5)
        self.assertAlmostEqual(report["usable_runs"]["longest"]["duration_s"], 2.98, places=6)
        write_motive(self.take, up="Y")
        self.assertTrue(any("expected -Z" in b for b in check_take.check(self.take, min_window_s=1.8)["blocking"]))


if __name__ == "__main__":
    unittest.main()

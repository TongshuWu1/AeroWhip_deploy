"""Report whether a Motive CSV export is usable for fitting, right after capture.

Checks export settings, native frame rate, frame/timestamp gaps, labels,
per-marker completeness, the hanging-cable up-axis and the longest complete
native-rate run. It reads only; nothing is cleaned, resampled or written.
Exit status is 1 when a blocking problem is found.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

from aerowhip.paths import ROOT
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from aerowhip.takes.paths import resolve_recording_path
from aerowhip.takes.take_data import load_take

RATE_HZ = 100.
STEP_TOLERANCE_S = 1e-6
EXPECTED_EXPORT = {"Length Units": "Meters", "Coordinate Space": "Global", "Rotation Type": "Quaternion"}
CABLE = [f"cable:c{i}" for i in range(1, 11)]


def read_export(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    metadata = dict(zip(rows[0][::2], rows[0][1::2]))
    h = next(i for i, row in enumerate(rows) if row[:2] == ["Frame", "Time (Seconds)"])
    types, labels, kinds, axes = rows[h-4], rows[h-3], rows[h-1], rows[h]
    width = len(axes)
    data = np.array([[float(v) if v.strip() else np.nan for v in (row+[""]*width)[:width]]
                     for row in rows[h+1:] if row and row[0].strip()])
    fields = {}
    for i in range(2, width):
        fields.setdefault((labels[i], kinds[i]), {})[axes[i]] = i
    assets = {}
    for i in range(2, width):
        assets.setdefault(labels[i], types[i] if i < len(types) else "")
    return metadata, assets, fields, data


def block(fields, data, label, kind, components):
    ids = fields.get((label, kind), {})
    if not all(c in ids for c in components):
        return None
    return data[:, [ids[c] for c in components]]


def runs(mask):
    """Inclusive [first, last] index pairs of consecutive True values."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1))


def longest_false(mask, step):
    return max(((b-a+1)*step for a, b in runs(~mask)), default=0.)


def check(path, *, min_window_s):
    path = resolve_recording_path(path)
    metadata, assets, fields, data = read_export(path)
    blocking, warnings = [], []
    report = {"take": path.name}

    export = {key: metadata.get(key) for key in (*EXPECTED_EXPORT, "Capture Frame Rate", "Export Frame Rate",
                                                 "Format Version", "Capture Start Time", "Total Frames in Take",
                                                 "Total Exported Frames")}
    report["export"] = export
    for key, expected in EXPECTED_EXPORT.items():
        if export[key] != expected:
            blocking.append(f"export {key} is {export[key]!r}, expected {expected!r}")
    for key in ("Capture Frame Rate", "Export Frame Rate"):
        try:
            if float(export[key]) != RATE_HZ:
                blocking.append(f"{key} is {export[key]}, expected {RATE_HZ:g}")
        except (TypeError, ValueError):
            blocking.append(f"{key} missing or unreadable: {export[key]!r}")

    frame, time = data[:, 0], data[:, 1]
    dt, dframe = np.diff(time), np.diff(frame)
    step = 1/RATE_HZ
    off_step = np.abs(dt-step) > STEP_TOLERANCE_S
    jumps = dframe != 1
    report["timing"] = {
        "frames": int(len(frame)), "duration_s": float(time[-1]-time[0]),
        "median_rate_hz": float(1/np.median(dt)), "min_step_s": float(dt.min()), "max_step_s": float(dt.max()),
        "frame_jumps": int(jumps.sum()), "off_nominal_steps": int(off_step.sum()),
        "first_problems": [{"time_s": float(time[i]), "frame": int(frame[i]), "next_frame": int(frame[i+1]),
                            "step_s": float(dt[i])} for i in np.flatnonzero(jumps | off_step)[:10]],
    }
    if not np.isfinite(time).all() or np.any(dt <= 0):
        blocking.append("timestamps are not finite and strictly increasing")
    if jumps.any() or off_step.any():
        warnings.append(f"{int(jumps.sum())} frame jumps and {int(off_step.sum())} off-nominal steps; "
                        "fitting windows must avoid them")

    drones = sorted(label for label in assets if re.fullmatch(r"cf_?\d+", label, re.I))
    prefixes = sorted({label.split(":")[0] for label in assets
                       if re.fullmatch(r"[^:]+:c\d+", label) and not label.startswith(tuple(drones))})
    unlabeled = [label for label in assets if label.lower().startswith("unlabeled")]
    expected = {"", *drones, *CABLE}
    report["labels"] = {
        "drones": drones, "cable_prefixes": prefixes,
        "cable_markers_found": [label for label in CABLE if label in assets],
        "unlabeled_markers": len(unlabeled),
        "rigid_body_markers": len([label for label, kind in assets.items() if kind == "Rigid Body Marker"]),
        "other_assets": sorted(label for label in assets
                               if label not in expected and label not in unlabeled
                               and assets[label] != "Rigid Body Marker"),
    }
    if len(drones) != 1:
        blocking.append(f"expected one drone rigid body named cf_N, found {drones}")
    missing_cable = [label for label in CABLE if label not in assets]
    if missing_cable:
        blocking.append(f"missing cable labels: {missing_cable}")
    if prefixes not in (["cable"], []):
        warnings.append(f"cable marker prefixes {prefixes}; fitting expects exactly cable:c1..c10")

    complete = np.ones(len(frame), bool)
    completeness = {}
    if len(drones) == 1:
        position = block(fields, data, drones[0], "Position", "XYZ")
        rotation = block(fields, data, drones[0], "Rotation", "XYZW")
        if position is None or rotation is None:
            blocking.append(f"{drones[0]} lacks Position XYZ or Rotation XYZW columns")
            pose = np.zeros(len(frame), bool)
        else:
            pose = (np.isfinite(position).all(1) & np.isfinite(rotation).all(1)
                    & (np.linalg.norm(np.nan_to_num(rotation), axis=1) > 1e-12))
        completeness[drones[0]] = {"percent": float(100*pose.mean()), "longest_dropout_s": longest_false(pose, step)}
        complete &= pose
    markers = {}
    for label in CABLE:
        values = block(fields, data, label, "Position", "XYZ")
        valid = np.isfinite(values).all(1) if values is not None else np.zeros(len(frame), bool)
        markers[label] = values
        completeness[label] = {"percent": float(100*valid.mean()), "longest_dropout_s": longest_false(valid, step)}
        complete &= valid
    completeness["all_complete_percent"] = float(100*complete.mean())
    report["completeness"] = completeness

    # A window is usable only if every sample is complete and every step is one native frame.
    breaks = jumps | off_step
    spans = []
    for a, b in runs(complete):
        cuts = [a, *(np.flatnonzero(breaks[a:b])+a+1), b+1]
        spans += [(start, end-1) for start, end in zip(cuts[:-1], cuts[1:])]
    durations = [float(time[b]-time[a]) for a, b in spans]
    longest = int(np.argmax(durations)) if spans else None
    report["usable_runs"] = {
        "min_window_s": min_window_s,
        "count_at_least_min": int(sum(d >= min_window_s for d in durations)),
        "total_at_least_min_s": float(sum(d for d in durations if d >= min_window_s)),
        "longest": None if longest is None else {"start_s": float(time[spans[longest][0]]),
                                                 "end_s": float(time[spans[longest][1]]),
                                                 "duration_s": durations[longest]},
    }
    if not any(d >= min_window_s for d in durations):
        blocking.append(f"no complete native-rate run of at least {min_window_s} s")

    if len(drones) == 1 and position is not None and markers["cable:c10"] is not None:
        hanging = np.isfinite(position).all(1) & np.isfinite(markers["cable:c10"]).all(1)
        if hanging.any():
            direction = np.median(markers["cable:c10"][hanging]-position[hanging], axis=0)
            axis = int(np.argmax(np.abs(direction)))
            report["up_axis"] = {"median_drone_to_tip_m": direction.tolist(),
                                 "dominant": ("+" if direction[axis] > 0 else "-")+"XYZ"[axis]}
            if report["up_axis"]["dominant"] != "-Z":
                blocking.append(f"cable tip hangs along {report['up_axis']['dominant']}, expected -Z; "
                                "export with Z up")

    try:
        take = load_take(path)
        report["load_take"] = {"ok": True, "alignment_applied": take.alignment_applied, "status": take.status}
        if not take.alignment_applied:
            warnings.append("no saved clock alignment yet; run python -m aerowhip.takes.estimate_clock_alignment")
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        report["load_take"] = {"ok": False, "error": str(exc)}
        blocking.append(f"load_take rejects the export: {exc}")

    logger = path.with_name("experiment_"+path.name)
    report["logger_files"] = {name: logger.with_suffix(suffix).is_file() for name, suffix in
                              (("samples", ".csv"), ("commands", ".commands.csv"), ("motors", ".motors.csv"))}
    if not report["logger_files"]["samples"]:
        warnings.append(f"no logger file {logger.name} beside the take")
    report["blocking"], report["warnings"] = blocking, warnings
    return report


def show(report):
    t, u = report["timing"], report["usable_runs"]
    lines = [f"Take {report['take']}",
             "Export: " + ", ".join(f"{k}={v}" for k, v in report["export"].items()),
             f"Timing: {t['frames']} frames, {t['duration_s']:.2f} s, median {t['median_rate_hz']:.3f} Hz, "
             f"step {t['min_step_s']*1e3:.3f}-{t['max_step_s']*1e3:.3f} ms, "
             f"{t['frame_jumps']} frame jumps, {t['off_nominal_steps']} off-nominal steps"]
    lines += [f"  gap at {p['time_s']:.3f} s: frame {p['frame']} -> {p['next_frame']}, step {p['step_s']*1e3:.2f} ms"
              for p in t["first_problems"]]
    lab = report["labels"]
    lines.append(f"Labels: drones {lab['drones']}, cable {len(lab['cable_markers_found'])}/10 "
                 f"(prefixes {lab['cable_prefixes']}), {lab['unlabeled_markers']} unlabeled, "
                 f"{lab['rigid_body_markers']} rigid-body markers, other {lab['other_assets']}")
    lines.append("Completeness:")
    for label, value in report["completeness"].items():
        if isinstance(value, dict):
            lines.append(f"  {label:12s} {value['percent']:6.2f} %   longest dropout {value['longest_dropout_s']:.2f} s")
    lines.append(f"  all present  {report['completeness']['all_complete_percent']:6.2f} %")
    longest = u["longest"]
    lines.append(f"Usable runs >= {u['min_window_s']} s: {u['count_at_least_min']} "
                 f"({u['total_at_least_min_s']:.2f} s); longest "
                 + (f"{longest['duration_s']:.2f} s at {longest['start_s']:.2f}-{longest['end_s']:.2f} s"
                    if longest else "none"))
    if "up_axis" in report:
        lines.append(f"Hanging cable direction: {report['up_axis']['dominant']}")
    lines.append(f"load_take: {report['load_take']}")
    lines.append(f"Logger files: {report['logger_files']}")
    lines += [f"WARNING: {w}" for w in report["warnings"]]
    lines += [f"BLOCKING: {b}" for b in report["blocking"]]
    lines.append("RESULT: " + ("NOT USABLE" if report["blocking"] else "usable for fitting preparation"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", type=Path, help="Motive CSV export")
    parser.add_argument("--min-window", type=float, default=1.8,
                        help="Required complete run, s (default: 1.7 s fit window + 0.1 s history)")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)
    report = check(args.take, min_window_s=args.min_window)
    print(json.dumps(report, indent=2) if args.json else show(report))
    return 1 if report["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

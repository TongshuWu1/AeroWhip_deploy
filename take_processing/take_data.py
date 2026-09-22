"""Read-only native recording and explicitly linked reference loading."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(path, expected):
    if not expected or digest(path) != expected:
        raise ValueError(f"Unverified source: {path.name}")


def valid_time(t):
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")


@dataclass
class Take:
    path: Path
    time: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    cable: np.ndarray
    command_time: np.ndarray | None = None
    command: np.ndarray | None = None
    desired_time: np.ndarray | None = None
    desired: np.ndarray | None = None
    status: str = ""
    frame: np.ndarray | None = None
    source_time: np.ndarray | None = None
    alignment: dict | None = None
    alignment_applied: bool = False
    alignment_error: str = ""
    command_breaks: np.ndarray | None = None


def discover(folder):
    folder = Path(folder).resolve()
    search = folder / "data" if (folder / "data").is_dir() else folder
    candidates = []
    for path in search.rglob("*.csv"):
        if path.name.startswith("experiment_"):
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            if "Format Version" in header and "Capture Frame Rate" in header:
                candidates.append(path.resolve())
        except (OSError, UnicodeError, csv.Error):
            continue
    return sorted(set(candidates))


def load_take(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    metadata = dict(zip(rows[0][::2], rows[0][1::2]))
    if (metadata.get("Length Units") != "Meters" or
            metadata.get("Coordinate Space") != "Global" or
            metadata.get("Rotation Type") != "Quaternion"):
        raise ValueError("Expected global, metre, quaternion Motive export")
    h = next(i for i, row in enumerate(rows) if row[:2] == ["Frame", "Time (Seconds)"])
    labels, kinds, axes = rows[h-3], rows[h-1], rows[h]

    def columns(label, kind, components):
        result = []
        for component in components:
            ids = [i for i, name in enumerate(labels)
                   if name == label and kinds[i] == kind and axes[i] == component]
            if len(ids) != 1:
                raise ValueError(f"Missing or ambiguous field: {label}/{kind}/{component}")
            result.extend(ids)
        return result

    identity = read_json(path.with_suffix(".tracking.json"))
    if identity:
        checked_hash(path, identity.get("optitrack_sha256"))
        drone = identity["drone"]
    else:
        candidates = sorted({label for label in labels if re.fullmatch(r"cf_?\d+", label, re.I)})
        if len(candidates) != 1:
            raise ValueError(f"Drone identity is ambiguous: {candidates}")
        drone = candidates[0]
    prefixes = [prefix for prefix in ("cable1", "cable")
                if all(f"{prefix}:c{i}" in labels for i in range(1, 11))]
    if len(prefixes) != 1:
        raise ValueError("Expected one set of ten named cable markers")
    ids = [0, 1] + columns(drone, "Position", "XYZ") + columns(drone, "Rotation", "XYZW")
    ids += [j for i in range(1, 11) for j in columns(f"{prefixes[0]}:c{i}", "Position", "XYZ")]
    a = np.array([[float(row[i]) if i < len(row) and row[i].strip() else np.nan
                   for i in ids] for row in rows[h+1:] if row and row[0].strip()])
    valid_time(a[:, 1])
    if (not np.isfinite(a[:, 0]).all() or np.any(a[:, 0] != np.floor(a[:, 0]))
            or np.any(np.diff(a[:, 0]) <= 0)):
        raise ValueError("Invalid frame sequence")
    origin = a[0, 1]
    result = Take(path, a[:, 1]-origin, a[:, 2:5], a[:, 5:9], a[:, 9:].reshape(-1, 10, 3))
    result.frame = a[:, 0].astype(np.int64)
    result.source_time = a[:, 1].copy()
    notes = []
    batch = path.parent.parent
    logger = path.with_name("experiment_" + path.name)
    alignment = read_json(batch / "time_alignment.json").get(path.stem)
    result.alignment = alignment
    if not alignment:
        result.status = "Command and desired unavailable: no saved clock alignment"
        return result
    try:
        checked_hash(path, alignment.get("optitrack_sha256"))
        checked_hash(logger, alignment.get("controller_sha256"))
        offset = float(alignment["offset_s"])
        if not np.isfinite(offset):
            raise ValueError("Invalid alignment offset")
    except (OSError, ValueError) as exc:
        result.alignment_error = str(exc)
        result.status = str(exc) + "; measured only"
        return result
    result.alignment_applied = True
    notes.append("Saved clock alignment" if alignment.get("clock_verified") else "Estimated clock alignment")
    command_path = logger.with_suffix(".commands.csv")
    if command_path.is_file():
        try:
            commands = np.atleast_1d(np.genfromtxt(command_path, delimiter=",", names=True, encoding="utf-8-sig"))
            valid_time(commands["time_s"])
            positions = np.column_stack([commands[f"cmd_{axis}"] for axis in "xyz"])
            positions[commands["cmd_valid"] != 1] = np.nan
            result.command_time = commands["time_s"]-offset-origin
            result.command = positions
            if "cmd_sequence" in commands.dtype.names:
                result.command_breaks = np.diff(commands["cmd_sequence"]) != 1
                if result.command_breaks.any():
                    notes.append("Command sequence has gaps or duplicates")
        except (OSError, ValueError, KeyError) as exc:
            notes.append(f"Command unavailable: {exc}")
    else:
        notes.append("Command unavailable: no event log")
    protocol = read_json(batch / "protocol.json")
    segment = read_json(logger.with_suffix(".metadata.json")).get("segment", {})
    if protocol.get("rehearsal") and "onset_s" in segment:
        rehearsal = Path(protocol["rehearsal"])
        if not rehearsal.is_absolute():
            root = next((p for p in batch.parents if (p / "data").is_dir()), batch)
            rehearsal = root / rehearsal
        archive = rehearsal / "rehearsal.npz"
        try:
            checked_hash(archive, protocol.get("forecast_sha256"))
            with np.load(archive, allow_pickle=False) as data:
                if {"reference_time_s", "reference_origin_positions_m"}.issubset(data.files):
                    t = data["reference_time_s"].copy()
                    p = data["reference_origin_positions_m"].copy()
                    valid_time(t)
                    if p.shape != (len(t), 3) or not np.isfinite(p).all():
                        raise ValueError("Invalid desired reference")
                    onset = float(segment["onset_s"])
                    if not np.isfinite(onset):
                        raise ValueError("Invalid reference onset")
                    result.desired_time = t + onset-offset-origin
                    result.desired = p
        except (OSError, ValueError) as exc:
            notes.append(str(exc))
    if result.desired is None:
        notes.append("Desired unavailable: no linked, timed reference")
    result.status = " · ".join(notes)
    return result

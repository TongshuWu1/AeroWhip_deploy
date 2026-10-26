"""Estimate one take's logger-receipt minus Motive-source clock offset.

``offset_s`` satisfies logger ``time_s`` = Motive ``Time (Seconds)`` + offset_s,
the convention read by take_data.load_take and fitting.prepare_window. It is
estimated from drone position alone, so it includes transport/receipt latency
and is saved with ``clock_verified: false``. Measurements are never modified;
the logger poses are interpolated only to evaluate candidate offsets.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from fitting.artifacts import atomic_save
from take_processing.paths import resolve_recording_path
from take_processing.take_data import digest, load_take, read_json, valid_time

METHOD = "position_least_squares_v1"
COARSE_STEP_S = .01
FINE_STEP_S = 1e-4
MAX_INTERPOLATION_GAP_S = .05


def logger_poses(path):
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig"))
    missing = {"time_s", "x", "y", "z"} - set(rows.dtype.names)
    if missing:
        raise ValueError(f"Logger lacks columns: {sorted(missing)}")
    t = rows["time_s"]
    p = np.column_stack([rows[a] for a in "xyz"])
    keep = np.isfinite(t) & np.isfinite(p).all(1)
    if "pose_valid" in rows.dtype.names:
        keep &= rows["pose_valid"] == 1
    t, p = t[keep], p[keep]
    valid_time(t)
    return t, p


def residuals(lag, motive_t, motive_p, logger_t, logger_p):
    """Position error at Motive samples whose shifted time has bracketing logger poses."""
    q = motive_t + lag
    i = np.searchsorted(logger_t, q)
    inside = (i > 0) & (i < len(logger_t))
    inside[inside] &= logger_t[i[inside]] - logger_t[i[inside]-1] <= MAX_INTERPOLATION_GAP_S
    q = q[inside]
    estimate = np.column_stack([np.interp(q, logger_t, logger_p[:, k]) for k in range(3)])
    return np.linalg.norm(estimate - motive_p[inside], axis=1), inside


def estimate(take, logger_t, logger_p, *, max_lag_s, min_overlap_fraction, exclusion_s):
    finite = np.isfinite(take.position).all(1)
    motive_t, motive_p = take.source_time[finite], take.position[finite]
    if len(motive_t) < 2:
        raise ValueError("Motive export has no finite drone positions")
    shortest = min(motive_t[-1]-motive_t[0], logger_t[-1]-logger_t[0])
    needed = min_overlap_fraction*shortest
    # Centre the search on aligned starts so max_lag bounds the start mismatch, not absolute clocks.
    centre = logger_t[0] - motive_t[0]
    lags = centre + np.arange(-max_lag_s, max_lag_s+COARSE_STEP_S/2, COARSE_STEP_S)

    def score(lag):
        error, inside = residuals(lag, motive_t, motive_p, logger_t, logger_p)
        covered = motive_t[inside]
        if len(error) < 2 or covered[-1]-covered[0] < needed:
            return np.inf
        return float(np.sqrt(np.mean(error**2)))

    coarse = np.array([score(lag) for lag in lags])
    if not np.isfinite(coarse).any():
        raise ValueError("No candidate offset gives the required overlap; increase --max-lag")
    best = int(np.argmin(coarse))
    if best in (0, len(lags)-1):
        raise ValueError("Best offset lies on the search boundary; increase --max-lag")
    fine_lags = lags[best] + np.arange(-COARSE_STEP_S, COARSE_STEP_S+FINE_STEP_S/2, FINE_STEP_S)
    fine = np.array([score(lag) for lag in fine_lags])
    offset = float(fine_lags[np.argmin(fine)])
    far = np.abs(lags - offset) > exclusion_s
    runner = int(np.flatnonzero(far)[np.argmin(coarse[far])]) if far.any() else None
    error, inside = residuals(offset, motive_t, motive_p, logger_t, logger_p)
    covered = motive_t[inside]
    z, zt = motive_p[:, 2], motive_t
    vertical_speed = np.abs(np.diff(z)/np.diff(zt))
    return {
        "offset_s": offset,
        "residual_rms_m": float(np.sqrt(np.mean(error**2))),
        "residual_max_m": float(error.max()),
        "residual_p95_m": float(np.percentile(error, 95)),
        "compared_samples": int(len(error)),
        "overlap_s": float(covered[-1]-covered[0]),
        "runner_up_offset_s": float(lags[runner]) if runner is not None else None,
        "runner_up_rms_m": float(coarse[runner]) if runner is not None and np.isfinite(coarse[runner]) else None,
        "search_s": [float(lags[0]), float(lags[-1])],
        "vertical_range_m": float(z.max()-z.min()),
        "peak_vertical_speed_mps": float(vertical_speed.max(initial=0)),
    }


def check(result, *, max_rms_m, min_contrast):
    problems = []
    if result["residual_rms_m"] > max_rms_m:
        problems.append(f"residual RMS {result['residual_rms_m']:.4f} m exceeds {max_rms_m} m "
                        "(check up axis/frame, drone identity, or that the files belong together)")
    runner = result["runner_up_rms_m"]
    if runner is not None and runner <= min_contrast*result["residual_rms_m"]:
        problems.append(f"ambiguous: offset {result['runner_up_offset_s']:.3f} s fits nearly as well "
                        f"({runner:.4f} m); add a sharp vertical bump to the take")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", type=Path, help="Motive CSV export <batch>/flight_take/<take>.csv")
    parser.add_argument("--logger", type=Path, help="Logger CSV; default experiment_<take>.csv beside the take")
    parser.add_argument("--max-lag", type=float, default=30., help="Search half-width around aligned starts, s")
    parser.add_argument("--max-rms", type=float, default=.01, help="Largest accepted position residual RMS, m")
    parser.add_argument("--min-contrast", type=float, default=2.,
                        help="Required ratio of the best distant offset's RMS to the best RMS")
    parser.add_argument("--exclusion", type=float, default=.25, help="Distance defining a distant offset, s")
    parser.add_argument("--min-overlap", type=float, default=.5,
                        help="Required overlap as a fraction of the shorter recording")
    parser.add_argument("--replace", action="store_true", help="Replace an existing entry for this take")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing time_alignment.json")
    args = parser.parse_args(argv)

    path = resolve_recording_path(args.take)
    logger = resolve_recording_path(args.logger) if args.logger else path.with_name("experiment_"+path.name)
    take = load_take(path)
    logger_t, logger_p = logger_poses(logger)
    result = estimate(take, logger_t, logger_p, max_lag_s=args.max_lag,
                      min_overlap_fraction=args.min_overlap, exclusion_s=args.exclusion)
    problems = check(result, max_rms_m=args.max_rms, min_contrast=args.min_contrast)
    print(json.dumps(result, indent=2))
    if problems:
        print("Not saved:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    target = path.parent.parent/"time_alignment.json"
    document = read_json(target)
    if path.stem in document and not args.replace:
        print(f"Not saved: {target.name} already has {path.stem}; use --replace", file=sys.stderr)
        return 1
    document[path.stem] = {
        "offset_s": result["offset_s"],
        "optitrack_sha256": digest(path),
        "controller_sha256": digest(logger),
        "clock_verified": False,
        "method": METHOD,
        "convention": "logger time_s = Motive Time (Seconds) + offset_s; includes receipt latency",
        "tracking_file": path.name,
        "logger_file": logger.name,
        "estimate": {key: value for key, value in result.items() if key != "offset_s"},
        "acceptance": {"max_rms_m": args.max_rms, "min_contrast": args.min_contrast,
                       "exclusion_s": args.exclusion, "min_overlap_fraction": args.min_overlap},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    text = json.dumps(document, indent=2, allow_nan=False)+"\n"
    atomic_save(target, lambda temporary: temporary.write_text(text, encoding="utf-8"))
    print(f"Saved {path.stem} in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

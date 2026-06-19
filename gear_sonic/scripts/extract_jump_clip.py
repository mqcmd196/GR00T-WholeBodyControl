#!/usr/bin/env python3
"""Cut a jump (or any) clip out of a pico_manager recording using controller buttons.

`pico_manager_thread_server.py --record_dir DIR` writes one ``pose_NNNNNN.npz``
per *sent* pose chunk. Each chunk holds a sliding window of motion frames
(``smpl_pose``/``smpl_joints``/``body_quat_w``/``joint_pos``/...) plus the
*instantaneous* controller state at send time (``left_trigger``/``right_trigger``/
``left_grip``/``right_grip``/...).

This tool scans those chunks in order and keeps the segment between a LEFT-button
press (clip start) and a RIGHT-button press (clip end), copying the selected
chunks into an output directory with ``frame_index`` rebased to 0. The result is
a self-contained clip that ``replay_jump_clip.py`` re-sends verbatim over ZMQ.

Example:
    # record on the robot first:
    #   python gear_sonic/scripts/pico_manager_thread_server.py --manager --record_dir /tmp/rec
    # then cut the jump (press left trigger before, right trigger after):
    python gear_sonic/scripts/extract_jump_clip.py --record_dir /tmp/rec --out /tmp/jump_clip
    # inspect button timeline without cutting:
    python gear_sonic/scripts/extract_jump_clip.py --record_dir /tmp/rec --list
"""

import argparse
import glob
import os

import numpy as np

PRESS_FIELDS = ("left_trigger", "right_trigger", "left_grip", "right_grip")


def _chunk_paths(record_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(record_dir, "pose_*.npz")))
    if not paths:
        raise SystemExit(f"[extract] No pose_*.npz files found in {record_dir!r}")
    return paths


def _scalar(chunk: dict, field: str) -> float:
    """Read a per-chunk scalar field (stored as a length-1 array); 0.0 if absent."""
    if field not in chunk:
        return 0.0
    v = np.asarray(chunk[field]).ravel()
    return float(v[0]) if v.size else 0.0


def _rising_edge(values: list[float], thresh: float, after: int = -1) -> int | None:
    """Index of the first rising edge (below->>=thresh) strictly after ``after``."""
    prev = False
    for i, v in enumerate(values):
        pressed = v >= thresh
        if i > after and pressed and not prev:
            return i
        prev = pressed
    return None


def _load_chunk(path: str) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record_dir", required=True, help="Directory with pose_*.npz recordings")
    ap.add_argument("--out", default=None, help="Output clip dir (default: <record_dir>/jump_clip)")
    ap.add_argument("--start-field", default="left_trigger", choices=PRESS_FIELDS,
                    help="Controller field whose press marks the clip START (default: left_trigger)")
    ap.add_argument("--end-field", default="right_trigger", choices=PRESS_FIELDS,
                    help="Controller field whose press marks the clip END (default: right_trigger)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Press threshold (default: 0.5)")
    ap.add_argument("--range", type=int, nargs=2, metavar=("A", "B"), default=None,
                    help="Override markers: keep chunk indices [A, B] inclusive")
    ap.add_argument("--list", action="store_true",
                    help="Only print the per-chunk button timeline and exit")
    args = ap.parse_args()

    paths = _chunk_paths(args.record_dir)
    chunks = [_load_chunk(p) for p in paths]
    n = len(chunks)
    print(f"[extract] {n} chunks in {args.record_dir}")

    start_vals = [_scalar(c, args.start_field) for c in chunks]
    end_vals = [_scalar(c, args.end_field) for c in chunks]

    if args.list:
        print(f"{'idx':>5} {'frames':>8} {args.start_field:>14} {args.end_field:>14}")
        for i, c in enumerate(chunks):
            nf = int(np.asarray(c["frame_index"]).shape[0]) if "frame_index" in c else 0
            print(f"{i:5d} {nf:8d} {start_vals[i]:14.3f} {end_vals[i]:14.3f}")
        return

    if args.range is not None:
        a, b = args.range
        if not (0 <= a <= b < n):
            raise SystemExit(f"[extract] --range {a} {b} out of bounds [0, {n - 1}]")
        start_idx, end_idx = a, b
    else:
        start_idx = _rising_edge(start_vals, args.threshold)
        if start_idx is None:
            raise SystemExit(
                f"[extract] No START press detected on '{args.start_field}' "
                f"(>= {args.threshold}). Use --list to inspect or --range A B."
            )
        end_idx = _rising_edge(end_vals, args.threshold, after=start_idx)
        if end_idx is None:
            raise SystemExit(
                f"[extract] No END press detected on '{args.end_field}' after chunk "
                f"{start_idx}. Use --list to inspect or --range A B."
            )

    selected = chunks[start_idx:end_idx + 1]
    print(f"[extract] keeping chunks [{start_idx}, {end_idx}] -> {len(selected)} chunks")

    # Rebase frame_index so the clip starts at 0 (preserves overlap between chunks).
    min_fi = min(int(np.asarray(c["frame_index"]).min()) for c in selected if "frame_index" in c)

    out_dir = args.out or os.path.join(args.record_dir, "jump_clip")
    os.makedirs(out_dir, exist_ok=True)
    for j, c in enumerate(selected):
        out = dict(c)
        if "frame_index" in out:
            out["frame_index"] = (np.asarray(out["frame_index"]).astype(np.int64) - min_fi)
        np.savez(os.path.join(out_dir, f"pose_{j:06d}.npz"), **out)

    span = max(int(np.asarray(c["frame_index"]).max()) for c in selected if "frame_index" in c) - min_fi + 1
    print(f"[extract] wrote {len(selected)} chunks ({span} unique frames) to {out_dir}")
    print(f"[extract] replay with: python gear_sonic/scripts/replay_jump_clip.py --clip {out_dir}")


if __name__ == "__main__":
    main()

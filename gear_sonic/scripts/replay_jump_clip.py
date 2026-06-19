#!/usr/bin/env python3
"""Replay a recorded/extracted pose clip over ZMQ to the deploy's ZMQManager.

This makes the robot reproduce a motion (e.g. a jump) without a Pico headset.
It mimics ``pico_manager_thread_server.py``: bind a PUB socket, switch the deploy
to streamed-motion mode via a ``command`` message, then resend each recorded
pose chunk verbatim at the original cadence. Because the chunks are re-sent
exactly as recorded, the wire format matches production with no reconstruction.

The deploy must be running with ``--input-type manager`` (e.g. ``./deploy.sh sim``)
and connected to this publisher's port (default 5556).

Example:
    python gear_sonic/scripts/replay_jump_clip.py --clip /tmp/jump_clip
    python gear_sonic/scripts/replay_jump_clip.py --clip /tmp/jump_clip --loop 3 --fps 50
"""

import argparse
import glob
import os
import time

import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


def _load_clip(clip_dir: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(clip_dir, "pose_*.npz")))
    if not paths:
        raise SystemExit(f"[replay] No pose_*.npz found in {clip_dir!r}")
    chunks = []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            chunks.append({k: d[k] for k in d.files})
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="Clip directory of pose_*.npz (from extract_jump_clip.py)")
    ap.add_argument("--host", default="*", help="Bind host for the PUB socket (default: *)")
    ap.add_argument("--port", type=int, default=5556, help="ZMQ manager port (default: 5556)")
    ap.add_argument("--fps", type=float, default=50.0, help="Replay rate in Hz (default: 50)")
    ap.add_argument("--loop", type=int, default=1, help="How many times to replay the clip (default: 1)")
    ap.add_argument("--no-start", action="store_true",
                    help="Do not send the start/streamed-motion command (assume already started)")
    ap.add_argument("--warmup", type=float, default=0.5,
                    help="Seconds to wait after bind for the subscriber to connect (default: 0.5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Pack messages and report sizes without binding/sending (offline self-test)")
    args = ap.parse_args()

    chunks = _load_clip(args.clip)
    # frame_index span of the clip, used to keep indices monotonic across loops.
    has_fi = all("frame_index" in c for c in chunks)
    span = (max(int(np.asarray(c["frame_index"]).max()) for c in chunks) + 1) if has_fi else 0
    print(f"[replay] {len(chunks)} chunks, frame span={span}, fps={args.fps}, loop={args.loop}")

    pub = None
    if not args.dry_run:
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://{args.host}:{args.port}")
        print(f"[replay] bound PUB tcp://{args.host}:{args.port}; waiting {args.warmup}s for subscriber")
        time.sleep(args.warmup)
        if not args.no_start:
            pub.send(build_command_message(start=True, stop=False, planner=False))
            print("[replay] sent START (streamed-motion mode)")
            time.sleep(0.2)

    period = 1.0 / max(1e-6, args.fps)
    total_sent = 0
    for loop_i in range(args.loop):
        offset = loop_i * span  # keep frame_index increasing across repeats
        for c in chunks:
            msg_data = dict(c)
            if has_fi and offset:
                msg_data["frame_index"] = np.asarray(c["frame_index"]).astype(np.int64) + offset
            packed = pack_pose_message(msg_data, topic="pose")
            if args.dry_run or pub is None:
                if total_sent == 0:
                    print(f"[replay][dry-run] first message: {len(packed)} bytes, "
                          f"fields={list(msg_data.keys())}")
            else:
                pub.send(packed)
                time.sleep(period)
            total_sent += 1
        print(f"[replay] loop {loop_i + 1}/{args.loop} done ({len(chunks)} chunks)")

    print(f"[replay] sent {total_sent} pose messages total")
    if pub is not None:
        # Let the last message drain before closing the socket.
        time.sleep(0.2)
        pub.close()


if __name__ == "__main__":
    main()

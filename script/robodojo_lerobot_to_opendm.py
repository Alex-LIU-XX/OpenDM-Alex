#!/usr/bin/env python3
"""Convert a RoboDojo LeRobot v3.0 dataset into the OpenDM JSONL + video layout.

OpenDM reads robot demonstrations as one JSON object per frame (see
``docs/zh/data.md``). RoboDojo ships its raw data as a LeRobot v3.0 dataset:
parquet episodes plus chunked MP4 files, where a single MP4 file holds many
episodes back to back. This script bridges the two without re-encoding video:

* Each episode becomes ``<output-root>/jsonl/<task-slug>/episode_XXXXXXX.jsonl``.
* Every frame references the original chunked MP4 through ``url`` + ``frame_idx``,
  so no video is copied or transcoded. ``image_dir`` (registered in
  ``opendm/dataset/robodojo_ee.py``) points at ``<output-root>/video``, which is a
  symlink to ``<dataset-root>/videos``.

Frame addressing
----------------
For every episode and camera, LeRobot v3.0 stores ``from_timestamp`` /
``to_timestamp`` inside the shared MP4. These timestamps are exact multiples of
``1 / fps`` (verified: ``(to - from) * fps == length`` to ~1e-12), so the frame
offset of the episode inside the MP4 is ``round(from_timestamp * fps)`` and the
frame index of episode-relative frame ``i`` is ``round(from_timestamp * fps) + i``.

Action / state layout
---------------------
RoboDojo EE datasets store 16 dims per frame::

    [l_x, l_y, l_z, l_w, l_wx, l_wy, l_wz, l_g, r_x, r_y, r_z, r_w, r_wx, r_wy, r_wz, r_g]

i.e. position(3) + quaternion(w, x, y, z) + gripper(1) per arm. OpenDM's EE
convention is position(3) + axis-angle(3) + gripper(1) per arm (see
``ArrangeState`` and ``VLA_ARENA_EEF_STATE_DESC``), so each arm is converted to
7 dims and the output is 14 dims total. The quaternion convention here is
``(w, x, y, z)``, matching ``opendm/data/transforms.py``, so no component
reordering is needed.

Requirements: ``pandas``, ``pyarrow``, ``numpy`` (all pulled in by the project's
``datasets`` dependency).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import re
import sys

import numpy as np

DEFAULT_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
DEFAULT_IMAGE_KEYS = ["images_1", "images_2", "images_3"]
EE16_SLICES = ((0, 8), (8, 16))
EE16_DIM = 16
EEF14_DIM = 14

# Mirrors opendm/data/transforms.py so converted data matches training-time math.
QUAT_EPS = 1e-8


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), QUAT_EPS)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) quaternions to axis-angle vectors.

    Identical to ``opendm.data.transforms._quat_to_rotvec``: normalizes, flips
    the sign so ``w >= 0`` (avoiding the double cover), then scales the vector
    part by the rotation angle.
    """
    quat = _normalize_quat(np.asarray(quat, dtype=np.float32))
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    vec = quat[..., 1:4]
    vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vec_norm, quat[..., :1])
    rotvec = np.where(
        vec_norm > QUAT_EPS,
        vec * (angle / np.maximum(vec_norm, QUAT_EPS)),
        np.zeros_like(vec),
    )
    return rotvec.astype(np.float32)


def ee16_to_eef14(values: np.ndarray) -> np.ndarray:
    """Convert 16-dim RoboDojo EE rows to OpenDM's 14-dim EEF layout.

    ``[pos3, quat(wxyz), grip] * 2`` -> ``[pos3, axis-angle(3), grip] * 2``.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[-1] != EE16_DIM:
        raise ValueError(
            f"expected {EE16_DIM}-dim RoboDojo EE state/action, got {values.shape[-1]}. "
            "Datasets whose features are not pos+quaternion+gripper per arm "
            "(for example 14-dim joint datasets) need a different converter."
        )
    out = np.empty((values.shape[0], EEF14_DIM), dtype=np.float32)
    for arm, (start, _end) in enumerate(EE16_SLICES):
        base = arm * 7
        out[:, base + 0 : base + 3] = values[:, start + 0 : start + 3]
        out[:, base + 3 : base + 6] = _quat_to_rotvec(values[:, start + 3 : start + 7])
        out[:, base + 6] = values[:, start + 7]
    return out[0] if values.shape[0] == 1 else out


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:48] or "task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a RoboDojo LeRobot v3.0 dataset to OpenDM JSONL + video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="LeRobot v3.0 dataset root, e.g. /mnt/cfs/data/robodojo/RoboDojo_ee_cover_v30",
    )
    parser.add_argument(
        "--output-root",
        default="./data/robodojo_ee",
        help="Where to write jsonl/ and the video symlink.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Case-insensitive substring of the task instruction to convert. "
            "Omit to convert every episode in the dataset."
        ),
    )
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated LeRobot video feature keys, in image_keys order.",
    )
    parser.add_argument(
        "--image-keys",
        default=",".join(DEFAULT_IMAGE_KEYS),
        help="Comma-separated OpenDM JSONL keys, aligned with --cameras.",
    )
    parser.add_argument(
        "--task-slug",
        default=None,
        help=(
            "Directory name under jsonl/ for the converted episodes. Defaults to a "
            "slug of the task instruction."
        ),
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Only convert the first N matching episodes (smoke tests).",
    )
    parser.add_argument(
        "--video-mode",
        choices=["symlink", "absolute", "none"],
        default="symlink",
        help=(
            "symlink: link <output-root>/video to <dataset-root>/videos; "
            "absolute: write absolute paths into JSONL urls and skip the link; "
            "none: only write JSONL (register image_dir yourself)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing jsonl output directory / video symlink.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate everything, write nothing.",
    )
    return parser.parse_args()


def load_lerobot_metadata(dataset_root: str):
    import pandas as pd

    info_path = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"not a LeRobot v3 dataset (no meta/info.json): {dataset_root}")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        print(
            f"[warn] codebase_version={info.get('codebase_version')!r}; "
            "this converter targets LeRobot v3.x layouts.",
            file=sys.stderr,
        )

    tasks_path = os.path.join(dataset_root, "meta", "tasks.parquet")
    tasks_table = pd.read_parquet(tasks_path)
    # Column holds task_index; the task string is the frame index.
    index_col = tasks_table.columns[0]
    task_text_by_index = {
        int(row[index_col]): str(task)
        for task, row in tasks_table.iterrows()
    }

    episode_files = sorted(glob.glob(os.path.join(dataset_root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not episode_files:
        raise FileNotFoundError(f"no meta/episodes/**/*.parquet under {dataset_root}")
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    return info, task_text_by_index, episodes


def resolve_video_path(dataset_root: str, video_root: str, camera: str, chunk: int, file_idx: int) -> str:
    rel = os.path.join(camera, f"chunk-{chunk:03d}", f"file-{file_idx:03d}.mp4")
    return os.path.join(video_root, rel)


def main() -> int:
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    output_root = os.path.abspath(args.output_root)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    image_keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]
    if len(cameras) != len(image_keys):
        raise SystemExit("--cameras and --image-keys must have the same length")

    import pandas as pd

    info, task_text_by_index, episodes = load_lerobot_metadata(dataset_root)
    fps = float(info["fps"])
    features = info["features"]
    for camera in cameras:
        feature = features.get(camera)
        if feature is None or feature.get("dtype") != "video":
            raise SystemExit(f"{camera!r} is not a video feature of {dataset_root}")

    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    print(f"dataset      : {dataset_root}")
    print(f"fps          : {fps}")
    print(f"state/action : {state_dim} / {action_dim} dims -> {EEF14_DIM} dims (EEF)")
    print(f"cameras      : {cameras}")

    # ---- select episodes -------------------------------------------------
    selected = episodes
    if args.task is not None:
        needle = args.task.lower()
        matching_indices = {
            idx for idx, text in task_text_by_index.items() if needle in text.lower()
        }
        if not matching_indices:
            raise SystemExit(
                f"no task matches {args.task!r}. Available tasks:\n  "
                + "\n  ".join(f"[{i}] {t}" for i, t in sorted(task_text_by_index.items()))
            )
        for idx in sorted(matching_indices):
            print(f"task match   : [{idx}] {task_text_by_index[idx]}")
        keep = episodes["tasks"].apply(
            lambda texts: any(str(t) in {task_text_by_index[i] for i in matching_indices} for t in texts)
        )
        selected = episodes[keep]
    if selected.empty:
        raise SystemExit("no episodes selected")
    selected = selected.sort_values("episode_index").reset_index(drop=True)
    if args.limit_episodes is not None:
        selected = selected.head(args.limit_episodes).reset_index(drop=True)
    print(f"episodes     : {len(selected)} selected of {len(episodes)}")

    episode_indices = [int(v) for v in selected["episode_index"]]

    # ---- load frames for the selected episodes ---------------------------
    data_files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not data_files:
        raise SystemExit(f"no data/chunk-*/*.parquet under {dataset_root}")
    wanted = set(episode_indices)
    frames_by_episode: dict[int, pd.DataFrame] = {}
    for path in data_files:
        table = pd.read_parquet(
            path,
            columns=[
                "observation.state",
                "action",
                "frame_index",
                "episode_index",
                "task_index",
            ],
        )
        table = table[table["episode_index"].isin(wanted)]
        if table.empty:
            continue
        for episode_index, group in table.groupby("episode_index", sort=False):
            frames_by_episode.setdefault(int(episode_index), group)
    frames_by_episode = {
        k: v.sort_values("frame_index").reset_index(drop=True)
        for k, v in frames_by_episode.items()
    }
    missing = sorted(wanted - set(frames_by_episode))
    if missing:
        raise SystemExit(f"no frames found for episodes: {missing[:10]}")

    # ---- output layout ---------------------------------------------------
    video_root = os.path.join(dataset_root, "videos")
    jsonl_root = os.path.join(output_root, "jsonl")
    link_path = os.path.join(output_root, "video")
    if not args.dry_run:
        if args.force and os.path.isdir(jsonl_root):
            shutil.rmtree(jsonl_root)
        os.makedirs(jsonl_root, exist_ok=True)
        if args.video_mode == "symlink":
            if os.path.islink(link_path) or os.path.exists(link_path):
                if args.force:
                    os.remove(link_path) if os.path.islink(link_path) else shutil.rmtree(link_path)
            if not os.path.exists(link_path):
                os.symlink(video_root, link_path)
                print(f"video link   : {link_path} -> {video_root}")
            else:
                print(f"video link   : {link_path} (reused)")

    manifest = {
        "source_dataset": dataset_root,
        "lerobot_codebase_version": info.get("codebase_version"),
        "fps": fps,
        "robot_type": info.get("robot_type"),
        "cameras": cameras,
        "image_keys": image_keys,
        "source_state_dim": state_dim,
        "source_action_dim": action_dim,
        "output_dim": EEF14_DIM,
        "task_filter": args.task,
        "episodes": [],
    }

    total_frames = 0
    for _, record in selected.iterrows():
        episode_index = int(record["episode_index"])
        episode_task = str(record["tasks"][0]) if len(record["tasks"]) else ""
        if not episode_task and args.task is not None:
            episode_task = task_text_by_index[sorted(matching_indices)[0]]
        length = int(record["length"])
        group = frames_by_episode[episode_index]
        if len(group) != length:
            print(
                f"[warn] episode {episode_index}: parquet has {len(group)} rows but "
                f"meta says length={length}; using {len(group)}",
                file=sys.stderr,
            )
            length = len(group)
        if length < 2:
            print(f"[skip] episode {episode_index}: only {length} frame(s)", file=sys.stderr)
            continue

        states = np.stack(group["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(group["action"].to_numpy()).astype(np.float32)
        if states.shape != (length, state_dim) or actions.shape != (length, action_dim):
            raise SystemExit(
                f"episode {episode_index}: unexpected frame shapes "
                f"{states.shape} / {actions.shape}"
            )
        states14 = ee16_to_eef14(states)
        actions14 = ee16_to_eef14(actions)

        # Per-camera frame offsets inside the shared MP4 files.
        camera_meta = []
        for camera in cameras:
            chunk = int(record[f"videos/{camera}/chunk_index"])
            file_idx = int(record[f"videos/{camera}/file_index"])
            from_ts = float(record[f"videos/{camera}/from_timestamp"])
            raw_offset = from_ts * fps
            offset = int(round(raw_offset))
            if abs(raw_offset - offset) > 1e-3:
                raise SystemExit(
                    f"episode {episode_index} camera {camera}: from_timestamp "
                    f"{from_ts} is not an integer frame offset ({raw_offset})"
                )
            mp4_path = resolve_video_path(dataset_root, video_root, camera, chunk, file_idx)
            if args.video_mode == "absolute":
                url = mp4_path
            elif args.video_mode == "symlink":
                url = os.path.relpath(mp4_path, video_root)
            else:
                url = os.path.relpath(mp4_path, video_root)
            camera_meta.append((url, offset))
            if not args.dry_run and not os.path.isfile(mp4_path):
                raise SystemExit(f"missing video file: {mp4_path}")

        slug = args.task_slug or (slugify(episode_task) if episode_task else "task")
        episode_dir = os.path.join(jsonl_root, slug)
        if not args.dry_run:
            os.makedirs(episode_dir, exist_ok=True)
        jsonl_path = os.path.join(episode_dir, f"episode_{episode_index:07d}.jsonl")

        lines = []
        for i in range(length):
            frame = {
                key: {"type": "video", "url": url, "frame_idx": offset + i}
                for key, (url, offset) in zip(image_keys, camera_meta)
            }
            frame["state"] = [round(float(v), 6) for v in states14[i]]
            frame["action"] = [round(float(v), 6) for v in actions14[i]]
            frame["prompt"] = episode_task
            frame["is_robot"] = True
            lines.append(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
        if not args.dry_run:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        total_frames += length
        manifest["episodes"].append(
            {
                "episode_index": episode_index,
                "task": episode_task,
                "length": length,
                "jsonl": os.path.relpath(jsonl_path, output_root),
                "video_offsets": {
                    camera: {"url": url, "offset": offset}
                    for camera, (url, offset) in zip(cameras, camera_meta)
                },
            }
        )
        print(
            f"  episode {episode_index:5d} len={length:5d} -> "
            f"{os.path.relpath(jsonl_path, output_root)}"
        )

    if not args.dry_run:
        manifest_path = os.path.join(output_root, "convert_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"manifest     : {manifest_path}")

    print(
        f"done         : {len(manifest['episodes'])} episodes, {total_frames} frames"
        + (" (dry run, nothing written)" if args.dry_run else "")
    )
    print(
        "next         : register jsonl_dir="
        f"{os.path.relpath(jsonl_root, os.getcwd())} and image_dir="
        f"{os.path.relpath(link_path, os.getcwd()) if args.video_mode == 'symlink' else video_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

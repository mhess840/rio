# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Open-loop π₀.₅ vs dataset ground truth. No robot.

Run:

    uv run --group pi --no-group docs -m examples.diagnose_pi05_offline \\
        --policy-path ckpts/pi05 --device cuda:1

Joint checkpoint against the joint dataset:

    uv run --group pi --no-group docs -m examples.diagnose_pi05_offline \\
        --policy-path ckpts/pi05_joints --device cuda:1 \\
        --dataset-root data/lerobot/kassow_pi05_joints \\
        --repo-id local/kassow_pi05_joints \\
        --out-dir /tmp/pi05_diag/offline_joints

Writes ADE tables and xyz plots to ``/tmp/pi05_diag/offline/``.

Home/start frames (post-idle first frame, not episode middle):

    uv run --group pi --no-group docs -m examples.diagnose_pi05_offline \\
        --policy-path ckpts/pi05 --device cuda:1 \\
        --dataset-root data/lerobot/kassow_pi05_pick \\
        --repo-id local/kassow_pi05_pick \\
        --sample-at start \\
        --out-dir /tmp/pi05_diag/offline_pick_start

RTC-off live recipe (separate, E-Stop ready):

    POLICY=Pi05LerobotCfg uv run --group pi --no-group docs -m examples.policy_inference \\
        --policy-path ckpts/pi05 \\
        --instruction "pick up the blue cylinder and place it on the red plate" \\
        --policy-cfg.device cuda:1 \\
        --policy-node-cfg.no-use-rtc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from rio.policies.pi05 import Pi05, normalize_rgb_image, principal_rotvec, snap_gripper

DEFAULT_INSTRUCTION = "pick up the blue cylinder and place it on the red plate"
DEFAULT_DATASET = "data/lerobot/kassow_pi05"
DEFAULT_REPO_ID = "local/kassow_pi05"
DEFAULT_OUT = Path("/tmp/pi05_diag/offline")


def to_hwc_uint8(image) -> np.ndarray:
    """Dataset image to HWC uint8 for ``format_rio_obs``."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        peak = float(np.nanmax(arr)) if arr.size else 1.0
        if peak <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def xyz_ade_m(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Mean and max Euclidean xyz error in metres."""
    delta = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    return float(delta.mean()), float(delta.max())


def xyz_span_m(actions: np.ndarray) -> float:
    """Distance from first to last xyz."""
    xyz = np.asarray(actions, dtype=np.float32)[:, :3]
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(xyz[-1] - xyz[0]))


def joint_ade_rad(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Mean and max L2 error on the seven arm joints."""
    delta = np.linalg.norm(pred[:, :7] - gt[:, :7], axis=1)
    return float(delta.mean()), float(delta.max())


def joint_span_rad(actions: np.ndarray) -> float:
    """L2 path from first to last joint vector."""
    q = np.asarray(actions, dtype=np.float32)[:, :7]
    if len(q) < 2:
        return 0.0
    return float(np.linalg.norm(q[-1] - q[0]))


def rot_ade_rad(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Mean and max geodesic rotation error after folding rotvecs."""
    from scipy.spatial.transform import Rotation as R

    pa = principal_rotvec(pred[:, 3:6])
    ga = principal_rotvec(gt[:, 3:6])
    angles = (R.from_rotvec(pa).inv() * R.from_rotvec(ga)).magnitude()
    return float(np.mean(angles)), float(np.max(angles))


def gripper_match(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of snapped gripper values that match."""
    ps = snap_gripper(pred[:, -1])
    gs = snap_gripper(gt[:, -1])
    return float(np.mean(ps == gs))


def episode_mid_index(from_i: int, to_i: int, horizon: int) -> int:
    """Frame index in the episode middle that still has ``horizon`` GT actions."""
    length = int(to_i) - int(from_i)
    if length <= horizon:
        raise ValueError(f"episode length {length} < horizon {horizon}")
    mid = int(from_i) + length // 2
    latest = int(to_i) - horizon
    return min(mid, latest)


def episode_start_index(from_i: int, to_i: int, horizon: int) -> int:
    """First frame of the episode. Needs ``horizon`` GT actions after it."""
    return episode_frac_index(from_i, to_i, horizon, 0.0)


def episode_frac_index(from_i: int, to_i: int, horizon: int, frac: float) -> int:
    """Frame at ``frac`` of the usable episode (0=start, 1=last window)."""
    length = int(to_i) - int(from_i)
    if length <= horizon:
        raise ValueError(f"episode length {length} < horizon {horizon}")
    frac = float(frac)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    latest = int(to_i) - int(horizon)
    span = latest - int(from_i)
    return int(from_i) + round(frac * span)


def parse_sample_at(sample_at: str) -> list[float]:
    """Map ``start``/``mid`` or comma-separated fractions onto [0, 1]."""
    text = str(sample_at).strip().lower()
    if text == "start":
        return [0.0]
    if text == "mid":
        return [0.5]
    fracs = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not fracs:
        raise ValueError(f"sample_at must be 'start', 'mid', or fractions, got {sample_at!r}")
    for frac in fracs:
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"sample_at fraction {frac} is not in [0, 1]")
    return fracs


def resolve_instruction(frame: dict, override: str | None = None) -> str:
    """Use the frame's actual task label unless the caller explicitly overrides it."""
    task = override if override is not None else frame.get("task")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else None
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Offline inference needs the dataset task label or an explicit instruction")
    return task


def to_hwc_float32(image) -> np.ndarray:
    """Preserve normalized dataset pixels while converting CHW images to HWC."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    return normalize_rgb_image(array)


def rio_obs_from_frame(frame: dict, instruction: str) -> dict:
    """Build a Serve-style observation dict from a LeRobot frame."""
    state = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
    return {
        "camera_1": to_hwc_float32(frame["observation.images.camera_1"]),
        "camera_2": to_hwc_float32(frame["observation.images.camera_2"]),
        "proprio": state,
        "proprio_joints": state[:7],
        "gripper_position": float(state[-1]),
        "task": instruction,
    }


def _episode_bounds(ds, episode: int) -> tuple[int, int]:
    meta = ds.meta.episodes
    row = meta[episode]
    if "dataset_from_index" in row and "dataset_to_index" in row:
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])
    start = int(ds.episode_data_index["from"][episode])
    end = int(ds.episode_data_index["to"][episode])
    return start, end


def _action_window(ds, start: int, horizon: int) -> np.ndarray:
    hf = ds.hf_dataset
    rows = hf.select(range(int(start), int(start) + int(horizon)))
    acts = [np.asarray(a, dtype=np.float32).reshape(-1) for a in rows["action"]]
    return np.stack(acts, axis=0)


def _plot_xyz(pred: np.ndarray, gt: np.ndarray, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib missing; skip {}", path)
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    t = np.arange(len(pred))
    for i, name in enumerate("xyz"):
        axes[i].plot(t, gt[:, i], label="gt")
        axes[i].plot(t, pred[:, i], label="pred")
        axes[i].set_title(name)
        axes[i].set_xlabel("step")
        axes[i].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _load_dataset(root: Path, repo_id: str):
    from rio.data.convert_pi05 import _patch_lerobot_pyav

    _patch_lerobot_pyav()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=str(root), video_backend="pyav")


@dataclass
class DiagnoseOfflineCfg:
    policy_path: str = "ckpts/pi05"
    dataset_root: str = DEFAULT_DATASET
    repo_id: str = DEFAULT_REPO_ID
    instruction: str | None = None
    device: str = "cuda:1"
    out_dir: str = str(DEFAULT_OUT)
    num_samples: int = 20
    horizon: int | None = None
    # ``mid``, ``start``, or comma-separated fractions e.g. ``0,0.25,0.5,0.75``.
    sample_at: str = "mid"
    binarize_gripper: bool = True
    action_space: str | None = None


def main(cfg: DiagnoseOfflineCfg) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Loading dataset {}", cfg.dataset_root)
    ds = _load_dataset(Path(cfg.dataset_root), cfg.repo_id)
    n_eps = int(ds.meta.total_episodes)
    horizon = int(cfg.horizon) if cfg.horizon else 50
    ckpt_cfg = Path(cfg.policy_path) / "config.json"
    if ckpt_cfg.is_file() and cfg.horizon is None:
        import json

        data = json.loads(ckpt_cfg.read_text())
        horizon = int(data.get("n_action_steps") or data.get("chunk_size") or horizon)

    fracs = parse_sample_at(cfg.sample_at)
    stride = max(1, n_eps // cfg.num_samples)
    episodes = list(range(0, n_eps, stride))[: cfg.num_samples]
    logger.info("Evaluating {} episodes at frac={} horizon={}", len(episodes), fracs, horizon)
    write_plots = len(fracs) == 1

    policy = Pi05(
        policy_path=cfg.policy_path,
        device=cfg.device,
        binarize_gripper=cfg.binarize_gripper,
        use_rtc=False,
        action_space=cfg.action_space,
    )
    policy.set_instruction(cfg.instruction)
    policy.construct_policy()

    rows = []
    for ep in episodes:
        from_i, to_i = _episode_bounds(ds, ep)
        for frac in fracs:
            try:
                t0 = episode_frac_index(from_i, to_i, horizon, frac)
            except ValueError as exc:
                logger.warning("skip episode {} frac {}: {}", ep, frac, exc)
                continue
            frame = ds[t0]
            task = resolve_instruction(frame, cfg.instruction)
            if frame.get("task"):
                raw = frame["task"]
                task = raw[0] if isinstance(raw, list) else str(raw)
                if cfg.instruction is not None and task != cfg.instruction:
                    logger.warning("episode {} task {!r} != eval instruction", ep, task)
            task = resolve_instruction(frame, cfg.instruction)
            policy.set_instruction(task)
            obs = rio_obs_from_frame(frame, task)
            pred = policy.inference(obs)["actions"][:horizon]
            gt = _action_window(ds, t0, horizon)
            n = min(len(pred), len(gt))
            pred, gt = pred[:n], gt[:n]
            joint_space = getattr(policy, "action_space", "tcp") == "joints" or pred.shape[-1] >= 8
            grip = gripper_match(pred, gt)
            if joint_space:
                ade, mx = joint_ade_rad(pred, gt)
                row = {
                    "episode": ep,
                    "frac": frac,
                    "frame": t0,
                    "joint_ade_rad": ade,
                    "joint_max_rad": mx,
                    "pred_span_rad": joint_span_rad(pred),
                    "gt_span_rad": joint_span_rad(gt),
                    "gripper_match": grip,
                }
            else:
                ade, mx = xyz_ade_m(pred, gt)
                rot_m, rot_x = rot_ade_rad(pred, gt)
                row = {
                    "episode": ep,
                    "frac": frac,
                    "frame": t0,
                    "xyz_ade_m": ade,
                    "xyz_max_m": mx,
                    "pred_span_m": xyz_span_m(pred),
                    "gt_span_m": xyz_span_m(gt),
                    "rot_ade_rad": rot_m,
                    "rot_max_rad": rot_x,
                    "gripper_match": grip,
                }
            rows.append(row)
            np.savez(out / f"ep{ep:03d}_frac{frac:.2f}_frame{t0}.npz", pred=pred, gt=gt)
            if write_plots:
                _plot_xyz(pred, gt, out / f"ep{ep:03d}_xyz.png", f"ep {ep} frame {t0}")
            if joint_space:
                logger.info(
                    "ep {} frac {:.2f} ADE {:.3f} rad pred_span {:.3f} rad gt_span {:.3f} rad",
                    ep,
                    frac,
                    ade,
                    row["pred_span_rad"],
                    row["gt_span_rad"],
                )
            else:
                logger.info(
                    "ep {} frac {:.2f} ADE {:.3f} m pred_span {:.3f} m gt_span {:.3f} m",
                    ep,
                    frac,
                    ade,
                    row["pred_span_m"],
                    row["gt_span_m"],
                )

    if not rows:
        raise RuntimeError("no episodes evaluated")
    summary_path = out / "summary.csv"
    keys = list(rows[0].keys())
    with summary_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    logger.info("Wrote {}", summary_path)
    ade_key = "joint_ade_rad" if "joint_ade_rad" in rows[0] else "xyz_ade_m"
    pred_key = "pred_span_rad" if "pred_span_rad" in rows[0] else "pred_span_m"
    gt_key = "gt_span_rad" if "gt_span_rad" in rows[0] else "gt_span_m"
    unit = "rad" if ade_key.endswith("rad") else "m"
    for frac in fracs:
        part = [r for r in rows if abs(float(r["frac"]) - frac) < 1e-9]
        if not part:
            continue
        mean_ade = float(np.mean([r[ade_key] for r in part]))
        mean_pred = float(np.mean([r[pred_key] for r in part]))
        mean_gt = float(np.mean([r[gt_key] for r in part]))
        logger.info(
            "frac {:.2f}: ADE {:.3f} {}  pred span {:.3f} {}  GT span {:.3f} {}",
            frac,
            mean_ade,
            unit,
            mean_pred,
            unit,
            mean_gt,
            unit,
        )
        if unit == "m":
            if mean_ade < 0.02 and mean_pred > 0.05:
                logger.info("  Read: directed plan.")
            elif mean_pred < 0.04:
                logger.info("  Read: stay local — weak plan.")
            else:
                logger.info("  Read: mixed.")


if __name__ == "__main__":
    main(tyro.cli(DiagnoseOfflineCfg))

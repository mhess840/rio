# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollout settings, resolved and validated once from a station/policy config."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rio.gripper import GripperEncoding
from rio.policies.rtc import RTC_EXECUTION_HORIZON, RTC_QUEUE_THRESHOLD
from rio.schema import ArmField

from .diagnostics import default_diag_dir
from .safety import (
    MAX_POLICY_JOINT_SPEED,
    MAX_POLICY_POS_SPEED,
    MAX_POLICY_ROT_SPEED,
    RolloutSafetyCfg,
    require_speed_limit,
)


def chunk_size_from_ckpt(policy_path: str, default: int = 16) -> int:
    """Read ``n_action_steps`` / ``chunk_size`` from a LeRobot checkpoint."""
    cfg_file = Path(policy_path) / "config.json"
    if not cfg_file.is_file():
        return default
    data = json.loads(cfg_file.read_text())
    n = data.get("n_action_steps") or data.get("chunk_size")
    return int(n) if n else default


def resolve_n_execute(policy_path: str, override: int | None, default: int = 16) -> int:
    """How many predicted steps to play before the next request.

    The checkpoint still predicts ``chunk_size`` (often 50). Playing all of them
    lets later steps reverse toward grasp/home. ``override<=0`` means the ckpt
    ``n_action_steps``.
    """
    if override is not None and int(override) > 0:
        return int(override)
    return chunk_size_from_ckpt(policy_path, default=default)


def is_joint_serve(args) -> bool:
    """True when the station sends ``moveJ`` / 8D joint actions."""
    space = str(getattr(args, "action_space", "") or "").lower()
    policy_space = str(getattr(getattr(args, "policy_cfg", None), "action_space", "") or "").lower()
    return space == "joint_pos" or policy_space == "joints"


class RolloutCfg(Protocol):
    """The part of a station config every rollout needs; read directly, never defaulted.

    Optional extras, defaulted in ``RolloutSettings.from_args`` because not every station declares them:
    ``policy_path``, ``arm_cfg`` speed limits, ``arm_latency``, ``safety_cfg``, ``diag_dir``,
    ``action_space``, the RTC/queue fields of ``policy_node_cfg``, and the gripper fields of ``policy_cfg``.
    """

    @property
    def freq(self) -> float: ...
    @property
    def instruction(self) -> str: ...
    @property
    def policy_node_cfg(self) -> Any: ...
    @property
    def policy_cfg(self) -> Any: ...


@dataclass(frozen=True)
class RolloutSettings:
    """Everything ``policy_loop`` reads from the config, checked before the policy is used."""

    freq: float
    action_hz: float
    ticks_per_action: int
    action_dim: int
    use_rtc: bool
    n_execute: int
    execution_horizon: int
    queue_threshold: int
    chunk_pause_s: float
    binarize_gripper: bool
    gripper_encoding: GripperEncoding
    gripper_state_source: str
    joint_serve: bool
    max_pos_speed: float
    max_rot_speed: float
    max_joint_speed: float
    arm_latency: float
    safety: RolloutSafetyCfg
    diag_root: Path
    load_timeout: float
    policy_path: str
    instruction: str

    @property
    def dt(self) -> float:
        return 1.0 / self.freq

    @property
    def time_per_action(self) -> float:
        return 1.0 / self.action_hz

    @classmethod
    def from_args(cls, args: RolloutCfg) -> "RolloutSettings":
        """Resolve a station/policy config (``tyro`` dataclass or namespace).

        Args:
            args: Station config; see ``RolloutCfg`` for required and optional fields.

        Raises:
            ValueError: If a rate, timeout, or speed limit is invalid or above its reviewed ceiling.
        """
        node = args.policy_node_cfg
        policy = args.policy_cfg
        load_timeout = float(getattr(node, "timeout", 180.0))
        if not np.isfinite(load_timeout) or load_timeout <= 0.0:
            raise ValueError("The policy loading timeout must be finite and positive")
        freq = float(args.freq)
        if not np.isfinite(freq) or freq <= 0:
            raise ValueError("The control frequency must be positive")
        action_hz = float(getattr(node, "action_hz", 20) or 20)
        if not np.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("action_hz must be finite and positive")
        ticks_per_action = max(1, round(freq / action_hz))
        if not math.isclose(ticks_per_action * action_hz, freq):
            raise ValueError("The control frequency must be an integer multiple of action_hz")
        arm_cfg = getattr(args, "arm_cfg", None)
        policy_path = str(getattr(args, "policy_path", "") or "")
        return cls(
            freq=freq,
            action_hz=action_hz,
            ticks_per_action=ticks_per_action,
            action_dim=int(node.action_dim),
            use_rtc=bool(getattr(node, "use_rtc", False)),
            n_execute=resolve_n_execute(policy_path, getattr(node, "n_execute", None)),
            execution_horizon=int(getattr(node, "rtc_execution_horizon", RTC_EXECUTION_HORIZON) or RTC_EXECUTION_HORIZON),
            queue_threshold=int(getattr(node, "rtc_queue_threshold", RTC_QUEUE_THRESHOLD) or RTC_QUEUE_THRESHOLD),
            chunk_pause_s=max(0.0, float(getattr(node, "chunk_pause_s", 0.0) or 0.0)),
            binarize_gripper=bool(getattr(policy, "binarize_gripper", True)),
            gripper_encoding={
                "gripper_close": float(getattr(policy, "gripper_close", 0.0)),
                "gripper_open": float(getattr(policy, "gripper_open", 0.7)),
                "gripper_threshold": getattr(policy, "gripper_threshold", None),
            },
            gripper_state_source=str(getattr(policy, "gripper_state_source", "measured")),
            joint_serve=is_joint_serve(args),
            max_pos_speed=require_speed_limit(
                "arm_cfg.max_pos_speed", getattr(arm_cfg, "max_pos_speed", 0.45) or 0.45, MAX_POLICY_POS_SPEED
            ),
            max_rot_speed=require_speed_limit(
                "arm_cfg.max_rot_speed", getattr(arm_cfg, "max_rot_speed", 1.0) or 1.0, MAX_POLICY_ROT_SPEED
            ),
            max_joint_speed=require_speed_limit(
                "arm_cfg.max_motor_speed", getattr(arm_cfg, "max_motor_speed", 1.0) or 1.0, MAX_POLICY_JOINT_SPEED
            ),
            arm_latency=float(getattr(args, "arm_latency", 0.0)),
            safety=getattr(args, "safety_cfg", None) or RolloutSafetyCfg(),
            diag_root=Path(getattr(args, "diag_dir", None) or default_diag_dir()),
            load_timeout=load_timeout,
            policy_path=policy_path,
            instruction=str(args.instruction),
        )

    def required_arm_fields(self, *, plan_done: bool) -> tuple[ArmField, ...]:
        """Arm state keys this rollout depends on; ``joint_qd`` only for plan-completion stops."""
        tracked = "joint_q" if self.joint_serve else "eef_pose"
        return ("timestamp", tracked, "joint_qd") if plan_done else ("timestamp", tracked)

"""Joint-velocity teleop for Kassow via directJControl (VelCmd).

Maps Spacemouse / Keyboard / Gamepad axes (6D, roughly [-1, 1]) onto the first
six joint velocities. Joint 7 stays at zero. Hands-off sends zero qd so the C++
RT loop keeps holding with directJControl.

Example:
  STATION=KassowStation uv run -m examples.teleop_joint_vel \\
    --action-space joint_vel \\
    --arm-cfg.max-motor-speed 0.3 \\
    --arm-cfg.log-diagnostics \\
    --teleop Spacemouse
"""

import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env
from rio.envs.poll import Interface, TeleopMode

TELEOP_IDLE_EPS = 0.05


def _axes_to_joint_qd(delta_axes: np.ndarray, max_motor_speed: float, num_joints: int) -> np.ndarray:
    """Map teleop axes onto joint velocities (rad/s)."""
    qd = np.zeros(num_joints, dtype=np.float64)
    n = min(6, num_joints, len(delta_axes))
    qd[:n] = np.asarray(delta_axes[:n], dtype=np.float64) * max_motor_speed
    return qd


def teleop_joint_vel(args, env, teleop, visualizer=None):
    if env.robot.arm is None:
        raise RuntimeError("teleop_joint_vel requires an arm")

    num_joints = env.robot.arm_num_joints
    max_motor_speed = float(getattr(args.arm_cfg, "max_motor_speed", 0.5))
    teleop_mode = TeleopMode.TRANSLATION
    t_last_mode_change = time.now()
    last_gripper_cmd = float(env.robot.gripper.get_state()["gripper_position"]) if env.robot.gripper else 0.0
    if args.teleop.lower() in ("keyboard", "sshkeyboard"):
        Interface.set_keyboard_gripper_state(teleop, last_gripper_cmd)

    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
        logger.debug(f"Visualizer: set robot model to {env.robot.urdf_path}")

    logger.info(
        f"Kassow joint_vel teleop: axes → joints[0:{min(6, num_joints)}] at ±{max_motor_speed:.2f} rad/s via directJControl"
    )
    input("Press Enter to start")

    freq = args.freq
    dt = 1.0 / freq
    command_latency = dt / 2
    t_start = time.now()
    it = 0
    env.set_start_time(t_start)
    env.set_instruction(args.instruction)

    try:
        while True:
            t_cycle_end = t_start + (it + 1) * dt
            t_sample = t_cycle_end - command_latency
            t_cmd_target = t_cycle_end + dt

            time.precise_wait(t_sample)

            delta_axes, gripper_pos, t_last_mode_change, teleop_mode = Interface.poll(
                args.teleop, teleop, t_sample, t_last_mode_change, teleop_mode
            )
            if gripper_pos is not None:
                last_gripper_cmd = gripper_pos

            if np.max(np.abs(delta_axes)) < TELEOP_IDLE_EPS:
                qd = np.zeros(num_joints, dtype=np.float64)
            else:
                qd = _axes_to_joint_qd(delta_axes, max_motor_speed, num_joints)

            t_target = t_cmd_target + args.arm_latency
            action = env.robot.build_action(qd, gripper_cmd=last_gripper_cmd)
            # Always send (including zeros) so Vel mode stays latched and holds.
            env.move(action, t_cmd_target=t_target)
            step = env.get_state(action=action)

            if env.recorder:
                env.recorder.record_step(step)
            if visualizer:
                visualizer.log_env_state("env", step)

            if it % freq == 0:
                state = env.robot.arm.get_state()
                print(
                    f"t: {t_cycle_end - t_start:.3f}s",
                    "|",
                    f"mode: {teleop_mode.name}",
                    "|",
                    f"qd: {np.array2string(qd, precision=3)}",
                    f"q: {np.array2string(np.asarray(state['joint_q']), precision=3)}",
                )
            time.precise_wait(t_cycle_end)
            it += 1
    except KeyboardInterrupt:
        # Park with zero velocity before teardown.
        try:
            zeros = np.zeros(num_joints, dtype=np.float64)
            env.move(env.robot.build_action(zeros, gripper_cmd=last_gripper_cmd), t_cmd_target=time.now() + dt)
        except Exception:
            pass
    finally:
        if env.recorder:
            env.recorder.save(wait=True)


def main(args):
    # Force the joint-velocity path even if the station defaults to task_pos.
    args.action_space = "joint_vel"
    args.arm_cfg.robot_controller = "joint_vel"

    servers, clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())):
        with (
            env,
            clients["teleop"]() as teleop,
            clients["visualizer"]() if clients["visualizer"] else nullcontext() as visualizer,
        ):
            try:
                teleop_joint_vel(args, env, teleop, visualizer)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    from examples import get_station_cfg

    args = tyro.cli(get_station_cfg())
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)

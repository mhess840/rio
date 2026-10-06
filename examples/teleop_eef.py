import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env
from rio.envs.poll import Interface, TeleopMode

# Spacemouse / gamepad axes are roughly in [-1, 1]. Below this, treat the device
# as released so the target does not keep leading the measured pose.
TELEOP_IDLE_EPS = 0.05


def teleop_eef(args, env, teleop, visualizer=None):
    # Initialize target states
    arm_target_pose = env.robot.arm.get_state()["eef_pose"].copy() if env.robot.arm else None
    last_gripper_cmd = float(env.robot.gripper.get_state()["gripper_position"]) if env.robot.gripper else 0.0
    if args.teleop.lower() in ("keyboard", "sshkeyboard"):
        Interface.set_keyboard_gripper_state(teleop, last_gripper_cmd)

    teleop_mode = TeleopMode.TRANSLATION
    t_last_mode_change = time.now()

    max_pos_speed = args.arm_cfg.max_pos_speed
    max_rot_speed = args.arm_cfg.max_rot_speed

    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
        logger.debug(f"Visualizer: set robot model to {env.robot.urdf_path}")
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

            # Poll teleop interface
            delta_tcp_pose, gripper_pos, t_last_mode_change, teleop_mode = Interface.poll(
                args.teleop, teleop, t_sample, t_last_mode_change, teleop_mode
            )
            if gripper_pos is not None:
                last_gripper_cmd = gripper_pos

            # Build arm command
            if env.robot.arm:
                t_target = t_cmd_target + args.arm_latency
                if np.max(np.abs(delta_tcp_pose)) < TELEOP_IDLE_EPS:
                    # Hands off: drop any lead and do not stream, so the arm
                    # stops instead of catching up / low-pass coasting.
                    arm_target_pose = env.robot.arm.get_state()["eef_pose"].copy()
                    action = env.robot.build_action(arm_target_pose, gripper_cmd=last_gripper_cmd)
                    if gripper_pos is not None:
                        env.robot.move_gripper(last_gripper_cmd, t_cmd_target=t_target)
                else:
                    arm_target_pose = env.robot.make_teleop_eef_cmd(
                        freq, teleop_mode, delta_tcp_pose, arm_target_pose, max_pos_speed, max_rot_speed
                    )
                    action = env.robot.build_action(arm_target_pose, gripper_cmd=last_gripper_cmd)
                    env.move(action, t_cmd_target=t_target)
            step = env.get_state(action=action)

            if env.recorder:
                env.recorder.record_step(step)
            if visualizer:
                visualizer.log_frame("world/teleop_target", arm_target_pose, axis_length=0.08)
                visualizer.log_env_state("env", step)

            # Logging
            if it % freq == 0:
                print(
                    f"t: {t_cycle_end - t_start:.3f}s",
                    "|",
                    f"mode: {teleop_mode.name}",
                    "|",
                    f"delta: {delta_tcp_pose[:3]}",
                    f"robot_tcp: {env.robot.arm.get_state()['eef_pose'] if env.robot.arm else None}",
                    "|",
                    f"gripper: {env.robot.gripper.get_state()['gripper_position'] if env.robot.gripper else None}",
                )
            time.precise_wait(t_cycle_end)
            it += 1
    except KeyboardInterrupt:
        pass
    finally:
        if env.recorder:
            env.recorder.save(wait=True)


def main(args):
    servers, clients, env = make_env(args)
    with ServerManager(args.mw, list(servers.values())):
        with (
            env,
            clients["teleop"]() as teleop,
            clients["visualizer"]() if clients["visualizer"] else nullcontext() as visualizer,
        ):
            try:
                teleop_eef(args, env, teleop, visualizer)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    from examples import get_station_cfg

    args = tyro.cli(get_station_cfg())
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)

"""Control a station's gripper from the keyboard without starting its arm."""

import multiprocessing as mp

import tyro
from rio_hw import time
from rio_hw.middleware import ServerManager

import rio.envs.factory as F
from rio.envs.poll import Interface, TeleopMode


def teleop_gripper_keyboard(args, gripper, keyboard) -> None:
    """Toggle the gripper with Space while leaving all other station nodes off."""
    current_position = float(gripper.get_state()["gripper_position"])
    Interface.set_keyboard_gripper_state(keyboard, current_position)
    teleop_mode = TeleopMode.TRANSLATION
    t_last_mode_change = time.now()

    print(f"Gripper position: {current_position:.3f}")
    print("Space: toggle open/closed | Ctrl+C: exit")

    dt = 1.0 / args.freq
    gripper_dt = 1.0 / args.gripper_cfg.freq
    status_target = None
    next_status_time = 0.0
    status_deadline = 0.0
    try:
        while True:
            t_now = time.now()
            t_cycle_end = t_now + dt
            _, gripper_command, t_last_mode_change, teleop_mode = Interface.poll(
                "SshKeyboard",
                keyboard,
                t_now,
                t_last_mode_change,
                teleop_mode,
            )
            if gripper_command is not None:
                # Keep the target ahead of the slower gripper node's request cycle.
                target_time = time.now() + 2.0 * gripper_dt
                gripper.moveG([gripper_command], target_time)
                state = "open" if gripper_command == 1.0 else "closed"
                print(f"Gripper target: {state}")
                status_target = gripper_command
                next_status_time = t_now + 0.5
                status_deadline = t_now + 5.0
            if status_target is not None and t_now >= next_status_time:
                measured_position = float(gripper.get_state()["gripper_position"])
                print(f"Gripper measured position: {measured_position:.3f}")
                reached_target = abs(measured_position - status_target) <= 0.01
                if reached_target or t_now >= status_deadline:
                    status_target = None
                else:
                    next_status_time = t_now + 0.5
            time.precise_wait(t_cycle_end)
    except KeyboardInterrupt:
        pass


def main(args) -> None:
    if args.gripper is None:
        raise RuntimeError("The selected station has no gripper configured")

    gripper_server, gripper_client = F.make_node(
        args.mw,
        "robots",
        args.gripper,
        F.dataclass_to_dict(args.gripper_cfg),
    )
    keyboard_server, keyboard_client = F.make_node(
        args.mw,
        "interfaces",
        "SshKeyboard",
        F.dataclass_to_dict(args.teleop_cfg),
    )
    if gripper_client is None or keyboard_client is None:
        raise RuntimeError("Failed to create gripper or keyboard client")

    with ServerManager(args.mw, [gripper_server, keyboard_server]):
        with gripper_client() as gripper, keyboard_client() as keyboard:
            teleop_gripper_keyboard(args, gripper, keyboard)


if __name__ == "__main__":
    from examples import get_station_cfg

    args = tyro.cli(get_station_cfg())
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)

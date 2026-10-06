"""Teleoperation input mappings."""

import numpy as np
import pytest

from rio.envs.poll import Interface, TeleopMode


class StubKeyboard:
    def __init__(self, *, alphanumeric: tuple[int, ...] = (), special: tuple[int, ...] = ()):
        self.alphanumeric = alphanumeric
        self.special = special

    def get_state(self) -> dict[str, np.ndarray]:
        return {
            "alphanumeric_state": np.asarray(self.alphanumeric, dtype=np.byte),
            "special_state": np.asarray(self.special, dtype=int),
        }


@pytest.mark.parametrize(
    "keyboard",
    [
        StubKeyboard(alphanumeric=(ord(" "),)),
        StubKeyboard(special=(0x2C,)),
    ],
)
def test_keyboard_space_toggles_gripper_on_press_edge(keyboard):
    Interface.set_keyboard_gripper_state(keyboard, gripper_position=1.0)

    _, first_press, _, _ = Interface.poll_keyboard(
        keyboard,
        t_sample=0.0,
        t_last_mode_change=0.0,
        teleop_mode=TeleopMode.TRANSLATION,
    )
    _, held, _, _ = Interface.poll_keyboard(
        keyboard,
        t_sample=0.1,
        t_last_mode_change=0.0,
        teleop_mode=TeleopMode.TRANSLATION,
    )

    keyboard.alphanumeric = ()
    keyboard.special = ()
    Interface.poll_keyboard(
        keyboard,
        t_sample=0.2,
        t_last_mode_change=0.0,
        teleop_mode=TeleopMode.TRANSLATION,
    )

    keyboard.alphanumeric = (ord(" "),)
    _, second_press, _, _ = Interface.poll_keyboard(
        keyboard,
        t_sample=0.3,
        t_last_mode_change=0.0,
        teleop_mode=TeleopMode.TRANSLATION,
    )

    assert first_press == 0.0
    assert held is None
    assert second_press == 1.0


def test_keyboard_right_bracket_opens_gripper():
    _, gripper, _, _ = Interface.poll_keyboard(
        StubKeyboard(alphanumeric=(ord("]"),)),
        t_sample=0.0,
        t_last_mode_change=0.0,
        teleop_mode=TeleopMode.TRANSLATION,
    )

    assert gripper == 1.0

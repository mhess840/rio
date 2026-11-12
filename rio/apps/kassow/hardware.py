# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Machine-specific values of the Kassow cell, read once at import.

Set the environment variable before starting to point the code at another cell;
a command-line flag still wins over both the variable and the default.
"""

import os

# Controller of the Kassow arm.
ROBOT_IP = os.environ.get("RIO_KASSOW_IP", "192.168.1.44")
# Serial port of the AG95 gripper.
GRIPPER_PORT = os.environ.get("RIO_GRIPPER_PORT", "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG004P3Y-if00-port0")
# RealSense serials: wrist camera, default overview camera, and the camera only Jev reads.
WRIST_SERIAL = os.environ.get("RIO_WRIST_SERIAL", "112322070077")
OVERVIEW_SERIAL = os.environ.get("RIO_OVERVIEW_SERIAL", "262522075082")
JEV_SERIAL = os.environ.get("RIO_JEV_SERIAL", "262722073415")

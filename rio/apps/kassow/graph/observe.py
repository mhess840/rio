# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Let a planner see the cell before it writes a graph: camera photos and Jev answers.

Only the cameras run; the arm is not connected. A RealSense can be open only
once, so do not run these while ``graph run --no-dry-run`` or the orchestrator
holds the cameras.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from rio.jev import encode_jpeg

from .check import question_problems
from .locate import locate_questions, read_box, read_noul
from .model import OBJECT_MAX_CHARS
from .table_map import TableMap
from .vla import GRAPH_STATION

CAMERAS = ("jev", "overview", "wrist")
JevAsk = Callable[[bytes, dict], dict]


def look_cameras() -> dict:
    """RealSense specs of the Jev camera and the two policy cameras of the graph station."""
    from rio.apps.kassow.experiments import jev_camera, resolve_experiment
    from rio.apps.kassow.hardware import WRIST_SERIAL
    from rio.cfg import Camera

    def realsense(addr: str, serial: str) -> Camera:
        return Camera(
            addr=addr,
            cam_type="Realsense",
            serial=serial,
            model="D400",
            enable_depth=False,
            hardware_reset=False,
            resolution=(480, 640),
            resolution_depth=(480, 640),
        )

    return {
        "jev": jev_camera(),
        "overview": realsense("127.0.0.1:5131", resolve_experiment(GRAPH_STATION).overview_serial),
        "wrist": realsense("127.0.0.1:5130", WRIST_SERIAL),
    }


def look(root: Path, *, cameras: tuple[str, ...] = CAMERAS, mw: str = "Thread") -> dict[str, str]:
    """One settled photo per camera as ``<root>/<camera>.jpg``. Returns camera to path.

    Raises:
        ValueError: An unknown camera name.
        RuntimeError: A camera delivers no fresh frame.
    """
    from rio_hw import time
    from rio_hw.middleware import ServerManager

    from rio.apps.kassow.jev_ask import CAMERA_WARMUP_S, fresh_frame
    from rio.envs.factory import make_cameras

    unknown = set(cameras) - set(CAMERAS)
    if unknown:
        raise ValueError(f"cameras must be among {CAMERAS}, got {sorted(unknown)}")
    specs = {name: spec for name, spec in look_cameras().items() if name in cameras}
    servers, clients = make_cameras(mw, specs)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    with ExitStack() as stack:
        stack.enter_context(ServerManager(mw, list(servers.values())))
        handles = {name: stack.enter_context(clients[name]()) for name in specs}
        for handle in handles.values():
            fresh_frame(handle, max_age_s=1.0)
        time.sleep(CAMERA_WARMUP_S)
        for name, handle in handles.items():
            path = root / f"{name}.jpg"
            path.write_bytes(encode_jpeg(fresh_frame(handle, max_age_s=0.5)))
            paths[name] = str(path)
    return paths


def ask_photo(
    jpeg: bytes, ask: JevAsk, *, questions: dict | None = None, locate: str = "", table: TableMap | None = None
) -> dict:
    """Jev answers about one photo, or where an object is.

    With ``locate`` and a table map (the photo must come from the Jev camera), the
    result also holds the target in base coordinates and whether it is inside the
    calibrated area.

    Raises:
        ValueError: Neither or both of ``questions`` and ``locate``, an invalid question,
            or an object description that a locate node would refuse.
    """
    if (questions is None) == (not locate):
        raise ValueError("pass either questions or locate")
    if questions is not None:
        problems = question_problems(questions)
        if problems:
            raise ValueError("\n".join(problems))
        return {"answers": ask(jpeg, questions)}
    if len(locate) > OBJECT_MAX_CHARS or "`" in locate or not locate.strip():
        raise ValueError(f"locate needs a description of at most {OBJECT_MAX_CHARS} characters, no backticks")
    answers = ask(jpeg, locate_questions(locate))
    box = read_box(answers)
    result: dict = {"answers": answers, "visible": read_noul(answers), "box": asdict(box)}
    if table is not None:
        target = table.target(box, locate)
        result["target"] = {
            **target.to_dict(),
            "inside_area": table.contains(target.x, target.y),
            "grasp_z": round(table.z_at(target.x, target.y), 4),
        }
    return result

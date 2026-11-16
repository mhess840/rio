# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""The Kassow cell's tools for v3 workflows (``rio.graph``).

Each tool wraps an existing skill, Jev, a VLA, or the operator. Arm skills
report ``True``, ``False`` (refused or missed) or an alarm string. A tool turns
an alarm into an exception, so a subgraph's ``on_error`` catches it, and turns
``True``/``False`` into an ``exit`` value where the planner needs to branch.

    registry = cell_tools(skills, ask, table=table, run_vla=run_vla, ask_operator=ask_operator)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from rio.graph import MOTION, OPERATOR, PERCEPTION, Param, Tool, ToolRegistry, current_node

from .check import BOX_YES, NOUL_YES, _check_question, pose_is_file, resolve_pose
from .locate import BOX_KEY, VISIBLE_KEY, Box, locate_questions, read_box, read_noul
from .model import (
    APPROACH_M,
    APPROACH_M_RANGE,
    HOVER_M,
    HOVER_M_RANGE,
    INSTRUCTION_MAX_CHARS,
    OBJECT_MAX_CHARS,
    PLACE_Z_OFFSET_M,
    PLACE_Z_OFFSET_M_RANGE,
    VLA_TIMEOUT_S,
    VLA_TIMEOUT_S_RANGE,
    Z_OFFSET_M_RANGE,
)

if TYPE_CHECKING:
    from rio.sam import SamReply

    from .handeye import HandEye
    from .table_map import TableMap, Target
    from .walk import Skills
    from .wristeye import WristEye

Ask = Callable[[str, dict, str], dict]
"""``ask(node_path, questions, note)`` returns Jev's ``answers`` object."""
AskOperator = Callable[[str, str, list[str]], str]
"""``ask_operator(node_path, question, answers)`` returns one of ``answers``."""
RunVla = Callable[[str, str, float], object]


@dataclass(frozen=True)
class Shot:
    """One camera frame: the JPEG, depth in metres aligned to it if streamed, and the camera pose at capture.

    ``view`` is a calibrated camera in the base frame (``HandEye``): the fixed Jev or overview
    calibration, or the wrist camera at the TCP pose of the photo. None means the Jev calibration.
    """

    jpeg: bytes
    depth_m: np.ndarray | None = None
    view: HandEye | None = None


Photo = Callable[[str], "bytes | Shot"]
"""``photo(node_path)`` takes a fresh Jev photo and keeps it with the run."""
Segment = Callable[[bytes, str], "SamReply"]
"""``segment(jpeg, noun_phrase)`` asks SAM for every instance."""

TARGET = "Target"
LIFT_M = 0.1
TABLE_Z_M = -0.020
"""Table surface in the base frame: wrist-calibration marker plane (-0.021) and Jev depth (-0.025), 2026-11-16."""
OBJECT_HEIGHT_M = 0.04
LOOK_HEIGHT_M = 0.30
SEARCH_RADIUS_M = 0.10
SEARCH_VIEWS = 7
CAMERAS = ("jev", "overview", "wrist")
OPERATOR_MAX_ANSWERS = 9
"""The operator answers with one key, 1 to 9."""
SAM_YES = 0.8
"""SAM score that counts as found; 0.69 was a false hit on the orange gripper (2026-11-16)."""
SAM_MAX_PROMPTS = 6
SAM_PROMPT_MAX_CHARS = 60


class SkillAlarm(RuntimeError):
    """The arm reported an alarm; a subgraph's ``on_error`` takes it."""


def cell_tools(
    skills: Skills,
    ask: Ask,
    *,
    table: TableMap | None = None,
    run_vla: RunVla | None = None,
    ask_operator: AskOperator | None = None,
    photo: Photo | None = None,
    segment: Segment | None = None,
    handeye: HandEye | None = None,
    jaw_yaw: Callable[[], float | None] | None = None,
    views: dict[str, Photo] | None = None,
    wrist: WristEye | None = None,
    pose_exists: Callable[[str], bool] = pose_is_file,
) -> ToolRegistry:
    """Every tool of the cell, bound to live or dry-run parts.

    Args:
        skills: ``KassowSkills`` on the cell, ``DrySkills`` in a dry run.
        ask: Jev questions about a fresh Jev photo.
        table: Image-to-table map; ``jev.locate`` needs it.
        run_vla: ``run_vla(experiment, instruction, timeout_s)``; ``vla.run`` needs it.
        ask_operator: Asks the human at the cell; ``operator.ask`` needs it.
        photo: Fresh Jev photo; ``sam.locate`` needs it.
        segment: SAM on one photo and one noun phrase; ``sam.locate`` needs it.
        handeye: Camera pose; with depth in the photo ``sam.locate`` measures height and yaw.
        jaw_yaw: Jaw reference yaw for ``pick_at align_yaw`` (default: ``jaw_yaw.json`` next to the table map).
        views: More cameras for ``sam.locate camera=...``: ``overview`` and ``wrist`` photo functions whose
            shots carry their camera pose (``Shot.view``).
        wrist: Wrist-camera calibration; ``kassow.look_at`` needs it.
        pose_exists: Whether a home file exists (tests pass a stub).
    """

    def move_to(pose: str) -> dict:
        _must(skills.move_to(resolve_pose(pose)), f"move_to {pose}")
        return {}

    def lift(meters: float = LIFT_M) -> dict:
        _must(skills.lift(meters), "lift")
        return {}

    def gripper_open() -> dict:
        _must(skills.gripper("open"), "gripper open")
        return {}

    def gripper_close() -> dict:
        _must(skills.gripper("close"), "gripper close")
        return {}

    def gripper_grasp() -> dict:
        held = skills.pick()
        _alarm(held)
        return {"exit": "held" if held is True else "empty"}

    def pick_at(target: Target, approach_m: float = APPROACH_M, z_offset_m: float = 0.0, align_yaw: bool = False) -> dict:
        turn = 0.0
        if align_yaw and target.yaw_deg is not None:
            from .table_map import load_jaw_yaw, wrist_turn_deg

            reference = (jaw_yaw or load_jaw_yaw)()
            if reference is None:
                raise RuntimeError("align_yaw needs the jaw reference (depth_check --jaw-reference after one mean grasp)")
            turn = wrist_turn_deg(target.yaw_deg, target.yaw_period_deg, reference)
        if turn:
            held = skills.pick_at(target, approach_m=approach_m, z_offset_m=z_offset_m, yaw_offset_deg=turn)
        else:
            held = skills.pick_at(target, approach_m=approach_m, z_offset_m=z_offset_m)
        _alarm(held)
        return {"exit": "held" if held is True else "missed", "wrist_turn_deg": round(turn, 1)}

    def place_at(target: Target, approach_m: float = APPROACH_M, z_offset_m: float = PLACE_Z_OFFSET_M) -> dict:
        _must(skills.place_at(target, approach_m=approach_m, z_offset_m=z_offset_m), f"place_at {target.name}")
        return {}

    def move_above(target: Target, height_m: float = HOVER_M) -> dict:
        _must(skills.move_above(target, height_m=height_m), f"move_above {target.name}")
        return {}

    def locate(object: str, threshold: float = BOX_YES, visible_threshold: float = NOUL_YES, note: str = "") -> dict:
        if table is None:
            raise RuntimeError("jev.locate needs a table map")
        answers = ask(current_node(), locate_questions(object), note)
        visible = read_noul(answers, VISIBLE_KEY)
        box = read_box(answers, BOX_KEY)
        seen = {
            "confidence": box.confidence,
            "visible": visible,
            "box": {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom},
        }
        if visible < visible_threshold:
            return {"exit": "not_found", "detail": f"not visible ({visible:.2f})", **seen}
        if box.confidence < threshold:
            return {"exit": "not_found", "detail": f"box confidence {box.confidence:.2f}", **seen}
        target = table.target(box, object)
        if not table.contains(target.x, target.y):
            detail = f"x={target.x:.3f} y={target.y:.3f} is outside the calibrated area"
            return {"exit": "outside_area", "target": target, "detail": detail, **seen}
        return {"exit": "found", "target": target, "detail": json.dumps(target.to_dict()), **seen}

    def sam_locate(
        prompts: list[str],
        threshold: float = SAM_YES,
        use_depth: bool = False,
        camera: str = "jev",
        height_m: float = OBJECT_HEIGHT_M,
    ) -> dict:
        if table is None:
            raise RuntimeError("sam.locate needs a table map")
        if segment is None:
            raise RuntimeError("sam.locate needs SAM")
        if camera != "jev":
            return _locate_in_view(camera, prompts, threshold, height_m)
        if photo is None:
            raise RuntimeError("sam.locate needs the Jev camera")
        shot = photo(current_node())
        jpeg, depth = (shot.jpeg, shot.depth_m) if isinstance(shot, Shot) else (shot, None)
        tried: dict[str, float] = {}
        for prompt in prompts:
            reply = segment(jpeg, prompt)
            best = reply.best()
            score = float(best.score or 0.0) if best is not None else 0.0
            tried[prompt] = round(score, 3)
            if best is None or score < threshold:
                continue
            left, top, right, bottom = best.fractions()
            target = table.target(Box(left, top, right, bottom, score), prompt)
            measured = None
            if depth is not None and handeye is not None:
                measured = _measure(depth, best.box, handeye, table, target, jpeg)
                if use_depth and measured is not None:
                    target = _depth_target(measured, prompt, score)
                if measured is not None and measured.get("yaw_deg") is not None:
                    target = replace(target, yaw_deg=measured["yaw_deg"], yaw_period_deg=measured["period_deg"])
            seen = {
                "measured": measured,
                "source": "depth" if use_depth and measured is not None else "image",
                "prompt": prompt,
                "confidence": score,
                "box": {"left": left, "top": top, "right": right, "bottom": bottom},
                "count": sum(1 for i in reply.instances if (i.score or 0.0) >= threshold),
                "tried": tried,
                "target": target,
            }
            if not table.contains(target.x, target.y):
                detail = f"{prompt}: x={target.x:.3f} y={target.y:.3f} is outside the calibrated area"
                return {"exit": "outside_area", "detail": detail, **seen}
            return {"exit": "found", "detail": json.dumps(target.to_dict()), **seen}
        return {"exit": "not_found", "detail": f"no prompt reached {threshold}: {tried}", "tried": tried}

    def _locate_in_view(camera: str, prompts: list[str], threshold: float, height_m: float) -> dict:
        """SAM in a non-Jev camera: the box centre's ray meets the plane at the object's mid-height."""
        assert table is not None and segment is not None
        take = (views or {}).get(camera)
        if take is None:
            raise RuntimeError(f"sam.locate camera {camera!r} is not connected")
        shot = take(current_node())
        if not isinstance(shot, Shot) or shot.view is None:
            raise RuntimeError(f"camera {camera!r} gave no calibrated view")
        return _find_in_shot(shot, camera, prompts, threshold, height_m)

    def _find_in_shot(shot: Shot, camera: str, prompts: list[str], threshold: float, height_m: float) -> dict:
        """The first prompt whose best instance reaches ``threshold``, located through the shot's camera pose."""
        assert table is not None and segment is not None and shot.view is not None
        view = shot.view
        tried: dict[str, float] = {}
        for prompt in prompts:
            reply = segment(shot.jpeg, prompt)
            best = reply.best()
            score = float(best.score or 0.0) if best is not None else 0.0
            tried[prompt] = round(score, 3)
            if best is None or score < threshold:
                continue
            x1, y1, x2, y2 = best.box
            z = TABLE_Z_M + height_m / 2.0
            centre = view.pixel_to_plane((x1 + x2) / 2.0, (y1 + y2) / 2.0, z)
            left = view.pixel_to_plane(x1, (y1 + y2) / 2.0, z)
            right = view.pixel_to_plane(x2, (y1 + y2) / 2.0, z)
            if centre is None:
                continue
            tcp = centre[:2] - table.tool_xy()
            width = float(np.linalg.norm(right[:2] - left[:2])) if left is not None and right is not None else 0.0
            from .table_map import Target

            target = Target(prompt, float(tcp[0]), float(tcp[1]), width, width, score)
            measured = None
            if shot.depth_m is not None:
                measured = _measure(shot.depth_m, best.box, view, table, target, shot.jpeg)
                if measured is not None and measured.get("yaw_deg") is not None:
                    target = replace(target, yaw_deg=measured["yaw_deg"], yaw_period_deg=measured["period_deg"])
            w, h = best.size
            seen = {
                "camera": camera,
                "object_xyz": [round(float(v), 4) for v in centre],
                "measured": measured,
                "source": "image",
                "prompt": prompt,
                "confidence": score,
                "box": {"left": x1 / w, "top": y1 / h, "right": (x2 + 1) / w, "bottom": (y2 + 1) / h},
                "count": sum(1 for i in reply.instances if (i.score or 0.0) >= threshold),
                "tried": tried,
                "target": target,
            }
            if not table.contains(target.x, target.y):
                detail = f"{prompt} ({camera}): x={target.x:.3f} y={target.y:.3f} is outside the calibrated area"
                return {"exit": "outside_area", "detail": detail, **seen}
            return {"exit": "found", "detail": json.dumps(target.to_dict()), **seen}
        return {
            "exit": "not_found",
            "detail": f"{camera}: no prompt reached {threshold}: {tried}",
            "tried": tried,
            "camera": camera,
        }

    def search(
        prompts: list[str],
        target: Target | None = None,
        x: float | None = None,
        y: float | None = None,
        radius_m: float = SEARCH_RADIUS_M,
        height_m: float = LOOK_HEIGHT_M,
        object_height_m: float = OBJECT_HEIGHT_M,
        max_views: int = SEARCH_VIEWS,
        threshold: float = SAM_YES,
    ) -> dict:
        """Look around a rough hint with the wrist camera until SAM finds the object, then look again from straight above."""
        if wrist is None or table is None:
            raise RuntimeError("kassow.search needs the wrist calibration and the table map")
        take = (views or {}).get("wrist")
        if take is None:
            raise RuntimeError("kassow.search needs the wrist camera")
        if target is not None:
            hint = np.array([target.x, target.y]) + table.tool_xy()  # TCP target → object point
        elif x is not None and y is not None:
            hint = np.array([x, y])
        else:
            raise RuntimeError("kassow.search needs a target or x and y")
        looked: list[dict] = []
        found: dict | None = None
        for spot in search_spots(hint, radius_m, max_views):
            try:
                look_at(float(spot[0]), float(spot[1]), height_m)
            except RuntimeError as error:
                looked.append({"x": round(float(spot[0]), 3), "y": round(float(spot[1]), 3), "skipped": str(error)})
                continue
            result = _find_in_shot(_calibrated(take(current_node()), "wrist"), "wrist", prompts, threshold, object_height_m)
            looked.append({"x": round(float(spot[0]), 3), "y": round(float(spot[1]), 3), "tried": result.get("tried")})
            if result["exit"] != "not_found":
                found = result
                break
        if found is None:
            return {"exit": "not_found", "detail": f"{len(looked)} views around {hint.round(3).tolist()}", "views": looked}
        centre = np.asarray(found["object_xyz"][:2])
        try:  # sharpen: look again from straight above the object
            look_at(float(centre[0]), float(centre[1]), height_m)
            shot = _calibrated(take(current_node()), "wrist")
            again = _find_in_shot(shot, "wrist", [found["prompt"]], threshold * 0.8, object_height_m)
            if again["exit"] != "not_found":
                found = again
        except RuntimeError as error:
            looked.append({"x": round(float(centre[0]), 3), "y": round(float(centre[1]), 3), "skipped": str(error)})
        return {**found, "views": looked}

    def look_at(x: float, y: float, height_m: float = LOOK_HEIGHT_M) -> dict:
        """Move so the wrist camera looks straight at table point ``x, y`` from ``height_m`` above it."""
        if wrist is None or table is None:
            raise RuntimeError("kassow.look_at needs the wrist calibration and the table map")
        xyz, rotvec = wrist_view_pose(wrist, table.rotvec, np.array([x, y, TABLE_Z_M]), height_m)
        _must(skills.move_tcp(xyz, rotvec), f"look_at x={x:.3f} y={y:.3f}")
        return {"tcp": [round(float(v), 4) for v in xyz]}

    def jev_ask(question: dict, threshold: float | None = None, note: str = "") -> dict:
        answers = ask(current_node(), {"q": question}, note)
        answer = answers.get("q") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise RuntimeError("Jev gave no answer")
        return {"exit": _jev_exit(question, answer, threshold), "answer": answer}

    def operator_ask(question: str, answers: list[str] | None = None) -> dict:
        if ask_operator is None:
            raise RuntimeError("operator.ask needs an operator (collaborative mode)")
        options = list(answers or ["yes", "no"])
        reply = ask_operator(current_node(), question, options)
        if reply not in options:
            raise RuntimeError(f"operator answer {reply!r} is not one of {options}")
        return {"exit": reply}

    def vla(experiment: str, instruction: str = "", timeout_s: float = VLA_TIMEOUT_S) -> dict:
        if run_vla is None:
            raise RuntimeError("vla.run needs a VLA runner")
        result = run_vla(experiment, instruction, timeout_s)
        return {"exit": "finished" if result is True else "stopped", "detail": "" if result is True else str(result)}

    def check_pose(inputs: dict[str, Any]) -> list[str]:
        pose = inputs.get("pose")
        if isinstance(pose, str) and not pose_exists(pose):
            return [f"pose {pose!r} is not a home file (describe lists the poses)"]
        return []

    target_param = Param(TARGET, doc="target output of jev.locate")
    approach = Param("float", False, f"metres above the target before descending, {list(APPROACH_M_RANGE)}")
    return ToolRegistry(
        [
            Tool(
                "kassow.move_to",
                move_to,
                "moveJ to a saved joint pose (park, home, a VLA start)",
                inputs={"pose": Param("str", doc="home file name from describe")},
                tags=(MOTION,),
                check=check_pose,
            ),
            Tool(
                "kassow.lift",
                lift,
                "raise the TCP straight up",
                inputs={"meters": Param("float", False, "default 0.1")},
                tags=(MOTION,),
                check=_ranges({"meters": (0.01, 0.3)}),
            ),
            Tool(
                "kassow.pick_at",
                pick_at,
                "open, move above the target, descend to the calibrated grasp height, close, lift; "
                "exit held only if the object is still in the jaws",
                inputs={
                    "target": target_param,
                    "approach_m": approach,
                    "z_offset_m": Param("float", False, f"raise the grasp, {list(Z_OFFSET_M_RANGE)}"),
                    "align_yaw": Param(
                        "bool",
                        False,
                        "turn the wrist (at most 45 deg) to square the jaws to a square object's yaw from "
                        "sam.locate depth; round objects grasp as usual; default false",
                    ),
                },
                outputs={"exit": "str", "wrist_turn_deg": "float"},
                exits=("held", "missed"),
                tags=(MOTION,),
                check=_ranges({"approach_m": APPROACH_M_RANGE, "z_offset_m": Z_OFFSET_M_RANGE}),
            ),
            Tool(
                "kassow.place_at",
                place_at,
                "move above the target, descend to grasp height plus z_offset_m, open, lift; raises when nothing is held",
                inputs={
                    "target": target_param,
                    "approach_m": approach,
                    "z_offset_m": Param("float", False, f"height of what it is set on, {list(PLACE_Z_OFFSET_M_RANGE)}"),
                },
                tags=(MOTION,),
                check=_ranges({"approach_m": APPROACH_M_RANGE, "z_offset_m": PLACE_Z_OFFSET_M_RANGE}),
            ),
            Tool(
                "kassow.move_above",
                move_above,
                "move to height_m above the calibrated grasp height at the target",
                inputs={"target": target_param, "height_m": Param("float", False, f"{list(HOVER_M_RANGE)}")},
                tags=(MOTION,),
                check=_ranges({"height_m": HOVER_M_RANGE}),
            ),
            Tool("gripper.open", gripper_open, "open the jaws", tags=(MOTION,)),
            Tool("gripper.close", gripper_close, "close the jaws without checking for an object", tags=(MOTION,)),
            Tool(
                "gripper.grasp",
                gripper_grasp,
                "close the jaws; exit held only if an object blocks them",
                outputs={"exit": "str"},
                exits=("held", "empty"),
                tags=(MOTION,),
            ),
            Tool(
                "jev.locate",
                locate,
                "find one object in a fresh Jev photo and map it onto the table",
                inputs={
                    "object": Param("str", doc=f"what to find, at most {OBJECT_MAX_CHARS} characters"),
                    "threshold": Param("float", False, f"box confidence, default {BOX_YES}"),
                    "visible_threshold": Param("float", False, f"visibility noul, default {NOUL_YES}"),
                    "note": Param("str", False, "context for Jev"),
                },
                outputs={
                    "exit": "str",
                    "target": TARGET,
                    "detail": "str",
                    "confidence": "float",
                    "visible": "float",
                    "box": "dict",
                },
                exits=("found", "not_found", "outside_area"),
                tags=(PERCEPTION,),
                check=_check_locate,
            ),
            Tool(
                "sam.locate",
                sam_locate,
                "find one object in a fresh Jev photo with SAM 3.1 and map it onto the table; tries the noun "
                "phrases in order and takes the first whose best instance reaches the threshold",
                inputs={
                    "prompts": Param(
                        "list",
                        doc=f"1 to {SAM_MAX_PROMPTS} short noun phrases with the colour SAM sees, e.g. "
                        "['orange plate', 'plate']; test them with --command ask --sam first",
                    ),
                    "threshold": Param("float", False, f"SAM score, default {SAM_YES}"),
                    "use_depth": Param(
                        "bool",
                        False,
                        "take the target from depth (top-surface centre) instead of the image box; "
                        "default false until checked on the cell",
                    ),
                    "camera": Param(
                        "str",
                        False,
                        "jev (default, top-down), overview (fixed, from the front: sees what Jev misses) or "
                        "wrist (after kassow.look_at)",
                    ),
                    "height_m": Param("float", False, f"object height for overview/wrist, default {OBJECT_HEIGHT_M}"),
                },
                outputs={
                    "exit": "str",
                    "target": TARGET,
                    "detail": "str",
                    "prompt": "str",
                    "confidence": "float",
                    "box": "dict",
                    "count": "int",
                    "tried": "dict",
                    "measured": "dict",
                    "source": "str",
                    "camera": "str",
                    "object_xyz": "list",
                },
                exits=("found", "not_found", "outside_area"),
                tags=(PERCEPTION,),
                check=_check_sam,
            ),
            Tool(
                "kassow.search",
                search,
                "search with the wrist camera around a rough hint (a target from a coarse sam.locate, or x/y): look at "
                "the hint, then at a ring around it, stop at the first SAM hit and look again from straight above it",
                inputs={
                    "prompts": Param("list", doc="noun phrases as for sam.locate"),
                    "target": Param(TARGET, False, "hint: target output of a coarse sam.locate (e.g. camera overview)"),
                    "x": Param("float", False, "hint: table point x (base frame) when there is no target"),
                    "y": Param("float", False, "hint: table point y"),
                    "radius_m": Param("float", False, f"ring radius, default {SEARCH_RADIUS_M}"),
                    "height_m": Param("float", False, f"camera height above the table, default {LOOK_HEIGHT_M}"),
                    "object_height_m": Param("float", False, f"object height, default {OBJECT_HEIGHT_M}"),
                    "max_views": Param("int", False, f"views at most (centre + ring), default {SEARCH_VIEWS}"),
                    "threshold": Param("float", False, f"SAM score, default {SAM_YES}"),
                },
                outputs={
                    "exit": "str",
                    "target": TARGET,
                    "detail": "str",
                    "prompt": "str",
                    "confidence": "float",
                    "box": "dict",
                    "object_xyz": "list",
                    "camera": "str",
                    "views": "list",
                    "tried": "dict",
                    "measured": "dict",
                },
                exits=("found", "not_found", "outside_area"),
                tags=(MOTION, PERCEPTION),
                check=_check_search,
            ),
            Tool(
                "kassow.look_at",
                look_at,
                "move the arm so the wrist camera looks at a table point from above; then sam.locate camera=wrist",
                inputs={
                    "x": Param("float", doc="table point, base frame (m)"),
                    "y": Param("float", doc="table point, base frame (m)"),
                    "height_m": Param("float", False, f"camera height above the table, default {LOOK_HEIGHT_M}"),
                },
                outputs={"tcp": "list"},
                tags=(MOTION,),
                check=_ranges({"height_m": (0.15, 0.6)}),
            ),
            Tool(
                "jev.ask",
                jev_ask,
                "ask Jev one typed question about a fresh Jev photo; exit yes/no (noul, box), "
                "high/low (score) or the chosen criterion (choice)",
                inputs={
                    "question": Param("dict", doc="one Jev question: type, instructions, criteria"),
                    "threshold": Param("float", False, "noul/box default 0.7/0.6; score required"),
                    "note": Param("str", False, "context for Jev"),
                },
                outputs={"exit": "str", "answer": "dict"},
                tags=(PERCEPTION,),
                check=_check_jev,
            ),
            Tool(
                "operator.ask",
                operator_ask,
                "ask the operator at the cell; the answer is the exit (collaborative mode only)",
                inputs={
                    "question": Param("str", doc="one short question about what the operator sees"),
                    "answers": Param("list", False, f"2 to {OPERATOR_MAX_ANSWERS} answers, default yes/no"),
                },
                outputs={"exit": "str"},
                tags=(OPERATOR,),
                check=_check_operator,
            ),
            Tool(
                "vla.run",
                vla,
                "moveJ to the VLA's start pose, run the learned policy until it finishes, s, x, or the timeout",
                inputs={
                    "experiment": Param("str", doc="a VLA with usable true in describe"),
                    "instruction": Param("str", False, f"one of its instructions, at most {INSTRUCTION_MAX_CHARS} characters"),
                    "timeout_s": Param("float", False, f"{list(VLA_TIMEOUT_S_RANGE)}, default {VLA_TIMEOUT_S}"),
                },
                outputs={"exit": "str", "detail": "str"},
                exits=("finished", "stopped"),
                tags=(MOTION,),
                check=_check_vla,
            ),
        ]
    )


def _alarm(outcome: object) -> None:
    if isinstance(outcome, str):
        raise SkillAlarm(outcome)


def _must(outcome: object, what: str) -> None:
    _alarm(outcome)
    if outcome is not True:
        raise RuntimeError(f"{what} failed")


def _ranges(bounds: dict[str, tuple[float, float]]) -> Callable[[dict[str, Any]], list[str]]:
    def check(inputs: dict[str, Any]) -> list[str]:
        problems = []
        for key, (low, high) in bounds.items():
            if key not in inputs:
                continue
            value = inputs[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                problems.append(f"{key} must be a number from {low} to {high}")
        return problems

    return check


def _check_locate(inputs: dict[str, Any]) -> list[str]:
    problems = _ranges({"threshold": (0.0, 1.0), "visible_threshold": (0.0, 1.0)})(inputs)
    what = inputs.get("object")
    if what is not None and (not isinstance(what, str) or not what.strip() or len(what) > OBJECT_MAX_CHARS or "`" in what):
        problems.append(f"object must be text of at most {OBJECT_MAX_CHARS} characters without backticks")
    return problems


def wrist_view_pose(wrist: WristEye, rotvec: np.ndarray, point: np.ndarray, height_m: float) -> tuple[np.ndarray, np.ndarray]:
    """TCP pose (grasp orientation ``rotvec``) that puts the wrist camera's optical axis through ``point``
    with the camera ``height_m`` above it."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    mount = np.asarray(wrist.tcp_from_camera, dtype=float)
    axis = rotation @ mount[:3, :3] @ np.array([0.0, 0.0, 1.0])
    if axis[2] > -0.5:
        raise RuntimeError("the wrist camera does not look down in the grasp orientation")
    distance = height_m / -axis[2]
    camera = np.asarray(point, dtype=float) - distance * axis
    return camera - rotation @ mount[:3, 3], np.asarray(rotvec, dtype=float)


def _tool_xy(table: TableMap) -> np.ndarray:
    """Base-frame offset from the TCP to the grasped object's centre in x, y (mean grasp)."""
    from scipy.spatial.transform import Rotation

    offset = float(table.stats.get("tool_offset_m", 0.0) or 0.0)
    return Rotation.from_rotvec(table.rotvec).apply([0.0, 0.0, offset])[:2]


def _measure(
    depth: np.ndarray, box: tuple, handeye: HandEye, table: TableMap, target: Target, jpeg: bytes | None = None
) -> dict | None:
    """Depth measurement of the object in ``box``, with its TCP target and distance to the image target.

    With the colour photo, the yaw comes from the top face in colour (``yaw_source: colour``):
    depth edges are too smeared for small objects. Round objects (no yaw in depth) keep none.
    """
    from .depth import measure, top_face_yaw

    m = measure(depth, box, handeye)
    if m is None:
        return None
    tcp = np.array([m.x, m.y]) - _tool_xy(table)
    out = {
        **m.to_dict(),
        "tcp_x": round(float(tcp[0]), 4),
        "tcp_y": round(float(tcp[1]), 4),
        "d_image_mm": round(float(np.hypot(tcp[0] - target.x, tcp[1] - target.y)) * 1000, 1),
        "yaw_source": "depth",
    }
    image = _decode(jpeg)
    face = top_face_yaw(image, box, handeye, m.top_z) if image is not None else None
    if face is not None:
        out["depth_yaw_deg"] = out["yaw_deg"]
        out["top_face"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in face.items()}
        yaw = face["yaw_deg"]
        out |= {"yaw_deg": None if yaw is None else round(yaw, 2), "period_deg": face["period_deg"], "yaw_source": "colour"}
    return out


def _decode(jpeg: bytes | None) -> np.ndarray | None:
    if not jpeg:
        return None
    import cv2

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def _depth_target(measured: dict, name: str, score: float) -> Target:
    from .table_map import Target

    return Target(name, measured["tcp_x"], measured["tcp_y"], measured["length_m"], measured["width_m"], score)


def _calibrated(shot: bytes | Shot, camera: str) -> Shot:
    if not isinstance(shot, Shot) or shot.view is None:
        raise RuntimeError(f"camera {camera!r} gave no calibrated view")
    return shot


def search_spots(hint: np.ndarray, radius_m: float, max_views: int) -> list[np.ndarray]:
    """The hint, then six points on a ring around it, then a second ring, in a short walk."""
    spots = [np.asarray(hint, dtype=float)]
    for ring, count in ((1, 6), (2, 12)):
        for k in range(count):
            angle = 2 * np.pi * k / count
            spots.append(hint + ring * radius_m * np.array([np.cos(angle), np.sin(angle)]))
    return spots[: max(1, max_views)]


def _check_search(inputs: dict[str, Any]) -> list[str]:
    problems = _check_sam({k: v for k, v in inputs.items() if k in ("prompts", "threshold")})
    problems += _ranges({"radius_m": (0.02, 0.3), "height_m": (0.15, 0.6), "object_height_m": (0.005, 0.4)})(inputs)
    views = inputs.get("max_views")
    if views is not None and (not isinstance(views, int) or not 1 <= views <= 19):
        problems.append("max_views must be 1 to 19")
    if "target" not in inputs and not ("x" in inputs and "y" in inputs):
        problems.append("give a target ($ref of a coarse locate) or x and y")
    return problems


def _check_sam(inputs: dict[str, Any]) -> list[str]:
    problems = _ranges({"threshold": (0.0, 1.0), "height_m": (0.005, 0.4)})(inputs)
    if "camera" in inputs and inputs["camera"] not in CAMERAS:
        problems.append(f"camera must be one of {list(CAMERAS)}")
    prompts = inputs.get("prompts")
    if prompts is None:
        return problems
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= SAM_MAX_PROMPTS:
        return [*problems, f"prompts must be a list of 1 to {SAM_MAX_PROMPTS} noun phrases"]
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > SAM_PROMPT_MAX_CHARS:
            problems.append(f"prompt {prompt!r} must be text of at most {SAM_PROMPT_MAX_CHARS} characters")
        elif any(mark in prompt for mark in (",", ";", "+", "<|")) or " and " in f" {prompt} ":
            problems.append(f"prompt {prompt!r} must be one noun phrase (SAM returns nothing for lists)")
    return problems


def _check_jev(inputs: dict[str, Any]) -> list[str]:
    question = inputs.get("question")
    if question is None:
        return []
    problems = [p.replace("node 'jev.ask' ", "") for p in _check_question("jev.ask", "question", question)]
    kind = question.get("type") if isinstance(question, dict) else None
    if kind == "score" and not isinstance(inputs.get("threshold"), (int, float)):
        problems.append("a score question needs a numeric threshold")
    if kind in ("noul", "box"):
        problems += _ranges({"threshold": (0.0, 1.0)})(inputs)
    return problems


def _check_operator(inputs: dict[str, Any]) -> list[str]:
    answers = inputs.get("answers")
    if answers is None:
        return []
    ok = (
        isinstance(answers, list)
        and 2 <= len(answers) <= OPERATOR_MAX_ANSWERS
        and all(isinstance(a, str) and a for a in answers)
    )
    if not ok or len(set(answers)) != len(answers):
        return [f"answers must be 2 to {OPERATOR_MAX_ANSWERS} distinct names"]
    return []


def _check_vla(inputs: dict[str, Any]) -> list[str]:
    from rio.apps.kassow.experiments import EXPERIMENTS, resolve_experiment

    from .vla import GRAPH_STATION, vla_usable

    problems = _ranges({"timeout_s": VLA_TIMEOUT_S_RANGE})(inputs)
    experiment = inputs.get("experiment")
    if experiment is None:
        return problems
    if experiment not in EXPERIMENTS:
        return [*problems, f"experiment must be one of {sorted(EXPERIMENTS)}"]
    if not vla_usable(str(experiment)):
        serial = resolve_experiment(str(experiment)).overview_serial
        problems.append(f"vla {experiment!r} needs overview camera {serial}; the graph station runs {GRAPH_STATION!r}")
    instruction = inputs.get("instruction", "")
    if not isinstance(instruction, str) or len(instruction) > INSTRUCTION_MAX_CHARS:
        return [*problems, f"instruction must be text of at most {INSTRUCTION_MAX_CHARS} characters"]
    try:
        EXPERIMENTS[str(experiment)].select_home(instruction)
    except ValueError as error:
        problems.append(str(error))
    return problems


def _jev_exit(question: dict, answer: dict, threshold: float | None) -> str:
    kind = answer.get("type")
    if kind == "noul":
        return "yes" if float(answer["noul"]) >= (NOUL_YES if threshold is None else threshold) else "no"
    if kind == "box":
        return "yes" if float(answer.get("confidence", 0.0)) >= (BOX_YES if threshold is None else threshold) else "no"
    if kind == "score":
        if threshold is None:
            raise RuntimeError("a score question needs a threshold")
        return "high" if float(answer["score"]) >= threshold else "low"
    if kind == "choice":
        choice = str(answer.get("choice") or "")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or choice not in criteria:
            raise RuntimeError(f"Jev chose {choice!r}, not a criterion")
        return choice
    raise RuntimeError(f"Jev answer has type {kind!r}")

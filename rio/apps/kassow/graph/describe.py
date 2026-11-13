# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""What a planning agent may use on this cell, as one JSON object.

Node kinds and limits come from the same constants ``check`` enforces; poses,
VLAs, and the table map are read from this station's files.
"""

from __future__ import annotations

from rio.apps.kassow.experiments import EXPERIMENTS, PACKAGED_HOMES_DIR, homes_dir, load_state
from rio.jev.client import IMAGE_KEY

from .check import BOX_YES, NOUL_YES
from .model import (
    APPROACH_M,
    APPROACH_M_RANGE,
    GRIPPER_STATES,
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
from .table_map import TableMap, table_map_path
from .vla import GRAPH_STATION, vla_usable

OK_FAIL = ["ok", "fail"]


def node_kinds() -> dict[str, dict]:
    """Every node kind ``check`` accepts: what it does, its fields, and its edges."""
    return {
        "move_to": {"does": "moveJ to a saved joint pose", "fields": {"pose": "one of poses"}, "edges": OK_FAIL},
        "lift": {"does": "raise the TCP straight up", "fields": {"meters": "optional, > 0, default 0.1"}, "edges": OK_FAIL},
        "pick": {"does": "close the gripper; ok only if an object blocks the jaws", "fields": {}, "edges": OK_FAIL},
        "place": {"does": "open the gripper", "fields": {}, "edges": OK_FAIL},
        "gripper": {
            "does": "open or close the jaws without checking for an object",
            "fields": {"state": f"one of {list(GRIPPER_STATES)}"},
            "edges": OK_FAIL,
        },
        "locate": {
            "does": "Jev boxes the object in the Jev photo; the table map turns the box into base x, y",
            "fields": {
                "object": f"what to find, at most {OBJECT_MAX_CHARS} characters, no backticks",
                "store": "lower-case name the target is kept under",
                "threshold": f"optional box confidence, default {BOX_YES}",
                "visible_threshold": f"optional noul, default {NOUL_YES}",
                "note": "optional context for Jev",
            },
            "edges": ["found", "missing", "fail (optional, Jev error)"],
        },
        "pick_at": {
            "does": "open, move above the target, descend to the calibrated grasp height, close, lift; "
            "ok only if the object is still held",
            "fields": {
                "target": "store of a locate node",
                "approach_m": f"optional {list(APPROACH_M_RANGE)}, default {APPROACH_M}",
                "z_offset_m": f"optional {list(Z_OFFSET_M_RANGE)}, default 0; only raises the grasp",
            },
            "edges": OK_FAIL,
        },
        "place_at": {
            "does": "move above the target, descend to grasp height plus z_offset_m, open, lift; refused when nothing is held",
            "fields": {
                "target": "store of a locate node",
                "approach_m": f"optional {list(APPROACH_M_RANGE)}, default {APPROACH_M}",
                "z_offset_m": f"optional {list(PLACE_Z_OFFSET_M_RANGE)}, default {PLACE_Z_OFFSET_M}; "
                "height of what the object is set on",
            },
            "edges": OK_FAIL,
        },
        "move_above": {
            "does": "move to height_m above the calibrated grasp height at the target",
            "fields": {"target": "store of a locate node", "height_m": f"optional {list(HOVER_M_RANGE)}, default {HOVER_M}"},
            "edges": OK_FAIL,
        },
        "vla": {
            "does": "moveJ to the VLA's start pose, then run the learned policy until it finishes, s, x, or the timeout",
            "fields": {
                "experiment": "one of vla with usable true",
                "instruction": f"one of its instructions, at most {INSTRUCTION_MAX_CHARS} characters; "
                "empty takes the model's first",
                "timeout_s": f"optional {list(VLA_TIMEOUT_S_RANGE)}, default {VLA_TIMEOUT_S}",
            },
            "edges": OK_FAIL,
        },
        "jev": {
            "does": "ask Jev typed questions about the Jev photo and route on one answer",
            "fields": {
                "questions": "map of key to question (see jev)",
                "route": "key of the question that picks the edge",
                "threshold": "noul/box: 0..1, score: required number",
                "when": "choice: map of option to node",
                "note": "optional context",
            },
            "edges": ["noul/box: yes, no", "score: high, low", "choice: when map, optional else", "fail (optional)"],
        },
        "stop": {"does": "end the walk", "fields": {"reason": "text"}, "edges": []},
    }


def poses() -> list[str]:
    """Pose file names a ``move_to`` may use; captured ones override packaged ones."""
    names = {path.name for path in PACKAGED_HOMES_DIR.glob("*.json")}
    if homes_dir().is_dir():
        names |= {path.name for path in homes_dir().glob("*.json")}
    return sorted(names)


def vlas() -> dict[str, dict]:
    """Registered VLAs, their instructions, and whether the graph station can run them."""
    out = {}
    for name, experiment in EXPERIMENTS.items():
        if experiment.homes:
            instructions = [home.instruction for home in experiment.homes]
        else:
            state = load_state(experiment)
            model = state.get("model")
            meta = state["models"].get(model, {}) if model else {}
            instructions = list(meta.get("instructions") or [])
        out[name] = {"usable": vla_usable(name), "instructions": instructions}
    return out


def table_status() -> dict:
    """Whether locate and the target skills can run."""
    path = table_map_path()
    if not path.is_file():
        return {"ready": False, "path": str(path), "fix": "record (graph.calibrate), label and fit (graph.table_map)"}
    try:
        table = TableMap.load(path)
    except ValueError as error:
        return {"ready": False, "path": str(path), "fix": str(error)}
    return {
        "ready": True,
        "path": str(path),
        "camera": table.camera,
        "samples": table.stats.get("n"),
        "loo_max_m": table.stats.get("loo_max_m"),
        "area_xy": table.hull.round(3).tolist(),
    }


def describe() -> dict:
    """Everything a planner needs to write a graph for this cell."""
    return {
        "graph": {
            "fields": {
                "task": "what the graph should achieve",
                "start": "first node",
                "max_steps": "upper bound on visited nodes",
                "nodes": "map of node name to node",
            },
            "rules": [
                "every edge names a node; every node is reachable from start",
                "a target skill needs a locate node that stores its target",
                "keep max_steps small; loops end there",
            ],
        },
        "nodes": node_kinds(),
        "jev": {
            "image": f"refer to the photo as `{IMAGE_KEY}` in every instruction",
            "types": {
                "noul": "is this statement true? returns noul 0..1",
                "choice": "criteria map of option to description; returns choice and probabilities",
                "score": "criteria list of 2..10 levels; returns score",
                "box": "where is it? returns edges as image fractions and confidence; always a box, so ask a noul too",
            },
        },
        "poses": poses(),
        "vla": vlas(),
        "graph_station": GRAPH_STATION,
        "cameras": {
            "jev": "fixed, top-down; locate and jev nodes read it",
            "overview": "fixed; VLA input",
            "wrist": "on the gripper; VLA input",
        },
        "table_map": table_status(),
        "operator": "on the cell every target move and VLA waits for space; q ends the walk, s ends and x aborts a VLA",
    }

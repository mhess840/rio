# GaP v3 workflows on this cell

Full example that ran on the cell (v1 form) and dry-runs in v3:
`rio/apps/kassow/graph/workflows/cylinder_to_plate.json`. Code: `rio/graph/`
(format, validator, executor), `rio/apps/kassow/graph/cell.py` (tools).

## Shape

```json
{
  "version": 3,
  "meta": {"name": "<task>", "description": "<one sentence>"},
  "nodes": {
    "<stage>": {"type": "subgraph", "ref": "<stage>_sg"},
    "done":    {"type": "end", "status": "success"},
    "abort":   {"type": "end", "status": "failure",
                "recovery": [{"tool": "gripper.open", "inputs": {}},
                             {"tool": "kassow.move_to", "inputs": {"pose": "home_organize.json"}}]}
  },
  "edges": [["START", "<first stage>"]],
  "conditional_edges": {
    "<stage>": {"router_field": "exit", "mapping": {"<exit>": "<next stage or end>"}}
  },
  "subgraphs": {"<stage>_sg": { "...": "see below" }}
}
```

- Top level: `subgraph`, `tool`, `end` nodes. Inside a subgraph: `tool`, `noop`.
  `script`, `router` and streaming nodes are refused.
- Unknown keys are errors. `START`, `END`, `in` are reserved names.
- One node at a time: at most one static edge out of a node, and a node has
  either a static edge or a conditional edge.
- The top level ends on an `end` node, never on `END`.

## Subgraph

```json
"grasp_sg": {
  "skill": "grasping-top-down",
  "inputs":  {"cube": "Target"},
  "outputs": {},
  "nodes": {
    "pick":    {"type": "tool", "tool": "kassow.pick_at",
                "inputs": {"target": {"$ref": "in.cube"}, "approach_m": 0.15, "z_offset_m": 0.02}},
    "grasped": {"type": "noop"},
    "missed":  {"type": "noop"}
  },
  "edges": [["START", "pick"], ["grasped", "END"], ["missed", "END"]],
  "conditional_edges": {"pick": {"router_field": "exit", "mapping": {"held": "grasped", "missed": "missed"}}},
  "exit": {"router_field": null, "success_values": ["grasped", "missed"]},
  "on_error": "failed"
}
```

- The exit value is the name of the `noop` that reached `END`; list every such
  noop in `success_values` ("success" here means "a planned exit", including `missed`).
- `on_error` is the exit when any tool inside raises (arm alarm, Jev error,
  refused move). It must not be a node name. The top level must map it.
- The top-level mapping of a subgraph must cover exactly its
  `success_values` + `on_error`; the validator reports missing and extra ones.

## Data

- `{"$ref": "node.field"}` reads a node's output in the same scope;
  `{"$ref": "in.name"}` reads a subgraph input.
- Across subgraphs data moves only by name: a subgraph `outputs`
  `{"cube": {"$ref": "find.target"}}`, a later subgraph declares input
  `"cube": "Target"` and gets the latest value. Inputs must match an output
  name of some subgraph.
- Types come from the catalog (`Target` from `jev.locate`); the validator
  checks a `$ref`'s type against the input it feeds.

## Patterns

- **Park** (keep the arm out of the Jev photo before locating, and at the end):
  `kassow.move_to {"pose": "home_organize.json"}` → `parked`; `on_error: stuck`.
  One `park_sg` may be used by two top-level nodes.
- **Perceive**: `sam.locate {"prompts": ["orange plate", "plate"]}` → noops
  `found` / `not_found` / `outside_area`; `on_error: no_answer`; output
  `find.target`. Default threshold 0.8 (0.69 was a false hit on the orange
  gripper). Prompts come from the Look step. Words that matched on 2026-11-13:
  plate "orange plate", cylinder "blue cap"/"blue object", cube "red object";
  "red plate", "red cube", "blue cylinder" found nothing. Park the arm first.
  `jev.locate` still exists (Jev box, less repeatable: ≈10 mm vs < 1 mm).
  With depth (live runs, or `run --image` next to a `jev.depth.npz`) the output
  `measured` gives the object's height above the table, footprint and yaw
  (None for round objects); use the height e.g. for `place_at.z_offset_m`.
  Keep `use_depth` false until it is checked on the cell. Absolute heights from
  depth are ≈ 18 mm off (2026-11-16); relative heights are fine.
- **Grasp**: `kassow.pick_at` (exits `held`/`missed`). `z_offset_m` 0.02 and
  `approach_m` 0.15 worked for the cylinder. For a square object (cube) set
  `"align_yaw": true`: the wrist turns at most 45° so the jaws close on two
  faces; the yaw comes from `sam.locate`'s depth (the target carries
  `yaw_deg`). It needs the jaw reference (`jaw_yaw.json`, measured once with
  `depth_check --jaw-reference`); round objects grasp as usual, elongated ones
  are refused. Not yet checked on the cell.
- **Transport**: `kassow.place_at` (raises when nothing is held) → `placed`;
  `z_offset_m` is the height of what the object is set on (plate: 0.03).
- **Check with Jev**: `jev.ask` with one question
  (`{"type": "noul", "instructions": "Is the cube on the plate in `crate`?"}`),
  exits `yes`/`no`; choice questions exit with the criterion name.
- **Operator as sensor** (collaborative only): `operator.ask`
  `{"question": "...", "answers": ["empty", "full"]}`; the answer is the exit.
- **VLA**: `vla.run {"experiment": "organize", "instruction": "...", "timeout_s": 120}`,
  exits `finished`/`stopped`. It moves to the VLA's start pose itself.
- **Loop over items**: a perceive subgraph whose `not_found` exit means
  "nothing left" routes to `done`; after `transport` route back to the
  perceive node. The executor reruns the loop body; the visit cap (100 per
  scope) ends a loop that never exits.
- **Failure ends**: perception failures before any motion → a failure end
  without recovery (`nothing_to_do`); failures after motion → `abort` with recovery.

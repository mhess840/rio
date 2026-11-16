# VLA or graph tools?

Decide per stage, and write the reason into the `choose` event of `plan.jsonl`.

In the GaP paper π0.5 fell to ≈0.20 success when object poses varied while
the graph held 0.93–0.97. Learned policies earn their place where scripts are
hard to write, not for plain pick and place.

## Graph tools when

- the object is rigid and Jev finds it reliably (`ask --locate` confidence ≥ 0.4,
  inside the calibrated area);
- the motion is approach, grasp, lift, place at a located target;
- success can be checked (gripper `held`, a Jev question afterwards).

## VLA when

- contact and fine motion decide the outcome: inserting into the inlay,
  pushing, tidying a full crate;
- objects are deformable, or geometry and pose are hard to model;
- a usable VLA in `describe.vla` has a matching instruction (use it verbatim).

## Hybrid ("steered policy")

The graph does the coarse part (perceive, move above), the VLA the last
centimetres. Smaller start variation makes the VLA more reliable. On this cell
a `vla.run` always starts from the experiment's start pose, so today a hybrid
is: graph stages, then `vla.run`, then a Jev check.

## When neither fits

Do not improvise with the tools you have. Write a `recommend` event:

```json
{"event": "recommend", "stage": "insert", "kind": "vla",
 "instruction": "insert the part into the inlay slot", "start_pose": "home_organize.json",
 "cameras": ["overview", "wrist"], "demos": "about 50 teleop demos from the start pose",
 "why": "contact-rich insertion; no tool controls force or slides along the slot"}
```

`kind` is `vla` (train a policy), `tool` (a skill is missing, e.g. yaw-aware
grasp) or `sensor` (the cell cannot see something, e.g. crate fill level).
Then stop and report. Evidence that strengthens a recommendation: a graph tool
that failed repeatedly at the same node, or an `operator.ask` that no sensor
can answer yet.

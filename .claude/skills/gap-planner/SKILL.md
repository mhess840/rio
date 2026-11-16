---
name: gap-planner
description: Plan, check and run a task on the Kassow robot cell as a Graph-as-Policy workflow (GaP v3), either autonomously or with the operator as a sensor. Use when the user gives the cell a task ("put the cube on the plate", "sort the parts into the crate", "what is on the table?"), asks to plan, dry-run or run a graph on the arm, or asks why a graph run failed.
argument-hint: "<task> [autonomous|collaborative]"
---

# GaP planner for the Kassow cell

You plan one task as a typed graph of the cell's tools, check it, dry-run it,
and run it on the arm with the operator's approval. The graph is the policy:
it is the artifact that gets compared, improved and shown. The format and
roles follow GaP (Goldberg et al., arXiv 2607.05369) — see `TODO_graph_agent.md`.

Answer in the user's language. All commands run from the repository root with
`uv run`; the entry point is `uv run python -m rio.apps.kassow.graph --command <cmd>`.

## Modes

The user names the mode; ask once if they did not.

- **autonomous**: you use only the cell's sensors (cameras, Jev, gripper,
  measured TCP). Do not ask the operator what they see, and do not put
  `operator.ask` nodes in the graph. The operator only approves motions.
  When the cell cannot sense something the task needs, stop and report it.
- **collaborative**: the operator is one of your sensors. Ask them in chat
  while planning, and use `operator.ask` nodes where the graph needs their
  eyes at run time. Every such question is logged (see `references/plan_log.md`):
  it marks a sensor or skill the cell still lacks.

Approval of every motion is the same in both modes; "autonomous" never means
moving without the operator's go.

## Process

Keep a task folder `experiments/graph/plans/<task>/` and log each step to its
`plan.jsonl` as you go (`references/plan_log.md`); the live view reads it.

1. **Inventory.** `--command describe`. Read `v3.tools` (the catalog: inputs,
   outputs, exits, tags), `v3.rules`, `poses`, `vla` (usable + instructions)
   and `table_map.ready`. Use nothing that is not listed there.
2. **Look.** `--command look`, then view the `jev`, `overview` and `wrist` photos.
   Ground every object the task names: `--command ask --locate "<object>"
   --image <look folder>/jev.jpg`; note `visible`, the box confidence and
   `target.inside_area`.
3. **Decompose** (coordinator). Split the task into 2–6 stages, each one
   subgraph owned by one skill: e.g. park → perceive object → perceive
   destination → grasp → transport → park. Repeated work ("all items") is a
   loop: a conditional back edge, not copied nodes.
4. **Choose per stage** between graph tools and a VLA with
   `references/vla_or_graph.md`. If no tool and no usable VLA fits, do not
   improvise: write a recommendation (`recommend` event) and stop.
5. **Fill each subgraph** (subgraph agent) from the catalog, following the
   patterns in `references/workflow_v3.md`. Name success exits after what
   happened (`found`, `grasped`, `placed`), give every exit a target, and
   end failures after motion on an `abort` end with recovery
   (`gripper.open`, then `kassow.move_to` a free pose).
6. **Check** (validator): write `workflow.json`, run `--command check --graph
   <file>`, fix every error, at most two fix rounds, then report.
7. **Dry-run**: write `dry_script.json` (answers keyed by node path, e.g.
   `"cube/find"`) and run every path you expect at least once: the success
   path and each failure exit. `--command run --graph <file> --script <script>`.
8. **Ask for the go.** Show the graph as stages with their tools and targets,
   the expected path, and what the operator must watch. Wait for an explicit
   go in chat for this exact file.
9. **Run** live (`references/live_runs.md`): describe each motion before the
   operator approves it; `q` stops; only the E-stop halts a `moveJ` already sent.
10. **Read the run**: `events.jsonl` in the printed log folder, `look` again,
    and report the end node, the path, and what the cameras show.
11. **Improve** (on failure): find the failing node in the events, propose one
    change as a new version `workflow.v<n>.json` with a one-line reason
    (threshold, prompt, offset, other tool), and ask before running it.

## Hard rules

- Never change code under `rio/`, `scripts/` or `examples/`; write only into
  `experiments/graph/plans/<task>/`.
- Never run `--no-dry-run` before check and dry-run passed and the user agreed
  to this exact file in chat. Never pass `--no-confirm`.
- Never invent tools, inputs, poses, VLA instructions or limits; ranges come
  from the catalog and `check` enforces them.
- Do not start or touch the crate orchestrator; do not `look`, `ask` or run
  while it or another run holds the cameras.
- At most two replans of a failing run, then report what is missing.
- Photos go to the hosted Jev only if the user allowed it (`--jev.backend local` otherwise).

## Output

- **Plan**: mode, stages with the chosen skill and why (2–6 lines).
- **Graph**: file path, and the stage list with tools and targets.
- **Result**: end node and status, log folder, what the cameras show, and in
  collaborative mode the operator questions that a sensor should replace.

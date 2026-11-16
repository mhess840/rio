# Running on the cell

## Commands

```sh
G="uv run python -m rio.apps.kassow.graph"
$G --command describe
$G --command look                                   # photos: experiments/graph/look/<time>/
$G --command ask --locate "the red cube" --image <look>/jev.jpg
$G --command check --graph experiments/graph/plans/<task>/workflow.json
$G --command run   --graph <file> --script <dry_script.json>          # dry run (default)
$G --command run   --graph <file> --no-dry-run --gripper-open 0.5 --home-joint-speed 0.3
```

Options that mattered on 2026-11-13: `--gripper-open 0.5` (wider than the
`open` experiment's 0.3), `--home-joint-speed` 0.2–0.3 rad/s for first runs,
`--jev.backend local` when photos must stay in the network.

## Dry-run script

JSON object keyed by node path (top-level node / inner node):

```json
{
  "cube/find":  {"visible": {"type": "noul", "noul": 0.95},
                 "where": {"type": "box", "box": {"left": 0.346, "top": 0.600, "right": 0.386, "bottom": 0.650},
                           "confidence": 0.9}},
  "check/q":    {"q": {"type": "noul", "noul": 0.8}},
  "crate/ask":  "full"
}
```

`jev.locate` nodes take a `visible` noul and a `where` box (image fractions;
the box above maps inside the calibrated area). `jev.ask` nodes take
`{"q": <answer>}`. `operator.ask` nodes take the answer string. Write one
script per path you want to test (success, not found, missed).

## Driving the keys yourself

The operator approves in chat; you press the keys. Run the program in a pty
with stdin from a key file, and append keys when the operator says so:

```sh
K=experiments/graph/plans/<task>/keys; : > $K
script -qfec "uv run python -m rio.apps.kassow.graph --command run --graph <file> --no-dry-run ..." /dev/null \
  < <(tail -f $K) > experiments/graph/plans/<task>/live.log 2>&1 &
printf ' ' >> $K     # go (only after the operator's explicit go for this motion)
printf 'q' >> $K     # stop
printf '2' >> $K     # operator.ask: second answer
```

Watch `live.log` (Monitor). Before each motion, tell the operator the exact
target (node path, tool, x/y) printed at the prompt and wait for "weiter"/"go".
`s` ends and `x` aborts a running VLA. Only the E-stop halts a `moveJ` already
sent to the controller — say so before the first motion.

## After the run

The log folder (`experiments/graph/<time>/`) holds `workflow.json` (the exact
graph that ran), `events.jsonl` (one executor event per line: `tool_start`,
`tool_end`, `route`, `subgraph_end` with `exit`, `tool_error`, `end`,
`run_end`) and one Jev photo per perception node (`<stage>__<node>.jpg`).
Link the folder in the task's `plan.jsonl` (`run` event).

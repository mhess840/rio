# Task folder and planning log

Everything about one task lives in `experiments/graph/plans/<task>/`:

| File | Content |
|---|---|
| `plan.jsonl` | one planning event per line, appended as you go |
| `workflow.json` | the current graph; earlier versions `workflow.v1.json`, `workflow.v2.json`, … |
| `dry_script*.json` | dry-run answers per tested path |
| `report.md` | short result: mode, graph, runs, open questions, recommendations |

`plan.jsonl` is how the planning becomes visible (live view, Phase M) and how
graphs from different modes are compared (Phase J). Append with one `printf`/
`echo` per event, or write the whole line with your file tool; every line is
one JSON object with `t` (ISO time) and `event`:

| event | fields |
|---|---|
| `task` | `task`, `mode` (`autonomous`/`collaborative`) |
| `inventory` | `tools` (names), `vla` (usable names), `table_ready` |
| `observe` | `photos` (paths), `objects`: `[{object, visible, confidence, inside_area, x, y}]` |
| `operator` | `question`, `answer`, `why` (collaborative only; what a sensor would need to tell) |
| `decompose` | `stages`: `[{name, skill, goal}]` |
| `choose` | `stage`, `kind` (`tools`/`vla`/`hybrid`), `tools`, `why` |
| `recommend` | see `vla_or_graph.md` |
| `write` | `file`, `version` |
| `check` | `file`, `ok`, `issues` (strings) |
| `dry_run` | `file`, `script`, `status`, `end`, `log` |
| `approve` | `file`, `by` (`operator`), `note` |
| `run` | `file`, `status`, `end`, `log`, `reason` |
| `diagnose` | `log`, `node`, `cause` |
| `revise` | `from`, `to`, `change`, `why` |

Example:

```json
{"t": "2026-11-16T14:02:11", "event": "decompose", "stages": [{"name": "clear", "skill": "parking", "goal": "arm out of the Jev photo"}, {"name": "cube", "skill": "perceiving-objects", "goal": "find the red cube"}]}
{"t": "2026-11-16T14:02:40", "event": "choose", "stage": "grasp", "kind": "tools", "tools": ["kassow.pick_at"], "why": "rigid cube, Jev confidence 0.9, inside the area"}
```

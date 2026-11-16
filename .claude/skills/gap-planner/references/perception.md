# Perception: which sensor for which question

**SAM finds, Jev judges.** Decided with the operator on 2026-11-16 after the
SAM-vs-Jev comparison (`docs/sam-api.md`, `TODO_graph_agent.md` phase B).

## Finding where something is → SAM (`sam.locate`, `kassow.search`)

- Use the colour the camera sees and a short noun phrase; test the words in
  the look step (`--command ask --sam "a|b"`). One concept per prompt.
- SAM found every visible object Jev missed and repeats a position within
  1 mm (Jev about 10 mm). Do not use `jev.locate` for targets any more.
- Camera choice:
  - `jev` (default): fixed, top-down, depth for height and yaw.
  - `overview`: fixed, from the front at an angle; sees what Jev cannot (in
    front of the crate, occluded). Give `height_m`.
  - `wrist`: after `kassow.look_at`, close and sharp; or let
    `kassow.search` walk views around a rough hint and sharpen.
- Pattern: coarse `sam.locate` (any camera) → `kassow.search` with its
  `target` → `kassow.pick_at`.

## Judging, deciding, describing → Jev (`jev.ask`)

SAM only names things. Jev answers typed questions about a photo, with
thresholds the graph can route on at run time without an LLM:

- **Checks after an action** (checkpoints): `noul` "Is the small box inside
  the top-left compartment of the crate in `crate`?" → `yes`/`no`.
- **States and choices**: `noul` "Is compartment 2 empty?"; `choice`
  "Which compartment is empty?"; `score` "How tidy is the inlay?" (the crate
  orchestrator already relies on these).
- **Descriptions, not nouns**: "the top-left compartment", "the box closest
  to the crate", "the part without a label" → Jev `box` for a rough region,
  then SAM / `kassow.search` / depth for the precise point.
- **Safety before a motion**: "Is anything on the place spot?"

Refer to the photo as `` `crate` `` in Jev instructions (the client's image key).

## Not needed at run time

Choosing words, understanding the scene, picking the strategy: the planner
does that once while planning, looking at the photos itself.

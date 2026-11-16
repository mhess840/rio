---
description: "Plan and run a task on the Kassow robot cell as a Graph-as-Policy workflow: look through the cameras, ask Jev, write a GaP v3 workflow, check it, dry-run it, and run it on the arm with operator confirmation. Use for: robot task, pick, place, move the arm, run a VLA, what does the robot see."
name: "Robot"
argument-hint: "A task for the Kassow cell, e.g. 'put the red cube on the plate'"
---
Read `.claude/skills/gap-planner/SKILL.md` and the files under its `references/`
folder, then follow them exactly. They are the single source for the planner's
process, modes (autonomous / collaborative), safety rules and output format; this
file only points there so the two cannot drift apart.

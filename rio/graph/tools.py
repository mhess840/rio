# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Tools a workflow may call: one typed callable each, dispatched by a flat name.

A cell registers its tools (``kassow.move_to``, ``jev.locate``, ...). The same
declarations drive validation, the executor and the planner's catalog, so the
catalog cannot drift from what runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ANY = "Any"
MOTION = "motion"
"""Tag of tools that move the arm or run a policy; the executor asks before each call."""
PERCEPTION = "perception"
OPERATOR = "operator"
"""Tag of tools that ask the human at the cell (operator as a sensor)."""


@dataclass(frozen=True)
class Param:
    """One tool input."""

    type: str = ANY
    required: bool = True
    doc: str = ""


@dataclass(frozen=True)
class Tool:
    """One callable. ``fn(**inputs)`` returns a dict whose keys are ``outputs``.

    Attributes:
        name: Flat dispatch name, ``<provider>.<function>``.
        fn: The callable.
        summary: One line for the planner's catalog.
        inputs: Input name to its parameter.
        outputs: Output field to type name.
        exits: Values of the ``exit`` output, when the tool reports one to route on.
        tags: ``motion``, ``perception``, ``operator`` and the like.
    """

    name: str
    fn: Callable[..., dict[str, Any] | None]
    summary: str
    inputs: dict[str, Param] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    exits: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def describe(self) -> dict:
        """The catalog entry the planner reads."""
        entry: dict[str, Any] = {
            "summary": self.summary,
            "inputs": {
                name: {"type": p.type, "required": p.required, **({"doc": p.doc} if p.doc else {})}
                for name, p in self.inputs.items()
            },
            "outputs": dict(self.outputs),
        }
        if self.exits:
            entry["exits"] = list(self.exits)
        if self.tags:
            entry["tags"] = list(self.tags)
        return entry


class ToolRegistry:
    """Flat name to tool. Names are unique."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        """Register one tool.

        Raises:
            ValueError: The name is taken or has no ``provider.`` prefix.
        """
        if "." not in tool.name:
            raise ValueError(f"tool name {tool.name!r} needs a provider prefix like 'kassow.'")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is registered twice")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """The tool called ``name``.

        Raises:
            KeyError: Nothing is registered under ``name``.
        """
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"no tool {name!r}") from error

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> dict[str, dict]:
        """Every tool's catalog entry, by name."""
        return {name: self._tools[name].describe() for name in self.names()}


def types_match(produced: str, wanted: str) -> bool:
    """Whether a value of type ``produced`` may feed an input of type ``wanted``."""
    if ANY in (produced, wanted) or "dict" in (produced, wanted):
        return True
    if {produced, wanted} <= {"int", "float"}:
        return True
    return produced == wanted

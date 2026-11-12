# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Fail before any node starts when a hardware node's guarded driver import failed.

rio_hw guards every driver import of a node module::

    try:
        import evdev
    except ImportError:
        evdev = None

so a missing driver would only surface inside the already started node process.
``require_node_drivers`` reads those guards from the source of the module that
defines the node and reports each one whose bound names are still ``None``.
New rio_hw nodes are covered without a table entry.
"""

from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Callable
from functools import cache
from importlib import import_module
from importlib.util import resolve_name
from types import ModuleType

# Packages whose node modules follow the guard convention. rio's own nodes guard
# genuinely optional extras (psutil, robot_descriptions) and warn by themselves.
CHECKED_PACKAGES = frozenset({"rio_hw"})

# Guarded imports a node can run without, so they must not block it.
OPTIONAL_DRIVER_IMPORTS: dict[str, frozenset[str]] = {
    # Only the camera enumeration helper uses it; the node opens a device index.
    "Uvc": frozenset({"cv2_enumerate_cameras"}),
}

# Helper modules whose guarded imports the node needs. Helpers are not scanned by
# default because some of their guards are optional (``bleak`` for ViveTracker).
REQUIRED_HELPER_MODULES: dict[str, tuple[str, ...]] = {
    "KassowArm": ("rio_hw.robots.kassow_kinematics",),
}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(kind, ast.Name) and kind.id in {"ImportError", "ModuleNotFoundError"} for kind in kinds)


@cache
def guarded_imports(module_name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """List the top-level ``try: import … except ImportError`` statements of a loaded module.

    Args:
        module_name: Name of a module in ``sys.modules``.

    Returns:
        ``(imported module, names it binds)`` per guarded import statement.
    """
    module = sys.modules[module_name]
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        return ()
    package = module.__package__ or ""
    found: list[tuple[str, tuple[str, ...]]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Try) or not any(_catches_import_error(h) for h in statement.handlers):
            continue
        for inner in statement.body:
            if isinstance(inner, ast.Import):
                found += [(alias.name, (alias.asname or alias.name.partition(".")[0],)) for alias in inner.names]
            elif isinstance(inner, ast.ImportFrom):
                name = inner.module or ""
                if inner.level:
                    name = resolve_name("." * inner.level + name, package)
                bound = tuple(alias.asname or alias.name for alias in inner.names if alias.name != "*")
                if bound:
                    found.append((name, bound))
    return tuple(found)


def missing_guarded_imports(module: ModuleType, optional: frozenset[str] = frozenset()) -> list[str]:
    """Guarded imports of ``module`` whose names were left unset or ``None``.

    Args:
        module: A loaded module.
        optional: Imported module names that may be missing.

    Returns:
        Imported module names, in source order, without duplicates.
    """
    missing: list[str] = []
    for name, bound in guarded_imports(module.__name__):
        if name in optional or name in missing:
            continue
        if any(getattr(module, binding, None) is None for binding in bound):
            missing.append(name)
    return missing


def _failure(name: str) -> str:
    """``name`` plus the error of importing it now, so the operator sees the real cause."""
    try:
        import_module(name)
    except Exception as error:
        return f"{name} ({type(error).__name__}: {error})"
    return name


def require_node_drivers(node: str, node_server: Callable, package: str) -> None:
    """Raise before any node starts when a driver of ``node`` could not be imported.

    Args:
        node: Node class prefix, e.g. ``"Clicker"``.
        node_server: The node's server factory; its module is the one scanned.
        package: Package the node comes from.

    Raises:
        ImportError: Naming every missing driver module and the import error behind it.
    """
    if package not in CHECKED_PACKAGES:
        return
    modules = [sys.modules.get(getattr(node_server, "__module__", "") or "")]
    modules += [import_module(name) for name in REQUIRED_HELPER_MODULES.get(node, ())]
    optional = OPTIONAL_DRIVER_IMPORTS.get(node, frozenset())
    missing: list[str] = []
    for module in modules:
        if module is not None:
            missing += [name for name in missing_guarded_imports(module, optional) if name not in missing]
    if not missing:
        return
    hint = "rebuild/install the native KORD bridge" if "_kord_bridge" in missing else "install the station extra"
    raise ImportError(
        f"Node {node} needs the Python module(s) {', '.join(_failure(name) for name in missing)}; {hint} "
        "(Kassow station: uv sync --frozen --extra kassow --group pi). "
        "If the node can run without one of them, list it in rio.envs.drivers.OPTIONAL_DRIVER_IMPORTS."
    )

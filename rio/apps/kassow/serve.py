# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""π₀.₅ on the Kassow station; ``POLICY`` picks the config (default ``Pi05LerobotCfg``)::

uv run --group pi --no-group docs -m rio.apps.kassow.serve --policy-cfg.device cuda:1
POLICY=Pi05DaggerCfg uv run --group pi --no-group docs -m rio.apps.kassow.serve --dagger-round 1
"""

import os

from rio.apps.kassow.dagger import open_dagger
from rio.apps.kassow.policy_cfgs.pi05_dagger import Pi05DaggerCfg
from rio.apps.kassow.policy_cfgs.pi05_lerobot import Pi05LerobotCfg
from rio.apps.kassow.policy_cfgs.pi05_lerobot_joint import Pi05LerobotJointCfg
from rio.policies.serve import cli as serve_cli

POLICY_CFGS: dict[str, type] = {
    "Pi05LerobotCfg": Pi05LerobotCfg,
    "Pi05LerobotJointCfg": Pi05LerobotJointCfg,
    "Pi05DaggerCfg": Pi05DaggerCfg,
}


def cli() -> None:
    """Serve the ``POLICY`` config with DAgger takeover available."""
    name = os.environ.get("POLICY", "Pi05LerobotCfg")
    if name not in POLICY_CFGS:
        raise SystemExit(f"Unknown POLICY {name!r}; choose one of {sorted(POLICY_CFGS)}")
    serve_cli(POLICY_CFGS[name], open_dagger=open_dagger)


if __name__ == "__main__":
    cli()

#!/usr/bin/env bash
# Clear latched Kassow alarms after an E-stop, without sudo.
# Root is only needed for kord-clean-alarm's optional `-r <prio>`, which is not used here.
# Usage: scripts/kassow_clear_alarms.sh [controller_ip] [kord-clean-alarm options, default --all]
set -euo pipefail

ip="${1:-192.168.1.44}"
shift || true
[[ $# -eq 0 ]] && set -- --all

# The binary was built with gcov; redirect its .gcda writes so it doesn't spam warnings.
export GCOV_PREFIX="${TMPDIR:-/tmp}/kord-gcov"
export GCOV_PREFIX_STRIP=10

exec kord-clean-alarm -c "$ip" "$@"

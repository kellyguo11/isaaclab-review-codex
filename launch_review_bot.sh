#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

: "${HOME:?HOME must be set}"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DEFAULT_ENV_FILE="${HOME}/.config/isaaclab-review-bot/environment"
readonly REVIEW_BOT_ENV_FILE="${ISAACLAB_REVIEW_BOT_ENV_FILE:-${DEFAULT_ENV_FILE}}"

fail() {
    printf 'isaaclab-review-bot: %s\n' "$1" >&2
    exit 1
}

command -v uv >/dev/null 2>&1 || fail "uv is not installed or is not on PATH"
command -v stat >/dev/null 2>&1 || fail "stat is not installed or is not on PATH"
[[ -f "${REVIEW_BOT_ENV_FILE}" ]] || fail "environment file not found: ${REVIEW_BOT_ENV_FILE}"
[[ -r "${REVIEW_BOT_ENV_FILE}" ]] || fail "environment file is not readable: ${REVIEW_BOT_ENV_FILE}"

env_file_mode="$(stat -c '%a' -- "${REVIEW_BOT_ENV_FILE}")"
if (( (8#${env_file_mode} & 8#077) != 0 )); then
    fail "environment file must be owner-only; run: chmod 600 ${REVIEW_BOT_ENV_FILE}"
fi

# Never allow a personal GitHub credential inherited from the interactive shell
# or accidentally added to the private environment file into the bot process.
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
set -a
# shellcheck disable=SC1090
source "${REVIEW_BOT_ENV_FILE}"
set +a
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN

if (( $# == 0 )); then
    set -- --watch
fi

exec uv --directory "${SCRIPT_DIR}" run --no-project python "${SCRIPT_DIR}/local_review_bot.py" "$@"

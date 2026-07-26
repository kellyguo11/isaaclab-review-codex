# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the executable review-bot launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_LAUNCHER = Path(__file__).parent / "launch_review_bot.sh"


def _make_fake_uv(tmp_path: Path) -> Path:
    """Create a fake uv executable that reports its arguments and GitHub token state."""
    executable = tmp_path / "bin" / "uv"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env bash\nprintf 'args=%s\\n' \"$*\"\nprintf 'gh_token=%s\\n' \"${GH_TOKEN-unset}\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_launcher_defaults_to_watch_and_clears_personal_github_token(tmp_path: Path) -> None:
    """The direct launcher should safely start continuous watching by default."""
    fake_uv = _make_fake_uv(tmp_path)
    environment_file = tmp_path / ".config" / "isaaclab-review-bot" / "environment"
    environment_file.parent.mkdir(parents=True)
    environment_file.write_text("GH_TOKEN=personal-token-from-file\n", encoding="utf-8")
    environment_file.chmod(0o600)
    environment = os.environ.copy()
    environment.update(
        {
            "GH_TOKEN": "personal-token-from-shell",
            "HOME": str(tmp_path),
            "PATH": f"{fake_uv.parent}:{environment['PATH']}",
        }
    )

    result = subprocess.run([str(_LAUNCHER)], capture_output=True, check=False, env=environment, text=True)

    assert result.returncode == 0
    assert f"--directory {_LAUNCHER.parent}" in result.stdout
    assert f"python {_LAUNCHER.parent / 'local_review_bot.py'} --watch" in result.stdout
    assert "gh_token=unset" in result.stdout


def test_launcher_requires_an_owner_only_environment_file(tmp_path: Path) -> None:
    """The direct launcher should reject an environment file readable by other users."""
    fake_uv = _make_fake_uv(tmp_path)
    environment_file = tmp_path / "environment"
    environment_file.write_text("REVIEW_BOT_TEST_SETTING=enabled\n", encoding="utf-8")
    environment_file.chmod(0o644)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path),
            "ISAACLAB_REVIEW_BOT_ENV_FILE": str(environment_file),
            "PATH": f"{fake_uv.parent}:{environment['PATH']}",
        }
    )

    result = subprocess.run([str(_LAUNCHER)], capture_output=True, check=False, env=environment, text=True)

    assert result.returncode == 1
    assert "environment file must be owner-only" in result.stderr

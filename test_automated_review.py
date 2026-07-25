# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the automated GitHub App pull-request reviewer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = Path(__file__).with_name("automated_review.py")


def _load_review_module() -> ModuleType:
    """Load the review runner from the local GitHub scripts directory."""
    spec = importlib.util.spec_from_file_location("automated_review", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_changed_right_lines_tracks_additions_across_hunks() -> None:
    """Added right-side lines should be recovered from every diff hunk."""
    reviewer = _load_review_module()
    patch = """@@ -10,4 +10,5 @@ def update():
 context
-old_value = 1
+new_value = 1
+extra_value = 2
 tail
@@ -30,2 +31,2 @@ def close():
-return old
+return new
"""

    assert reviewer._changed_right_lines(patch) == {11, 12, 31}
    assert reviewer._format_line_ranges({11, 12, 31}) == "11-12,31"


def test_validate_findings_filters_invalid_locations_and_duplicates() -> None:
    """Only unique findings on added lines should be accepted."""
    reviewer = _load_review_module()
    findings = [
        {
            "path": "source/example.py",
            "line": 8,
            "severity": "warning",
            "title": "Real issue",
            "body": "The reset path leaves stale state.",
            "suggestion": "state = 0",
        },
        {
            "path": "source/example.py",
            "line": 8,
            "severity": "critical",
            "title": "Duplicate",
            "body": "Duplicate location.",
            "suggestion": "",
        },
        {
            "path": "source/example.py",
            "line": 9,
            "severity": "warning",
            "title": "Context line",
            "body": "This line was not added.",
            "suggestion": "",
        },
        {
            "path": "source/other.py",
            "line": 8,
            "severity": "warning",
            "title": "Wrong file",
            "body": "The file was not changed.",
            "suggestion": "",
        },
    ]

    validated = reviewer._validate_findings(findings, {"source/example.py": {8}})

    assert validated == [findings[0]]


def test_extract_structured_output_parses_output_text() -> None:
    """A completed structured response should decode its JSON output text."""
    reviewer = _load_review_module()
    response = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"summary":"ok","findings":[]}'}],
            },
        ],
    }

    assert reviewer._extract_structured_output(response) == {"summary": "ok", "findings": []}


def test_existing_review_requires_bot_login_and_matching_sha(monkeypatch) -> None:
    """A contributor-controlled marker must not suppress the bot review."""
    reviewer = _load_review_module()
    marker = "<!-- isaaclab-review-bot:sha=abc123 -->"
    reviews = [
        {"user": {"login": "contributor"}, "body": marker},
        {"user": {"login": "isaaclab-review-bot[bot]"}, "body": "different commit"},
    ]
    monkeypatch.setattr(reviewer, "_github_paginate", lambda path, token: reviews)

    assert not reviewer._has_existing_review("isaac-sim/IsaacLab", 10, marker, "token")

    reviews.append({"user": {"login": "isaaclab-review-bot[bot]"}, "body": marker})
    assert reviewer._has_existing_review("isaac-sim/IsaacLab", 10, marker, "token")


def test_fetch_raw_file_refuses_non_github_hosts() -> None:
    """Changed-file reads should never send requests to contributor-selected hosts."""
    reviewer = _load_review_module()

    assert reviewer._fetch_raw_file("https://example.com/payload.py") == ("", False)


def test_installation_token_must_be_scoped_only_to_target_repository(monkeypatch) -> None:
    """An installation token with access to another repository should be rejected."""
    reviewer = _load_review_module()
    monkeypatch.setattr(
        reviewer,
        "_github_paginate",
        lambda path, token: [
            {"full_name": "isaac-sim/IsaacLab"},
            {"full_name": "kellyguo11/private-repository"},
        ],
    )

    with pytest.raises(RuntimeError, match="scoped only"):
        reviewer._verify_installation_token("isaac-sim/IsaacLab", "installation-token")


def test_post_review_always_uses_comment_event(monkeypatch) -> None:
    """The runner should never approve a pull request or request changes."""
    reviewer = _load_review_module()
    captured = {}

    def fake_github_json(path, token, method="GET", payload=None):
        captured.update({"path": path, "token": token, "method": method, "payload": payload})
        return {"id": 42}

    monkeypatch.setattr(reviewer, "_github_json", fake_github_json)
    finding = {
        "path": "source/example.py",
        "line": 8,
        "severity": "critical",
        "title": "Incorrect reset",
        "body": "This retains state from the prior episode.",
        "suggestion": "state[env_ids] = 0",
    }

    response = reviewer._post_review(
        "isaac-sim/IsaacLab",
        10,
        "abc123",
        "Review body",
        [finding],
        "app-token",
    )

    assert response == {"id": 42}
    assert captured["payload"]["event"] == "COMMENT"
    assert captured["payload"]["commit_id"] == "abc123"
    assert captured["payload"]["comments"] == [
        {
            "path": "source/example.py",
            "line": 8,
            "side": "RIGHT",
            "body": (
                "🔴 Critical — **Incorrect reset**\n\n"
                "This retains state from the prior episode.\n\n"
                "```suggestion\nstate[env_ids] = 0\n```"
            ),
        }
    ]


def test_dry_run_prints_preview_without_posting(monkeypatch, capsys) -> None:
    """A dry run should print the proposed review and return a preview status."""
    reviewer = _load_review_module()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GitHub review posting must not run")

    monkeypatch.setattr(reviewer, "_post_review", fail_if_called)
    finding = {
        "path": "source/example.py",
        "line": 8,
        "severity": "warning",
        "title": "Incorrect reset",
        "body": "This retains state from the prior episode.",
        "suggestion": "",
    }

    status = reviewer._publish_or_preview(
        "isaac-sim/IsaacLab",
        10,
        "abc123",
        "Review body",
        [finding],
        "read-token",
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "Proposed review for isaac-sim/IsaacLab#10" in output
    assert "Proposed inline comment at source/example.py:8" in output
    assert "no GitHub review was posted" in output
    assert status is reviewer.ReviewStatus.PREVIEWED

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime regressions for non-blocking inline nits and finding severity."""

from __future__ import annotations

import pytest

from test_automated_review import _load_review_module


def _finding(severity: str, category: str = "style_consistency") -> dict:
    return {
        "path": "source/example.py",
        "line": 8,
        "side": "RIGHT",
        "category": category,
        "severity": severity,
        "title": "Use the adjacent naming convention",
        "body": "Rename the added variable to match the documented convention used by this module.",
    }


def test_suggestion_only_review_posts_inline_nit_without_blocking_verdict(monkeypatch) -> None:
    """A critic's stale strong verdict must not turn accepted nits into blocking issues."""
    reviewer = _load_review_module()
    monkeypatch.setattr(
        reviewer,
        "_request_verification_completion",
        lambda *args: {"accepted_finding_ids": [0], "verdict": "Needs rework"},
    )
    verified = reviewer._review_candidate_review("{}", {"findings": [_finding("suggestion")]}, ("model",), "key")
    validated = reviewer._validate_findings(verified["findings"], {"source/example.py": {"RIGHT": {8}}})
    body = reviewer._build_review_body(verified, validated, "<!-- marker -->", False)
    captured = {}

    def capture_post(endpoint, token, *, method, payload):
        captured.update(payload)
        return {"id": 1}

    monkeypatch.setattr(reviewer, "_github_json", capture_post)
    reviewer._post_review("owner/repo", 1, "head", body, validated, "token")

    assert verified["verdict"] == "No blocking issues"
    assert "**No blocking issues.**" in captured["body"]
    assert captured["event"] == "COMMENT"
    assert len(captured["comments"]) == 1
    comment = captured["comments"][0]
    assert (comment["path"], comment["line"], comment["side"]) == ("source/example.py", 8, "RIGHT")
    assert comment["body"].startswith("nit:")
    assert "Rename the added variable" in comment["body"]


@pytest.mark.parametrize("nit_first", [True, False])
def test_nit_cannot_suppress_critical_issue_at_same_location(nit_first) -> None:
    """Input ordering must not hide a serious finding behind a nit at the same line."""
    reviewer = _load_review_module()
    nit = _finding("suggestion")
    critical = {**_finding("critical", "implementation"), "title": "Reset leaves stale state"}
    findings = [nit, critical] if nit_first else [critical, nit]

    validated = reviewer._validate_findings(findings, {"source/example.py": {"RIGHT": {8}}})
    comments = reviewer._build_inline_comments(validated)

    assert len(comments) == 1
    assert "Critical" in comments[0]["body"]
    assert "Reset leaves stale state" in comments[0]["body"]
    assert not comments[0]["body"].startswith("nit:")


@pytest.mark.parametrize(
    ("severity", "category", "expected_verdict"),
    [
        ("warning", "implementation", "Minor fixes needed"),
        ("critical", "implementation", "Significant concerns"),
        ("suggestion", "compatibility", "Minor fixes needed"),
    ],
)
def test_retained_serious_findings_override_non_blocking_critic_verdict(
    monkeypatch, severity, category, expected_verdict
) -> None:
    """Warnings, critical issues, and compatibility breaks retain their severity through publishing."""
    reviewer = _load_review_module()
    monkeypatch.setattr(
        reviewer,
        "_request_verification_completion",
        lambda *args: {"accepted_finding_ids": [0], "verdict": "No blocking issues"},
    )
    verified = reviewer._review_candidate_review("{}", {"findings": [_finding(severity, category)]}, ("model",), "key")
    findings = reviewer._validate_findings(verified["findings"], {"source/example.py": {"RIGHT": {8}}})
    body = reviewer._build_review_body(verified, findings, "<!-- marker -->", False)

    assert verified["verdict"] == expected_verdict
    assert f"**{expected_verdict}.**" in body
    assert not reviewer._build_inline_comments(findings)[0]["body"].startswith("nit:")

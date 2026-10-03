# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime regressions for published finding severity."""

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


def test_suggestion_is_promoted_to_inline_warning(monkeypatch) -> None:
    """Legacy suggestion output should publish as a warning."""
    reviewer = _load_review_module()
    monkeypatch.setattr(
        reviewer,
        "_request_verification_completion",
        lambda *args: {"accepted_finding_ids": [0], "verdict": "No blocking issues"},
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

    assert verified["findings"][0]["severity"] == "warning"
    assert verified["verdict"] == "Minor fixes needed"
    assert "**Minor fixes needed.**" in captured["body"]
    assert captured["event"] == "COMMENT"
    assert len(captured["comments"]) == 1
    comment = captured["comments"][0]
    assert (comment["path"], comment["line"], comment["side"]) == ("source/example.py", 8, "RIGHT")
    assert comment["body"].startswith("🟡 Warning")
    assert "Rename the added variable" in comment["body"]


@pytest.mark.parametrize("suggestion_first", [True, False])
def test_promoted_suggestion_cannot_suppress_critical_issue_at_same_location(suggestion_first) -> None:
    """Input ordering must not hide a critical finding behind a promoted warning."""
    reviewer = _load_review_module()
    suggestion = _finding("suggestion")
    critical = {**_finding("critical", "implementation"), "title": "Reset leaves stale state"}
    findings = [suggestion, critical] if suggestion_first else [critical, suggestion]

    validated = reviewer._validate_findings(findings, {"source/example.py": {"RIGHT": {8}}})
    comments = reviewer._build_inline_comments(validated)

    assert len(comments) == 1
    assert "Critical" in comments[0]["body"]
    assert "Reset leaves stale state" in comments[0]["body"]
    assert comments[0]["body"].startswith("🔴 Critical")


@pytest.mark.parametrize(
    ("severity", "category", "expected_verdict"),
    [
        ("warning", "implementation", "Minor fixes needed"),
        ("critical", "implementation", "Significant concerns"),
        ("suggestion", "style_consistency", "Minor fixes needed"),
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
    assert reviewer._build_inline_comments(findings)[0]["body"].startswith(("🟡 Warning", "🔴 Critical"))


def test_finding_schema_only_allows_warning_or_critical() -> None:
    """Models should not be asked to emit suggestion severity."""
    reviewer = _load_review_module()

    assert reviewer._finding_schema()["properties"]["severity"]["enum"] == ["critical", "warning"]


def test_author_facing_findings_are_hard_limited() -> None:
    """Validation should bound verbose model output before publication."""
    reviewer = _load_review_module()
    finding = {
        **_finding("suggestion"),
        "title": "This title contains far too many words for a useful inline review comment",
        "body": " ".join(f"word{index}" for index in range(60)),
    }

    validated = reviewer._validate_findings([finding], {"source/example.py": {"RIGHT": {8}}})
    comment = reviewer._build_inline_comments(validated)[0]["body"]

    assert validated[0]["severity"] == "warning"
    assert len(validated[0]["title"].removesuffix("…").split()) == reviewer._MAX_FINDING_TITLE_WORDS
    assert len(validated[0]["body"].removesuffix("…").split()) == reviewer._MAX_FINDING_BODY_WORDS
    assert comment.startswith("🟡 Warning")

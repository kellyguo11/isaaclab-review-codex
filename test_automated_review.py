# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the automated GitHub App pull-request reviewer."""

from __future__ import annotations

import importlib.util
import json
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
            "category": "implementation",
            "severity": "warning",
            "title": "Real issue",
            "body": "The reset path leaves stale state.",
            "suggestion": "state = 0",
        },
        {
            "path": "source/example.py",
            "line": 8,
            "category": "implementation",
            "severity": "critical",
            "title": "Duplicate",
            "body": "Duplicate location.",
            "suggestion": "",
        },
        {
            "path": "source/example.py",
            "line": 9,
            "category": "implementation",
            "severity": "warning",
            "title": "Context line",
            "body": "This line was not added.",
            "suggestion": "",
        },
        {
            "path": "source/other.py",
            "line": 8,
            "category": "implementation",
            "severity": "warning",
            "title": "Wrong file",
            "body": "The file was not changed.",
            "suggestion": "",
        },
        {
            "path": "source/example.py",
            "line": 10,
            "category": "test_coverage",
            "severity": "suggestion",
            "title": "Missing test",
            "body": "This needs another test.",
            "suggestion": "",
        },
    ]

    validated = reviewer._validate_findings(findings, {"source/example.py": {8, 10}})

    assert validated == [findings[0]]


def test_validate_findings_has_no_numeric_cap() -> None:
    """Every high-confidence finding on a distinct added line should survive."""
    reviewer = _load_review_module()
    findings = [
        {
            "path": "source/example.py",
            "line": line,
            "category": "implementation",
            "severity": "warning",
            "title": f"Issue {line}",
            "body": f"Added line {line} introduces a demonstrated issue.",
            "suggestion": "",
        }
        for line in range(1, 6)
    ]

    validated = reviewer._validate_findings(findings, {"source/example.py": set(range(1, 6))})

    assert validated == findings


def test_extract_chat_completion_output_parses_json_content() -> None:
    """An OpenAI-compatible chat response should decode its JSON content."""
    reviewer = _load_review_module()
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": '{"summary":"ok","findings":[]}'}},
        ],
    }

    assert reviewer._extract_chat_completion_output(response) == {"summary": "ok", "findings": []}


def test_extract_chat_completion_output_tolerates_json_fence() -> None:
    """Provider-added JSON fences should not make an otherwise valid result fail."""
    reviewer = _load_review_module()
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": '```json\n{"summary":"ok","findings":[]}\n```'}},
        ],
    }

    assert reviewer._extract_chat_completion_output(response) == {"summary": "ok", "findings": []}


def test_extract_chat_completion_output_diagnoses_exhausted_reasoning_budget() -> None:
    """An empty reasoning response should report safe metadata instead of model text."""
    reviewer = _load_review_module()
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "provider_specific_fields": {"thinking_blocks": [{"type": "thinking"}]},
                },
            },
        ],
        "usage": {
            "completion_tokens": 16_384,
            "completion_tokens_details": {"reasoning_tokens": 16_384},
        },
    }

    with pytest.raises(
        RuntimeError,
        match=r"finish_reason='length'.*completion_tokens=16384.*reasoning_tokens=16384.*thinking_blocks=1",
    ):
        reviewer._extract_chat_completion_output(response)


def test_aggregate_completion_uses_nvidia_endpoint_and_fallback(monkeypatch) -> None:
    """A failed primary aggregation should retry with the other ensemble model."""
    reviewer = _load_review_module()
    calls = []

    def fake_request(url, token, method="GET", payload=None, accept="application/json", extra_headers=None):
        calls.append((url, token, method, payload))
        if payload["model"] == "primary-model":
            raise RuntimeError("primary unavailable")
        return {
            "choices": [
                {"message": {"role": "assistant", "content": '{"summary":"fallback","findings":[]}'}},
            ],
        }

    monkeypatch.setattr(reviewer, "_request_json", fake_request)

    result = reviewer._request_aggregate_completion(
        ("primary-model", "ensemble-model"),
        "Review carefully.",
        '{"pull_request":{}}',
        reviewer._specialist_schema(),
        "nvidia-key",
    )

    assert result == {"summary": "fallback", "findings": []}
    assert [call[3]["model"] for call in calls] == ["primary-model", "ensemble-model"]
    assert all(call[:3] == (reviewer._NVIDIA_CHAT_COMPLETIONS_URL, "nvidia-key", "POST") for call in calls)
    assert calls[1][3]["messages"][0]["role"] == "system"
    assert "JSON Schema" in calls[1][3]["messages"][0]["content"]
    assert calls[1][3]["max_tokens"] == 65_536


def test_specialist_ensemble_runs_every_role_on_every_model(monkeypatch) -> None:
    """Each specialist role should receive independent results from both models."""
    reviewer = _load_review_module()
    calls = []

    def fake_review_pass(role_name, role_instructions, review_input, model, api_key):
        calls.append((role_name, model, review_input, api_key))
        return {"summary": f"{role_name} from {model}", "findings": []}

    monkeypatch.setattr(reviewer, "_run_review_pass", fake_review_pass)

    results = reviewer._run_specialist_reviews("review-input", ("opus-model", "gpt-model"), "nvidia-key")

    assert len(results) == 6
    assert {(role, model) for role, model, _, _ in calls} == {
        (role, model)
        for role in ("design_architecture", "api_contract", "implementation_quality")
        for model in ("opus-model", "gpt-model")
    }
    assert {(result["review_pass"], result["model"]) for result in results} == {
        (role, model)
        for role in ("design_architecture", "api_contract", "implementation_quality")
        for model in ("opus-model", "gpt-model")
    }


def test_specialist_prompt_defaults_to_no_speculative_findings(monkeypatch) -> None:
    """The review prompt should explicitly favor precision over hypothetical concerns."""
    reviewer = _load_review_module()
    captured = {}

    def fake_completion(model, system_prompt, user_input, output_schema, api_key):
        captured["system_prompt"] = system_prompt
        return {"summary": "No findings.", "findings": []}

    monkeypatch.setattr(reviewer, "_request_model_completion", fake_completion)

    reviewer._run_review_pass(
        "design_architecture",
        "Review architecture.",
        '{"pull_request":{}}',
        "model",
        "key",
    )

    prompt = captured["system_prompt"]
    assert "The correct default is zero findings" in prompt
    assert "Do not report hypothetical edge cases" in prompt
    assert "reasonable maintainers could disagree" in prompt
    assert "Return every finding that satisfies this high bar" in prompt
    assert "do not add filler" in prompt


def test_aggregation_prompt_rejects_subjective_and_test_only_findings(monkeypatch) -> None:
    """The final validator should treat specialist claims as untrusted hypotheses."""
    reviewer = _load_review_module()
    captured = {}

    def fake_aggregate(models, system_prompt, user_input, output_schema, api_key):
        captured["system_prompt"] = system_prompt
        return {
            "summary": "No findings.",
            "design_architecture": "No material concerns.",
            "api_assessment": "No material concerns.",
            "implementation_assessment": "No material concerns.",
            "verdict": "Ship it",
            "findings": [],
        }

    monkeypatch.setattr(reviewer, "_request_aggregate_completion", fake_aggregate)

    reviewer._aggregate_reviews("{}", [], ("primary", "fallback"), "key")

    prompt = captured["system_prompt"]
    assert "False positives are substantially worse than missed findings" in prompt
    assert "Specialist repetition is not proof" in prompt
    assert "Never turn a test-coverage observation" in prompt
    assert "into an inline finding" in prompt
    assert "When uncertain, output no findings" in prompt


def test_prepublication_critic_can_only_accept_candidate_findings(monkeypatch) -> None:
    """The critic should use the other model first and be unable to invent findings."""
    reviewer = _load_review_module()
    captured = {}
    candidates = [
        {
            "path": "source/example.py",
            "line": 8,
            "category": "api",
            "severity": "warning",
            "title": "Breaks the public contract",
            "body": "The changed return type breaks existing callers.",
            "suggestion": "",
        },
        {
            "path": "source/example.py",
            "line": 12,
            "category": "implementation",
            "severity": "suggestion",
            "title": "Optional cleanup",
            "body": "This could use a different helper.",
            "suggestion": "",
        },
    ]
    candidate_review = {
        "summary": "Two concerns.",
        "design_architecture": "No material concerns.",
        "api_assessment": "One API concern.",
        "implementation_assessment": "One implementation concern.",
        "verdict": "Minor fixes needed",
        "findings": candidates,
    }

    def fake_verification(models, system_prompt, user_input, output_schema, api_key):
        captured.update(
            {
                "models": models,
                "system_prompt": system_prompt,
                "user_input": user_input,
                "output_schema": output_schema,
                "api_key": api_key,
            }
        )
        return {
            "summary": "One demonstrated API concern.",
            "design_architecture": "No material concerns.",
            "api_assessment": "The return contract is broken.",
            "implementation_assessment": "No material concerns.",
            "verdict": "Minor fixes needed",
            "accepted_finding_ids": [0, 99, 0],
        }

    monkeypatch.setattr(reviewer, "_request_verification_completion", fake_verification)

    verified = reviewer._review_candidate_review(
        '{"pull_request":{},"files":[]}',
        candidate_review,
        ("opus-model", "gpt-model"),
        "nvidia-key",
    )

    assert captured["models"] == ("gpt-model", "opus-model")
    assert captured["api_key"] == "nvidia-key"
    assert "really needs" in captured["system_prompt"]
    assert "fixing" in captured["system_prompt"]
    assert "Never create a new finding" in captured["system_prompt"]
    assert captured["output_schema"] == reviewer._critic_schema()
    critic_input = json.loads(captured["user_input"])
    assert [finding["candidate_id"] for finding in critic_input["CANDIDATE_REVIEW"]["findings"]] == [0, 1]
    assert verified["findings"] == [candidates[0]]


def test_prepublication_critic_fails_closed(monkeypatch) -> None:
    """A review should not bypass verification when every verifier fails."""
    reviewer = _load_review_module()

    def fail_verification(*args, **kwargs):
        raise RuntimeError("all verification models failed")

    monkeypatch.setattr(reviewer, "_request_verification_completion", fail_verification)

    with pytest.raises(RuntimeError, match="all verification models failed"):
        reviewer._review_candidate_review(
            '{"pull_request":{},"files":[]}',
            {"findings": []},
            ("opus-model", "gpt-model"),
            "nvidia-key",
        )


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
        "category": "implementation",
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
                "🔴 Critical · Implementation — **Incorrect reset**\n\n"
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
        "category": "implementation",
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

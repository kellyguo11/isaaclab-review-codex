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
    """Changed left- and right-side lines should be recovered from every diff hunk."""
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

    assert reviewer._changed_diff_lines(patch) == ({11, 30}, {11, 12, 31})
    assert reviewer._changed_right_lines(patch) == {11, 12, 31}
    assert reviewer._format_line_ranges({11, 12, 31}) == "11-12,31"


def test_fair_text_budget_preserves_short_entries_before_splitting() -> None:
    """Short patches should remain complete while large patches share the remainder."""
    reviewer = _load_review_module()

    assert reviewer._allocate_fair_text_budgets([10, 100, 100], 110) == [10, 50, 50]
    assert reviewer._allocate_fair_text_budgets([10, 20], 100) == [10, 20]
    assert reviewer._allocate_fair_text_budgets([10, 20], 0) == [0, 0]


def test_changed_file_excerpt_includes_late_changed_regions() -> None:
    """Current-file context should follow changed lines instead of taking a file prefix."""
    reviewer = _load_review_module()
    current_file = "\n".join(f"line {line}" for line in range(1, 151))

    excerpt = reviewer._changed_file_excerpt(current_file, {90, 140})

    assert "90: line 90" in excerpt
    assert "140: line 140" in excerpt
    assert "\n1: line 1\n" not in excerpt
    assert "[current file lines 50-150]" in excerpt


def test_review_context_preserves_complete_patch_and_changed_line_excerpts(monkeypatch) -> None:
    """The context builder should keep the full diff and sample current code at changed lines."""
    reviewer = _load_review_module()
    patch = "@@ -89,1 +90,1 @@\n-old value\n+new value\n"
    current_file = "\n".join(f"line {line}" if line != 90 else "new value" for line in range(1, 151))
    pull_request = {
        "number": 10,
        "title": "Change late code",
        "body": "",
        "user": {"login": "author"},
        "base": {"ref": "develop", "sha": "base-sha"},
        "head": {"ref": "feature", "sha": "head-sha"},
    }
    changed_files = [
        {
            "filename": "source/example.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": patch,
            "raw_url": "https://raw.githubusercontent.com/example/repo/head/source/example.py",
        }
    ]
    monkeypatch.setattr(reviewer, "_fetch_repository_file", lambda *args: "Trusted instructions")
    monkeypatch.setattr(reviewer, "_fetch_raw_file", lambda url: (current_file, False))

    review_input = reviewer._build_review_input("example/repo", pull_request, changed_files, "token")
    serialized = json.loads(review_input.serialized)

    assert serialized["files"][0]["patch"] == patch
    assert "90: new value" in serialized["files"][0]["current_file"]
    assert serialized["files"][0]["valid_added_line_ranges"] == "90"
    assert serialized["files"][0]["valid_deleted_line_ranges"] == "89"
    assert serialized["repository_instructions"] == "Trusted instructions"
    assert serialized["contribution_guidance"] == "Trusted instructions"
    assert serialized["test_audit_guidance"] == "Trusted instructions"
    assert serialized["test_audit_context"]["changed_test_files"] == []
    assert serialized["patches_truncated"] is False
    assert review_input.patches_truncated is False
    assert len(review_input.serialized) <= reviewer._MAX_CONTEXT_CHARS


def test_review_context_adds_bounded_test_audit_evidence(monkeypatch) -> None:
    """Changed tests should include full local context and related base-branch tests."""
    reviewer = _load_review_module()
    test_path = "source/pkg/test/test_widget.py"
    patch = "@@ -1,1 +1,2 @@\n existing\n+def test_new_contract(): pass\n"
    pull_request = {
        "number": 10,
        "title": "Add a regression",
        "body": "",
        "user": {"login": "author"},
        "base": {"ref": "develop", "sha": "base-sha"},
        "head": {"ref": "feature", "sha": "head-sha"},
    }
    changed_files = [
        {
            "filename": test_path,
            "status": "modified",
            "additions": 1,
            "deletions": 0,
            "patch": patch,
            "raw_url": f"https://raw.githubusercontent.com/example/repo/head/{test_path}",
        }
    ]

    def fake_repository_file(repository, path, ref, token):
        files = {
            "AGENTS.md": "Trusted instructions",
            "docs/source/refs/contributing.rst": (
                "Introduction\n============\n\nCoding Style\n------------\nLean code.\n\n"
                "Unit Testing\n------------\nDistinct tests.\n\nTools\n-----\nRun lint."
            ),
            "skills/developer/test-audit/SKILL.md": "Reject duplicate test ownership.",
            "source/pkg/test/test_widget_existing.py": "def test_existing_contract():\n    assert True\n",
            "tools/test_settings.py": "TIMEOUTS = {}\n",
        }
        return files.get(path, "")

    monkeypatch.setattr(reviewer, "_fetch_repository_file", fake_repository_file)
    monkeypatch.setattr(
        reviewer,
        "_github_json",
        lambda path, token: {
            "tree": [
                {"type": "blob", "path": test_path},
                {"type": "blob", "path": "source/pkg/test/test_widget_existing.py"},
                {"type": "blob", "path": "source/other/test/test_other.py"},
            ],
            "truncated": False,
        },
    )
    current_test = "def test_existing_contract():\n    assert True\n\ndef test_new_contract():\n    assert True\n"
    monkeypatch.setattr(reviewer, "_fetch_raw_file", lambda url: (current_test, False))

    review_input = reviewer._build_review_input("example/repo", pull_request, changed_files, "token")
    serialized = json.loads(review_input.serialized)
    audit_context = serialized["test_audit_context"]

    assert serialized["contribution_guidance"].startswith("Coding Style")
    assert "Distinct tests." in serialized["contribution_guidance"]
    assert "Run lint." not in serialized["contribution_guidance"]
    assert serialized["test_audit_guidance"] == "Reject duplicate test ownership."
    assert audit_context["changed_test_files"] == [{"path": test_path, "current_file": current_test, "complete": True}]
    assert "source/pkg/test/test_widget_existing.py" in audit_context["repository_test_inventory"]
    assert audit_context["related_existing_tests"][0]["path"] == "source/pkg/test/test_widget_existing.py"
    assert audit_context["ci_test_routing"] == "TIMEOUTS = {}\n"


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

    validated = reviewer._validate_findings(
        findings,
        {"source/example.py": {"LEFT": set(), "RIGHT": {8, 10}}},
    )

    assert validated == [{**findings[0], "side": "RIGHT", "suggestion": ""}]


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

    validated = reviewer._validate_findings(
        findings,
        {"source/example.py": {"LEFT": set(), "RIGHT": set(range(1, 6))}},
    )

    assert validated == [{**finding, "side": "RIGHT"} for finding in findings]


def test_validate_findings_accepts_style_and_test_quality_categories() -> None:
    """Style and test-audit findings should survive location validation."""
    reviewer = _load_review_module()
    findings = [
        {
            "path": "source/example.py",
            "line": 4,
            "category": "style_consistency",
            "severity": "suggestion",
            "title": "Prefer direct attribute access",
            "body": "The known field is accessed reflectively, unlike the documented local pattern.",
            "suggestion": "",
        },
        {
            "path": "source/test_example.py",
            "line": 9,
            "category": "test_quality",
            "severity": "suggestion",
            "title": "Duplicates the owner test",
            "body": "This repeats the same contract and fixture already covered by the owner-boundary test.",
            "suggestion": "",
        },
    ]

    assert reviewer._validate_findings(
        findings,
        {
            "source/example.py": {"LEFT": set(), "RIGHT": {4}},
            "source/test_example.py": {"LEFT": set(), "RIGHT": {9}},
        },
    ) == [{**finding, "side": "RIGHT"} for finding in findings]


def test_validate_findings_accepts_breaking_change_on_deleted_line() -> None:
    """A removed public contract should remain commentable on the diff's left side."""
    reviewer = _load_review_module()
    finding = {
        "path": "source/public_api.py",
        "line": 12,
        "side": "LEFT",
        "category": "compatibility",
        "severity": "suggestion",
        "title": "Public API removed without deprecation",
        "body": "The exported function is deleted without retaining a warning shim for the deprecation cycle.",
        "suggestion": "",
    }

    validated = reviewer._validate_findings(
        [finding],
        {"source/public_api.py": {"LEFT": {12}, "RIGHT": set()}},
    )

    assert validated == [{**finding, "severity": "warning"}]
    assert reviewer._build_inline_comments(validated)[0]["side"] == "LEFT"


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
        calls.append((role_name, role_instructions, model, review_input, api_key))
        return {"summary": f"{role_name} from {model}", "findings": []}

    monkeypatch.setattr(reviewer, "_run_review_pass", fake_review_pass)

    results = reviewer._run_specialist_reviews("review-input", ("opus-model", "gpt-model"), "nvidia-key")

    roles = ("design_architecture", "api_contract", "implementation_quality", "style_consistency", "test_quality")
    assert len(results) == 10
    assert {(role, model) for role, _, model, _, _ in calls} == {
        (role, model) for role in roles for model in ("opus-model", "gpt-model")
    }
    instructions = {role: role_instructions for role, role_instructions, _, _, _ in calls}
    assert "coordinate-basis" in instructions["design_architecture"]
    assert "internal zero-copy wrapper escaping" in instructions["api_contract"]
    assert (
        "Classify every directly evidenced incompatible change as category compatibility"
        in instructions["api_contract"]
    )
    assert "A release note, changelog entry" in instructions["api_contract"]
    assert "old entry point or behavior to remain functional" in instructions["api_contract"]
    assert "still-fresh stale source" in instructions["implementation_quality"]
    assert "documentation includes" in instructions["implementation_quality"]
    assert "changelog fragments for every touched package" in instructions["implementation_quality"]
    assert "files converted into thin delegates" in instructions["implementation_quality"]
    assert "new lean-code guidance from PR 8117" in instructions["style_consistency"]
    assert "direct attribute access" in instructions["style_consistency"]
    assert "Apply test_audit_guidance in authoring mode" in instructions["test_quality"]
    assert "duplicate tests" in instructions["test_quality"]
    assert {(result["review_pass"], result["model"]) for result in results} == {
        (role, model) for role in roles for model in ("opus-model", "gpt-model")
    }


def test_specialist_prompt_requires_adversarial_review_before_no_findings(monkeypatch) -> None:
    """The review prompt should require exhaustive analysis without inviting speculation."""
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

    prompt = " ".join(captured["system_prompt"].split())
    assert "Zero findings is acceptable only after completing" in prompt
    assert "conservatism is not a substitute for analysis" in prompt
    assert "Do not report hypothetical edge cases" in prompt
    assert "runtime reproduction is not required" in prompt
    assert "incompatible change to an existing public type" in prompt
    assert "Always use an empty suggestion" in prompt
    assert "Return every finding that satisfies this high bar" in prompt
    assert "do not add filler" in prompt
    assert "Read every patch hunk" in prompt
    assert "Compare the replacement with deleted behavior statement by statement" in prompt
    assert "documentation include" in prompt
    assert "Perform a second adversarial pass" in prompt
    assert "source-inspection test" in prompt
    assert "breaks an unchanged caller" in prompt
    assert "contribution_guidance" in prompt
    assert "test_audit_guidance" in prompt
    assert "style, formatting, naming" in prompt
    assert "audit every added test case" in prompt
    assert "Build an explicit compatibility ledger" in prompt
    assert "valid_deleted_line_ranges (LEFT)" in prompt
    assert "A changelog, migration note, major-version claim" in prompt


def test_aggregation_prompt_rechecks_every_file_before_no_findings(monkeypatch) -> None:
    """The final validator should recheck cross-file obligations without adding noise."""
    reviewer = _load_review_module()
    captured = {}

    def fake_aggregate(models, system_prompt, user_input, output_schema, api_key):
        captured["system_prompt"] = system_prompt
        return {
            "summary": "The generator change keeps discovery template-driven.",
            "design_architecture": "The new template follows the existing generator boundary.",
            "api_assessment": "Existing algorithm names and CLI inputs remain accepted.",
            "compatibility_assessment": "Breaking changes: none identified. Existing contracts remain available.",
            "implementation_assessment": "Template discovery and rendering paths remain aligned.",
            "style_assessment": "The templates follow the adjacent naming and typing patterns.",
            "test_assessment": "No tests were added or changed.",
            "verdict": "No blocking issues",
            "findings": [],
        }

    monkeypatch.setattr(reviewer, "_request_aggregate_completion", fake_aggregate)

    reviewer._aggregate_reviews("{}", [], ("primary", "fallback"), "key")

    prompt = " ".join(captured["system_prompt"].split())
    assert "Precision remains mandatory" in prompt
    assert "Specialist repetition is not proof" in prompt
    assert "missing-test or generic test-coverage observation" in prompt
    assert "deterministic compatibility" in prompt
    assert "type-contract failure" in prompt
    assert "new opt-in wrapper API" in prompt
    assert "Same-timestamp writes" in prompt
    assert "Every changed file has been examined" in prompt
    assert "textual markers" in prompt
    assert "trusted changelog rules" in prompt
    assert "Hydra or preset forwarding" in prompt
    assert "actively try to falsify" in prompt
    assert "broken by an added line" in prompt
    assert "summary and all six assessments must remain useful" in prompt
    assert "main, style, and test audits are deliberately picky" in prompt
    assert "Every added or changed test passes the test-audit authoring gate" in prompt
    assert "does not duplicate existing tests" in prompt
    assert "Every incompatible change" in prompt
    assert "old-versus-new compatibility ledger" in prompt
    assert "all six assessments" in prompt
    assert "Breaking changes: none identified." in prompt
    assert "valid_deleted_line_ranges (LEFT)" in prompt
    assert '"No blocking issues"' in prompt


def test_prepublication_critic_can_only_accept_candidate_findings(monkeypatch) -> None:
    """The critic should use the other model first and be unable to invent findings."""
    reviewer = _load_review_module()
    captured = {}
    candidates = [
        {
            "path": "source/example.py",
            "line": 8,
            "side": "RIGHT",
            "category": "api",
            "severity": "warning",
            "title": "Breaks the public contract",
            "body": "The changed return type breaks existing callers.",
            "suggestion": "",
        },
        {
            "path": "source/example.py",
            "line": 12,
            "side": "RIGHT",
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
        "compatibility_assessment": "Breaking changes: the public return contract changed without deprecation.",
        "implementation_assessment": "One implementation concern.",
        "style_assessment": "One style concern.",
        "test_assessment": "No tests changed.",
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
            "compatibility_assessment": "Breaking changes: the return type changed without a deprecation bridge.",
            "implementation_assessment": "No material concerns.",
            "style_assessment": "No demonstrated style violation.",
            "test_assessment": "No tests changed.",
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
    prompt = " ".join(captured["system_prompt"].split())
    assert "concrete design, architecture, API" in prompt
    assert "valid evidence" in prompt
    assert "without a runtime reproduction" in prompt
    assert "documentation integration failures" in prompt
    assert "impact appears in an unchanged caller" in prompt
    assert "Never create a new finding" in prompt
    assert "maintainability concern that warrants maintainer action" in prompt
    assert "do not raise the bar" in prompt
    assert "Do not reject a finding merely because it is style-only or test-only" in prompt
    assert "LEFT-side findings on deleted lines" in prompt
    assert "old-contract-functional" in prompt
    assert "all six assessments" in prompt
    assert "preserve useful" in prompt
    assert "PR-specific feedback" in prompt
    assert captured["output_schema"] == reviewer._critic_schema()
    critic_input = json.loads(captured["user_input"])
    assert [finding["candidate_id"] for finding in critic_input["CANDIDATE_REVIEW"]["findings"]] == [0, 1]
    assert verified["findings"] == [candidates[0]]


def test_prepublication_critic_preserves_specific_feedback_without_findings(monkeypatch) -> None:
    """A zero-finding result should retain the critic's pull-request-specific assessment."""
    reviewer = _load_review_module()

    def fake_verification(models, system_prompt, user_input, output_schema, api_key):
        return {
            "summary": "The template remains the source of truth for RSL-RL algorithm discovery.",
            "design_architecture": "The new distillation template stays within the existing generator boundary.",
            "api_assessment": "Existing PPO names and CLI inputs remain accepted.",
            "compatibility_assessment": "Breaking changes: none identified. Existing contracts remain available.",
            "implementation_assessment": "Discovery, rendering, and generated config naming were traced together.",
            "style_assessment": "The template follows adjacent naming and structure.",
            "test_assessment": "No tests were added or changed.",
            "verdict": "No blocking issues",
            "accepted_finding_ids": [],
        }

    monkeypatch.setattr(reviewer, "_request_verification_completion", fake_verification)

    verified = reviewer._review_candidate_review(
        '{"pull_request":{},"files":[]}',
        {
            "summary": "One candidate concern.",
            "design_architecture": "Review the template boundary.",
            "api_assessment": "Review the CLI contract.",
            "compatibility_assessment": "Review the compatibility ledger.",
            "implementation_assessment": "Review discovery and rendering.",
            "style_assessment": "Review adjacent style.",
            "test_assessment": "Review changed tests.",
            "verdict": "Minor fixes needed",
            "findings": [],
        },
        ("opus-model", "gpt-model"),
        "nvidia-key",
    )

    assert verified == {
        "summary": "The template remains the source of truth for RSL-RL algorithm discovery.",
        "design_architecture": "The new distillation template stays within the existing generator boundary.",
        "api_assessment": "Existing PPO names and CLI inputs remain accepted.",
        "compatibility_assessment": "Breaking changes: none identified. Existing contracts remain available.",
        "implementation_assessment": "Discovery, rendering, and generated config naming were traced together.",
        "style_assessment": "The template follows adjacent naming and structure.",
        "test_assessment": "No tests were added or changed.",
        "verdict": "No blocking issues",
        "findings": [],
    }


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


def test_review_body_keeps_specific_feedback_without_inline_findings() -> None:
    """A clean review should explain what was checked instead of posting boilerplate."""
    reviewer = _load_review_module()

    body = reviewer._build_review_body(
        {
            "summary": "The template remains the source of truth for RSL-RL algorithm discovery.",
            "design_architecture": "The new template follows the existing generator boundary.",
            "api_assessment": "Existing algorithm names and CLI inputs remain accepted.",
            "compatibility_assessment": "Breaking changes: none identified. Existing contracts remain available.",
            "implementation_assessment": "Discovery, rendering, and config naming were traced together.",
            "style_assessment": "The template follows adjacent naming and structure.",
            "test_assessment": "No tests were added or changed.",
            "verdict": "No blocking issues",
        },
        [],
        "<!-- marker -->",
        context_truncated=False,
    )

    assert "The template remains the source of truth" in body
    assert "**Compatibility and deprecation:** Breaking changes: none identified." in body
    assert "**Style consistency:** The template follows adjacent naming and structure." in body
    assert "**Test quality:** No tests were added or changed." in body
    assert "**No blocking issues.**" in body
    assert "No inline issue met the actionable-evidence threshold" in body
    assert "No material issues were identified" not in body
    assert "No material concerns" not in body
    assert "_Automated review; human maintainers own approval decisions._" in body


def test_review_body_does_not_call_actionable_findings_non_blocking() -> None:
    """A retained finding should not be presented as a no-blocking-issues result."""
    reviewer = _load_review_module()

    body = reviewer._build_review_body(
        {
            "summary": "A public contract was removed.",
            "design_architecture": "The module boundary is otherwise unchanged.",
            "api_assessment": "The old entry point is no longer available.",
            "compatibility_assessment": "Breaking changes: the entry point was removed without deprecation.",
            "implementation_assessment": "The replacement path is internally coherent.",
            "style_assessment": "The replacement follows local style.",
            "test_assessment": "No tests were added or changed.",
            "verdict": "No blocking issues",
        },
        [
            {
                "path": "source/public_api.py",
                "line": 12,
                "side": "LEFT",
                "category": "compatibility",
                "severity": "warning",
                "title": "Public API removed without deprecation",
                "body": "The old entry point was removed immediately.",
                "suggestion": "",
            }
        ],
        "<!-- marker -->",
        context_truncated=False,
    )

    assert "**Minor fixes needed.**" in body
    assert "**No blocking issues.**" not in body


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
        "side": "RIGHT",
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
            "body": "🔴 Critical · Implementation — **Incorrect reset**\n\nThis retains state from the prior episode.",
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
        "side": "RIGHT",
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

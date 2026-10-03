# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publication behavior for verified PR-description feedback."""

from __future__ import annotations

import json

import pytest

import automated_review as reviewer


@pytest.mark.parametrize("changed_field", [None, "title", "body"])
def test_description_feedback_is_verified_and_only_published_for_current_metadata(monkeypatch, changed_field) -> None:
    """Description feedback survives an empty inline review, but edits invalidate the reviewed metadata."""
    pull_request = {
        "state": "open",
        "draft": False,
        "title": "Reduce reset allocations",
        "body": "The reset is twice as fast. This also changes sensor processing.",
        "head": {"sha": "head-sha"},
    }
    latest_pull_request = dict(pull_request)
    if changed_field is not None:
        latest_pull_request[changed_field] = "Updated after review started"
    metadata = iter([pull_request, latest_pull_request])
    published = []
    schemas = []
    verified_description = (
        "Remove the sensor-processing claim: the diff only changes reset allocation. "
        "For the speedup claim, add a baseline/candidate comparison with revisions, workload, "
        "hardware, backend, environment count, warmup, and reset latency."
    )
    review_input = json.dumps({"pull_request": pull_request, "files": []})

    def github_json(path, token, method="GET", payload=None):
        if method == "POST":
            published.append(payload)
            return {"id": 123}
        return next(metadata)

    def aggregate_completion(models, prompt, serialized, schema, key):
        schemas.append(schema)
        return {
            "summary": "Reset reuses its buffer.",
            "description_assessment": "Unverified proposed feedback.",
            "findings": [],
            "verdict": "No blocking issues",
        }

    def verification_completion(models, prompt, serialized, schema, key):
        schemas.append(schema)
        candidate = json.loads(serialized)["CANDIDATE_REVIEW"]
        assert candidate["description_assessment"] == "Unverified proposed feedback."
        return {
            "summary": "Reset reuses its buffer.",
            "description_assessment": verified_description,
            "accepted_finding_ids": [],
            "verdict": "No blocking issues",
        }

    monkeypatch.setattr(reviewer, "_verify_installation_token", lambda *args: None)
    monkeypatch.setattr(reviewer, "_has_existing_review", lambda *args: False)
    monkeypatch.setattr(reviewer, "_add_review_start_reaction", lambda *args, **kwargs: None)
    monkeypatch.setattr(reviewer, "_github_json", github_json)
    monkeypatch.setattr(reviewer, "_github_paginate", lambda *args: [{"filename": "reset.py"}])
    monkeypatch.setattr(
        reviewer,
        "_build_review_input",
        lambda *args: reviewer.ReviewInput(review_input, {}, False, False),
    )
    monkeypatch.setattr(
        reviewer,
        "_run_specialist_reviews",
        lambda *args, **kwargs: [
            {"review_pass": "pr_description", "description_assessment": "Check scope and timing.", "findings": []}
        ],
    )
    monkeypatch.setattr(reviewer, "_request_aggregate_completion", aggregate_completion)
    monkeypatch.setattr(reviewer, "_request_verification_completion", verification_completion)

    status = reviewer.review_pull_request("example/repo", 10, "token", "inference-key")

    for schema in schemas:
        assert "description_assessment" in schema["schema"]["required"]
        assert schema["schema"]["properties"]["description_assessment"]["type"] == "string"
    if changed_field is not None:
        assert status is reviewer.ReviewStatus.STALE
        assert published == []
    else:
        assert status is reviewer.ReviewStatus.POSTED
        assert len(published) == 1
        assert published[0]["comments"] == []
        assert f"**PR description:** {verified_description}" in published[0]["body"]
        assert "Unverified proposed feedback" not in published[0]["body"]

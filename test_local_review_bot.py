# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the local GitHub App review-bot daemon."""

from __future__ import annotations

import base64
import datetime
import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_DIRECTORY = Path(__file__).parent


def _load_module(name: str, filename: str) -> ModuleType:
    """Load one review-bot module from the local tools directory."""
    spec = importlib.util.spec_from_file_location(name, _DIRECTORY / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_bot() -> ModuleType:
    """Load the reviewer and local daemon with their normal module names."""
    _load_module("automated_review", "automated_review.py")
    return _load_module("local_review_bot", "local_review_bot.py")


def _decode_jwt_part(value: str) -> dict:
    """Decode one JSON JWT segment."""
    padding = "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode(value + padding))


def _configuration(local_bot, tmp_path: Path):
    """Create a minimal test configuration."""
    return local_bot.BotConfiguration(
        repository="isaac-sim/IsaacLab",
        client_id="client-id",
        private_key_path=tmp_path / "private-key.pem",
        inference_api_key="nvidia-key",
        review_models=("opus-test", "gpt-test"),
        max_concurrent_model_requests=10,
    )


def test_parse_arguments_uses_short_pr_option() -> None:
    """The one-shot command should expose the concise ``--pr`` option."""
    local_bot = _load_local_bot()

    arguments = local_bot._parse_arguments(["--pr", "6704", "--dry-run"])

    assert arguments.pull_request_number == 6704
    assert arguments.dry_run


def test_rejects_inherited_personal_github_token(monkeypatch) -> None:
    """The daemon should fail closed when a personal GitHub token is inherited."""
    local_bot = _load_local_bot()
    monkeypatch.setenv("GH_TOKEN", "personal-token")

    with pytest.raises(RuntimeError, match="Refusing to start"):
        local_bot._reject_personal_github_credentials()


def test_configuration_is_locked_to_isaaclab_repository(monkeypatch) -> None:
    """The runtime should refuse a repository outside the intended App installation."""
    local_bot = _load_local_bot()
    monkeypatch.setenv("GITHUB_REPOSITORY", "kellyguo11/private-repository")

    with pytest.raises(RuntimeError, match="locked to isaac-sim/IsaacLab"):
        local_bot._load_configuration()


def test_configuration_uses_nvidia_inference_models(monkeypatch, tmp_path) -> None:
    """The daemon should use only the dedicated NVIDIA inference credential."""
    local_bot = _load_local_bot()
    monkeypatch.setenv("ISAACLAB_REVIEW_APP_PRIVATE_KEY_PATH", str(tmp_path / "app.pem"))
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "nvidia-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    configuration = local_bot._load_configuration()

    assert configuration.inference_api_key == "nvidia-key"
    assert configuration.review_models == (
        "azure/anthropic/claude-opus-5",
        "azure/openai/gpt-5.6-sol",
    )
    assert configuration.max_concurrent_model_requests == 10


def test_configuration_accepts_bounded_model_concurrency(monkeypatch, tmp_path) -> None:
    """Specialist concurrency should be configurable without exceeding the request count."""
    local_bot = _load_local_bot()
    monkeypatch.setenv("ISAACLAB_REVIEW_APP_PRIVATE_KEY_PATH", str(tmp_path / "app.pem"))
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "nvidia-key")
    monkeypatch.setenv("NVIDIA_REVIEW_MAX_CONCURRENT_REQUESTS", "6")

    assert local_bot._load_configuration().max_concurrent_model_requests == 6

    monkeypatch.setenv("NVIDIA_REVIEW_MAX_CONCURRENT_REQUESTS", "11")
    with pytest.raises(RuntimeError, match="must be between 1 and 10"):
        local_bot._load_configuration()


def test_create_app_jwt_uses_expected_claims_and_rs256(monkeypatch, tmp_path) -> None:
    """App JWTs should use GitHub's short-lived RS256 claim contract."""
    local_bot = _load_local_bot()
    captured = {}

    def fake_run(command, input, capture_output, check):
        captured.update(
            {
                "command": command,
                "input": input,
                "capture_output": capture_output,
                "check": check,
            }
        )
        return SimpleNamespace(returncode=0, stdout=b"signed", stderr=b"")

    monkeypatch.setattr(local_bot.subprocess, "run", fake_run)
    private_key_path = tmp_path / "app.pem"

    token = local_bot._create_app_jwt("client-id", private_key_path, now=10_000)

    header, payload, signature = token.split(".")
    assert _decode_jwt_part(header) == {"alg": "RS256", "typ": "JWT"}
    assert _decode_jwt_part(payload) == {"exp": 10_540, "iat": 9_940, "iss": "client-id"}
    assert base64.urlsafe_b64decode(signature + "==") == b"signed"
    assert captured["command"] == ["openssl", "dgst", "-sha256", "-sign", str(private_key_path)]
    assert captured["input"] == f"{header}.{payload}".encode()


def test_token_provider_requests_one_repository_and_read_only_permissions(monkeypatch, tmp_path) -> None:
    """A polling token should be limited to IsaacLab and read-only PR access."""
    local_bot = _load_local_bot()
    private_key_path = tmp_path / "app.pem"
    private_key_path.write_text("private key fixture", encoding="utf-8")
    private_key_path.chmod(0o600)
    calls = []

    monkeypatch.setattr(local_bot, "_create_app_jwt", lambda client_id, key_path: "app-jwt")

    def fake_github_app_json(path, token, method="GET", payload=None):
        calls.append((path, token, method, payload))
        if path == "/app":
            return {"slug": "isaaclab-review-bot"}
        if path.endswith("/installation"):
            return {"id": 123}
        return {
            "token": "installation-token",
            "expires_at": "2099-01-01T00:00:00Z",
            "permissions": {"contents": "read", "pull_requests": "read"},
        }

    monkeypatch.setattr(local_bot, "_github_app_json", fake_github_app_json)
    monkeypatch.setattr(local_bot.automated_review, "_verify_installation_token", lambda repository, token: None)
    provider = local_bot.GitHubAppTokenProvider(
        "isaac-sim/IsaacLab",
        "client-id",
        private_key_path,
    )

    assert provider.get_token(write=False) == "installation-token"
    assert calls[-1] == (
        "/app/installations/123/access_tokens",
        "app-jwt",
        "POST",
        {
            "repositories": ["IsaacLab"],
            "permissions": {"contents": "read", "pull_requests": "read"},
        },
    )
    assert stat.S_IMODE(private_key_path.stat().st_mode) == 0o600


def test_write_token_includes_issue_reaction_permission(monkeypatch, tmp_path) -> None:
    """A posting token should be able to acknowledge review starts on PR comments."""
    local_bot = _load_local_bot()
    private_key_path = tmp_path / "app.pem"
    private_key_path.write_text("private key fixture", encoding="utf-8")
    private_key_path.chmod(0o600)
    calls = []

    monkeypatch.setattr(local_bot, "_create_app_jwt", lambda client_id, key_path: "app-jwt")

    def fake_github_app_json(path, token, method="GET", payload=None):
        calls.append((path, token, method, payload))
        if path == "/app":
            return {"slug": "isaaclab-review-bot"}
        if path.endswith("/installation"):
            return {"id": 123}
        return {
            "token": "installation-token",
            "expires_at": "2099-01-01T00:00:00Z",
            "permissions": {"contents": "read", "issues": "write", "pull_requests": "write"},
        }

    monkeypatch.setattr(local_bot, "_github_app_json", fake_github_app_json)
    monkeypatch.setattr(local_bot.automated_review, "_verify_installation_token", lambda repository, token: None)
    provider = local_bot.GitHubAppTokenProvider("isaac-sim/IsaacLab", "client-id", private_key_path)

    assert provider.get_token(write=True) == "installation-token"
    assert calls[-1][3] == {
        "repositories": ["IsaacLab"],
        "permissions": {"contents": "read", "issues": "write", "pull_requests": "write"},
    }


def test_token_provider_rejects_a_different_github_app(monkeypatch, tmp_path) -> None:
    """The private key must authenticate the expected review-bot App."""
    local_bot = _load_local_bot()
    private_key_path = tmp_path / "app.pem"
    private_key_path.write_text("private key fixture", encoding="utf-8")
    private_key_path.chmod(0o600)
    monkeypatch.setattr(local_bot, "_create_app_jwt", lambda client_id, key_path: "app-jwt")
    monkeypatch.setattr(local_bot, "_github_app_json", lambda *args, **kwargs: {"slug": "different-app"})
    provider = local_bot.GitHubAppTokenProvider(
        "isaac-sim/IsaacLab",
        "client-id",
        private_key_path,
    )

    with pytest.raises(RuntimeError, match="does not authenticate isaaclab-review-bot"):
        provider.get_token(write=False)


def test_first_poll_baselines_existing_pull_requests_without_reviewing(monkeypatch, tmp_path) -> None:
    """First startup should not unexpectedly review the current PR backlog."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    provider = SimpleNamespace(get_token=lambda write: "read-token")
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(
        local_bot,
        "_list_open_pull_requests",
        lambda repository, token: [
            local_bot.PullRequestHead(number=10, head_sha="head-10"),
            local_bot.PullRequestHead(number=20, head_sha="head-20"),
        ],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("review must not run while initializing the baseline")

    monkeypatch.setattr(local_bot.automated_review, "review_pull_request", fail_if_called)

    initialized = local_bot._poll_once(configuration, provider, state_file, backfill=False)

    assert initialized
    assert local_bot._load_state(state_file, configuration.repository) == {10: "head-10", 20: "head-20"}
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600


def test_poll_does_not_review_new_commit_on_seen_pull_request(monkeypatch, tmp_path) -> None:
    """A later head on a known PR should wait for an explicit review command."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    state_file = tmp_path / "state.json"
    local_bot._save_state(state_file, configuration.repository, {10: "old-head"})
    token_requests = []
    provider = SimpleNamespace(get_token=lambda write: token_requests.append(write) or "read-token")
    monkeypatch.setattr(
        local_bot,
        "_list_open_pull_requests",
        lambda repository, token: [local_bot.PullRequestHead(number=10, head_sha="new-head")],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a commit on an existing PR must not trigger a full review")

    monkeypatch.setattr(local_bot.automated_review, "review_pull_request", fail_if_called)

    initialized = local_bot._poll_once(configuration, provider, state_file, backfill=False)

    assert not initialized
    assert token_requests == [False]
    assert local_bot._load_state(state_file, configuration.repository) == {10: "old-head"}


def test_poll_reviews_new_pull_request_once_and_preserves_prior_state(monkeypatch, tmp_path) -> None:
    """A new PR number should be reviewed while seen and closed PRs stay recorded."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    state_file = tmp_path / "state.json"
    local_bot._save_state(state_file, configuration.repository, {10: "initial-head", 11: "closed-head"})
    token_requests = []
    provider = SimpleNamespace(
        get_token=lambda write: token_requests.append(write) or ("write-token" if write else "read-token")
    )
    monkeypatch.setattr(
        local_bot,
        "_list_open_pull_requests",
        lambda repository, token: [
            local_bot.PullRequestHead(number=10, head_sha="later-head"),
            local_bot.PullRequestHead(number=20, head_sha="new-head"),
        ],
    )
    reviews = []

    def fake_review(repository, number, token, api_key, models, max_concurrent_model_requests):
        reviews.append((repository, number, token, api_key, models, max_concurrent_model_requests))
        return local_bot.automated_review.ReviewStatus.POSTED

    monkeypatch.setattr(local_bot.automated_review, "review_pull_request", fake_review)

    initialized = local_bot._poll_once(configuration, provider, state_file, backfill=False)

    assert not initialized
    assert token_requests == [False, True]
    assert reviews == [
        ("isaac-sim/IsaacLab", 20, "write-token", "nvidia-key", ("opus-test", "gpt-test"), 10),
    ]
    assert local_bot._load_state(state_file, configuration.repository) == {
        10: "initial-head",
        11: "closed-head",
        20: "new-head",
    }


def test_review_command_matching_and_authorization(monkeypatch) -> None:
    """Exact commands should allow the PR author and users with write-level access."""
    local_bot = _load_local_bot()
    pull_request = {"user": {"login": "external-author"}}
    permission_calls = []

    def fake_github_json(path, token):
        permission_calls.append((path, token))
        permission = "admin" if path.endswith("/maintainer/permission") else "read"
        return {"permission": permission}

    monkeypatch.setattr(local_bot.automated_review, "_github_json", fake_github_json)

    assert local_bot._is_review_command("@isaaclab-review-bot review")
    assert local_bot._is_review_command("  /isaaclab-review  ")
    assert not local_bot._is_review_command("please @isaaclab-review-bot review this")
    assert local_bot._is_authorized_review_request(
        {"user": {"login": "external-author"}, "author_association": "NONE"},
        pull_request,
        "isaac-sim/IsaacLab",
        "read-token",
    )
    assert local_bot._is_authorized_review_request(
        {"user": {"login": "maintainer"}, "author_association": "CONTRIBUTOR"},
        pull_request,
        "isaac-sim/IsaacLab",
        "read-token",
    )
    assert not local_bot._is_authorized_review_request(
        {"user": {"login": "unrelated-user"}, "author_association": "MEMBER"},
        pull_request,
        "isaac-sim/IsaacLab",
        "read-token",
    )
    assert not local_bot._is_authorized_review_request(
        {"user": {"login": "isaaclab-review-bot[bot]"}, "author_association": "MEMBER"},
        pull_request,
        "isaac-sim/IsaacLab",
        "read-token",
    )
    assert permission_calls == [
        ("/repos/isaac-sim/IsaacLab/collaborators/maintainer/permission", "read-token"),
        ("/repos/isaac-sim/IsaacLab/collaborators/unrelated-user/permission", "read-token"),
    ]


def test_first_command_poll_baselines_without_processing_old_comments(tmp_path) -> None:
    """A fresh command cursor should ignore comments made before monitoring starts."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    command_state_file = tmp_path / "state-commands.json"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("initializing the command cursor must not call GitHub")

    provider = SimpleNamespace(get_token=fail_if_called)

    local_bot._poll_review_commands(configuration, provider, command_state_file)

    state = local_bot._load_command_state(command_state_file, configuration.repository)
    assert state is not None
    assert not state.processed_comment_ids
    assert stat.S_IMODE(command_state_file.stat().st_mode) == 0o600


def test_pr_author_command_reviews_current_head_once(monkeypatch, tmp_path) -> None:
    """A PR author conversation command should launch one current-head review."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    command_state_file = tmp_path / "state-commands.json"
    local_bot._save_command_state(
        command_state_file,
        configuration.repository,
        local_bot._CommandPollState(
            updated_after=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            processed_comment_ids=frozenset(),
        ),
    )
    comment = {
        "id": 123,
        "body": "@isaaclab-review-bot review",
        "issue_url": "https://api.github.com/repos/isaac-sim/IsaacLab/issues/20",
        "user": {"login": "external-author"},
        "author_association": "NONE",
    }
    monkeypatch.setattr(
        local_bot,
        "_list_updated_issue_comments",
        lambda repository, token, updated_after: [comment],
    )
    monkeypatch.setattr(
        local_bot.automated_review,
        "_github_json",
        lambda path, token: {"state": "open", "head": {"sha": "current-head"}, "user": {"login": "external-author"}},
    )
    token_requests = []
    provider = SimpleNamespace(
        get_token=lambda write: token_requests.append(write) or ("write-token" if write else "read-token")
    )
    reviews = []

    def fake_review(
        repository,
        number,
        token,
        api_key,
        models,
        max_concurrent_model_requests,
        acknowledgement_comment_id,
    ):
        reviews.append(
            (
                repository,
                number,
                token,
                api_key,
                models,
                max_concurrent_model_requests,
                acknowledgement_comment_id,
            )
        )
        return local_bot.automated_review.ReviewStatus.POSTED

    monkeypatch.setattr(local_bot.automated_review, "review_pull_request", fake_review)

    local_bot._poll_review_commands(configuration, provider, command_state_file)

    assert token_requests == [False, True]
    assert reviews == [
        ("isaac-sim/IsaacLab", 20, "write-token", "nvidia-key", ("opus-test", "gpt-test"), 10, 123),
    ]
    state = local_bot._load_command_state(command_state_file, configuration.repository)
    assert state is not None
    assert state.processed_comment_ids == frozenset({123})


def test_untrusted_commenter_cannot_trigger_review(monkeypatch, tmp_path) -> None:
    """A non-author without repository association should not spend inference."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    command_state_file = tmp_path / "state-commands.json"
    local_bot._save_command_state(
        command_state_file,
        configuration.repository,
        local_bot._CommandPollState(
            updated_after=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            processed_comment_ids=frozenset(),
        ),
    )
    monkeypatch.setattr(
        local_bot,
        "_list_updated_issue_comments",
        lambda repository, token, updated_after: [
            {
                "id": 456,
                "body": "/isaaclab-review",
                "issue_url": "https://api.github.com/repos/isaac-sim/IsaacLab/issues/20",
                "user": {"login": "unrelated-user"},
                "author_association": "NONE",
            }
        ],
    )
    monkeypatch.setattr(
        local_bot.automated_review,
        "_github_json",
        lambda path, token: (
            {"permission": "read"}
            if path.endswith("/collaborators/unrelated-user/permission")
            else {"state": "open", "head": {"sha": "current-head"}, "user": {"login": "pr-author"}}
        ),
    )
    token_requests = []
    provider = SimpleNamespace(get_token=lambda write: token_requests.append(write) or "read-token")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("an untrusted commenter must not trigger model inference")

    monkeypatch.setattr(local_bot.automated_review, "review_pull_request", fail_if_called)

    local_bot._poll_review_commands(configuration, provider, command_state_file)

    assert token_requests == [False]
    state = local_bot._load_command_state(command_state_file, configuration.repository)
    assert state is not None
    assert state.processed_comment_ids == frozenset({456})

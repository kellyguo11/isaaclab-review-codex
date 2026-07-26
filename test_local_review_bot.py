# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the local GitHub App review-bot daemon."""

from __future__ import annotations

import base64
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
    )


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


def test_poll_reviews_new_heads_with_write_installation_token(monkeypatch, tmp_path) -> None:
    """An unseen PR head should be reviewed and persisted."""
    local_bot = _load_local_bot()
    configuration = _configuration(local_bot, tmp_path)
    state_file = tmp_path / "state.json"
    local_bot._save_state(state_file, configuration.repository, {10: "old-head"})
    token_requests = []
    provider = SimpleNamespace(
        get_token=lambda write: token_requests.append(write) or ("write-token" if write else "read-token")
    )
    monkeypatch.setattr(
        local_bot,
        "_list_open_pull_requests",
        lambda repository, token: [local_bot.PullRequestHead(number=10, head_sha="new-head")],
    )
    reviews = []

    def fake_review(repository, number, token, api_key, models):
        reviews.append((repository, number, token, api_key, models))
        return local_bot.automated_review.ReviewStatus.POSTED

    monkeypatch.setattr(local_bot.automated_review, "review_pull_request", fake_review)

    initialized = local_bot._poll_once(configuration, provider, state_file, backfill=False)

    assert not initialized
    assert token_requests == [False, True]
    assert reviews == [
        ("isaac-sim/IsaacLab", 10, "write-token", "nvidia-key", ("opus-test", "gpt-test")),
    ]
    assert local_bot._load_state(state_file, configuration.repository) == {10: "new-head"}

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Isaac Lab review bot locally using only GitHub App credentials."""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import automated_review

_DEFAULT_REPOSITORY = "isaac-sim/IsaacLab"
_DEFAULT_CLIENT_ID = "Iv23liPhQICNbdPQ9bBU"
_EXPECTED_APP_SLUG = "isaaclab-review-bot"
_DEFAULT_POLL_INTERVAL_SECONDS = 60
_TOKEN_REFRESH_BUFFER_SECONDS = 300
_FORBIDDEN_USER_TOKEN_ENV_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)


@dataclass(frozen=True)
class PullRequestHead:
    """Identity of one non-draft open pull-request revision."""

    number: int
    head_sha: str


@dataclass(frozen=True)
class BotConfiguration:
    """Runtime configuration loaded without personal GitHub credentials."""

    repository: str
    client_id: str
    private_key_path: Path
    inference_api_key: str
    review_models: tuple[str, ...]


@dataclass(frozen=True)
class _CachedToken:
    """Short-lived GitHub App installation token."""

    value: str
    expires_at: float


class GitHubAppTokenProvider:
    """Mint repository-scoped GitHub App installation tokens."""

    def __init__(self, repository: str, client_id: str, private_key_path: Path):
        """Initialize the token provider.

        Args:
            repository: Repository in ``owner/name`` form.
            client_id: GitHub App client ID.
            private_key_path: App private key PEM file.
        """
        _split_repository(repository)
        _validate_private_key(private_key_path)
        self._repository = repository
        self._client_id = client_id
        self._private_key_path = private_key_path
        self._tokens: dict[bool, _CachedToken] = {}

    def get_token(self, write: bool) -> str:
        """Return a current installation token with the requested access level.

        Args:
            write: Whether the token may write pull-request reviews.

        Returns:
            A repository-scoped GitHub App installation token.
        """
        cached = self._tokens.get(write)
        if cached is not None and cached.expires_at > time.time() + _TOKEN_REFRESH_BUFFER_SECONDS:
            return cached.value
        token = self._mint_token(write)
        self._tokens[write] = token
        return token.value

    def _mint_token(self, write: bool) -> _CachedToken:
        """Mint one repository-scoped installation token."""
        app_jwt = _create_app_jwt(self._client_id, self._private_key_path)
        app = _github_app_json("/app", app_jwt)
        if not isinstance(app, dict) or app.get("slug") != _EXPECTED_APP_SLUG:
            raise RuntimeError(f"Private key does not authenticate {_EXPECTED_APP_SLUG}.")
        owner, repository_name = _split_repository(self._repository)
        installation = _github_app_json(
            f"/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repository_name, safe='')}/installation",
            app_jwt,
        )
        if not isinstance(installation, dict) or not isinstance(installation.get("id"), int):
            raise RuntimeError(f"GitHub App is not installed on {self._repository}.")

        pull_request_access = "write" if write else "read"
        response = _github_app_json(
            f"/app/installations/{installation['id']}/access_tokens",
            app_jwt,
            method="POST",
            payload={
                "repositories": [repository_name],
                "permissions": {
                    "contents": "read",
                    "pull_requests": pull_request_access,
                },
            },
        )
        if not isinstance(response, dict):
            raise RuntimeError("GitHub returned an invalid installation-token response.")
        token = str(response.get("token") or "")
        expires_at = _parse_github_timestamp(str(response.get("expires_at") or ""))
        permissions = response.get("permissions")
        if not token or not isinstance(permissions, dict):
            raise RuntimeError("GitHub did not return a usable installation token.")
        if permissions.get("contents") != "read" or permissions.get("pull_requests") != pull_request_access:
            raise RuntimeError(f"GitHub returned unexpected installation-token permissions: {permissions}.")

        automated_review._verify_installation_token(self._repository, token)
        return _CachedToken(value=token, expires_at=expires_at)


def main() -> None:
    """Run a one-shot review or continuously monitor for new pull-request revisions."""
    arguments = _parse_arguments()
    _reject_personal_github_credentials()
    configuration = _load_configuration()
    provider = GitHubAppTokenProvider(
        configuration.repository,
        configuration.client_id,
        configuration.private_key_path,
    )

    if arguments.pull_request_number is not None:
        token = provider.get_token(write=not arguments.dry_run)
        automated_review.review_pull_request(
            configuration.repository,
            arguments.pull_request_number,
            token,
            configuration.inference_api_key,
            models=configuration.review_models,
            dry_run=arguments.dry_run,
        )
        return

    state_file = arguments.state_file or _default_state_file()
    _watch_pull_requests(
        configuration,
        provider,
        state_file,
        poll_interval_seconds=arguments.poll_interval,
        backfill=arguments.backfill,
    )


def _parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse local review-bot command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true", help="continuously monitor non-draft open pull requests")
    mode.add_argument("--pr-number", dest="pull_request_number", type=_positive_integer, help="review one pull request")
    parser.add_argument("--dry-run", action="store_true", help="print a one-shot review using a read-only App token")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="review all open pull requests when initializing a missing monitor state file",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_integer,
        default=_DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"poll interval in seconds (default: {_DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument("--state-file", type=Path, help="override the persistent monitor state file")
    parsed = parser.parse_args(arguments)
    if parsed.dry_run and parsed.pull_request_number is None:
        parser.error("--dry-run requires --pr-number")
    if parsed.backfill and not parsed.watch:
        parser.error("--backfill requires --watch")
    return parsed


def _positive_integer(value: str) -> int:
    """Parse a positive integer for :mod:`argparse`."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _reject_personal_github_credentials() -> None:
    """Refuse to start when a personal GitHub credential could be inherited."""
    present = [name for name in _FORBIDDEN_USER_TOKEN_ENV_VARS if os.environ.get(name)]
    if present:
        names = ", ".join(present)
        raise RuntimeError(
            f"Refusing to start while personal GitHub credential variables are set: {names}. "
            "Unset them; this bot authenticates only from its GitHub App private key."
        )


def _load_configuration() -> BotConfiguration:
    """Load and validate runtime configuration from environment variables."""
    repository = os.environ.get("GITHUB_REPOSITORY", _DEFAULT_REPOSITORY).strip()
    client_id = os.environ.get("ISAACLAB_REVIEW_APP_CLIENT_ID", _DEFAULT_CLIENT_ID).strip()
    primary_model = os.environ.get("NVIDIA_REVIEW_MODEL", automated_review._DEFAULT_MODEL).strip()
    ensemble_model = os.environ.get(
        "NVIDIA_REVIEW_FALLBACK_MODEL",
        automated_review._DEFAULT_ENSEMBLE_MODEL,
    ).strip()
    if repository != _DEFAULT_REPOSITORY:
        raise RuntimeError(f"This bot is locked to {_DEFAULT_REPOSITORY}, not {repository}.")
    if client_id != _DEFAULT_CLIENT_ID:
        raise RuntimeError(f"This bot is locked to GitHub App client ID {_DEFAULT_CLIENT_ID}.")
    review_models = (primary_model, ensemble_model)
    if any(not model for model in review_models) or len(set(review_models)) != len(review_models):
        raise RuntimeError(
            "NVIDIA_REVIEW_MODEL and NVIDIA_REVIEW_FALLBACK_MODEL must name two distinct, non-empty models."
        )
    private_key_path = Path(_required_env("ISAACLAB_REVIEW_APP_PRIVATE_KEY_PATH")).expanduser().resolve()
    return BotConfiguration(
        repository=repository,
        client_id=client_id,
        private_key_path=private_key_path,
        inference_api_key=_required_env("NVIDIA_INFERENCE_API_KEY"),
        review_models=review_models,
    )


def _required_env(name: str) -> str:
    """Return one required, non-empty environment variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def _split_repository(repository: str) -> tuple[str, str]:
    """Split an ``owner/name`` repository identifier."""
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid GitHub repository: {repository!r}.")
    return parts[0], parts[1]


def _validate_private_key(private_key_path: Path) -> None:
    """Validate that the App private key is a protected regular file."""
    if not private_key_path.is_file():
        raise RuntimeError(f"GitHub App private key does not exist: {private_key_path}.")
    if os.name == "posix":
        mode = stat.S_IMODE(private_key_path.stat().st_mode)
        if mode & 0o077:
            raise RuntimeError(
                f"GitHub App private key permissions must be owner-only, not {mode:04o}: {private_key_path}."
            )
    if shutil.which("openssl") is None:
        raise RuntimeError("The openssl executable is required to sign GitHub App JWTs.")


def _create_app_jwt(client_id: str, private_key_path: Path, now: int | None = None) -> str:
    """Create a short-lived RS256 JWT for GitHub App authentication."""
    issued_at = int(time.time()) if now is None else now
    header = _base64url_json({"alg": "RS256", "typ": "JWT"})
    payload = _base64url_json({"iat": issued_at - 60, "exp": issued_at + 540, "iss": client_id})
    signing_input = f"{header}.{payload}".encode()
    try:
        result = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", str(private_key_path)],
            input=signing_input,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"Could not execute openssl to sign the GitHub App JWT: {error}.") from error
    if result.returncode != 0 or not result.stdout:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not sign the GitHub App JWT with openssl: {details[:500]}.")
    return f"{header}.{payload}.{_base64url(result.stdout)}"


def _base64url_json(value: dict[str, Any]) -> str:
    """Encode compact JSON using unpadded URL-safe base64."""
    return _base64url(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _base64url(value: bytes) -> str:
    """Encode bytes using unpadded URL-safe base64."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _github_app_json(
    path: str,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    """Call a GitHub endpoint using an App JWT."""
    return automated_review._request_json(
        f"{automated_review._GITHUB_API_URL}{path}",
        token,
        method=method,
        payload=payload,
        accept="application/vnd.github+json",
        extra_headers={"X-GitHub-Api-Version": automated_review._GITHUB_API_VERSION},
    )


def _parse_github_timestamp(value: str) -> float:
    """Parse an ISO 8601 timestamp returned by GitHub."""
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"GitHub returned an invalid token expiration timestamp: {value!r}.") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"GitHub returned a timezone-free token expiration timestamp: {value!r}.")
    return parsed.timestamp()


def _watch_pull_requests(
    configuration: BotConfiguration,
    provider: GitHubAppTokenProvider,
    state_file: Path,
    poll_interval_seconds: int,
    backfill: bool,
) -> None:
    """Continuously poll for new non-draft pull-request revisions."""
    print(
        f"Monitoring {configuration.repository} every {poll_interval_seconds}s as the GitHub App installation. "
        f"State: {state_file}"
    )
    first_poll = True
    while True:
        try:
            initialized = _poll_once(configuration, provider, state_file, backfill=backfill and first_poll)
            first_poll = False
            if initialized:
                print("Monitor baseline initialized; future PRs and head updates will be reviewed.")
        except Exception as error:
            print(f"Review-bot poll failed: {error}", file=sys.stderr)
        time.sleep(poll_interval_seconds)


def _poll_once(
    configuration: BotConfiguration,
    provider: GitHubAppTokenProvider,
    state_file: Path,
    backfill: bool,
) -> bool:
    """Poll once and review every unseen non-draft pull-request head.

    Returns:
        Whether a new state file was initialized without backfilling.
    """
    read_token = provider.get_token(write=False)
    pull_requests = _list_open_pull_requests(configuration.repository, read_token)
    state = _load_state(state_file, configuration.repository)
    if state is None and not backfill:
        _save_state(state_file, configuration.repository, {item.number: item.head_sha for item in pull_requests})
        return True
    if state is None:
        state = {}

    open_numbers = {item.number for item in pull_requests}
    state = {number: head_sha for number, head_sha in state.items() if number in open_numbers}
    for item in pull_requests:
        if state.get(item.number) == item.head_sha:
            continue
        try:
            write_token = provider.get_token(write=True)
            status = automated_review.review_pull_request(
                configuration.repository,
                item.number,
                write_token,
                configuration.inference_api_key,
                models=configuration.review_models,
            )
        except Exception as error:
            print(f"Review of PR #{item.number} failed: {error}", file=sys.stderr)
            continue
        if status is not automated_review.ReviewStatus.STALE:
            state[item.number] = item.head_sha
            _save_state(state_file, configuration.repository, state)

    _save_state(state_file, configuration.repository, state)
    return False


def _list_open_pull_requests(repository: str, token: str) -> list[PullRequestHead]:
    """List every open, non-draft pull-request head."""
    response = automated_review._github_paginate(
        f"/repos/{repository}/pulls?state=open&sort=created&direction=asc",
        token,
    )
    pull_requests = []
    for item in response:
        if item.get("draft"):
            continue
        number = item.get("number")
        head = item.get("head")
        head_sha = head.get("sha") if isinstance(head, dict) else None
        if isinstance(number, int) and isinstance(head_sha, str) and head_sha:
            pull_requests.append(PullRequestHead(number=number, head_sha=head_sha))
    return pull_requests


def _default_state_file() -> Path:
    """Return the default persistent monitor state file."""
    state_root = os.environ.get("XDG_STATE_HOME")
    base_path = Path(state_root).expanduser() if state_root else Path.home() / ".local" / "state"
    return base_path / "isaaclab-review-bot" / "state.json"


def _load_state(state_file: Path, repository: str) -> dict[int, str] | None:
    """Load monitor state or return :obj:`None` when it has not been initialized."""
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load review-bot state from {state_file}: {error}.") from error
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("repository") != repository:
        raise RuntimeError(f"Review-bot state does not match repository {repository}: {state_file}.")
    heads = data.get("heads")
    if not isinstance(heads, dict):
        raise RuntimeError(f"Review-bot state has an invalid heads mapping: {state_file}.")
    try:
        return {int(number): str(head_sha) for number, head_sha in heads.items()}
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Review-bot state contains an invalid pull-request number: {state_file}.") from error


def _save_state(state_file: Path, repository: str, heads: dict[int, str]) -> None:
    """Atomically save monitor state with owner-only permissions."""
    state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "repository": repository,
        "heads": {str(number): head_sha for number, head_sha in sorted(heads.items())},
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_file.parent,
            prefix=".state-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, state_file)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save review-bot state to {state_file}: {error}.") from error


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Local review bot failed: {error}", file=sys.stderr)
        raise

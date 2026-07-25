# Isaac Lab review bot

This repository contains the local, always-on reviewer for new pull requests to
`isaac-sim/IsaacLab`. It polls GitHub from one maintainer-controlled machine,
runs four structured review passes through the OpenAI Responses API, and posts
one comment-only review as `isaaclab-review-bot[bot]`.

The bot does not use GitHub Actions, `gh auth`, a personal access token, or a
GitHub App user access token. It creates a GitHub App JWT from the
`isaaclab-review-bot` private key and exchanges that JWT for short-lived
installation tokens scoped to only `isaac-sim/IsaacLab`.

The poller normally uses a read-only installation token. It mints a separate
token with pull-request write permission only when a new non-draft PR or head
revision needs review.

## Requirements

- Linux with Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- OpenSSL
- A GitHub App installation on `isaac-sim/IsaacLab`
- A dedicated OpenAI API project service-account key

Do not copy a personal Codex login or `~/.codex/auth.json` into the service.
Although Codex CLI supports ChatGPT authentication, OpenAI advises against
ChatGPT-managed automation for public or open-source repositories. Sign in to
the OpenAI Platform with SSO, then create a project and service account named
`isaaclab-review-bot`. Store only that service account's API key in the bot
environment file.

## GitHub App configuration

Configure `isaaclab-review-bot` with these repository permissions:

- Contents: read
- Pull requests: read and write

Install the App with **Only select repositories** and select only
`isaac-sim/IsaacLab`. Do not enable user authorization or create a personal
token for the bot.

Generate an App private key and store it outside this repository:

```bash
install -d -m 700 /home/kellyg/.config/isaaclab-review-bot
install -m 600 /path/to/downloaded-private-key.pem \
  /home/kellyg/.config/isaaclab-review-bot/app-private-key.pem
```

## Private environment file

Create the service environment:

```bash
cd /home/kellyg/Documents/isaac/isaaclab-review-codex
install -m 600 environment.example \
  /home/kellyg/.config/isaaclab-review-bot/environment
```

Edit `/home/kellyg/.config/isaaclab-review-bot/environment` and set the
downloaded GitHub App key path and the dedicated OpenAI project
service-account key. Never add `GH_TOKEN`, `GITHUB_TOKEN`,
`GH_ENTERPRISE_TOKEN`, or `GITHUB_ENTERPRISE_TOKEN`; the process refuses to
start when any of them is present.

The default model is `gpt-5.6`, currently the alias for GPT-5.6 Sol. Change
`OPENAI_REVIEW_MODEL` only after validating another model on representative
Isaac Lab pull requests.

## One-shot dry run

Use an existing non-draft pull request to verify authentication and review
quality without allowing a GitHub write:

```bash
cd /home/kellyg/Documents/isaac/isaaclab-review-codex
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
set -a
source /home/kellyg/.config/isaaclab-review-bot/environment
set +a
uv run --no-project python local_review_bot.py --pr-number 1234 --dry-run
unset OPENAI_API_KEY
```

Replace `1234` with the test PR number. Dry-run mode authenticates as the App
but requests a read-only installation token, so it cannot post. It still makes
four billed model requests: three specialist passes and one final validation
pass.

## Continuous user service

Install and start the user service:

```bash
install -d -m 700 /home/kellyg/.config/systemd/user
install -d -m 700 /home/kellyg/.local/state/isaaclab-review-bot
install -m 600 isaaclab-review-bot.service \
  /home/kellyg/.config/systemd/user/isaaclab-review-bot.service

systemctl --user unset-environment \
  GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
systemctl --user daemon-reload
systemctl --user enable --now isaaclab-review-bot.service
loginctl enable-linger kellyg
systemctl --user status isaaclab-review-bot.service
```

Follow its logs with:

```bash
journalctl --user -u isaaclab-review-bot.service -f
```

User lingering keeps the service running after logout and starts the user
service manager during boot. The bot cannot run while the machine is powered
off, suspended, or disconnected from the network.

On first startup, the service records current non-draft open PR heads without
reviewing them. Later PRs and later head revisions are reviewed. To
intentionally review the existing backlog, stop the service, remove its state
file, and run the poller once with `--watch --backfill`.

The service restarts after failures, refreshes expiring App installation
tokens, rechecks each PR head before posting, and embeds the reviewed commit SHA
to prevent duplicate reviews after restarts.

## Development

Runtime code uses only the Python standard library. Install the development
environment and run all tests with:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

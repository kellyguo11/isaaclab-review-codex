# Isaac Lab review bot

This repository contains the local, always-on reviewer for new pull requests to
`isaac-sim/IsaacLab`. It polls GitHub from one maintainer-controlled machine,
runs four structured review passes through NVIDIA's OpenAI-compatible inference
API, and posts one comment-only review as `isaaclab-review-bot[bot]`.

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
- An NVIDIA inference API key

No OpenAI API key, Codex login, `~/.codex/auth.json`, or OpenClaw process is
used. The model credential is sent only to the fixed endpoint
`https://inference-api.nvidia.com/v1/chat/completions`.

The former OpenClaw configuration maps to this service as follows:

- Primary model: `azure/anthropic/claude-opus-4-6`
- Fallback model: `azure/anthropic/claude-sonnet-4-6`
- Three concurrent specialist passes
- 600-second inference request timeout

OpenClaw's workspace, memory search, image model, Slack channel, gateway,
session, and tool settings are not needed for pull-request reviews.

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
downloaded GitHub App key path and `NVIDIA_INFERENCE_API_KEY`. Never add
`GH_TOKEN`, `GITHUB_TOKEN`,
`GH_ENTERPRISE_TOKEN`, or `GITHUB_ENTERPRISE_TOKEN`; the process refuses to
start when any of them is present.

The base URL is intentionally not configurable so a typo or compromised
environment cannot redirect the NVIDIA key to another host. Set
`NVIDIA_REVIEW_MODEL` and `NVIDIA_REVIEW_FALLBACK_MODEL` only to model IDs
enabled for the NVIDIA key.

Verify the key and model IDs without printing the key:

```bash
set -a
source /home/kellyg/.config/isaaclab-review-bot/environment
set +a
curl --silent --show-error https://inference-api.nvidia.com/v1/models \
  --header "Authorization: Bearer ${NVIDIA_INFERENCE_API_KEY}" \
  | uv run --no-project python -c \
    'import json,sys; print("\n".join(item["id"] for item in json.load(sys.stdin)["data"]))'
unset NVIDIA_INFERENCE_API_KEY
```

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
unset NVIDIA_INFERENCE_API_KEY
```

Replace `1234` with the test PR number. Dry-run mode authenticates as the App
but requests a read-only installation token, so it cannot post. It normally
makes four model requests: three specialist passes and one final validation
pass. A failed primary request is retried with the configured fallback model.

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

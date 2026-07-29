# Isaac Lab review bot

This repository contains the local, always-on reviewer for new pull requests to
`isaac-sim/IsaacLab`. It polls GitHub from one maintainer-controlled machine
and posts one initial comment-only review as `isaaclab-review-bot[bot]`.
Subsequent commits are reviewed only when an authorized developer requests
another review from the PR conversation.

Every pull request is reviewed by a two-model ensemble:

- `azure/anthropic/claude-opus-5`
- `azure/openai/gpt-5.6-sol`

Each model independently runs three specialist passes: design and architecture,
API contracts, and implementation quality. Opus 5 then conservatively
aggregates all six results into one review; GPT-5.6 Sol handles aggregation if
Opus is unavailable. Before publication, GPT-5.6 Sol independently checks every
candidate issue against the diff and rejects anything that does not clearly
need fixing; Opus handles this verification if GPT is unavailable. The verifier
can accept or reject existing candidate IDs but cannot invent or relocate
findings. A normal review therefore makes eight NVIDIA inference requests. If
verification fails with both models, nothing is posted.
Each request allows up to 65,536 output tokens so reasoning models have enough
budget to produce their final structured answer.

Review context is capped at 2,100,000 characters, increased from the original
480,000-character budget. This leaves tokenizer headroom inside the configured
models' one-million-token input windows. The allocator preserves the complete PR
diff before using the remaining space for line-numbered current-file excerpts
around every changed region. If only supplemental context is limited, the
review explicitly states that the full diff was still reviewed.

The review policy prioritizes precision over recall. It reports every finding
that clears the high-confidence threshold, with no numerical cap, and rejects
hypothetical edge cases, missing-test observations, logging preferences,
optional hardening, alternative designs, formatting, and other subjective style
feedback. Every finding must demonstrate a concrete design, architecture, API,
or maintainability impact from an added line in the PR. Deterministic contract,
type, and producer/consumer failures do not require a runtime reproduction or a
specific external caller to be reported. The bot posts concise explanatory
comments and does not generate GitHub replacement-code suggestion blocks.

The bot calls NVIDIA's OpenAI-compatible
`https://inference-api.nvidia.com/v1/chat/completions` endpoint directly. It
does not require OpenClaw, Slack, an OpenClaw gateway, an OpenAI API key, a
Codex login, or `~/.codex/auth.json`.

## Requirements

- Linux with Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- OpenSSL
- Git and SSH access to this private repository
- A GitHub App installation on `isaac-sim/IsaacLab`
- An NVIDIA inference API key with access to both ensemble models

Only one machine should run the continuous watcher at a time.

## 1. Clone on a new machine

```bash
git clone git@github.com:kellyguo11/isaaclab-review-codex.git
cd isaaclab-review-codex
pwd -P
```

Keep the absolute path printed by `pwd -P`; it becomes
`ISAACLAB_REVIEW_BOT_DIR`.

Install the development dependencies and verify the checkout:

```bash
uv sync
uv run pytest
```

Runtime code itself uses only the Python standard library.

## 2. Configure the GitHub App

Configure `isaaclab-review-bot` with these repository permissions:

- Contents: read
- Pull requests: read and write

The Pull requests permission also allows the App to read PR conversation
comments through GitHub's
[shared issue-comments endpoint](https://docs.github.com/en/rest/issues/comments#list-issue-comments-for-a-repository).
No Issues permission or user authorization is required.

Install it with **Only select repositories** and select only
`isaac-sim/IsaacLab`. Do not enable user authorization and do not create a
personal access token for the bot.

From the GitHub App's settings, generate a private key. Store the downloaded
PEM outside the repository with owner-only permissions:

```bash
install -d -m 700 "${HOME}/.config/isaaclab-review-bot"
install -m 600 /path/to/downloaded-private-key.pem \
  "${HOME}/.config/isaaclab-review-bot/app-private-key.pem"
```

Replace `/path/to/downloaded-private-key.pem` with the downloaded file.

## 3. Create the private environment file

From the cloned repository:

```bash
install -m 600 environment.example \
  "${HOME}/.config/isaaclab-review-bot/environment"
```

Edit `${HOME}/.config/isaaclab-review-bot/environment`:

```text
ISAACLAB_REVIEW_BOT_DIR=/absolute/path/from/pwd
ISAACLAB_REVIEW_APP_PRIVATE_KEY_PATH=/home/your-user/.config/isaaclab-review-bot/app-private-key.pem
ISAACLAB_REVIEW_APP_CLIENT_ID=Iv23liPhQICNbdPQ9bBU
GITHUB_REPOSITORY=isaac-sim/IsaacLab
NVIDIA_INFERENCE_API_KEY=<PASTE-THE-NVIDIA-KEY-HERE>
NVIDIA_REVIEW_MODEL=azure/anthropic/claude-opus-5
NVIDIA_REVIEW_FALLBACK_MODEL=azure/openai/gpt-5.6-sol
```

Despite its legacy name, `NVIDIA_REVIEW_FALLBACK_MODEL` is always used as the
second ensemble reviewer. It is also the fallback for final aggregation when
the primary model fails and the first-choice pre-publication verifier.

Do not put quotes around values unless a path contains spaces. Do not commit
this environment file or paste either private key into an issue, pull request,
or chat.

Enforce owner-only permissions:

```bash
chmod 600 \
  "${HOME}/.config/isaaclab-review-bot/environment" \
  "${HOME}/.config/isaaclab-review-bot/app-private-key.pem"
```

The NVIDIA endpoint is intentionally fixed in the code so an environment typo
cannot redirect the inference key to another host.

The executable `launch_review_bot.sh` loads this owner-only environment file,
clears any personal GitHub tokens inherited from the shell, and starts the bot
from the correct repository directory. With no arguments, it starts the
continuous watcher. To keep the environment file elsewhere, set
`ISAACLAB_REVIEW_BOT_ENV_FILE` to its absolute path before launching.

## 4. Verify NVIDIA access

Load the private environment into the current shell:

```bash
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
set -a
source "${HOME}/.config/isaaclab-review-bot/environment"
set +a
```

List the model IDs enabled for the key:

```bash
curl --silent --show-error https://inference-api.nvidia.com/v1/models \
  --header "Authorization: Bearer ${NVIDIA_INFERENCE_API_KEY}" \
  | uv run --no-project python -c \
    'import json,sys; print("\n".join(item["id"] for item in json.load(sys.stdin)["data"]))'
```

Confirm that both configured model IDs appear. When finished with the
interactive shell:

```bash
unset NVIDIA_INFERENCE_API_KEY
```

## 5. Run a safe dry run

A dry run reads one non-draft PR, runs the full eight-request ensemble, and
prints the proposed review. It requests a read-only GitHub App token and cannot
post:

```bash
cd /absolute/path/to/isaaclab-review-codex
./launch_review_bot.sh --pr 1234 --dry-run
```

Replace `1234` with an existing non-draft IsaacLab PR number.

## 6. Run one posting review

After inspecting a dry run, omit `--dry-run` to post one review as the GitHub
App:

```bash
cd /absolute/path/to/isaaclab-review-codex
./launch_review_bot.sh --pr 1234
```

The bot posts a `COMMENT` review only. It never approves a PR or requests
changes.

## 7. Run the watcher in the foreground

For a temporary foreground session:

```bash
cd /absolute/path/to/isaaclab-review-codex
./launch_review_bot.sh
```

On first startup, the watcher records all current non-draft open PR numbers
without reviewing them. Each later PR is reviewed once when it first becomes
open and ready for review. New commits and force-pushes on a known PR do not
trigger automatic full reviews.
The bot prints timestamped progress for every poll, GitHub authentication step,
specialist model request, aggregation request, verification request, and posting
attempt. Model calls can take several minutes; the bot prints a waiting
heartbeat every 30 seconds and completion messages include elapsed time.

To intentionally review every currently open non-draft PR when initializing a
new state file, use `./launch_review_bot.sh --watch --backfill`. This can post
many reviews and incur many inference requests, so stop the systemd service
first and use it only deliberately.

### Request another review from a PR

After pushing new commits, the PR author or a repository owner, member, or
collaborator can add this exact conversation comment:

```text
@isaaclab-review-bot review
```

`/isaaclab-review` is also accepted. The watcher sees the command on its next
poll and reviews the PR's current head. Put the command in a normal PR
conversation comment, not an inline code-review comment. A command for a head
the bot already reviewed is acknowledged in the local log and skipped before
inference, preventing duplicate reviews and model spend.

Press `Ctrl+C` to stop a foreground watcher.

## 8. Install the always-on user service

From the cloned repository:

```bash
install -d -m 700 "${HOME}/.config/systemd/user"
install -d -m 700 "${HOME}/.local/state/isaaclab-review-bot"
install -m 600 isaaclab-review-bot.service \
  "${HOME}/.config/systemd/user/isaaclab-review-bot.service"

systemctl --user unset-environment \
  GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GITHUB_ENTERPRISE_TOKEN
systemctl --user daemon-reload
systemctl --user enable --now isaaclab-review-bot.service
loginctl enable-linger "${USER}"
```

Check service state and follow logs:

```bash
systemctl --user status isaaclab-review-bot.service
journalctl --user -u isaaclab-review-bot.service -f
```

Common operations:

```bash
systemctl --user stop isaaclab-review-bot.service
systemctl --user start isaaclab-review-bot.service
systemctl --user restart isaaclab-review-bot.service
systemctl --user disable --now isaaclab-review-bot.service
```

User lingering keeps the service running after logout and starts the user
service manager during boot. The bot cannot run while the machine is powered
off, suspended, or disconnected from the network.

## Updating an installed machine

```bash
cd /absolute/path/to/isaaclab-review-codex
git pull --ff-only origin main
uv sync
uv run pytest
systemctl --user restart isaaclab-review-bot.service
```

Review the diff before restarting when an update changes environment variables
or service configuration.

## Runtime behavior and recovery

- The poller uses a read-only installation token while listing PRs.
- It mints a separate repository-scoped write token only when a review must be
  posted.
- Installation tokens expire and are refreshed automatically.
- State is stored at
  `${HOME}/.local/state/isaaclab-review-bot/state.json` with mode `0600`.
- The review-command cursor is stored beside it in `state-commands.json`, also
  with mode `0600`.
- Seen PR numbers remain in state, so later commits do not trigger automatic
  reviews.
- Only a PR author or a repository owner, member, or collaborator can trigger
  an on-demand review comment.
- The reviewed commit SHA is embedded in each bot review to prevent duplicate
  reviews after ordinary restarts.
- A PR head is rechecked immediately before posting; a result is discarded if
  the PR changed during inference.
- The service restarts automatically after transient failures.

The process refuses to start if `GH_TOKEN`, `GITHUB_TOKEN`,
`GH_ENTERPRISE_TOKEN`, or `GITHUB_ENTERPRISE_TOKEN` is present. GitHub activity
is authenticated only through the `isaaclab-review-bot` App installation and
is attributed to `isaaclab-review-bot[bot]`.

## Development

This utility repository is maintained with direct pushes to `main`; do not open
a pull request unless a maintainer explicitly asks for one.

Run the validation suite before pushing:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
bash -n launch_review_bot.sh
systemd-analyze --user verify isaaclab-review-bot.service
```

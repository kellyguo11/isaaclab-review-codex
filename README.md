# Isaac Lab review bot

This repository contains the local, always-on reviewer for new pull requests to
`isaac-sim/IsaacLab`. It polls GitHub from one maintainer-controlled machine
and posts one initial comment-only review as `isaaclab-review-bot[bot]`.
Subsequent commits are reviewed only when an authorized developer requests
another review from the PR conversation.

Every pull request is reviewed by a two-model ensemble:

- `azure/anthropic/claude-opus-5`
- `azure/openai/gpt-5.6-sol`

Each model independently runs five specialist passes: design and architecture,
API contracts, implementation quality, style consistency, and test quality.
The three main passes use a skeptical maintainer standard: they compare deleted
behavior with its replacement line by line, trace boundary and failure paths,
and report directly evidenced semantic, integration, architectural, API, and
maintainability defects even when they do not produce an immediate crash.
The API pass builds an old-versus-new compatibility ledger for every touched
contract, including import paths, signatures, defaults, runtime types, shapes,
units, ordering, configuration keys, CLI flags, registry IDs, exceptions,
side effects, and serialized forms.
The style pass deliberately applies the current Isaac Lab contribution guide
and adjacent code patterns strictly, including small consistency and
maintainability issues. The test pass applies the repository's `test-audit`
authoring gate to every added or changed test case, checking that it owns a
distinct contract and does not duplicate existing tests, fixtures, scenes,
backends, or parameter axes. Opus 5 then conservatively aggregates all ten
results into one review; GPT-5.6 Sol handles aggregation if
Opus is unavailable. Before publication, GPT-5.6 Sol independently checks every
candidate issue against the diff and rejects anything that does not clearly
need fixing; Opus handles this verification if GPT is unavailable. The verifier
can accept or reject existing candidate IDs but cannot invent or relocate
findings. A normal review therefore makes twelve NVIDIA inference requests. If
verification fails with both models, nothing is posted.
All ten specialist requests run concurrently by default, reducing that stage
from four waves to one while preserving every role and both ensemble models.
Aggregation and pre-publication verification remain sequential because each
depends on the preceding result.
Each request allows up to 65,536 output tokens so reasoning models have enough
budget to produce their final structured answer. Model responses may take up
to 15 minutes before the bot treats the connection as stalled. A timed-out
specialist is not retried because its counterpart model is already reviewing
the same role; aggregation and verification instead try the other configured
model. This avoids turning one provider stall into another 15-minute wait.

Review context is capped at 2,100,000 characters, increased from the original
480,000-character budget. This leaves tokenizer headroom inside the configured
models' one-million-token input windows. The allocator preserves the complete PR
diff before using the remaining space for line-numbered current-file excerpts
around every changed region. The trusted context includes the base branch's
coding and unit-testing contribution guidance and its `test-audit` skill. When
tests change, the bot also includes bounded full changed-test content, the base
test inventory, related existing tests, and CI test routing so duplication and
test ownership findings have repository evidence. That expanded test evidence
is sent only to the dedicated test-quality specialist; other specialists still
receive the changed-file patches and excerpts but do not receive redundant full
test files and inventory data. If only supplemental context is limited, the
review explicitly states that the full diff was still reviewed.

The review policy requires every specialist to inspect every patch hunk, compare
deleted behavior with its replacement, trace changed producers and consumers,
and perform a second adversarial pass before returning no findings. The review
explicitly audits downstream callers, public exports and lazy-loading stubs,
registrations, configuration and CLI forwarding, templates, examples,
documentation includes, and per-package changelog obligations. Files converted
to thin delegates, moved modules, and renamed symbols receive extra scrutiny.

It reports every finding that clears the evidence threshold, with no numerical
cap, and rejects hypothetical edge cases, generic missing-test requests,
logging preferences, optional hardening, alternative designs, and unsupported
personal preferences. Exact formatting, naming, typing, documentation, local
style, and test-value violations are intentionally reportable even when their
appropriate severity is only a suggestion. An unchanged downstream consumer
broken by an added or deleted line is considered introduced by the PR and remains
reportable. Breaking changes are highlighted in a dedicated compatibility and
deprecation assessment and are warnings at minimum when they lack a transition.
A changelog, migration note, major-version claim, or replacement API alone does
not count as a deprecation cycle: the old contract must remain functional for
the repository-prescribed window, use the established targeted warning with
replacement and removal guidance, document migration, and cover both paths
during the transition. The bot can anchor removal findings directly to deleted
lines instead of dropping deletion-only API breaks. Deterministic contract,
repository-rule, style, documentation-integration, test-audit, type, and
producer/consumer failures do
not require a runtime reproduction or a specific external caller to be
reported. The bot posts concise explanatory comments and does not generate
GitHub replacement-code suggestion blocks.

When no inline finding clears the evidence threshold, the bot still posts
pull-request-specific feedback: the design approach reviewed, the exact API or
compatibility surface checked, the implementation paths traced, and concrete
non-blocking tradeoffs or residual risks. It uses `No blocking issues` for that
outcome rather than treating the automated review as an approval or saying
`Ship it`.

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
- Issues: read and write
- Pull requests: read and write

The Pull requests permission allows the App to read PR conversation comments
through GitHub's shared issue-comments endpoint. Issues write access is used
only to add an eyes reaction when a review starts. No user authorization is
required.

For an existing installation, add **Issues: Read and write** in the GitHub
App's repository permissions, save the App settings, and approve the requested
permission change for the `isaac-sim` installation before restarting the bot.

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
NVIDIA_REVIEW_MAX_CONCURRENT_REQUESTS=10
```

Despite its legacy name, `NVIDIA_REVIEW_FALLBACK_MODEL` is always used as the
second ensemble reviewer. It is also the fallback for final aggregation when
the primary model fails and the first-choice pre-publication verifier.
`NVIDIA_REVIEW_MAX_CONCURRENT_REQUESTS` accepts `1` through `10`. The default
of `10` starts all specialist requests in one wave. Reduce it if the NVIDIA
endpoint consistently returns rate-limit responses.

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

After pushing new commits, the PR author or a user with effective `write`,
`maintain`, or `admin` permission on the repository can add this exact
conversation comment:

```text
@isaaclab-review-bot review
```

`/isaaclab-review` is also accepted. The watcher sees the command on its next
poll, immediately adds an 👀 reaction to the accepted command, and reviews the
PR's current head. Put the command in a normal PR
conversation comment, not an inline code-review comment. A command for a head
the bot already reviewed is acknowledged in the local log and skipped before
inference, preventing duplicate reviews and model spend.

For a new PR reviewed automatically, the bot adds the 👀 reaction to the pull
request itself before starting model inference.

The bot verifies effective access with GitHub's repository-permission endpoint.
It does not use the comment's coarse `author_association` label, which may say
`CONTRIBUTOR` even when a user has elevated access through an organization,
team, or enterprise role.

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

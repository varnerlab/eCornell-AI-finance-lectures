# S4 cron setup — new-machine checklist

These steps install the intraday production cron (22 fires/weekday) on a fresh
machine. Some files are gitignored on purpose — they hold credentials or
machine-specific secrets and must be created by hand on each machine.

## What this cron does, briefly

- 14 engine fires per weekday (09:30 open, every :00/:30 from 10:00–15:30, 16:00 close) — submits paper Alpaca orders, queues compliance exceptions.
- 7 news fires per weekday (10:00–16:00 hourly) — calls Anthropic to fetch + score real headlines.
- 1 next-day execution fire (09:35) — submits the prior session's class-signed ticket.

All times are in the **system timezone**. Cron does not know about ET — the
host machine must be set to America/New_York for the schedule to align with
US equity market hours.

## Prerequisites

- macOS or Linux with `cron` available (use `crontab -l` to confirm).
- System timezone set to America/New_York. Verify with `date` — the output line should end in `EST` or `EDT`. If it doesn't:
  - macOS: `sudo systemsetup -settimezone America/New_York`
  - Linux: `sudo timedatectl set-timezone America/New_York`
- Julia 1.12 installed and on `PATH`. Verify with `which julia`.
- The repository cloned to a known location (referenced below as `$REPO_ROOT`).

## Step 1: instantiate the session-4 Julia environment

```bash
cd $REPO_ROOT/lectures/session-4
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This pulls in the local `code/` package, the Alpaca SDK, JSON, JLD2, and the
rest of the session deps. First run may take ~5 minutes for compilation.

## Step 2: create `config/credentials.toml`

This file is gitignored — it holds your Alpaca API key pair. On the new
machine:

```bash
cp $REPO_ROOT/lectures/session-4/config/credentials.toml.example \
   $REPO_ROOT/lectures/session-4/config/credentials.toml
```

Then open `credentials.toml` in an editor and fill in your paper trading key
and secret under the `[Credentials]` section. You can copy these from your
existing machine if you have them, or generate fresh ones at
`https://app.alpaca.markets/paper/dashboard/overview` -> API Keys.

Verify the credentials work:

```bash
julia --project=. scripts/verify_alpaca_bars.jl
```

You should see your account number, $100K cash, and 30-min bars for AAPL/MSFT/SPY.

## Step 3: create `~/.ecornell-s4-env` for cron secrets

Cron does not source `~/.zshrc` or `~/.bash_profile`, so any environment
variables you depend on must be set explicitly. The setup script reads from
a single dedicated env file:

```bash
cat > ~/.ecornell-s4-env <<'EOF'
# Loaded by S4 cron entries (see lectures/session-4/scripts/setup_cron.sh).
# This file is intentionally outside the repo so secrets never get committed.
export ANTHROPIC_API_KEY="sk-ant-..."   # from console.anthropic.com -> API Keys
EOF
chmod 600 ~/.ecornell-s4-env
```

Replace `sk-ant-...` with your actual key. The `chmod 600` ensures only your
user can read it.

If you ever rotate the key, edit this one file — no need to reinstall the
cron entries.

## Step 4: smoke test the news scorer

Before installing the cron, run one news fire by hand and confirm it
completes against the real Anthropic API:

```bash
source ~/.ecornell-s4-env
cd $REPO_ROOT/lectures/session-4
julia --project=. scripts/news_scorer.jl --mode=hourly
```

Expected behavior:

- Logs `fire at YYYY-MM-DDTHH:MM:SS source=anthropic_web tickers=5`.
- Fetches headlines for AAPL, MSFT, GOOG, AMZN, NVDA over ~10 minutes (Tier 1 rate limits).
- Writes `data/news/news-YYYY-MM-DD-HH.jld2` and `data/news/budget-YYYY-MM-DD.json`.
- Logs `N items, M scored, K searches (today K/100)`.

A run with all 0 scored items usually means the API key is wrong or the
account is out of credit. Check `data/production-log.txt` for the underlying
HTTP status.

## Step 5: install the cron entries

```bash
cd $REPO_ROOT/lectures/session-4/scripts
chmod +x setup_cron.sh
./setup_cron.sh
```

The script will:

1. Verify `~/.ecornell-s4-env` exists (refuses to install otherwise).
2. Locate `julia`.
3. Append 5 cron schedule lines to your existing crontab, each tagged with `[AI-FINANCE]`.
4. Print the installed entries for review.

To inspect what got installed:

```bash
crontab -l | grep '\[AI-FINANCE\]'
```

To remove all AI-FINANCE entries (does not touch other cron jobs you have):

```bash
crontab -l | grep -v '\[AI-FINANCE\]' | crontab -
```

## Step 6 (macOS only): grant Full Disk Access to cron

Modern macOS sandboxes `cron` jobs by default — they can run but get
permission-denied when reading files outside `~/`. To allow the cron to read
the repo and write to `data/`:

1. Open **System Settings -> Privacy & Security -> Full Disk Access**.
2. Click `+`, then `Cmd+Shift+G` and paste `/usr/sbin/cron`.
3. Add it. You may also need to add your terminal app (Terminal.app or iTerm).
4. Reboot the machine for the change to take effect cleanly.

If you skip this step, fires will appear to run but the log will show
"Operation not permitted" or empty output.

## Step 7: monitor the first few fires

After install, watch the log live:

```bash
tail -f $REPO_ROOT/lectures/session-4/data/production-log.txt
```

Wait for the next scheduled minute boundary (`:00` or `:30` during market
hours, weekday only). You should see entries appear in real time:

- `[NEWS-HOURLY] fire at ...` followed by either successful items or fallback messages.
- `[ENGINE] equity=$... bandit=...` from the production_runner fires.

If nothing appears at the expected time, check:

- `crontab -l | grep AI-FINANCE` — entries actually installed
- `date` — system clock and timezone correct
- `pgrep cron` — cron daemon running (Linux: `systemctl status cron`)
- `~/.ecornell-s4-env` — file exists, has `ANTHROPIC_API_KEY` exported, mode 600
- (macOS) Full Disk Access granted to `/usr/sbin/cron`

## Files that DO transfer via git

These are tracked in the repo and will be present after `git clone`:

- `lectures/session-4/scripts/setup_cron.sh`
- `lectures/session-4/scripts/production_runner.jl`
- `lectures/session-4/scripts/news_scorer.jl`
- `lectures/session-4/scripts/verify_alpaca_bars.jl`
- `lectures/session-4/config/production-config.toml`
- `lectures/session-4/config/news-source.toml`
- `lectures/session-4/config/credentials.toml.example`
- `lectures/session-4/data/news/synthetic-corpus-2026-05.jld2`
- `code/scripts/generate-synthetic-news.jl`

## Files that do NOT transfer (must be created on each machine)

- `~/.ecornell-s4-env` (Anthropic API key)
- `lectures/session-4/config/credentials.toml` (Alpaca API keys)
- `lectures/session-4/data/news/news-*.jld2` (cron-generated, regenerated each fire)
- `lectures/session-4/data/news/budget-*.json` (cron-generated)

## Cost expectation

- Alpaca paper trading: free.
- Anthropic API on Tier 1 (default for new accounts): roughly $0.30 per news fire including web search fees and scoring tokens. With 7 fires/weekday for 5 days, total ~$10. Tier 2 (auto-promoted after $40 cumulative spend AND 7 days since first deposit) drops costs and removes the Tier 1 rate-limit fragility.
- Set a monthly usage limit at console.anthropic.com -> Plans & Billing as a safety net.

# Tools Used in the eCornell AI in Finance Lectures

This file lists every tool, package, and external service that the four sessions depend on, along with download links and installation instructions. The format is meant to be read both by humans and by a coding assistant (Claude, Gemini, ChatGPT) that is asked to set up a clean machine for the course.

---

## Instructions for an LLM Reading This File

If you are an LLM and the user has asked you to install everything in this file, do the following before running any command:

1. Detect the host operating system. Prefer reading `uname -a` (macOS or Linux) or `systeminfo` (Windows). If a shell command is not available, ask the user which OS they are on: macOS, Windows, or Linux.
2. Detect the architecture: Apple Silicon (`arm64`) vs Intel (`x86_64`) on macOS; `x86_64` vs `aarch64` on Linux.
3. Detect whether the user already has Homebrew (macOS), winget or Chocolatey (Windows), or apt / dnf (Linux). Prefer the package manager that is already installed.
4. Ask one clarifying question only if the user has not already answered it: do they want the optional external accounts (Anthropic API, Alpaca Markets paper trading) configured now, or skipped. Sessions 1 and 2 run entirely offline; Sessions 3 and 4 require both.
5. Install in the order given below. Each section is a prerequisite for the section after it.
6. After every install step, run the verification command listed in that section. Do not move on until verification passes.
7. Never run `sudo` on macOS unless the user explicitly approves. Use Homebrew without `sudo`.

If a command fails, report the failure and stop. Do not silently swap in a different version or a different package manager.

---

## 1. GitHub

What it is: the web service that hosts the course repository at <https://github.com/varnerlab/eCornell-AI-finance-lectures>, plus a desktop app (GitHub Desktop) that does the cloning, pulling, and committing without a terminal. GitHub Desktop ships with its own embedded git, so there is no need to install git separately on macOS or Windows.

Links:

- Web service: <https://github.com/>
- GitHub Desktop: <https://desktop.github.com/>

### Create a GitHub account

1. Open <https://github.com/join> in a browser.
2. Pick a username, enter an email, choose a password.
3. Verify the email (GitHub sends a confirmation link).
4. Skip the paid plan upgrade. The Free tier is sufficient for this course.

### Install GitHub Desktop

- macOS: `brew install --cask github` or download from <https://desktop.github.com/>.
- Windows: `winget install --id GitHub.GitHubDesktop -e` or download from <https://desktop.github.com/>.
- Linux: GitHub Desktop is not officially supported on Linux. Use the community fork at <https://github.com/shiftkey/desktop>, or install standalone git (`sudo apt-get install -y git` on Debian / Ubuntu, `sudo dnf install -y git` on Fedora / RHEL) and clone with the command line shown below.

After install:

1. Launch GitHub Desktop.
2. File -> Options (Windows) or GitHub Desktop -> Settings (macOS) -> "Sign in to GitHub.com" and authorize with the account from the previous step.
3. Under "Git", set the author name and email to match the GitHub account.

### Clone the course repository

In GitHub Desktop:

1. File -> Clone repository -> URL tab.
2. Paste `https://github.com/varnerlab/eCornell-AI-finance-lectures` into the URL field.
3. Choose a local path (e.g. `~/Documents/GitHub/eCornell-AI-finance-lectures` on macOS, or `C:\Users\<you>\Documents\GitHub\eCornell-AI-finance-lectures` on Windows).
4. Click "Clone".

Equivalent command line if not using the desktop app:

```bash
git clone https://github.com/varnerlab/eCornell-AI-finance-lectures.git
```

Verify: the local clone has `lectures/session-1` through `lectures/session-4` and a `code/` folder.

---

## 2. Julia

What it is: the language all course code is written in. The course is tested on Julia 1.12.5 and requires 1.10 or newer.

Link: <https://julialang.org/downloads/>

Install (preferred, all platforms): use the official version manager `juliaup`. It handles upgrades cleanly and lets you pin a channel.

- macOS / Linux: `curl -fsSL https://install.julialang.org | sh`
- Windows: `winget install --id JuliaLang.Juliaup -e`

After `juliaup` is installed:

```bash
juliaup add 1.12
juliaup default 1.12
```

Alternative: download the platform-specific installer from the link above and add `julia` to `PATH`.

Verify:

```bash
julia --version
```

Expected: `julia version 1.12.x` or newer.

---

## 3. Jupyter

What it is: the notebook server that renders the `.ipynb` lecture and example files. The course uses the classic notebook or JupyterLab interface; either is fine. VS Code can also open `.ipynb` files directly without a separate Jupyter install, but `nbconvert` (used in some of the build steps documented in `CLAUDE.md`) requires a Python Jupyter install.

Link: <https://jupyter.org/install>

Install (requires Python 3.9 or newer):

- macOS: `brew install python && python3 -m pip install --user jupyterlab nbconvert`
- Windows: install Python from <https://www.python.org/downloads/> (check the box "Add python.exe to PATH"), then `py -m pip install --user jupyterlab nbconvert`.
- Linux: `sudo apt-get install -y python3 python3-pip && python3 -m pip install --user jupyterlab nbconvert`

Verify:

```bash
jupyter --version
jupyter lab --version
```

Both commands should print a version, not an error.

---

## 4. VS Code

What it is: the recommended editor for the course. It opens `.ipynb` notebooks natively, runs Julia cells, and has good Git integration.

Link: <https://code.visualstudio.com/>

Install:

- macOS: `brew install --cask visual-studio-code` or download from the link above.
- Windows: `winget install --id Microsoft.VisualStudioCode -e`.
- Linux (Debian / Ubuntu): follow <https://code.visualstudio.com/docs/setup/linux>, or `sudo snap install --classic code`.

Then install three extensions from inside VS Code (Extensions panel, or `code --install-extension`):

```bash
code --install-extension julialang.language-julia
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.python
```

- `julialang.language-julia` — Julia language server, REPL, plot pane.
- `ms-toolsai.jupyter` — run `.ipynb` files inside VS Code.
- `ms-python.python` — needed by the Jupyter extension.

Verify: open any `.ipynb` file in this repo. The kernel picker in the top right should list `julia-1.12` after Section 5 below.

---

## 5. IJulia (Julia Kernel for Jupyter)

What it is: the Julia kernel that lets Jupyter and VS Code execute Julia cells. The course expects the kernel to be named `julia-1.12`. The repo memory notes: notebooks were built against `--ExecutePreprocessor.kernel_name=julia-1.12`, so the kernel name matters if you plan to run `nbconvert`.

Link: <https://github.com/JuliaLang/IJulia.jl>

Install from a Julia REPL:

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
IJulia.installkernel("Julia 1.12")
```

Verify:

```bash
jupyter kernelspec list
```

The output should include a line like `julia-1.12  /Users/<you>/Library/Jupyter/kernels/julia-1.12`.

---

## 6. Instantiate the Julia Environments

Each session has its own `Project.toml` that depends on the shared `eCornellAIFinance` package in `code/`. The first instantiate downloads every Julia dependency, including `JumpHMM.jl` from GitHub and the Alpaca SDK from a varnerlab fork. Plan for ~2 GB on disk and 10 to 20 minutes the first time.

```bash
for s in 1 2 3 4; do
  pushd lectures/session-$s
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  popd
done
```

Also instantiate the shared package's test environment:

```bash
cd code
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
cd ..
```

Verify: the test suite at `code/test/runtests.jl` should pass without errors.

---

## 7. Julia Packages Used by the Course

Listed by purpose. All are installed transitively by Section 6; the table is for reference, not for manual `Pkg.add` calls.

### Course package (local path or GitHub)

| Package | Source | Used in | Purpose |
|:--------|:-------|:--------|:--------|
| `eCornellAIFinance` | local path `code/` | all sessions | Types, factories, compute kernels, I/O. |
| `JumpHMM` | <https://github.com/varnerlab/JumpHMM.jl> | all sessions | Synthetic market data generator and HMM tools. |
| `Alpaca` | <https://github.com/varnerlab/alpaca-markets-sdk> | sessions 3, 4 | Julia SDK for Alpaca Markets paper trading. |

### Optimization

| Package | Docs | Used for |
|:--------|:-----|:---------|
| `JuMP` | <https://jump.dev/JuMP.jl/stable/> | Algebraic modeling for portfolio QPs and LPs. |
| `Clarabel` | <https://clarabel.org/stable/> | Conic and quadratic solver used by min-variance optimization. |
| `Ipopt` | <https://coin-or.github.io/Ipopt/> | Nonlinear interior-point solver used by utility-based allocators. |

### Data and I/O

| Package | Docs | Used for |
|:--------|:-----|:---------|
| `CSV` | <https://csv.juliadata.org/> | Read and write CSVs (tickers, configs, daily bars). |
| `DataFrames` | <https://dataframes.juliadata.org/stable/> | In-memory tabular data. |
| `JLD2` | <https://juliaio.github.io/JLD2.jl/stable/> | Save and load synthetic datasets and trained models. |
| `JSON` | <https://github.com/JuliaIO/JSON.jl> | Parse Alpaca and Anthropic API responses. |
| `HTTP` | <https://juliaweb.github.io/HTTP.jl/stable/> | REST calls to Anthropic and Alpaca. |
| `FileIO` | <https://juliaio.github.io/FileIO.jl/stable/> | File-format dispatch used by `JLD2`. |
| `Dates`, `LinearAlgebra`, `Statistics`, `Random` | Julia stdlib | Numerics and time handling. |

### Statistics and ML

| Package | Docs | Used for |
|:--------|:-----|:---------|
| `Distributions` | <https://juliastats.org/Distributions.jl/stable/> | Probability distributions for SIM residuals and Monte Carlo. |
| `StatsBase` | <https://juliastats.org/StatsBase.jl/stable/> | Sample statistics, weighting, sampling. |
| `HypothesisTests` | <https://juliastats.org/HypothesisTests.jl/stable/> | t-tests and friends for the Session 3 validation gates. |
| `Flux` | <https://fluxml.ai/Flux.jl/stable/> | Neural networks (Session 3 REINFORCE, DQN; Session 4 GNN). |
| `GraphNeuralNetworks` | <https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/> | GNN layers for the Session 4 money-laundering detection optional notebook. |
| `Graphs` | <https://juliagraphs.org/Graphs.jl/stable/> | Graph data structures used by the GNN example. |

### Plotting and tables

| Package | Docs | Used for |
|:--------|:-----|:---------|
| `Plots` | <https://docs.juliaplots.org/stable/> | All figures. |
| `StatsPlots` | <https://docs.juliaplots.org/stable/generated/statsplots/> | Distribution and group plots. |
| `Colors` | <https://juliagraphics.github.io/Colors.jl/stable/> | Color definitions for figures. |
| `PrettyTables` | <https://ronisbr.github.io/PrettyTables.jl/stable/> | Notebook-rendered tables. |

### Notebook environments

| Package | Docs | Used for |
|:--------|:-----|:---------|
| `IJulia` | <https://julialang.github.io/IJulia.jl/stable/> | Jupyter kernel. |
| `Pluto` | <https://plutojl.org/> | Reactive notebooks used in a few Session 1 introductions. |
| `PlutoUI` | <https://github.com/JuliaPluto/PlutoUI.jl> | Interactive widgets for Pluto. |
| `HypertextLiteral` | <https://github.com/JuliaPluto/HypertextLiteral.jl> | HTML rendering helper used inside Pluto cells. |

---

## 8. Anthropic API Account (Sessions 3 and 4)

What it is: the API that powers Claude. The Session 4 news-sentiment pipeline (`score_news_with_claude!` in `code/src/Compute.jl`) calls the Anthropic Messages API directly over HTTPS. Sessions 1 and 2 do not need this.

Link: <https://console.anthropic.com/>

### Create the account

1. Open <https://console.anthropic.com/> and click "Sign up".
2. Sign up with email + password or with a Google account.
3. Verify the email; complete the phone-number SMS verification (Anthropic requires it to issue an API key).
4. On the welcome screen, name the workspace something like `ecornell-ai-finance`.

### Add billing and set a monthly cap

1. Console -> Settings -> Plans & Billing.
2. Click "Add payment method" and enter a card.
3. Purchase initial credits. $25 is enough to complete the course; the repo estimates ~$10 total spend across the cron-driven news fires (see `lectures/session-4/scripts/cron_setup_instructions.md`).
4. While still on the Billing page, click "Usage limits" and set a monthly cap (e.g. $50). This is a hard ceiling that prevents runaway spend.

Rate-limit note: brand-new accounts are placed in Tier 1 (10K input tokens/min and 5 requests/min for Sonnet 4.6). Promotion to Tier 2 requires at least $40 cumulative spend and at least 7 days since the first deposit. If you plan to run the full intraday cron from day one, deposit the $40 up front.

### Generate the API key

1. Console -> Settings -> API Keys -> "Create key".
2. Name it `ecornell-s4` (or similar) and copy the value. The console only shows it once.
3. Treat the key like a password. Do not paste it into a notebook cell, commit it, or share it.

### Store the key in `~/.ecornell-s4-env`

The course does **not** put the Anthropic key in `~/.zshrc`. Cron jobs do not source `.zshrc`, so the key is kept in a dedicated env file that the cron entries source explicitly:

```bash
cat > ~/.ecornell-s4-env <<'EOF'
# Loaded by the S4 cron entries (lectures/session-4/scripts/setup_cron.sh).
# Kept outside the repo so the key is never committed.
export ANTHROPIC_API_KEY="sk-ant-..."
EOF
chmod 600 ~/.ecornell-s4-env
```

Replace `sk-ant-...` with the value from the previous step. `chmod 600` restricts the file to the current user.

For interactive Julia work outside cron (running a notebook by hand), either source the file in the current shell or add a one-liner to `~/.zshrc`:

```bash
# Option A: source on demand
source ~/.ecornell-s4-env

# Option B: load automatically in every interactive shell
echo '[ -f ~/.ecornell-s4-env ] && source ~/.ecornell-s4-env' >> ~/.zshrc
```

Windows PowerShell equivalent (no cron, see Section 10):

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```

The default model used by the course code is `claude-sonnet-4-6`. The model id can be overridden per-call.

Verify (from a Julia REPL with the project activated, after sourcing the env file):

```julia
using HTTP, JSON
r = HTTP.post(
    "https://api.anthropic.com/v1/messages",
    [
        "x-api-key" => ENV["ANTHROPIC_API_KEY"],
        "anthropic-version" => "2023-06-01",
        "content-type" => "application/json",
    ],
    JSON.json(Dict(
        "model" => "claude-sonnet-4-6",
        "max_tokens" => 16,
        "messages" => [Dict("role" => "user", "content" => "ping")],
    )),
)
println(r.status)   # 200
```

---

## 9. Alpaca Markets Paper-Trading Account (Sessions 3 and 4)

What it is: a brokerage with a free paper-trading sandbox. The Session 4 production engine sends paper orders, pulls daily bars, and verifies fills against an Alpaca paper account. Sessions 1 and 2 do not need this.

Link: <https://alpaca.markets/>

### Create the account

1. Open <https://alpaca.markets/> and click "Sign up".
2. Provide email, password, and basic identity info. Alpaca asks for full name and address even for paper accounts (required by US regulations).
3. Verify the email link Alpaca sends.
4. After login, the dashboard at <https://app.alpaca.markets/dashboard/overview> has a "Paper" / "Live" toggle in the top-left. Make sure it is set to **Paper**. The course never touches a live account.
5. The paper account is funded automatically with $100,000 in virtual cash.

### Generate paper-trading API keys

1. In the Paper dashboard, open the "Home" tab and find the "API Keys" panel on the right, or go directly to <https://app.alpaca.markets/paper/dashboard/overview>.
2. Click "Generate New Key". A modal appears showing the **API Key ID** (starts with `PK...`) and the **Secret Key**. The secret is only shown once; copy it immediately.
3. If you want multiple paper books (the course supports this), repeat to generate a second key pair. Each key pair is its own paper account.

### Store the credentials in `credentials.toml`

The course does **not** put Alpaca keys in environment variables. The Julia SDK reads them from a TOML file. The template is committed; your filled-in copy is gitignored.

```bash
cd lectures/session-4/config
cp credentials.toml.example credentials.toml
```

Open `credentials.toml` in an editor. The default section is named `[Credentials]`:

```toml
[Credentials]
endpoint = "https://paper-api.alpaca.markets"
key      = "PK...your key id..."
secret   = "...your secret..."
```

For multiple paper accounts, add named sections and select them at call time with `Alpaca.load_client(path; section = "paper_production")`. The template file has commented-out examples.

Verify (from a Julia REPL with `lectures/session-4` activated):

```bash
cd lectures/session-4
julia --project=. scripts/verify_alpaca_bars.jl
```

Expected output: your account number, $100K cash, and a few 30-minute OHLC bars for AAPL, MSFT, and SPY.

### Where transaction and audit files are stored

All paper-trading artifacts land under `lectures/session-4/data/`. Most are written by the cron and are gitignored:

| Path | Written by | Purpose |
|:-----|:-----------|:--------|
| `data/production-log.txt` | every cron fire | Plain-text log of each engine and news fire. |
| `data/production-state.jld2` | engine fires | Latest engine state (positions, parameter snapshots, bandit state). |
| `data/intraday-tape/` | engine fires | One JLD2 file per fire with full decision context. |
| `data/decisions/` | engine fires | Per-fire decision audit records (target weights, deltas, exception flags). |
| `data/queue/` | engine fires | Compliance-exception queue for trades that need human sign-off. |
| `data/tickets/` | engine close (16:00) | Next-day execution tickets requiring a class signature. |
| `data/paper-trading-execution.jld2` | execution fires | Alpaca order IDs, fill prices, slippage vs. target. |
| `data/news/news-YYYY-MM-DD-HH.jld2` | hourly news cron | Headlines + Claude scores for the hour. |
| `data/news/budget-YYYY-MM-DD.json` | hourly news cron | Daily Anthropic spend tracker (search count vs. cap). |
| `data/daily-baseline-bars.jld2` | `pull_daily_bars.jl` | Cached daily OHLC bars for the offline daily-cadence baseline. |
| `data/daily-baseline-tape.jld2` | `daily_baseline.jl` | Per-day decisions from the daily-cadence counterfactual. |

To inspect what an engine fire wrote without reading JLD2 directly:

```bash
tail -f lectures/session-4/data/production-log.txt
```

To wipe state and start the paper book over (after rotating an Alpaca key, for example): delete the JLD2 files under `data/` (keep the directory) and re-run `scripts/deploy_initial_allocation.jl` to seed a fresh book.

---

## 10. Scheduler for the Session 4 Intraday Loop

What it is: the Session 4 production engine fires 22 times per weekday during US market hours (14 engine fires + 7 hourly news fires + 1 next-day execution). On macOS and Linux this is wired up with `cron`. On Windows the equivalent is Task Scheduler.

The full reference is `lectures/session-4/scripts/cron_setup_instructions.md`. The summary below is enough to install and verify; defer to that file when anything below is ambiguous.

Prerequisites (both platforms):

- Sessions 3 and 4 fully instantiated (Section 6).
- `~/.ecornell-s4-env` exists and contains `ANTHROPIC_API_KEY` (Section 8).
- `lectures/session-4/config/credentials.toml` exists and verifies (Section 9).
- The system timezone is `America/New_York`. Cron does not know about ET; the host clock has to be on it. Verify with `date` — the trailing letters should be `EST` or `EDT`.
  - macOS: `sudo systemsetup -settimezone America/New_York`
  - Linux: `sudo timedatectl set-timezone America/New_York`
  - Windows: Settings -> Time & language -> Date & time -> "(UTC-05:00) Eastern Time".

### macOS and Linux: install via `setup_cron.sh`

```bash
cd lectures/session-4/scripts
chmod +x setup_cron.sh
./setup_cron.sh
```

The script refuses to install if `~/.ecornell-s4-env` is missing or if `julia` is not on `PATH`. On success it appends five tagged lines to your existing crontab, each marked with `[AI-FINANCE]`.

Inspect what landed:

```bash
crontab -l | grep '\[AI-FINANCE\]'
```

Remove all course entries (leaves any unrelated cron jobs intact):

```bash
crontab -l | grep -v '\[AI-FINANCE\]' | crontab -
```

### macOS only: grant Full Disk Access to cron

Modern macOS sandboxes `cron`. Without this step the fires will run but the log will be full of "Operation not permitted" errors.

1. System Settings -> Privacy & Security -> Full Disk Access.
2. Click `+`, then `Cmd+Shift+G` and paste `/usr/sbin/cron`.
3. Add it. Also add Terminal.app or iTerm, whichever you launch Julia from.
4. Reboot for the change to take effect cleanly.

### Windows: use Task Scheduler instead

Windows has no `cron`. The course's `setup_cron.sh` is bash-only and will not run on Windows. The closest equivalent is the built-in Task Scheduler (`taskschd.msc`), but it has two caveats worth flagging before you start:

- Task Scheduler does not have a single `0,30 10-15 * * 1-5` syntax. You have to create one task per fire (15+ tasks) or use a single repeating task with a 30-minute interval and a custom market-hours guard inside the runner.
- The course does not ship a `setup_task_scheduler.ps1`. You will build the schedule by hand.

If you still want to run the Session 4 cron on Windows, follow this outline:

1. Install the Windows Subsystem for Linux (WSL2) and Ubuntu (`wsl --install`). Inside WSL run `setup_cron.sh` exactly as on Linux. **This is the recommended path** — it reuses the existing scripts unchanged.
2. If WSL is not available, open Task Scheduler (`Win+R`, `taskschd.msc`) and create one "Basic Task" per fire:
   - Trigger: Weekly, Mon-Fri, at the fire time (09:30, 10:00, 10:30, ..., 16:00).
   - Action: Start a program.
     - Program: `C:\Path\To\julia.exe`
     - Arguments: `--project="C:\path\to\lectures\session-4" "C:\path\to\lectures\session-4\scripts\production_runner.jl" --mode=engine`
     - Start in: `C:\path\to\lectures\session-4\scripts`
   - In "Conditions", uncheck "Start only if on AC power" if running on a laptop.
   - In "Settings", set "If the task fails, restart every 1 minute, up to 3 times".
3. Repeat for `--mode=engine_close` at 16:00, `--mode=execute_signed_ticket` at 09:35, and `news_scorer.jl --mode=hourly` at the seven news fires.
4. Environment variables: Task Scheduler does not source `~/.ecornell-s4-env`. Either set `ANTHROPIC_API_KEY` as a User-level environment variable in Windows (`setx ANTHROPIC_API_KEY "sk-ant-..."`), or wrap each action in a PowerShell command that loads a `.env` file.

Verify on Windows:

```powershell
Get-ScheduledTask | Where-Object {$_.TaskName -like "*AI-FINANCE*"}
```

If running an interactive course session and you do not need the cron loop, you can skip this whole section. The notebooks run fine without a scheduler; only the live intraday demo depends on it.

### Verify the schedule is firing

After install, watch the log live for the next scheduled minute boundary (`:00` or `:30` during market hours, weekday only):

```bash
tail -f lectures/session-4/data/production-log.txt
```

You should see lines like `[ENGINE] equity=$... bandit=...` and `[NEWS-HOURLY] fire at ...` appear in real time.

---

## 11. Build the API Docs (optional)

```bash
cd code/docs
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. make.jl
open build/index.html   # macOS; use xdg-open on Linux or start on Windows
```

The hosted version is at <https://varnerlab.org/eCornell-AI-finance-lectures/dev/>.

---

## Final Verification Checklist

Run each line and confirm a clean result before declaring the install done.

```bash
julia --version                                 # 1.10+, ideally 1.12.x
jupyter --version                               # Python Jupyter present
jupyter kernelspec list | grep julia-1.12       # IJulia kernel registered
code --version                                  # VS Code CLI works
julia --project=lectures/session-1 -e 'using eCornellAIFinance; println("ok")'
julia --project=lectures/session-4 -e 'using Alpaca; println("ok")'

# Sessions 3-4 only: credentials in place
test -f lectures/session-4/config/credentials.toml && echo "alpaca creds present"
test -f ~/.ecornell-s4-env && echo "anthropic env file present"
source ~/.ecornell-s4-env && echo "${ANTHROPIC_API_KEY:0:7}"   # prints sk-ant-

# Sessions 3-4 only: live API smoke tests
cd lectures/session-4 && julia --project=. scripts/verify_alpaca_bars.jl
```

If every line above returns the expected output, the machine is ready to run all four sessions. If the Session 4 intraday cron is in scope as well, follow Section 10 and confirm `crontab -l | grep '\[AI-FINANCE\]'` shows five entries.

#!/usr/bin/env julia
#
# production_runner.jl -- Intraday production runner for the AI finance engine.
#
# Modes:
#   --mode=engine                  -- 30-min intraday fire during market hours.
#                                     Pulls latest bars, updates EWLS, runs the
#                                     bandit + allocator, splits proposed trades
#                                     into auto-execute (submit to Alpaca paper)
#                                     and queued (append to today's queue file).
#   --mode=engine_close            -- 16:00 fire. Same as engine, plus finalizes
#                                     the EOD tape and writes tomorrow's ticket
#                                     (data/tickets/ticket-YYYY-MM-DD.jld2).
#   --mode=execute_signed_ticket   -- 9:35 next-morning fire. Loads
#                                     data/tickets/signed-YYYY-MM-DD.jld2 from
#                                     today (the class signed it last night) and
#                                     submits it to Alpaca paper.
#
# Cron schedule is in setup_cron.sh. Configuration is in config/.
#

using Dates, Statistics, TOML

const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
const CONFIG_DIR = joinpath(SESSION_DIR, "config");
const DATA_DIR = joinpath(SESSION_DIR, "data");
const TAPE_DIR = joinpath(DATA_DIR, "intraday-tape");
const QUEUE_DIR = joinpath(DATA_DIR, "queue");
const TICKET_DIR = joinpath(DATA_DIR, "tickets");
const NEWS_DIR = joinpath(DATA_DIR, "news");

using Pkg;
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance
using Alpaca

const STATE_PATH = joinpath(DATA_DIR, "production-state.jld2");
const LOG_PATH = joinpath(DATA_DIR, "production-log.txt");
const CREDS_PATH = joinpath(CONFIG_DIR, "credentials.toml");
const CONFIG_PATH = joinpath(CONFIG_DIR, "production-config.toml");
const S1_ALLOCATION_PATH = joinpath(SESSION_DIR, "..", "session-1", "data", "minvar-allocation.jld2");
const POLICY_ARTIFACT_PATH = joinpath(SESSION_DIR, "..", "session-3", "data", "policy-eta-results.jld2");

# ──────────────────────────────────────────────────────────────────────────────
# CLI + logging
# ──────────────────────────────────────────────────────────────────────────────
"""
    parse_args() -> String

Parse the `--mode=` CLI argument and return the requested fire mode
("engine", "engine_close", or "execute_signed_ticket"). Unknown or missing
args default to "engine" (the every-30-min intraday fire). Used by `main`.
"""
function parse_args()
    mode = "engine";
    for arg in ARGS
        if startswith(arg, "--mode=")
            mode = String(split(arg, "=")[2]);   # split returns SubString; log_entry wants String
        end
    end
    return mode;
end

"""
    log_entry(tag, msg)

Append one timestamped line to the shared production log
(`data/production-log.txt`) and echo it to stdout so the cron's stderr/out
captures the same record. The line format is `yyyy-mm-dd HH:MM:SS [<TAG>]
<msg>`; `tag` is upper-cased for readability.
"""
function log_entry(tag::String, msg::String)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS");
    line = "$(ts) [$(uppercase(tag))] $(msg)";
    println(line);
    mkpath(DATA_DIR);
    open(LOG_PATH, "a") do f
        println(f, line);
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
"""
    load_config() -> NamedTuple

Read `config/production-config.toml` and return the engine configuration
as a typed NamedTuple. Resolves four config blocks: `Engine` (bar_minutes,
EMA windows, half-life, GAIN, B0), `Compliance` (concentration cap,
trade-size cap, news-severity threshold, flag-severity threshold for
ticket assembly), `CircuitBreakers` (sentiment-driven λ override, max
drawdown), and `Schedule` (bar minutes again, redundant for sanity).
Resolves the ticker universe from either `Tickers.source = "session-1"`
(loading from the S1 minvar artifact) or the inline universe array.
"""
function load_config()
    isfile(CONFIG_PATH) || error("Config not found: $(CONFIG_PATH)");
    cfg = TOML.parsefile(CONFIG_PATH);
    eng = cfg["Engine"];
    comp = cfg["Compliance"];
    cb = cfg["CircuitBreakers"];
    news = cfg["News"];
    sched = cfg["Schedule"];
    bar_minutes = Int(eng["bar_minutes"]);

    source = String(get(cfg["Tickers"], "source", "manual"));
    tickers = if source == "session-1"
        isfile(S1_ALLOCATION_PATH) || error(
            "Tickers.source=\"session-1\" but $(S1_ALLOCATION_PATH) is missing. " *
            "Run the S1 BuildMinVariancePortfolio notebook to produce it, or set " *
            "Tickers.source=\"manual\" in $(CONFIG_PATH).");
        String.(load_results(S1_ALLOCATION_PATH)["my_tickers"])
    else
        String.(cfg["Tickers"]["universe"])
    end;
    log_entry("config", "tickers source=$(source) -> $(length(tickers)) names");

    eta_source = String(get(eng, "eta_source", "heuristic"));
    eta_source in ("heuristic", "policy") ||
        error("Engine.eta_source must be \"heuristic\" or \"policy\", got \"$(eta_source)\"");

    return (
        tickers = tickers,
        eta_source = eta_source,
        B₀ = Float64(eng["B0"]),
        bar_minutes = bar_minutes,
        half_life_calendar_days = Float64(eng["half_life_calendar_days"]),
        prior_weight_calendar_days = Float64(eng["prior_weight_calendar_days"]),
        N_short = Int(eng["N_short"]),
        N_long = Int(eng["N_long"]),
        GAIN = Float64(eng["GAIN"]),
        N_growth = Int(eng["N_growth"]),
        cost_bps = Float64(eng["cost_bps"]),
        epsilon = Float64(eng["epsilon"]),
        n_bandit_iters = Int(eng["n_bandit_iters"]),
        max_bandit_churn = Int(eng["max_bandit_churn"]),
        compliance = build_compliance_config(
            concentration_cap = Float64(comp["concentration_cap"]),
            position_size_limit = Float64(comp["position_size_limit"]),
            turnover_limit = Float64(comp["turnover_limit"])),
        news_severity_queue_threshold = Float64(comp["news_severity_queue_threshold"]),
        max_drawdown = Float64(cb["max_drawdown"]),
        sentiment_threshold = Float64(cb["sentiment_threshold"]),
        sentiment_override_lambda = Float64(cb["sentiment_override_lambda"]),
        flag_severity_threshold = Float64(news["flag_severity_threshold"]),
        market_open = String(sched["market_open"]),
        market_close = String(sched["market_close"]),
    );
end

# ──────────────────────────────────────────────────────────────────────────────
# Policy-eta shadow logging (REINFORCE policy from S3 Optional PolicyEtaLearning)
# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 of the S3 -> S4 handoff: load the trained policy if its artifact
# exists, compute eta_policy alongside the heuristic at every fire, and
# persist both into engine_snapshot. The actual decision still uses the
# heuristic; we only swap behind a config flag once the shadow record clears
# the validation tournament.

"""
    load_trained_policy() -> Union{MyREINFORCEPolicy, Nothing}

Load the S3 REINFORCE policy artifact if present. The artifact provides
the trained weights for the shadow-eta calculation: at every fire we
compute eta_policy alongside the heuristic eta and persist both into the
engine_snapshot. The live decision still uses the heuristic; we only swap
the trained policy in behind a config flag once the shadow record clears
the S3 validation tournament. Returns `nothing` (logged) if the artifact
is missing or fails to deserialize, in which case `compute_shadow_eta_policy`
returns `NaN` and the engine continues with the heuristic only.
"""
function load_trained_policy()
    isfile(POLICY_ARTIFACT_PATH) || return nothing;
    try
        d = load_results(POLICY_ARTIFACT_PATH);
        return build(MyREINFORCEPolicy, (
            n_features = length(d["policy_w_mu"]),
            eta_min    = Float64(d["policy_eta_min"]),
            eta_max    = Float64(d["policy_eta_max"]),
            w_mu       = Vector{Float64}(d["policy_w_mu"]),
            b_mu       = Float64(d["policy_b_mu"]),
            log_sigma  = Float64(d["policy_log_sigma"]),
        ));
    catch err
        log_entry("policy-eta", "load failed at $(POLICY_ARTIFACT_PATH): $(err)");
        return nothing;
    end
end

# Cross-asset annualized realized vol from the intraday market series.
"""
    intraday_realized_vol(intraday_market_prices; window = 21) -> Float64

Annualized realized volatility from the rolling intraday market-price
series. Computes per-bar log returns, takes the last `window` of them,
and returns `std(returns) * sqrt(252)`. Used as one of the six policy-state
features fed into the trained REINFORCE policy in
`compute_shadow_eta_policy`.

Returns 0.0 when the buffer has fewer than two bars (cold-start) so the
shadow eta is well-defined on the first fire of the day.
"""
function intraday_realized_vol(intraday_market_prices::Vector{Float64};
    window::Int = 21)::Float64
    n = length(intraday_market_prices);
    n < 2 && return 0.0;
    rets = diff(log.(max.(intraday_market_prices, 1e-9)));
    isempty(rets) && return 0.0;
    win = rets[max(1, end-window+1):end];
    length(win) < 2 && return 0.0;
    return std(win) * sqrt(252);
end

# Compute eta_policy if the policy is available; otherwise NaN (signals
# missing artifact in production-state).
"""
    compute_shadow_eta_policy(policy, λ_short, λ_long, regime, drawdown,
                              realized_vol, news_flag_count) -> Float64

Evaluate the trained REINFORCE policy on the current engine state and
return its eta recommendation as a shadow signal (not used for the live
decision; persisted to the engine snapshot for validation analysis).
Returns `NaN` when `policy === nothing`, which is how the runner signals
that the S3 policy artifact is missing or failed to load.
"""
function compute_shadow_eta_policy(policy, λ_short::Float64, λ_long::Float64,
    regime::Symbol, drawdown::Float64, realized_vol::Float64,
    news_flag_count::Float64)::Float64
    policy === nothing && return NaN;
    s = build_policy_state(
        lambda_short = λ_short, lambda_long = λ_long,
        regime = regime, drawdown = drawdown,
        realized_vol = realized_vol, news_flag_count = news_flag_count,
    );
    return policy_eta(policy, s);
end

# ──────────────────────────────────────────────────────────────────────────────
# Path helpers (per-day artifacts)
# ──────────────────────────────────────────────────────────────────────────────
todays_tape_path(d::Date) = joinpath(TAPE_DIR, "tape-$(Dates.format(d, "yyyy-mm-dd")).jld2");
todays_queue_path(d::Date) = joinpath(QUEUE_DIR, "queue-$(Dates.format(d, "yyyy-mm-dd")).jld2");
ticket_path(d::Date) = joinpath(TICKET_DIR, "ticket-$(Dates.format(d, "yyyy-mm-dd")).jld2");
signed_ticket_path(d::Date) = joinpath(TICKET_DIR, "signed-$(Dates.format(d, "yyyy-mm-dd")).jld2");

# ──────────────────────────────────────────────────────────────────────────────
# News integration
# ──────────────────────────────────────────────────────────────────────────────
"""
    latest_news_artifact(fire_time, tickers) -> NamedTuple

Scan `data/news/` for `news-YYYY-MM-DD-HH.jld2` files whose timestamp is
at or before `fire_time` and return the most recent one's per-ticker
sentiment and severity dicts, plus the absolute path on disk for the
audit log. Returns zero-filled dicts and an empty path when the directory
is missing or no eligible file is found, so the engine always has a
well-defined snapshot to consult.
"""
function latest_news_artifact(fire_time::DateTime, tickers::Vector{String})
    isdir(NEWS_DIR) || return (
        sentiment = Dict{String,Float64}(t => 0.0 for t in tickers),
        severity = Dict{String,Float64}(t => 0.0 for t in tickers),
        path = "");
    pattern = r"^news-(\d{4}-\d{2}-\d{2})-(\d{2})\.jld2$";
    candidates = filter(f -> occursin(pattern, f), readdir(NEWS_DIR));
    isempty(candidates) && return (
        sentiment = Dict{String,Float64}(t => 0.0 for t in tickers),
        severity = Dict{String,Float64}(t => 0.0 for t in tickers),
        path = "");

    fire_stamp = Dates.format(fire_time, "yyyy-mm-dd-HH");
    eligible = sort(filter(f -> begin
        m = match(pattern, f);
        stamp = "$(m[1])-$(m[2])";
        stamp <= fire_stamp
    end, candidates));
    isempty(eligible) && return (
        sentiment = Dict{String,Float64}(t => 0.0 for t in tickers),
        severity = Dict{String,Float64}(t => 0.0 for t in tickers),
        path = "");

    latest = joinpath(NEWS_DIR, last(eligible));
    data = load_results(latest);
    sentiment = Dict{String,Float64}(t => get(data["sentiment"], t, 0.0) for t in tickers);
    severity = Dict{String,Float64}(t => get(data["severity"], t, 0.0) for t in tickers);
    return (sentiment = sentiment, severity = severity, path = latest);
end

# ──────────────────────────────────────────────────────────────────────────────
# State init / load
# ──────────────────────────────────────────────────────────────────────────────
"""
    initialize_state(client, cfg) -> Dict{String,Any}

First-time bootstrap of the persistent engine state. Seeds one
`MyEWLSState` per ticker from the frozen SIM calibration
(`code/src/data/sim-calibration.jld2`), with `half_life` and
`prior_weight` translated from calendar days to bars via
`intraday_half_life`. Initializes `peak_wealth` to the larger of the
config's starting balance (`B0`) and Alpaca's reported account equity,
sets `last_bar_timestamp` to the Unix epoch (so the first bar fetch is
unbounded on the left), and starts with empty intraday-market-price /
timestamp buffers. Saves the state to `STATE_PATH` and returns it.
"""
function initialize_state(client, cfg)
    tickers = cfg.tickers;
    K = length(tickers);
    Δt = intraday_dt(cfg.bar_minutes);
    half_life_bars = intraday_half_life(cfg.half_life_calendar_days, cfg.bar_minutes);
    prior_weight_bars = intraday_half_life(cfg.prior_weight_calendar_days, cfg.bar_minutes);

    log_entry("init", "Initializing state: bar_minutes=$(cfg.bar_minutes), " *
        "half_life_bars=$(round(half_life_bars, digits=1)), tickers=$(K)");

    # Seed SIM params from frozen calibration. EWLS will then track from there.
    cal = load_results(joinpath(eCornellAIFinance._PATH_TO_DATA, "sim-calibration.jld2"));
    all_tickers = cal["tickers"];
    tidx = Dict(all_tickers[i] => i for i in eachindex(all_tickers));
    for t in tickers
        haskey(tidx, t) || error("Ticker $(t) not in SIM calibration set.");
    end

    ewls_states = Dict{String,MyEWLSState}();
    for t in tickers
        i = tidx[t];
        ewls_states[t] = ewls_init(cal["alpha"][i], cal["beta"][i], cal["sigma_eps"][i];
            half_life = half_life_bars, prior_weight = prior_weight_bars);
    end

    acct = Alpaca.get_account(client);
    peak_wealth = max(cfg.B₀, acct.equity);

    state = Dict(
        "tickers" => tickers,
        "ewls_states" => ewls_states,
        "Δt" => Δt,
        "bar_minutes" => cfg.bar_minutes,
        "last_bar_timestamp" => DateTime(1970, 1, 1),
        "peak_wealth" => peak_wealth,
        "intraday_market_prices" => Float64[],   # rolling SPY close per bar, today only
        "intraday_market_timestamps" => DateTime[],
        "prior_action" => ones(Int, K),
    );
    save_results(STATE_PATH, state);
    return state;
end

"""
    load_or_init_state(client, cfg) -> Dict{String,Any}

Return the persisted engine state from `STATE_PATH` if it exists,
otherwise bootstrap a fresh one via `initialize_state`. Called by `main`
at the start of every fire so the cron is stateful across runs.
"""
function load_or_init_state(client, cfg)
    isfile(STATE_PATH) ? load_results(STATE_PATH) : initialize_state(client, cfg);
end

# ──────────────────────────────────────────────────────────────────────────────
# Bar fetch
# ──────────────────────────────────────────────────────────────────────────────
"""
    bar_interval_string(bar_minutes) -> String

Format a bar-interval integer as the string Alpaca's `get_bars` expects.
60-minute bars get the special form `"1Hour"`; every other interval is
expressed as `"<n>Min"` (e.g. 30 -> `"30Min"`). Used in `fetch_latest_bars`.
"""
function bar_interval_string(bar_minutes::Int)
    bar_minutes == 60 && return "1Hour";
    return "$(bar_minutes)Min";
end

"""
    fetch_latest_bars(client, tickers, bar_minutes, last_seen, fire_time)

Call Alpaca's `get_bars` for every ticker (plus SPY for market growth)
between `last_seen` (or one trading day before `fire_time`, whichever is
later) and `fire_time`. The one-day floor guarantees we always capture
today's bars even if `last_seen` is stale. Passes raw `DateTime`s
through: the Alpaca SDK formats them as RFC3339 internally, while
pre-formatting without a timezone marker triggers a 400 from the data API.
"""
function fetch_latest_bars(client, tickers::Vector{String}, bar_minutes::Int,
        last_seen::DateTime, fire_time::DateTime)
    interval = bar_interval_string(bar_minutes);
    # Look back up to one trading day so we always capture today's bars.
    # Pass DateTime directly: the SDK formats it as RFC3339 (yyyy-mm-ddTHH:MM:SSZ);
    # pre-formatting without a timezone returns HTTP 400.
    start_dt = max(last_seen, fire_time - Day(1));
    return Alpaca.get_bars(client, vcat(tickers, ["SPY"]), interval;
        start = start_dt, finish = fire_time);
end

# ──────────────────────────────────────────────────────────────────────────────
# Engine step (per fire)
# ──────────────────────────────────────────────────────────────────────────────
"""
    run_engine_step(client, cfg, state, fire_time; is_close = false) -> state

One full intraday engine fire. The function is long because one fire does
nine things in order:

  1. Fetch new bars from Alpaca for the trading universe plus SPY.
  2. For every new bar, append the SPY close to the rolling intraday-market
     buffer and update each ticker's EWLS state with the per-bar
     `(g_i, g_mkt)` growth pair.
  3. Compute live sentiment from the buffer, EMA-crossover `λ_t`, and
     apply the sentiment-driven `λ` override if the live sentiment is
     below `cfg.sentiment_threshold`.
  4. Pull current prices, positions, and cash from Alpaca, then mark
     wealth and drawdown to market.
  5. Drawdown circuit breaker: if realized drawdown exceeds
     `cfg.max_drawdown`, close all positions, persist state with zero
     shares, and return. No allocation runs today.
  6. Run the Cobb-Douglas allocator on the EWLS-tracked SIM parameters,
     sentiment-adjusted `λ_eff`, and the smoothed market growth, to get
     the per-ticker target weight vector.
  7. Convert target weights to integer share counts at the latest bar
     close, compute `Δshares`, and read per-ticker news severity from
     the latest hourly news artifact.
  8. Apply the compliance gate (concentration cap, dollar trade-size cap,
     news severity threshold) to split proposed trades into the
     auto-execute set `A_t` (submitted to Alpaca paper) and the queue set
     `Q_t` (appended to today's queue file as `MyComplianceQueueItem`s).
  9. Append the fire's snapshot to today's tape file with all gate
     decisions, the shadow eta from the trained policy, sentiment, λ,
     regime, drawdown, wealth, and the news artifact path for audit.

Persists the updated EWLS states, `last_bar_timestamp`, `peak_wealth`,
and the intraday buffers to `STATE_PATH` and returns the state dict for
chaining into `run_engine_close`. The `is_close` flag is passed through to
audit logging only; ticket assembly happens in `run_engine_close`.
"""
function run_engine_step(client, cfg, state, fire_time::DateTime; is_close::Bool = false)
    tickers = state["tickers"]::Vector{String};
    K = length(tickers);
    Δt = state["Δt"]::Float64;
    bar_minutes = state["bar_minutes"]::Int;
    ewls_states = state["ewls_states"]::Dict{String,MyEWLSState};
    last_seen = state["last_bar_timestamp"]::DateTime;
    intraday_market_prices = state["intraday_market_prices"]::Vector{Float64};
    intraday_market_timestamps = state["intraday_market_timestamps"]::Vector{DateTime};
    peak_wealth = state["peak_wealth"]::Float64;

    today = Date(fire_time);
    # New trading day -> reset intraday rolling buffers.
    if !isempty(intraday_market_timestamps) && Date(last(intraday_market_timestamps)) != today
        intraday_market_prices = Float64[];
        intraday_market_timestamps = DateTime[];
    end

    # 1) Fetch latest bars.
    bars = fetch_latest_bars(client, tickers, bar_minutes, last_seen, fire_time);
    spy_bars = bars["SPY"];
    new_bars = filter(b -> DateTime(b.t) > last_seen, spy_bars);
    n_new = length(new_bars);

    if n_new == 0
        log_entry("engine", "no new bars since $(last_seen); skipping engine update.");
        return state;
    end

    # 2) For each new bar, update EWLS and append to rolling intraday buffer.
    for b in new_bars
        bts = DateTime(b.t);
        push!(intraday_market_prices, b.c);
        push!(intraday_market_timestamps, bts);
        if length(intraday_market_prices) >= 2
            spy_prev = intraday_market_prices[end - 1];
            gm_t = (1.0 / Δt) * log(b.c / spy_prev);
            for ticker in tickers
                tb = bars[ticker];
                # find ticker bar at this timestamp
                idx = findfirst(x -> DateTime(x.t) == bts, tb);
                idx === nothing && continue;
                if idx >= 2
                    gi_t = (1.0 / Δt) * log(tb[idx].c / tb[idx - 1].c);
                    ewls_update!(ewls_states[ticker], gi_t, gm_t);
                end
            end
        end
    end
    last_seen = last(intraday_market_timestamps);

    # 3) Sentiment + lambda from rolling intraday tape (treats each 30-min bar
    #    like a "day" in the EMA windows; keep windows in trading days but
    #    recognize that intraday EMA is responsive).
    sentiment = compute_live_sentiment(intraday_market_prices);
    ema_s = compute_ema(intraday_market_prices; window = cfg.N_short);
    ema_l = compute_ema(intraday_market_prices; window = cfg.N_long);
    λ_t = compute_lambda(ema_s, ema_l; G = cfg.GAIN);
    λ_eff = λ_t[end];
    if sentiment < cfg.sentiment_threshold
        λ_eff = λ_eff * cfg.sentiment_override_lambda;
    end

    # 4) Pull current prices + positions from Alpaca.
    current_prices = zeros(K);
    for (k, t) in enumerate(tickers)
        tb = bars[t];
        if !isempty(tb)
            current_prices[k] = tb[end].c;
        end
    end
    positions = Alpaca.list_positions(client);
    current_shares = zeros(Float64, K);
    for (k, t) in enumerate(tickers)
        for p in positions
            if p.symbol == t
                current_shares[k] = Float64(p.qty);
                break;
            end
        end
    end
    acct = Alpaca.get_account(client);
    current_cash = Float64(acct.cash);
    current_wealth = sum(current_shares .* current_prices) + current_cash;
    peak_wealth = max(peak_wealth, current_wealth);
    drawdown = peak_wealth > 0 ? (peak_wealth - current_wealth) / peak_wealth : 0.0;

    # 5) Circuit breaker: drawdown.
    if drawdown > cfg.max_drawdown
        log_entry("engine", "DRAWDOWN BREACH ($(round(drawdown*100, digits=1))%) -- de-risking to cash.");
        Alpaca.close_all_positions(client; cancel_orders = true);
        # Persist state with zero positions and bail out (no allocation today).
        state["last_bar_timestamp"] = last_seen;
        state["peak_wealth"] = peak_wealth;
        state["intraday_market_prices"] = intraday_market_prices;
        state["intraday_market_timestamps"] = intraday_market_timestamps;
        save_results(STATE_PATH, state);
        return state;
    end

    # 6) Eta selection (heuristic decision + REINFORCE policy shadow log).
    regime = classify_regime(λ_eff);
    eta_heuristic = compute_adaptive_eta(λ_eff);

    # Shadow log: compute eta_policy from the S3 trained policy if available.
    # Decision stays on the heuristic; eta_policy is recorded for the
    # validation tournament that gates the production cutover.
    ema_xs_p = compute_ema(intraday_market_prices; window = max(5, cfg.N_short ÷ 4));
    λ_short_series = compute_lambda(ema_xs_p, ema_s; G = cfg.GAIN);
    λ_short = isempty(λ_short_series) ? 0.0 : λ_short_series[end];
    realized_vol = intraday_realized_vol(intraday_market_prices);
    trained_policy = load_trained_policy();
    eta_policy = compute_shadow_eta_policy(trained_policy, λ_short, λ_eff,
        regime, drawdown, realized_vol, 0.0);
    eta = if cfg.eta_source == "policy" && !isnan(eta_policy)
        eta_policy
    else
        if cfg.eta_source == "policy"
            log_entry("policy-eta", "eta_source=\"policy\" but no trained policy artifact; falling back to heuristic.");
        end
        eta_heuristic
    end;

    # 7) Allocator. Build a single-period rebalancing context using current
    # snapshot (frozen sim params from EWLS, current prices, λ_eff).
    sim_params_current = Dict{String,Tuple{Float64,Float64,Float64}}(
        t => (ewls_states[t].α, ewls_states[t].β, ewls_states[t].σ_ε) for t in tickers
    );
    gm_raw = compute_market_growth(intraday_market_prices; Δt = Δt);
    gm_smoothed = compute_ema(gm_raw; window = cfg.N_growth);
    gm_t = isempty(gm_smoothed) ? 0.0 : gm_smoothed[end];
    γ = compute_preference_weights(sim_params_current, tickers, gm_t, λ_eff);
    γ_sum = sum(γ);
    γ_sum > 0 || (γ = ones(K) ./ K);
    target_weights = γ ./ sum(γ);

    target_dollar = target_weights .* current_wealth;
    target_shares = target_dollar ./ max.(current_prices, 1e-8);
    delta_shares = round.(Int, target_shares .- current_shares);

    # 8) News severity per ticker.
    news = latest_news_artifact(fire_time, tickers);

    # 9) Build proposed-trade tuples (skip zero deltas).
    proposed_trades = NamedTuple[];
    for (k, ticker) in enumerate(tickers)
        delta = delta_shares[k];
        delta == 0 && continue;
        side = delta > 0 ? :buy : :sell;
        dollar_value = Float64(delta) * current_prices[k];  # signed
        post_weight = abs(target_dollar[k]) / max(current_wealth, 1e-8);
        push!(proposed_trades, (
            ticker = ticker,
            qty = abs(delta),
            side = side,
            dollar_value = dollar_value,
            post_trade_weight = post_weight,
        ));
    end

    # 10) Gate config + portfolio value for this fire.
    gate_config = copy(cfg.compliance);
    gate_config["news_severity_queue_threshold"] = cfg.news_severity_queue_threshold;
    gate_config["portfolio_value"] = current_wealth;

    snaps = Dict{String,NamedTuple}(
        t => (news_severity = news.severity[t],) for t in tickers
    );
    engine_snapshot = Dict{String,Any}(
        "eta" => eta,
        "eta_heuristic" => eta_heuristic,
        "eta_policy" => eta_policy,
        "lambda_eff" => λ_eff,
        "lambda_short" => λ_short,
        "realized_vol" => realized_vol,
        "sentiment" => sentiment,
        "regime" => string(regime),
        "drawdown" => drawdown,
        "wealth" => current_wealth,
        "news_path" => news.path,
    );

    auto_trades, queued_items = split_intraday_trades(proposed_trades, snaps,
        gate_config, fire_time, engine_snapshot);

    # 11) Submit auto-cleared trades to Alpaca paper.
    n_submitted = 0;
    submitted_ids = String[];
    for trade in auto_trades
        side_str = trade.side == :buy ? "buy" : "sell";
        try
            order = Alpaca.submit_order(client, trade.ticker, trade.qty, side_str;
                type = "market", time_in_force = "day");
            push!(submitted_ids, order.id);
            n_submitted += 1;
        catch err
            log_entry("engine", "Alpaca submit_order failed for $(trade.ticker): $(err)");
        end
    end

    # 12) Persist queued items to today's queue file.
    if !isempty(queued_items)
        mkpath(QUEUE_DIR);
        qpath = todays_queue_path(today);
        for q in queued_items
            append_queue_item!(qpath, q);
        end
    end

    # 13) Append today's tape entry.
    mkpath(TAPE_DIR);
    tape_path = todays_tape_path(today);
    tape_entries = isfile(tape_path) ? load_results(tape_path)["entries"] : NamedTuple[];
    entry = (
        fire_time = fire_time,
        is_close = is_close,
        last_bar = last_seen,
        sentiment = sentiment,
        lambda_eff = λ_eff,
        lambda_short = λ_short,
        realized_vol = realized_vol,
        regime = regime,
        eta = eta,
        eta_heuristic = eta_heuristic,
        eta_policy = eta_policy,
        target_weights = target_weights,
        proposed_n = length(proposed_trades),
        auto_n = length(auto_trades),
        queued_n = length(queued_items),
        submitted_ids = submitted_ids,
        wealth = current_wealth,
        drawdown = drawdown,
        news_path = news.path,
    );
    push!(tape_entries, entry);
    save_results(tape_path, Dict("entries" => tape_entries));

    eta_policy_log = isnan(eta_policy) ? "n/a" : string(round(eta_policy, digits=2));
    log_entry("engine",
        "fire=$(fire_time) " *
        "sent=$(round(sentiment, digits=3)) λ=$(round(λ_eff, digits=3)) " *
        "regime=$(regime) η=$(round(eta, digits=2)) " *
        "[shadow η_h=$(round(eta_heuristic, digits=2)) η_p=$(eta_policy_log)] " *
        "wealth=\$$(round(current_wealth, digits=2)) dd=$(round(drawdown*100, digits=1))% " *
        "auto=$(length(auto_trades)) queued=$(length(queued_items)) submitted=$(n_submitted)");

    # 14) Update mutable state.
    state["ewls_states"] = ewls_states;
    state["last_bar_timestamp"] = last_seen;
    state["peak_wealth"] = peak_wealth;
    state["intraday_market_prices"] = intraday_market_prices;
    state["intraday_market_timestamps"] = intraday_market_timestamps;
    save_results(STATE_PATH, state);
    return state;
end

# ──────────────────────────────────────────────────────────────────────────────
# Mode: engine_close — run engine step + write tomorrow's ticket
# ──────────────────────────────────────────────────────────────────────────────
"""
    run_engine_close(client, cfg, state, fire_time) -> state

The 16:00 ET engine fire. Runs `run_engine_step` with `is_close = true`
to update state and write today's tape entry, then recomputes the
sentiment / λ / regime / target weights one more time so the assembled
`MyTomorrowsTicket` reflects the EOD state. Pulls current prices,
positions, and cash from Alpaca, applies the news flag set (any ticker
whose severity meets `cfg.flag_severity_threshold` is flagged in the
ticket), and calls `build_tomorrows_ticket` to produce the artifact the
desk PM signs in the evening review. Saves the ticket under tomorrow's
business-date filename (skipping over weekends) so the next morning's
`execute_signed_ticket` cron can find it.
"""
function run_engine_close(client, cfg, state, fire_time::DateTime)
    state = run_engine_step(client, cfg, state, fire_time; is_close = true);

    # Recompute snapshot for ticket assembly.
    tickers = state["tickers"]::Vector{String};
    K = length(tickers);
    ewls_states = state["ewls_states"]::Dict{String,MyEWLSState};
    intraday_market_prices = state["intraday_market_prices"]::Vector{Float64};
    Δt = state["Δt"]::Float64;

    sentiment = compute_live_sentiment(intraday_market_prices);
    ema_s = compute_ema(intraday_market_prices; window = cfg.N_short);
    ema_l = compute_ema(intraday_market_prices; window = cfg.N_long);
    λ_t = compute_lambda(ema_s, ema_l; G = cfg.GAIN);
    λ_eff = λ_t[end];
    if sentiment < cfg.sentiment_threshold
        λ_eff = λ_eff * cfg.sentiment_override_lambda;
    end
    regime = classify_regime(λ_eff);
    eta = compute_adaptive_eta(λ_eff);

    sim_params_current = Dict{String,Tuple{Float64,Float64,Float64}}(
        t => (ewls_states[t].α, ewls_states[t].β, ewls_states[t].σ_ε) for t in tickers
    );
    gm_raw = compute_market_growth(intraday_market_prices; Δt = Δt);
    gm_smoothed = compute_ema(gm_raw; window = cfg.N_growth);
    gm_t = isempty(gm_smoothed) ? 0.0 : gm_smoothed[end];
    γ = compute_preference_weights(sim_params_current, tickers, gm_t, λ_eff);
    γ_sum = sum(γ);
    γ_sum > 0 || (γ = ones(K) ./ K);
    target_weights = γ ./ sum(γ);

    # Pull final prices + positions for the ticket.
    bars = fetch_latest_bars(client, tickers, state["bar_minutes"]::Int,
        state["last_bar_timestamp"]::DateTime - Hour(1), fire_time);
    current_prices = zeros(K);
    for (k, t) in enumerate(tickers)
        tb = bars[t];
        !isempty(tb) && (current_prices[k] = tb[end].c);
    end
    positions = Alpaca.list_positions(client);
    current_shares = zeros(Float64, K);
    for (k, t) in enumerate(tickers)
        for p in positions
            p.symbol == t && (current_shares[k] = Float64(p.qty); break);
        end
    end
    current_cash = Float64(Alpaca.get_account(client).cash);

    sentiment_signal = build(MySentimentSignal, (
        score = sentiment, source = "intraday-rolling",
        day = Dates.value(Date(fire_time) - Date(2026, 1, 1)),
    ));

    # News flags from latest news artifact.
    news = latest_news_artifact(fire_time, tickers);
    news_flags = String[t for t in tickers if news.severity[t] >= cfg.flag_severity_threshold];

    ticket = build_tomorrows_ticket(target_weights, tickers, current_shares, current_cash,
        current_prices, sentiment_signal, news_flags, eta, regime;
        generated_at = fire_time);

    # Save under tomorrow's date so the evening review notebook reads it.
    next_day = Date(fire_time) + Day(1);
    while dayofweek(next_day) > 5
        next_day += Day(1);
    end
    mkpath(TICKET_DIR);
    save_ticket!(ticket_path(next_day), ticket);
    log_entry("engine_close", "wrote ticket for $(next_day): " *
        "$(length(ticket.proposed_trades)) trades, η=$(round(eta, digits=2)), " *
        "regime=$(regime), news_flags=$(length(news_flags))");
    return state;
end

# ──────────────────────────────────────────────────────────────────────────────
# Mode: execute_signed_ticket
# ──────────────────────────────────────────────────────────────────────────────
"""
    run_execute_signed_ticket(client, _cfg, state, fire_time) -> state

The 9:35 ET next-morning fire. Loads the signed ticket written by the
desk PM (`data/tickets/signed-YYYY-MM-DD.jld2`) and submits each proposed
trade to Alpaca paper as a day market order, honoring per-trade
modifications: a modification with `modified_qty === nothing` rejects the
trade (skip), otherwise the order is submitted at the modified quantity.
Logs each rejection and each submission failure. Returns the state
unchanged (this mode does not touch persisted state).
"""
function run_execute_signed_ticket(client, _cfg, state, fire_time::DateTime)
    today = Date(fire_time);
    spath = signed_ticket_path(today);
    if !isfile(spath)
        log_entry("execute", "no signed ticket at $(spath); nothing to execute.");
        return state;
    end

    signed = load_signed_ticket(spath);
    ticket = signed.ticket;
    log_entry("execute", "loaded signed ticket: $(length(ticket.proposed_trades)) trades; " *
        "$(length(signed.modifications)) modifications signed by $(signed.signed_by) at $(signed.signed_at)");

    # Apply modifications: build a ticker -> modified_qty (or :reject) map.
    mods = Dict{String,Any}();
    for m in signed.modifications
        mods[m.ticker] = m.modified_qty === nothing ? :reject : m.modified_qty;
    end

    n_submitted = 0;
    for trade in ticket.proposed_trades
        qty = trade.qty;
        if haskey(mods, trade.ticker)
            v = mods[trade.ticker];
            if v === :reject
                log_entry("execute", "skipping $(trade.ticker) (rejected at sign-off).");
                continue;
            else
                qty = Int(v);
            end
        end
        side_str = trade.side == :buy ? "buy" : "sell";
        try
            Alpaca.submit_order(client, trade.ticker, qty, side_str;
                type = "market", time_in_force = "day");
            n_submitted += 1;
        catch err
            log_entry("execute", "submit_order failed for $(trade.ticker): $(err)");
        end
    end

    log_entry("execute", "submitted $(n_submitted) orders from signed ticket.");
    return state;
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
"""
    main()

Entry point for the cron. Parses the `--mode` flag, loads credentials and
config, opens the Alpaca paper client, and applies the market-hours gate
for `mode == "engine"` only (so an out-of-session 30-min fire skips with
a log line, while `engine_close` and `execute_signed_ticket` always run
even when Alpaca reports `is_open == false`). Dispatches on mode to
`run_engine_step` (intraday fire), `run_engine_close` (16:00 fire), or
`run_execute_signed_ticket` (9:35 fire). Errors on an unrecognized mode.
"""
function main()
    mode = parse_args();
    isfile(CREDS_PATH) || error("Credentials not found: $(CREDS_PATH). See credentials.toml.example.");

    cfg = load_config();
    client = Alpaca.load_client(CREDS_PATH);

    # Market-hours gate (skip silently outside session). engine_close runs
    # AT the 16:00 ET close where Alpaca already reports is_open=false, so
    # only the intraday engine fires gate on the clock; engine_close always
    # runs (its job is to write the EOD tape entry + tomorrow's ticket).
    clock = Alpaca.get_clock(client);
    fire_time = now(UTC);
    if mode == "engine" && !clock.is_open
        log_entry(mode, "market closed at $(fire_time); skipping.");
        return;
    end

    state = load_or_init_state(client, cfg);

    if mode == "engine"
        run_engine_step(client, cfg, state, fire_time; is_close = false);
    elseif mode == "engine_close"
        run_engine_close(client, cfg, state, fire_time);
    elseif mode == "execute_signed_ticket"
        run_execute_signed_ticket(client, cfg, state, fire_time);
    else
        error("Unknown mode: $(mode). Use --mode=engine|engine_close|execute_signed_ticket.");
    end
end

main();

#!/usr/bin/env julia
#
# daily_baseline.jl -- Offline counterfactual: run the S4 production engine
# at *daily* cadence over the same May 5–8 window the live 30-min engine
# already traded. Produces data/daily-baseline-tape.jld2 in the same schema
# as the 30-min tape so a downstream comparison can read both.
#
# Daily-cadence rules:
#   * EWLS, sentiment, lambda, eta all update on close-to-close growth (Δt=1/252).
#   * EMA windows interpreted as trading days (canonical short=5, long=21).
#   * Decision at day t close → execute Δshares at day t+1 *open* with 5 bps friction.
#   * Bootstrap matches deploy_initial_allocation.jl: S1 minvar weights × $100k
#     at the May 4 close, with 5 bps friction on the deploy.
#
# This script does NOT submit any orders. Reads the cached bars in
# data/daily-baseline-bars.jld2 (produced by pull_daily_bars.jl).
#
# Run from lectures/session-4:
#   julia --project=. scripts/daily_baseline.jl
#

using Pkg;
const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance, Dates, JLD2, Statistics, TOML

const BARS_PATH = joinpath(SESSION_DIR, "data", "daily-baseline-bars.jld2");
const S1_ALLOCATION_PATH = joinpath(SESSION_DIR, "..", "session-1", "data", "minvar-allocation.jld2");
const CONFIG_PATH = joinpath(SESSION_DIR, "config", "production-config.toml");
const OUT_PATH = joinpath(SESSION_DIR, "data", "daily-baseline-tape.jld2");

# Daily-cadence constants. EMA windows are canonical (5, 21) trading days,
# overriding the production-config values which were tuned for 30-min bars.
const DT_DAILY        = 1.0 / 252.0;
const N_SHORT_DAILY   = 5;
const N_LONG_DAILY    = 21;
const N_GROWTH_DAILY  = 10;
const COST_BPS        = 5.0;
const COST_RATE       = COST_BPS / 10_000.0;
const B0              = 100_000.0;
const CASH_BUFFER     = 0.95;          # matches deploy_initial_allocation.jl
const PRIOR_WEIGHT    = 63.0;
const HALF_LIFE_DAYS  = 63.0;

# Test-window dates (must exist in the bars cache).
const TEST_DATES = [Date(2026, 5, 4), Date(2026, 5, 5), Date(2026, 5, 6),
                    Date(2026, 5, 7), Date(2026, 5, 8)];

# Decision dates (close fires) and corresponding execution dates (next-day opens).
# Decision at TEST_DATES[i] close → execute at TEST_DATES[i+1] open.
# We *skip* a decision on May 4 close: both engines hold S1 bootstrap on May 5
# morning. First daily decision is May 5 close → execute May 6 open.
const DECISION_DATES = TEST_DATES[2:end-1];     # [May 5, May 6, May 7]
const EXECUTION_DATES = TEST_DATES[3:end];      # [May 6, May 7, May 8]

"""
    load_engine_config() -> NamedTuple

Read the production-engine config TOML and extract the four parameters the
daily-cadence baseline needs to mirror live behavior: `gain` (EMA-crossover
gain for the λ signal), `sentiment_threshold` and `sentiment_override_lambda`
(circuit breaker that scales λ when live sentiment falls below threshold),
and `max_drawdown` (cutover to cash when realized drawdown exceeds the gate).
"""
function load_engine_config()
    cfg = TOML.parsefile(CONFIG_PATH);
    return (
        gain                       = Float64(cfg["Engine"]["GAIN"]),
        sentiment_threshold        = Float64(cfg["CircuitBreakers"]["sentiment_threshold"]),
        sentiment_override_lambda  = Float64(cfg["CircuitBreakers"]["sentiment_override_lambda"]),
        max_drawdown               = Float64(cfg["CircuitBreakers"]["max_drawdown"]),
    );
end

"""
    load_bars() -> NamedTuple

Load the cached daily OHLC bars written by `pull_daily_bars.jl` and return a
NamedTuple of `dates::Vector{Date}`, `symbols::Vector{String}` (trading
universe plus SPY), `tickers::Vector{String}` (just the trading universe),
`open::Matrix{Float64}` and `close::Matrix{Float64}` (both shaped n_days x
n_symbols). Errors with a build-hint if the cache file does not exist.
"""
function load_bars()
    isfile(BARS_PATH) || error("Bars cache missing. Run scripts/pull_daily_bars.jl first.");
    d = load(BARS_PATH);
    return (
        dates   = Date.(d["dates"]),
        symbols = String.(d["symbols"]),
        tickers = String.(d["tickers"]),
        open    = Float64.(d["open"]),
        close   = Float64.(d["close"]),
    );
end

"""
    load_s1_priors() -> NamedTuple

Read the Session 1 minimum-variance allocation artifact and return the universe
tickers, target weights, and per-ticker SIM priors `(α, β, σ_ε)`. The SIM
priors seed the EWLS recursion in `main`; the weights bootstrap the May 4
close book in `bootstrap_book`. Asserts that the order of `sim_estimates`
matches `tickers` so the resulting `sim_priors` dict is keyed correctly.
"""
function load_s1_priors()
    s1 = load_results(S1_ALLOCATION_PATH);
    tickers = String.(s1["my_tickers"]);
    weights = Float64.(s1["allocation_weights"]);
    sim_estimates = s1["sim_estimates"];
    sim_priors = Dict{String,Tuple{Float64,Float64,Float64}}();
    for (i, est) in enumerate(sim_estimates)
        # SerializedVec reconstruction: tuple of (ticker, α, β, σ_ε, R²)
        ticker = String(est.ticker);
        α      = Float64(est.α);
        β      = Float64(est.β);
        σ_ε    = Float64(est.σ_ε);
        sim_priors[ticker] = (α, β, σ_ε);
        @assert ticker == tickers[i] "S1 sim_estimates order mismatch at $(i)";
    end
    return (tickers = tickers, weights = weights, sim_priors = sim_priors);
end

"""
    date_index(dates, d) -> Int

Return the row index of `d` in `dates`. Errors with the cache's first and
last date when the lookup misses, so the caller can fix the bars cache range
rather than chase a silent `nothing`.
"""
function date_index(dates::Vector{Date}, d::Date)
    idx = findfirst(==(d), dates);
    idx === nothing && error("Date $(d) not in bars cache. Cache spans $(dates[1]) → $(dates[end]).");
    return idx;
end

# --- Bootstrap: replicate deploy_initial_allocation.jl at May 4 close --------
"""
    bootstrap_book(close_row, weights, K) -> (shares, cash, deploy_trade_value)

Replicate the live deploy from `deploy_initial_allocation.jl` against the
May 4 daily-close prices: target the S1 minvar weights `weights` against a
deployable budget of `B0 * CASH_BUFFER` (= 95k against the \$100k book),
floor each per-ticker dollar target to integer shares at the close price,
debit cash, then apply 5 bps friction on the deployed notional. Returns the
integer share counts (as `Float64`), residual cash, and the total notional
traded during the deploy (used only for the deploy-day log line).
"""
function bootstrap_book(close_row::Vector{Float64}, weights::Vector{Float64}, K::Int)
    deployable = B0 * CASH_BUFFER;
    shares = zeros(Float64, K);
    cash = B0;
    deploy_trade_value = 0.0;
    for i in 1:K
        weights[i] > 1e-6 || continue;
        target_dollars = weights[i] * deployable;
        qty = floor(Int, target_dollars / close_row[i]);
        qty > 0 || continue;
        notional = qty * close_row[i];
        shares[i] = Float64(qty);
        cash -= notional;
        deploy_trade_value += notional;
    end
    cash -= COST_RATE * deploy_trade_value;
    return shares, cash, deploy_trade_value;
end

# --- One daily engine step at day t close, execute at day t+1 open ----------
"""
    daily_engine_step(ewls_states, tickers, close_history, spy_history,
                      exec_open, exec_close, shares, cash, peak_wealth, cfg)
        -> (shares, cash, peak_wealth, info::NamedTuple)

One daily engine fire: update EWLS state with the day-t close-to-close growth,
recompute sentiment / λ / regime from the daily-close history, mark wealth to
the day-t close, run the Cobb-Douglas allocator, then size `Δshares` against
the next-day open and execute with 5 bps friction.

Mirrors the production runner's `run_engine_step` but with daily-cadence
`Δt = 1/252` and EMA windows `(N_SHORT_DAILY, N_LONG_DAILY) = (5, 21)` trading
days instead of intraday bars. Triggers the drawdown circuit breaker (de-risk
to cash) when realized drawdown exceeds `cfg.max_drawdown`; in that branch
`info.regime == :derisk` and no allocator step runs.

The returned `info` NamedTuple carries the fields the tape consumes:
`target_weights, λ_eff, sentiment, drawdown, regime, eta, wealth, trade_value,
friction`.
"""
function daily_engine_step(
    ewls_states::Dict{String,eCornellAIFinance.MyEWLSState},
    tickers::Vector{String},
    close_history::Matrix{Float64},          # (n_days_seen, K)
    spy_history::Vector{Float64},
    exec_open::Vector{Float64},              # opens at execution day (day t+1)
    exec_close::Vector{Float64},             # closes at execution day (day t+1)
    shares::Vector{Float64},
    cash::Float64,
    peak_wealth::Float64,
    cfg::NamedTuple,
)
    K = length(tickers);

    # 1) EWLS update with day-t close-to-close growth.
    spy_t   = spy_history[end];
    spy_tm1 = spy_history[end - 1];
    gm_t = (1.0 / DT_DAILY) * log(spy_t / spy_tm1);
    for (k, ticker) in enumerate(tickers)
        p_t   = close_history[end,     k];
        p_tm1 = close_history[end - 1, k];
        gi_t = (1.0 / DT_DAILY) * log(p_t / p_tm1);
        ewls_update!(ewls_states[ticker], gi_t, gm_t);
    end

    # 2) Sentiment + EMAs from full daily-close history.
    sentiment = compute_live_sentiment(spy_history);
    ema_s = compute_ema(spy_history; window = N_SHORT_DAILY);
    ema_l = compute_ema(spy_history; window = N_LONG_DAILY);
    λ_series = compute_lambda(ema_s, ema_l; G = cfg.gain);
    λ_eff = isempty(λ_series) ? 0.0 : λ_series[end];
    if sentiment < cfg.sentiment_threshold
        λ_eff = λ_eff * cfg.sentiment_override_lambda;
    end

    # 3) Wealth at decision day close (mark-to-market at close[t]).
    close_row_t = close_history[end, :];
    wealth_at_close_t = sum(shares .* close_row_t) + cash;
    peak_wealth = max(peak_wealth, wealth_at_close_t);
    drawdown = peak_wealth > 0 ? (peak_wealth - wealth_at_close_t) / peak_wealth : 0.0;

    # 4) Drawdown circuit breaker — de-risk to cash if breached.
    if drawdown > cfg.max_drawdown
        derisk_trade_value = sum(shares .* close_row_t);
        cash = wealth_at_close_t - COST_RATE * derisk_trade_value;
        shares = zeros(Float64, K);
        return (shares, cash, peak_wealth,
                (target_weights = zeros(K), λ_eff = λ_eff, sentiment = sentiment,
                 drawdown = drawdown, regime = :derisk, eta = 0.0,
                 wealth = cash, trade_value = derisk_trade_value, friction = COST_RATE * derisk_trade_value));
    end

    # 5) Allocator (Cobb-Douglas, mirrors run_engine_step lines 416-431).
    sim_params_current = Dict{String,Tuple{Float64,Float64,Float64}}(
        t => (ewls_states[t].α, ewls_states[t].β, ewls_states[t].σ_ε) for t in tickers
    );
    gm_raw = compute_market_growth(spy_history; Δt = DT_DAILY);
    gm_smoothed = compute_ema(gm_raw; window = N_GROWTH_DAILY);
    gm_t_signal = isempty(gm_smoothed) ? 0.0 : gm_smoothed[end];
    γ = compute_preference_weights(sim_params_current, tickers, gm_t_signal, λ_eff);
    γ_sum = sum(γ);
    γ = γ_sum > 0 ? (γ ./ γ_sum) : (ones(K) ./ K);
    target_weights = γ;

    regime = classify_regime(λ_eff);
    eta = compute_adaptive_eta(λ_eff);

    # 6) Δshares sized at execution-day open price.
    target_dollar = target_weights .* wealth_at_close_t;
    target_shares_int = round.(Int, target_dollar ./ max.(exec_open, 1e-8));
    delta_shares = Float64.(target_shares_int) .- shares;

    # 7) Execute at exec-day open.
    realized_trade_value = sum(abs.(delta_shares) .* exec_open);
    cash -= sum(delta_shares .* exec_open);
    cash -= COST_RATE * realized_trade_value;
    shares = Float64.(target_shares_int);

    # 8) Roll holdings open → close on execution day.
    wealth_at_close_t1 = sum(shares .* exec_close) + cash;
    peak_wealth = max(peak_wealth, wealth_at_close_t1);

    return (shares, cash, peak_wealth,
            (target_weights = target_weights, λ_eff = λ_eff, sentiment = sentiment,
             drawdown = drawdown, regime = regime, eta = eta,
             wealth = wealth_at_close_t1, trade_value = realized_trade_value,
             friction = COST_RATE * realized_trade_value));
end

# --- Main --------------------------------------------------------------------
"""
    main()

Orchestrate the daily-cadence baseline run end-to-end. Load the engine
config, the cached bars, and the S1 priors; initialize per-ticker EWLS
states from the SIM priors and warm them up by feeding every daily growth
observation from row 2 through the day before May 4 (so May 4 itself is
the bootstrap day, not a decision day); bootstrap the book at the May 4
close; step the daily engine for May 5 / 6 / 7 close decisions, executed
at May 6 / 7 / 8 open respectively; and write the five-entry tape to
`data/daily-baseline-tape.jld2` in the same schema as the live 30-min
tape so `compare_daily_vs_intraday.jl` can read both.
"""
function main()
    cfg = load_engine_config();
    bars = load_bars();
    s1 = load_s1_priors();

    tickers = s1.tickers;
    K = length(tickers);

    # Subset bar matrices to ticker columns (drop SPY); keep SPY series separate.
    ticker_cols = [findfirst(==(t), bars.symbols) for t in tickers];
    @assert all(!isnothing, ticker_cols) "ticker not found in bars cache";
    spy_col = findfirst(==("SPY"), bars.symbols);

    open_t  = bars.open[:, ticker_cols];     # (n_days, K)
    close_t = bars.close[:, ticker_cols];    # (n_days, K)
    spy_close = bars.close[:, spy_col];      # (n_days,)

    # Locate test-window indices.
    idx_may4 = date_index(bars.dates, Date(2026, 5, 4));
    idx_may5 = date_index(bars.dates, Date(2026, 5, 5));

    # --- Initialize EWLS states from S1 SIM priors and warmup -----------------
    ewls_states = Dict{String,eCornellAIFinance.MyEWLSState}();
    for ticker in tickers
        (α₀, β₀, σ₀) = s1.sim_priors[ticker];
        ewls_states[ticker] = ewls_init(α₀, β₀, σ₀;
            half_life = HALF_LIFE_DAYS, prior_weight = PRIOR_WEIGHT);
    end

    # Warmup: feed daily g_i, g_m for t = 2..idx_may4-1 (everything strictly
    # before the May 4 close, since the May 4 EWLS update happens inside the
    # decision loop on May 5 below — but May 4 itself is bootstrap day, not
    # a decision day).
    println("Warmup: feeding $(idx_may4 - 2) daily growth observations into EWLS …");
    for t in 2:(idx_may4 - 1)
        gm = (1.0 / DT_DAILY) * log(spy_close[t] / spy_close[t - 1]);
        for (k, ticker) in enumerate(tickers)
            gi = (1.0 / DT_DAILY) * log(close_t[t, k] / close_t[t - 1, k]);
            ewls_update!(ewls_states[ticker], gi, gm);
        end
    end

    # --- Bootstrap at May 4 close --------------------------------------------
    close_may4 = close_t[idx_may4, :];
    shares, cash, deploy_value = bootstrap_book(close_may4, s1.weights, K);
    bootstrap_wealth = sum(shares .* close_may4) + cash;
    println("Bootstrap May 4 close: deployed \$$(round(deploy_value, digits=0)), residual cash=\$$(round(cash, digits=2)), wealth=\$$(round(bootstrap_wealth, digits=2))");

    peak_wealth = bootstrap_wealth;

    # --- Tape entries (5 wealth points: May 4, 5, 6, 7, 8 closes) ------------
    entries = NamedTuple[];
    push!(entries, (
        fire_time = DateTime(Date(2026, 5, 4), Time(16, 0, 0)),
        is_close = true,
        last_bar = DateTime(Date(2026, 5, 4), Time(16, 0, 0)),
        sentiment = 0.0,
        lambda_eff = 0.0,
        regime = :bootstrap,
        eta = 0.0,
        target_weights = copy(s1.weights),
        proposed_n = K,
        auto_n = K,
        queued_n = 0,
        submitted_ids = String[],
        wealth = bootstrap_wealth,
        drawdown = 0.0,
        news_path = "",
    ));

    # May 5 close: same bootstrap shares rolled by close-to-close growth, no
    # decision yet (first daily-engine decision is May 5 close → execute May 6).
    wealth_may5_close = sum(shares .* close_t[idx_may5, :]) + cash;
    peak_wealth = max(peak_wealth, wealth_may5_close);
    drawdown_may5 = peak_wealth > 0 ? (peak_wealth - wealth_may5_close) / peak_wealth : 0.0;
    push!(entries, (
        fire_time = DateTime(Date(2026, 5, 5), Time(16, 0, 0)),
        is_close = true,
        last_bar = DateTime(Date(2026, 5, 5), Time(16, 0, 0)),
        sentiment = 0.0,
        lambda_eff = 0.0,
        regime = :hold,
        eta = 0.0,
        target_weights = copy(s1.weights),
        proposed_n = 0,
        auto_n = 0,
        queued_n = 0,
        submitted_ids = String[],
        wealth = wealth_may5_close,
        drawdown = drawdown_may5,
        news_path = "",
    ));

    # --- Decision loop -------------------------------------------------------
    for (decision_date, exec_date) in zip(DECISION_DATES, EXECUTION_DATES)
        idx_dec = date_index(bars.dates, decision_date);
        idx_exec = date_index(bars.dates, exec_date);

        close_history = close_t[1:idx_dec, :];        # up to and including decision day close
        spy_history   = spy_close[1:idx_dec];

        exec_open  = open_t[idx_exec, :];
        exec_close = close_t[idx_exec, :];

        shares, cash, peak_wealth, info = daily_engine_step(
            ewls_states, tickers, close_history, spy_history,
            exec_open, exec_close, shares, cash, peak_wealth, cfg);

        wealth = info.wealth;
        drawdown = peak_wealth > 0 ? (peak_wealth - wealth) / peak_wealth : 0.0;
        push!(entries, (
            fire_time = DateTime(exec_date, Time(16, 0, 0)),
            is_close = true,
            last_bar = DateTime(exec_date, Time(16, 0, 0)),
            sentiment = info.sentiment,
            lambda_eff = info.λ_eff,
            regime = info.regime,
            eta = info.eta,
            target_weights = info.target_weights,
            proposed_n = K,
            auto_n = K,
            queued_n = 0,
            submitted_ids = String[],
            wealth = wealth,
            drawdown = drawdown,
            news_path = "",
        ));
        println("$(exec_date)  λ=$(round(info.λ_eff, digits=3))  sent=$(round(info.sentiment, digits=3))  η=$(round(info.eta, digits=2))  regime=$(info.regime)  trade=\$$(round(info.trade_value, digits=0))  friction=\$$(round(info.friction, digits=2))  wealth=\$$(round(wealth, digits=2))  dd=$(round(drawdown*100, digits=2))%");
    end

    save_results(OUT_PATH, Dict("entries" => entries));
    println("\nWrote $(OUT_PATH) ($(length(entries)) entries)");
    println("Bootstrap wealth (May 4 close): \$$(round(entries[1].wealth, digits=2))");
    println("Terminal  wealth (May 8 close): \$$(round(entries[end].wealth, digits=2))");
    total_return = (entries[end].wealth / entries[1].wealth - 1.0) * 100;
    println("Total return:                    $(round(total_return, digits=3))%");
end

main()

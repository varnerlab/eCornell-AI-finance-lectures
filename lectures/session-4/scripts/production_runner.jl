#!/usr/bin/env julia
#
# production_runner.jl — Automated daily production runner for the AI finance engine.
#
# Usage:
#   julia --project=../.. production_runner.jl --mode=production
#   julia --project=../.. production_runner.jl --mode=monitor
#
# Cron schedule (ET, weekdays only):
#   35 9  * * 1-5  ... --mode=production   # full pipeline + trade
#   0  12 * * 1-5  ... --mode=monitor      # safety check
#   0  14 * * 1-5  ... --mode=monitor      # safety check
#   50 15 * * 1-5  ... --mode=monitor      # end-of-day check
#

using Dates, Statistics, TOML

# --- Resolve paths relative to this script ---
const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
const CONFIG_DIR = joinpath(SESSION_DIR, "config");
const DATA_DIR = joinpath(SESSION_DIR, "data");

# Activate the session-4 environment -
using Pkg;
Pkg.activate(SESSION_DIR; io=devnull);

using eCornellAIFinance
using Alpaca

const STATE_PATH = joinpath(DATA_DIR, "production-state.jld2");
const LOG_PATH = joinpath(DATA_DIR, "production-log.txt");
const CREDS_PATH = joinpath(CONFIG_DIR, "credentials.toml");
const CONFIG_PATH = joinpath(CONFIG_DIR, "production-config.toml");

# ──────────────────────────────────────────────────────────────────────────────
# Parse command-line arguments
# ──────────────────────────────────────────────────────────────────────────────
function parse_args()
    mode = "production";
    for arg in ARGS
        if startswith(arg, "--mode=")
            mode = split(arg, "=")[2];
        end
    end
    return mode;
end

# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
function load_config()
    isfile(CONFIG_PATH) || error("Config not found: $(CONFIG_PATH)");
    cfg = TOML.parsefile(CONFIG_PATH);

    tickers = String.(cfg["Tickers"]["universe"]);
    eng = cfg["Engine"];
    risk = cfg["Risk"];

    return (
        tickers = tickers,
        B₀ = Float64(eng["B0"]),
        half_life = Float64(eng["half_life"]),
        offset = Int(eng["offset"]),
        N_short = Int(eng["N_short"]),
        N_long = Int(eng["N_long"]),
        GAIN = Float64(eng["GAIN"]),
        N_growth = Int(eng["N_growth"]),
        cost_bps = Float64(eng["cost_bps"]),
        epsilon = Float64(eng["epsilon"]),
        max_drawdown = Float64(risk["max_drawdown"]),
        max_turnover = Float64(risk["max_turnover"]),
        sentiment_threshold = Float64(risk["sentiment_threshold"]),
        sentiment_override_lambda = Float64(risk["sentiment_override_lambda"]),
        max_bandit_churn = Int(risk["max_bandit_churn"]),
        n_bandit_iters = Int(risk["n_bandit_iters"])
    );
end

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
function log_entry(mode::String, msg::String)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS");
    tag = uppercase(mode);
    line = "$(ts) [$(tag)] $(msg)";
    println(line);
    mkpath(DATA_DIR);
    open(LOG_PATH, "a") do f
        println(f, line);
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Initialize state on first run
# ──────────────────────────────────────────────────────────────────────────────
function initialize_state(client, cfg)
    tickers = cfg.tickers;
    K = length(tickers);
    Δt = 1.0 / 252.0;

    # load SIM params from calibration -
    cal = load_results(joinpath(eCornellAIFinance._PATH_TO_DATA, "sim-calibration.jld2"));
    all_tickers = cal["tickers"];
    tidx = Dict(all_tickers[i] => i for i in eachindex(all_tickers));
    for t in tickers
        haskey(tidx, t) || error("Ticker $(t) not in calibration set.");
    end
    sim_params = Dict{String,Tuple{Float64,Float64,Float64}}(
        t => (cal["alpha"][tidx[t]], cal["beta"][tidx[t]], cal["sigma_eps"][tidx[t]])
        for t in tickers
    );

    # fetch 1 year of daily bars -
    finish_date = Dates.format(today(), "yyyy-mm-dd");
    start_date = Dates.format(today() - Year(1), "yyyy-mm-dd");
    log_entry("init", "Fetching bars $(start_date) to $(finish_date) for $(K) tickers + SPY...");
    bars_dict = Alpaca.get_bars(client, vcat(tickers, ["SPY"]), "1Day";
        start = start_date, finish = finish_date);

    spy_bars = bars_dict["SPY"];
    n_days = length(spy_bars);
    market_prices = Float64[b.c for b in spy_bars];
    last_bar_date = Dates.format(spy_bars[end].t, "yyyy-mm-dd");

    price_matrix = zeros(n_days, K + 1);
    price_matrix[:, 1] = 1:n_days;
    for (k, t) in enumerate(tickers)
        tb = bars_dict[t];
        for i in 1:min(length(tb), n_days)
            price_matrix[i, k + 1] = tb[i].c;
        end
    end

    # initialize and warm up EWLS -
    ewls_states = Dict{String,MyEWLSState}(
        t => ewls_init(sim_params[t]...; half_life = cfg.half_life) for t in tickers
    );
    gm_raw = compute_market_growth(market_prices; Δt = Δt);
    for t_idx in 1:length(gm_raw)
        gm_t = gm_raw[t_idx];
        for (k, ticker) in enumerate(tickers)
            if t_idx + 1 <= n_days
                gi_t = (1.0 / Δt) * log(price_matrix[t_idx + 1, k + 1] / price_matrix[t_idx, k + 1]);
                ewls_update!(ewls_states[ticker], gi_t, gm_t);
            end
        end
    end

    # get current account equity as initial peak -
    acct = Alpaca.get_account(client);
    peak_wealth = max(cfg.B₀, acct.equity);

    state = Dict(
        "tickers" => tickers,
        "sim_params" => sim_params,
        "ewls_states" => ewls_states,
        "price_matrix" => price_matrix,
        "market_prices" => market_prices,
        "last_bar_date" => last_bar_date,
        "production_history" => MyLiveProductionDayResult[],
        "event_history" => MyEscalationEvent[],
        "peak_wealth" => peak_wealth,
        "prev_action" => ones(Int, K)
    );

    save_results(STATE_PATH, state);
    log_entry("init", "State initialized: $(n_days) bars, $(K) tickers, peak=\$$(round(peak_wealth, digits=2))");
    return state;
end

# ──────────────────────────────────────────────────────────────────────────────
# PRODUCTION MODE
# ──────────────────────────────────────────────────────────────────────────────
function run_production(client, cfg)
    Δt = 1.0 / 252.0;
    K = length(cfg.tickers);

    # load or initialize state -
    state = isfile(STATE_PATH) ? load_results(STATE_PATH) : initialize_state(client, cfg);

    tickers = state["tickers"];
    sim_params = state["sim_params"];
    ewls_states = state["ewls_states"];
    price_matrix = state["price_matrix"];
    market_prices = state["market_prices"];
    last_bar_date = state["last_bar_date"];
    history = state["production_history"];
    events = state["event_history"];
    peak_wealth = state["peak_wealth"];
    prev_action = state["prev_action"];

    # fetch new bars since last saved date -
    start_date = string(Date(last_bar_date) + Day(1));
    finish_date = Dates.format(today(), "yyyy-mm-dd");
    if start_date <= finish_date
        new_bars = Alpaca.get_bars(client, vcat(tickers, ["SPY"]), "1Day";
            start = start_date, finish = finish_date);

        n_new = length(new_bars["SPY"]);
        if n_new > 0
            # append new bars -
            new_spy = Float64[b.c for b in new_bars["SPY"]];
            market_prices = vcat(market_prices, new_spy);

            n_old = size(price_matrix, 1);
            new_rows = zeros(n_new, K + 1);
            for d in 1:n_new
                new_rows[d, 1] = n_old + d;
                for (k, t) in enumerate(tickers)
                    if d <= length(new_bars[t])
                        new_rows[d, k + 1] = new_bars[t][d].c;
                    end
                end
            end
            price_matrix = vcat(price_matrix, new_rows);
            last_bar_date = Dates.format(new_bars["SPY"][end].t, "yyyy-mm-dd");

            # update EWLS with new observations -
            gm_raw = compute_market_growth(market_prices; Δt = Δt);
            for d in (length(gm_raw) - n_new + 1):length(gm_raw)
                gm_t = gm_raw[d];
                for (k, ticker) in enumerate(tickers)
                    if d + 1 <= size(price_matrix, 1)
                        gi_t = (1.0 / Δt) * log(price_matrix[d + 1, k + 1] / price_matrix[d, k + 1]);
                        ewls_update!(ewls_states[ticker], gi_t, gm_t);
                    end
                end
            end
            log_entry("production", "Appended $(n_new) new bar(s) through $(last_bar_date)");
        end
    end

    # build production context -
    ctx = build(MyProductionContext, (
        tickers = tickers, sim_parameters = sim_params, B₀ = cfg.B₀,
        epsilon = cfg.epsilon, max_drawdown = cfg.max_drawdown,
        max_turnover = cfg.max_turnover, sentiment_threshold = cfg.sentiment_threshold,
        sentiment_override_lambda = cfg.sentiment_override_lambda,
        max_bandit_churn = cfg.max_bandit_churn
    ));

    # get current positions -
    positions = Alpaca.list_positions(client);
    current_shares = zeros(K);
    for (k, t) in enumerate(tickers)
        for p in positions
            if p.symbol == t
                current_shares[k] = p.qty;
                break;
            end
        end
    end
    acct = Alpaca.get_account(client);
    current_cash = acct.cash;

    # run production step -
    n_days = size(price_matrix, 1);
    (result, step_events) = run_production_step(ctx, ewls_states, price_matrix,
        market_prices, tickers, n_days;
        n_bandit_iters = cfg.n_bandit_iters, prev_action = prev_action,
        peak_wealth = peak_wealth, current_shares = current_shares,
        current_cash = current_cash, N_short = cfg.N_short, N_long = cfg.N_long,
        GAIN = cfg.GAIN, N_growth = cfg.N_growth);
    result.timestamp = string(now());

    # submit orders if rebalancing -
    n_orders = 0;
    if result.rebalanced && !any(e.severity == :critical for e in step_events)
        for (k, t) in enumerate(tickers)
            delta = result.shares[k] - current_shares[k];
            abs(delta) < 0.01 && continue;
            side = delta > 0 ? "buy" : "sell";
            qty = round(abs(delta), digits=2);
            order = Alpaca.submit_order(client, t, qty, side;
                type = "market", time_in_force = "day");
            push!(result.order_ids, order.id);
            n_orders += 1;
        end
        if n_orders > 0
            sleep(5.0);  # wait for fills
        end
    elseif any(e.severity == :critical for e in step_events)
        # critical: de-risk to cash -
        log_entry("production", "CRITICAL ESCALATION — de-risking to cash");
        Alpaca.close_all_positions(client; cancel_orders = true);
        n_orders = K;
    end

    # update peak wealth -
    acct = Alpaca.get_account(client);
    peak_wealth = max(peak_wealth, acct.equity);

    # save state -
    push!(history, result);
    append!(events, step_events);

    save_results(STATE_PATH, Dict(
        "tickers" => tickers,
        "sim_params" => sim_params,
        "ewls_states" => ewls_states,
        "price_matrix" => price_matrix,
        "market_prices" => market_prices,
        "last_bar_date" => last_bar_date,
        "production_history" => history,
        "event_history" => events,
        "peak_wealth" => peak_wealth,
        "prev_action" => result.bandit_action
    ));

    bandit_count = sum(result.bandit_action);
    log_entry("production",
        "equity=\$$(round(acct.equity, digits=2)) " *
        "sentiment=$(round(result.sentiment, digits=3)) " *
        "lambda=$(round(result.lambda, digits=3)) " *
        "bandit=$(bandit_count)/$(K) " *
        "rebalanced=$(result.rebalanced) " *
        "orders=$(n_orders) " *
        "escalated=$(result.escalated)");
end

# ──────────────────────────────────────────────────────────────────────────────
# MONITOR MODE
# ──────────────────────────────────────────────────────────────────────────────
function run_monitor(client, cfg)
    K = length(cfg.tickers);

    if !isfile(STATE_PATH)
        log_entry("monitor", "No state file — run production mode first.");
        return;
    end

    state = load_results(STATE_PATH);
    tickers = state["tickers"];
    peak_wealth = state["peak_wealth"];
    prev_action = state["prev_action"];
    market_prices = state["market_prices"];

    # fetch current account state -
    acct = Alpaca.get_account(client);
    equity = acct.equity;

    # compute intraday drawdown from peak -
    drawdown = peak_wealth > 0 ? (peak_wealth - equity) / peak_wealth : 0.0;

    # compute current sentiment from SPY -
    spy_q = Alpaca.get_latest_quote(client, "SPY");
    spy_price = (spy_q.ask_price + spy_q.bid_price) / 2.0;
    sentiment = compute_live_sentiment(vcat(market_prices, [spy_price]));

    # build context for trigger checking -
    ctx = build(MyProductionContext, (
        tickers = tickers,
        sim_parameters = state["sim_params"],
        B₀ = cfg.B₀,
        epsilon = cfg.epsilon,
        max_drawdown = cfg.max_drawdown,
        max_turnover = cfg.max_turnover,
        sentiment_threshold = cfg.sentiment_threshold,
        sentiment_override_lambda = cfg.sentiment_override_lambda,
        max_bandit_churn = cfg.max_bandit_churn
    ));

    # check escalation triggers -
    trigger_events = check_escalation_triggers(0, ctx, equity, peak_wealth,
        sentiment, prev_action, prev_action);

    has_critical = any(e.severity == :critical for e in trigger_events);
    trigger_status = isempty(trigger_events) ? "OK" : join([
        "$(e.trigger_type)=$(e.severity)" for e in trigger_events], ",");

    # if critical: de-risk immediately -
    if has_critical
        log_entry("monitor", "CRITICAL TRIGGER FIRED — de-risking to cash!");
        Alpaca.close_all_positions(client; cancel_orders = true);

        # update state -
        state["peak_wealth"] = max(peak_wealth, equity);
        append!(state["event_history"], trigger_events);
        save_results(STATE_PATH, state);
    end

    log_entry("monitor",
        "equity=\$$(round(equity, digits=2)) " *
        "drawdown=$(round(drawdown * 100, digits=1))% " *
        "sentiment=$(round(sentiment, digits=3)) " *
        "triggers=$(trigger_status)");
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
function main()
    mode = parse_args();
    isfile(CREDS_PATH) || error("Credentials not found: $(CREDS_PATH). See credentials.toml.example.");

    client = Alpaca.load_client(CREDS_PATH);
    cfg = load_config();

    # verify market is open -
    clock = Alpaca.get_clock(client);
    if !clock.is_open
        log_entry(mode, "Market is CLOSED. Skipping.");
        return;
    end

    if mode == "production"
        run_production(client, cfg);
    elseif mode == "monitor"
        run_monitor(client, cfg);
    else
        error("Unknown mode: $(mode). Use --mode=production or --mode=monitor");
    end
end

main();

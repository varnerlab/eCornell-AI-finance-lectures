# --- Import Session 2 types and functions ---
# We need the rebalancing engine from Session 2
const _PATH_TO_SESSION2 = joinpath(dirname(dirname(_ROOT)), "session-2", "L2");

# Include Session 2 types and compute (for the rebalancing engine) -
include(joinpath(_PATH_TO_SESSION2, "src", "Types.jl"));
include(joinpath(_PATH_TO_SESSION2, "src", "Factory.jl"));
include(joinpath(_PATH_TO_SESSION2, "src", "Compute.jl"));

"""
    generate_training_prices(; S₀, μ, σ, T, Δt, seed) -> Array{Float64,1}

Generate a synthetic "training" price path using GBM for fitting the HMM.
"""
function generate_training_prices(; S₀::Float64 = 100.0, μ::Float64 = 0.08,
    σ::Float64 = 0.18, T::Int = 1260, Δt::Float64 = 1.0/252.0,
    seed::Int = 42)::Array{Float64,1}

    Random.seed!(seed);
    prices = zeros(T);
    prices[1] = S₀;
    for t in 2:T
        z = randn();
        prices[t] = prices[t-1] * exp((μ - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * z);
    end

    return prices;
end

"""
    generate_hmm_scenario(model::JumpHiddenMarkovModel, tickers, sim_params;
        n_paths, n_steps, P₀_market, start_prices, rf, Δt, label) -> MyBacktestScenario

Generate a backtest scenario: use the HMM to generate market index paths,
then use SIM to generate per-ticker paths from each market path.
"""
function generate_hmm_scenario(model::JumpHiddenMarkovModel, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}};
    n_paths::Int = 100, n_steps::Int = 252, P₀_market::Float64 = 100.0,
    start_prices::Dict{String,Float64} = Dict("LargeCap" => 150.0, "SmallCap" => 45.0,
        "International" => 80.0, "Bond" => 100.0, "Commodity" => 60.0),
    rf::Float64 = 0.05, Δt::Float64 = 1.0/252.0,
    label::String = "HMM")::MyBacktestScenario

    K = length(tickers);

    # storage -
    market_paths = zeros(n_paths, n_steps);
    price_paths = zeros(n_paths, n_steps, K);

    for p in 1:n_paths

        # generate market path via HMM -
        result = hmm_simulate(model, n_steps; n_paths=1);
        G_market = result.paths[1].observations; # excess growth rates

        # convert to prices -
        mkt_prices = JumpHMM.prices_from_growth_rates(G_market, P₀_market; rf=rf, dt=Δt);
        market_paths[p, :] = mkt_prices;

        # compute market growth for SIM -
        gm = zeros(n_steps);
        for t in 2:n_steps
            gm[t] = (1.0 / Δt) * log(mkt_prices[t] / mkt_prices[t-1]);
        end

        # generate per-ticker paths via SIM -
        for (k, ticker) in enumerate(tickers)
            (αᵢ, βᵢ, σᵢ) = sim_params[ticker];
            price_paths[p, 1, k] = start_prices[ticker];
            for t in 2:n_steps
                gᵢ = αᵢ + βᵢ * gm[t] * Δt + σᵢ * sqrt(Δt) * randn();
                price_paths[p, t, k] = price_paths[p, t-1, k] * exp(gᵢ);
            end
        end
    end

    return build(MyBacktestScenario, (
        label = label,
        price_paths = price_paths,
        market_paths = market_paths,
        n_paths = n_paths,
        n_steps = n_steps
    ));
end

"""
    backtest_engine(scenario::MyBacktestScenario, tickers, sim_params, rules_params;
        B₀, offset, N_short, N_long, GAIN, N_growth) -> MyBacktestResult

Run the rebalancing engine from Session 2 across all paths in a scenario.
Returns aggregate performance statistics.
"""
function backtest_engine(scenario::MyBacktestScenario, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
    rules_params::NamedTuple;
    B₀::Float64 = 10000.0, offset::Int = 84,
    N_short::Int = 21, N_long::Int = 63,
    GAIN::Float64 = 10.0, N_growth::Int = 10)::MyBacktestResult

    Δt = 1.0 / 252.0;
    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);

    for p in 1:n_paths

        # extract this path's market prices -
        mkt = scenario.market_paths[p, :];

        # compute EMAs and lambda -
        ema_s = compute_ema(mkt; window=N_short);
        ema_l = compute_ema(mkt; window=N_long);
        λ = compute_lambda(ema_s, ema_l; G=GAIN);
        λ[1:offset] .= 0.0;

        # compute market growth EMA -
        gm_raw = compute_market_growth(mkt; Δt=Δt);
        gm_e = compute_ema(gm_raw; window=N_growth);

        # build price matrix (day_index, ticker_1, ..., ticker_K) -
        pmatrix = zeros(n_steps, K + 1);
        pmatrix[:, 1] = 1:n_steps;
        for k in 1:K
            pmatrix[:, k+1] = scenario.price_paths[p, :, k];
        end

        # build context -
        ctx = build(MyRebalancingContextModel, (
            B = B₀, tickers = tickers, marketdata = pmatrix,
            marketfactor = gm_e, sim_parameters = sim_params,
            lambda = 0.0, Δt = Δt, epsilon = 0.1
        ));

        # build rules -
        rules = build(MyTriggerRules, (
            max_drawdown = rules_params.max_drawdown,
            max_turnover = rules_params.max_turnover,
            rebalance_schedule = ones(Int, n_trading)
        ));

        # run engine -
        results = run_rebalancing_engine(ctx, rules, λ; offset=offset);
        wealth = compute_wealth_series(results, pmatrix, tickers; offset=offset);

        # metrics -
        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "AI Engine";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;

    return result;
end

"""
    backtest_buyhold(scenario::MyBacktestScenario, tickers;
        B₀, offset) -> MyBacktestResult

Run an equal-weight buy-and-hold strategy across all paths.
"""
function backtest_buyhold(scenario::MyBacktestScenario, tickers::Array{String,1};
    B₀::Float64 = 10000.0, offset::Int = 84)::MyBacktestResult

    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);

    for p in 1:n_paths

        # equal-weight buy at offset -
        budget_per = B₀ / K;
        shares = [budget_per / scenario.price_paths[p, offset, k] for k in 1:K];

        wealth = zeros(n_trading + 1);
        for d in 0:n_trading
            day = offset + d;
            wealth[d+1] = sum(shares[k] * scenario.price_paths[p, day, k] for k in 1:K);
        end

        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "Buy-and-Hold";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;

    return result;
end

"""
    backtest_ai_baseline(scenario::MyBacktestScenario, tickers;
        B₀, offset, lookback) -> MyBacktestResult

"Zero-Level AI Baseline" — a naive momentum strategy that mimics what a simple
AI advisor might suggest: rebalance monthly toward recent winners.
This provides a baseline for what "just asking AI" might produce.
"""
function backtest_ai_baseline(scenario::MyBacktestScenario, tickers::Array{String,1};
    B₀::Float64 = 10000.0, offset::Int = 84, lookback::Int = 21)::MyBacktestResult

    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);

    for p in 1:n_paths

        # start equal-weight -
        budget_per = B₀ / K;
        shares = [budget_per / scenario.price_paths[p, offset, k] for k in 1:K];
        cash = 0.0;

        wealth = zeros(n_trading + 1);
        wealth[1] = B₀;

        for d in 1:n_trading
            day = offset + d;

            # compute current wealth -
            total = cash;
            for k in 1:K
                total += shares[k] * scenario.price_paths[p, day, k];
            end
            wealth[d+1] = total;

            # monthly rebalance toward momentum -
            if d % 21 == 0 && day > lookback + 1
                # compute lookback returns -
                rets = zeros(K);
                for k in 1:K
                    rets[k] = scenario.price_paths[p, day, k] / scenario.price_paths[p, day - lookback, k] - 1.0;
                end

                # tilt toward winners: softmax-weighted rebalance -
                exp_rets = exp.(rets);
                weights = exp_rets ./ sum(exp_rets);

                # liquidate and re-allocate -
                for k in 1:K
                    shares[k] = weights[k] * total / scenario.price_paths[p, day, k];
                end
                cash = 0.0;
            end
        end

        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "AI Baseline (momentum)";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;

    return result;
end

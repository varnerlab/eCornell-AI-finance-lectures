# --- Session 1: Portfolio Optimization ------------------------------------------

"""
    solve_minvariance(problem::MyPortfolioAllocationProblem) -> MyPortfolioPerformanceResult

Solve the classical minimum-variance portfolio problem using JuMP.
Returns a `MyPortfolioPerformanceResult` with the optimal weights and portfolio statistics.
"""
function solve_minvariance(problem::MyPortfolioAllocationProblem)::MyPortfolioPerformanceResult

    # unpack -
    μ = problem.μ;
    Σ = problem.Σ;
    bounds = problem.bounds;
    R = problem.R;
    N = length(μ);

    # setup the JuMP model -
    model = Model();

    # decision variables: portfolio weights
    @variable(model, bounds[i,1] <= w[i=1:N] <= bounds[i,2]);

    # constraints -
    @constraint(model, sum(w) == 1.0);                  # fully invested
    @constraint(model, dot(μ, w) >= R);                  # target return

    # objective: minimize portfolio variance -
    @objective(model, Min, dot(w, Σ * w));

    # solve -
    optimize!(model);

    # package results -
    result = MyPortfolioPerformanceResult();
    result.weights = value.(w);
    result.expected_return = dot(μ, result.weights);
    result.variance = dot(result.weights, Σ * result.weights);

    # return -
    return result;
end

# --- Shared Portfolio Metrics ---------------------------------------------------

"""
    compute_drawdown(returns::Array{Float64,1}) -> Float64

Compute the maximum drawdown from a return series.
"""
function compute_drawdown(returns::Array{Float64,1})::Float64

    # compute cumulative wealth -
    wealth = cumprod(1.0 .+ returns);
    peak = accumulate(max, wealth);
    drawdowns = (peak .- wealth) ./ peak;

    # return max drawdown -
    return maximum(drawdowns);
end

"""
    compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1}) -> Float64

Compute portfolio turnover between two weight vectors.
"""
function compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1})::Float64
    return sum(abs.(w_new .- w_old));
end

# --- Session 2: Signal Computation ----------------------------------------------

"""
    compute_ema(prices::Array{Float64,1}; window::Int = 21) -> Array{Float64,1}

Compute the exponential moving average of a price series.
Uses smoothing factor alpha = 2/(window + 1).
"""
function compute_ema(prices::Array{Float64,1}; window::Int = 21)::Array{Float64,1}

    # setup -
    T = length(prices);
    ema = zeros(T);
    α = 2.0 / (window + 1.0);

    # initialize with first price -
    ema[1] = prices[1];

    # compute -
    for t in 2:T
        ema[t] = α * prices[t] + (1 - α) * ema[t - 1];
    end

    # return -
    return ema;
end

"""
    compute_lambda(short_ema::Array{Float64,1}, long_ema::Array{Float64,1};
        G::Float64 = 10.0) -> Array{Float64,1}

Compute the sentiment/risk-aversion signal lambda_t from short and long EMA crossover.

    lambda_t = -G * (EMA_short_t / EMA_long_t - 1)

Positive lambda -> bearish (short < long), negative lambda -> bullish (short > long).
"""
function compute_lambda(short_ema::Array{Float64,1}, long_ema::Array{Float64,1};
    G::Float64 = 10.0)::Array{Float64,1}

    # setup -
    T = length(short_ema);
    λ = zeros(T);

    # compute -
    for t in 1:T
        λ[t] = -G * (short_ema[t] / long_ema[t] - 1.0);
    end

    # return -
    return λ;
end

"""
    compute_market_growth(prices::Array{Float64,1}; Δt::Float64 = 1.0/252.0) -> Array{Float64,1}

Compute the log growth rate of a market price series: g_m = (1/Δt) * log(S_t/S_{t-1}).
Returns an array of length T-1.
"""
function compute_market_growth(prices::Array{Float64,1}; Δt::Float64 = 1.0/252.0)::Array{Float64,1}

    # setup -
    T = length(prices);
    gm = zeros(T - 1);

    # compute -
    for t in 2:T
        gm[t - 1] = (1.0 / Δt) * log(prices[t] / prices[t - 1]);
    end

    # return -
    return gm;
end

"""
    compute_preference_weights(sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}},
        tickers::Array{String,1}, gm_t::Float64, lambda::Float64) -> Array{Float64,1}

Compute Cobb-Douglas preference exponents gamma_i from SIM parameters and sentiment.

    gamma_i = tanh(alpha_i / beta_i^lambda + beta_i^(1-lambda) * E[g_m])

where alpha_i is the firm-specific intercept (Jensen's alpha), beta_i is the market sensitivity,
and lambda is the sentiment/risk-aversion parameter.
"""
function compute_preference_weights(sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}},
    tickers::Array{String,1}, gm_t::Float64, lambda::Float64)::Array{Float64,1}

    # setup -
    K = length(tickers);
    gamma = zeros(K);

    # compute preference weights -
    for i in 1:K
        (αᵢ, βᵢ, _) = sim_parameters[tickers[i]];
        RF = max(abs(βᵢ)^lambda, 1e-8); # risk factor, clamped
        g_hat = αᵢ / RF + (βᵢ / RF) * gm_t;
        gamma[i] = tanh(g_hat);
    end

    # return -
    return gamma;
end

# --- Session 2: Utility-Based Allocation Functions ------------------------------

"""
    allocate_cobb_douglas(problem::MyCobbDouglasChoiceProblem) -> (shares::Array{Float64,1}, cash::Float64)

Solve the Cobb-Douglas utility maximization problem analytically.

    maximize   kappa(gamma) * prod(n_i^gamma_i)
    subject to B = sum(n_i * p_i)

Analytical solution for preferred assets (gamma_i > 0):

    n_i* = (gamma_i / sum_{j in S+} gamma_j) * (B_adj / p_i)

Non-preferred assets (gamma_i <= 0) receive epsilon shares.
"""
function allocate_cobb_douglas(problem::MyCobbDouglasChoiceProblem)::Tuple{Array{Float64,1},Float64}

    # unpack -
    gamma = problem.gamma;
    prices = problem.prices;
    B = problem.B;
    ε = problem.epsilon;
    K = length(gamma);

    # separate preferred (gamma > 0) and non-preferred (gamma <= 0) -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate budget -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to minimum epsilon shares -
    for i in non_preferred
        shares[i] = ε;
        remaining_B -= ε * prices[i];
    end

    # preferred: proportional allocation from analytical solution -
    cash = 0.0;
    if length(preferred) > 0 && remaining_B > 0
        gamma_bar = sum(gamma[preferred]);
        for i in preferred
            shares[i] = (gamma[i] / gamma_bar) * (remaining_B / prices[i]);
        end
    else
        cash = remaining_B;
    end

    # return -
    return (shares, cash);
end

"""
    allocate_ces(problem::MyCESChoiceProblem) -> (shares::Array{Float64,1}, cash::Float64)

Solve the CES utility maximization problem analytically.

    maximize   (sum gamma_i * n_i^rho)^(1/rho)
    subject to B = sum(n_i * p_i)

where rho = (sigma - 1)/sigma. Analytical solution for preferred assets:

    n_i* = (gamma_i^sigma / p_i^sigma) / sum_{j in S+}(gamma_j^sigma / p_j^(sigma-1)) * B_adj

As sigma -> 1, this converges to Cobb-Douglas. As sigma -> inf, converges to linear (all-in on best).
"""
function allocate_ces(problem::MyCESChoiceProblem)::Tuple{Array{Float64,1},Float64}

    # unpack -
    gamma = problem.gamma;
    prices = problem.prices;
    B = problem.B;
    ε = problem.epsilon;
    σ = problem.sigma;
    K = length(gamma);

    # separate preferred and non-preferred -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to epsilon -
    for i in non_preferred
        shares[i] = ε;
        remaining_B -= ε * prices[i];
    end

    # preferred: CES analytical solution -
    cash = 0.0;
    if length(preferred) > 0 && remaining_B > 0
        # weight_i = gamma_i^sigma / p_i^sigma (demand function from CES)
        weights = zeros(length(preferred));
        for (j, i) in enumerate(preferred)
            weights[j] = (gamma[i]^σ) / (prices[i]^σ);
        end
        denom = sum((gamma[preferred[j]]^σ) / (prices[preferred[j]]^(σ - 1)) for j in 1:length(preferred));
        for (j, i) in enumerate(preferred)
            shares[i] = (weights[j] / denom) * remaining_B;
        end
    else
        cash = remaining_B;
    end

    # return -
    return (shares, cash);
end

"""
    allocate_log_linear(problem::MyLogLinearChoiceProblem) -> (shares::Array{Float64,1}, cash::Float64)

Solve the log-linear utility maximization problem analytically.

    maximize   sum gamma_i * log(n_i)
    subject to B = sum(n_i * p_i)

Analytical solution for preferred assets (gamma_i > 0):

    n_i* = (gamma_i / sum_{j in S+} gamma_j) * (B_adj / p_i)

Note: This yields the same allocation as Cobb-Douglas — the log transform preserves
the optimum. The utility *values* differ, which matters for bandit reward signals.
"""
function allocate_log_linear(problem::MyLogLinearChoiceProblem)::Tuple{Array{Float64,1},Float64}

    # unpack -
    gamma = problem.gamma;
    prices = problem.prices;
    B = problem.B;
    ε = problem.epsilon;
    K = length(gamma);

    # separate preferred and non-preferred -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to epsilon -
    for i in non_preferred
        shares[i] = ε;
        remaining_B -= ε * prices[i];
    end

    # preferred: proportional (same allocation as Cobb-Douglas) -
    cash = 0.0;
    if length(preferred) > 0 && remaining_B > 0
        gamma_bar = sum(gamma[preferred]);
        for i in preferred
            shares[i] = (gamma[i] / gamma_bar) * (remaining_B / prices[i]);
        end
    else
        cash = remaining_B;
    end

    # return -
    return (shares, cash);
end

# --- Session 2: Utility Evaluation Functions ------------------------------------

"""
    evaluate_cobb_douglas(shares::Array{Float64,1}, gamma::Array{Float64,1}) -> Float64

Compute Cobb-Douglas utility value: U = kappa * prod(n_i^gamma_i).
kappa = +1 if all gamma_i >= 0, else -1.
"""
function evaluate_cobb_douglas(shares::Array{Float64,1}, gamma::Array{Float64,1})::Float64

    κ = any(gamma .< 0) ? -1.0 : 1.0;
    U = κ;
    for i in 1:length(shares)
        if shares[i] > 0
            U *= shares[i]^gamma[i];
        end
    end

    return U;
end

"""
    evaluate_ces(shares::Array{Float64,1}, gamma::Array{Float64,1}; sigma::Float64 = 2.0) -> Float64

Compute CES utility value: U = (sum gamma_i * n_i^rho)^(1/rho) where rho = (sigma-1)/sigma.
"""
function evaluate_ces(shares::Array{Float64,1}, gamma::Array{Float64,1}; sigma::Float64 = 2.0)::Float64

    ρ = (sigma - 1.0) / sigma;
    preferred = findall(gamma .> 0);
    if length(preferred) == 0
        return 0.0;
    end
    inner = sum(gamma[i] * max(shares[i], 1e-8)^ρ for i in preferred);
    return inner > 0 ? inner^(1.0 / ρ) : 0.0;
end

"""
    evaluate_log_linear(shares::Array{Float64,1}, gamma::Array{Float64,1}) -> Float64

Compute log-linear utility value: U = sum gamma_i * log(n_i).
"""
function evaluate_log_linear(shares::Array{Float64,1}, gamma::Array{Float64,1})::Float64

    preferred = findall(gamma .> 0);
    if length(preferred) == 0
        return 0.0;
    end
    return sum(gamma[i] * log(max(shares[i], 1e-8)) for i in preferred);
end

# --- Session 2: Rebalancing Engine ----------------------------------------------

"""
    allocate_shares(t::Int, context::MyRebalancingContextModel;
        allocator::Symbol = :cobb_douglas) -> MyRebalancingResult

Allocate shares at time step t using the specified utility function.
This is the bridge between the utility-based allocators and the rebalancing engine.

### Allocator options
- `:cobb_douglas` — Cobb-Douglas utility (default)
- `:ces` — CES utility with sigma = 2.0
- `:log_linear` — Log-linear utility
"""
function allocate_shares(t::Int, context::MyRebalancingContextModel;
    allocator::Symbol = :cobb_douglas)::MyRebalancingResult

    # unpack -
    B = context.B;
    tickers = context.tickers;
    marketdata = context.marketdata;
    gm = context.marketfactor;
    sim_params = context.sim_parameters;
    λ = context.lambda;
    ε = context.epsilon;
    K = length(tickers);

    # compute preference weights -
    gm_t = gm[min(t, length(gm))];
    gamma = compute_preference_weights(sim_params, tickers, gm_t, λ);

    # get current prices -
    prices = [marketdata[t, i + 1] for i in 1:K];

    # dispatch to the chosen allocator -
    if allocator == :cobb_douglas
        problem = MyCobbDouglasChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices;
        problem.B = B;
        problem.epsilon = ε;
        (shares, cash) = allocate_cobb_douglas(problem);
    elseif allocator == :ces
        problem = MyCESChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices;
        problem.B = B;
        problem.epsilon = ε;
        problem.sigma = 2.0;
        (shares, cash) = allocate_ces(problem);
    elseif allocator == :log_linear
        problem = MyLogLinearChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices;
        problem.B = B;
        problem.epsilon = ε;
        (shares, cash) = allocate_log_linear(problem);
    else
        error("Unknown allocator: $allocator. Use :cobb_douglas, :ces, or :log_linear.");
    end

    # package result -
    result = MyRebalancingResult();
    result.shares = shares;
    result.cash = cash;
    result.gamma = gamma;

    # return -
    return result;
end

"""
    run_rebalancing_engine(context::MyRebalancingContextModel, rules::MyTriggerRules,
        lambda_series::Array{Float64,1}; offset::Int = 84,
        allocator::Symbol = :cobb_douglas) -> Dict{Int, MyRebalancingResult}

Run the AI rebalancing engine over the trading period using the specified utility allocator.

Daily loop:
1. Check rebalance schedule and trigger rules
2. If rebalance: liquidate, update budget and lambda, re-allocate via utility maximization
3. If hold: propagate prior positions
4. If drawdown exceeded: de-risk to cash
"""
function run_rebalancing_engine(context::MyRebalancingContextModel, rules::MyTriggerRules,
    lambda_series::Array{Float64,1}; offset::Int = 84,
    allocator::Symbol = :cobb_douglas)::Dict{Int,MyRebalancingResult}

    # setup -
    tickers = context.tickers;
    marketdata = context.marketdata;
    K = length(tickers);
    schedule = rules.rebalance_schedule;
    T = length(schedule);

    # results storage -
    results = Dict{Int,MyRebalancingResult}();

    # initial allocation at offset -
    results[0] = allocate_shares(offset, context; allocator = allocator);

    # track peak wealth for drawdown -
    peak_wealth = context.B;

    # daily loop -
    for day in 1:T

        actual_day = offset + day;
        action = schedule[day];

        if action == 1
            # liquidate current position -
            prev = results[day - 1];
            liquidation_value = prev.cash;
            for i in 1:K
                price = marketdata[actual_day, i + 1];
                liquidation_value += prev.shares[i] * price;
            end

            # check drawdown trigger -
            peak_wealth = max(peak_wealth, liquidation_value);
            drawdown = (peak_wealth - liquidation_value) / peak_wealth;

            if drawdown > rules.max_drawdown
                # de-risk: move to cash -
                derisk_result = MyRebalancingResult();
                derisk_result.shares = zeros(K);
                derisk_result.cash = liquidation_value;
                derisk_result.gamma = zeros(K);
                results[day] = derisk_result;
            else
                # rebalance via utility maximization -
                ctx = deepcopy(context);
                ctx.B = liquidation_value;
                ctx.lambda = lambda_series[min(actual_day, length(lambda_series))];

                new_result = allocate_shares(actual_day, ctx; allocator = allocator);

                # check turnover cap -
                if day > 0 && haskey(results, day - 1)
                    old_shares = results[day - 1].shares;
                    new_shares = new_result.shares;

                    # compute turnover as fraction of portfolio -
                    old_value = sum(old_shares[i] * marketdata[actual_day, i + 1] for i in 1:K);
                    trade_value = sum(abs(new_shares[i] - old_shares[i]) * marketdata[actual_day, i + 1] for i in 1:K);
                    turnover_frac = old_value > 0 ? trade_value / old_value : 0.0;

                    if turnover_frac > rules.max_turnover
                        # cap: scale trades down -
                        scale = rules.max_turnover / turnover_frac;
                        for i in 1:K
                            new_result.shares[i] = old_shares[i] + scale * (new_shares[i] - old_shares[i]);
                        end
                    end
                end

                results[day] = new_result;
            end
        else
            # hold: propagate prior -
            results[day] = results[day - 1];
        end
    end

    # return -
    return results;
end

"""
    compute_wealth_series(results::Dict{Int, MyRebalancingResult},
        marketdata::Array{Float64,2}, tickers::Array{String,1};
        offset::Int = 84) -> Array{Float64,1}

Compute the total wealth at each time step from a rebalancing results dictionary.
"""
function compute_wealth_series(results::Dict{Int,MyRebalancingResult},
    marketdata::Array{Float64,2}, tickers::Array{String,1};
    offset::Int = 84)::Array{Float64,1}

    # setup -
    T = maximum(keys(results));
    K = length(tickers);
    wealth = zeros(T + 1);

    # compute -
    for day in 0:T
        actual_day = offset + day;
        r = results[day];
        total = r.cash;
        for i in 1:K
            total += r.shares[i] * marketdata[actual_day, i + 1];
        end
        wealth[day + 1] = total;
    end

    # return -
    return wealth;
end

# --- Session 3: Training Data Generation ----------------------------------------

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

# --- Session 3: Scenario Generation ---------------------------------------------

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

# --- Session 3: Backtesting Functions -------------------------------------------

"""
    backtest_engine(scenario::MyBacktestScenario, tickers, sim_params, rules_params;
        B₀, offset, N_short, N_long, GAIN, N_growth) -> MyBacktestResult

Run the Cobb-Douglas rebalancing engine from Session 2 across all paths in a scenario.
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

        # run engine with Cobb-Douglas allocator -
        results = run_rebalancing_engine(ctx, rules, λ; offset=offset, allocator=:cobb_douglas);
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
    result.strategy_label = "Cobb-Douglas Engine";
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

# --- Session 3: Bandit Functions ------------------------------------------------

"""
    bandit_world(action::Array{Int,1}, context::MyBanditContext) -> (utility::Float64, shares::Array{Float64,1}, gamma::Array{Float64,1})

The "world" function for the combinatorial bandit. Given a binary action vector
(1 = include asset, 0 = exclude), computes the Cobb-Douglas utility-maximizing
allocation over the included assets.

Returns the Cobb-Douglas utility value as the reward signal, plus the share allocation
and preference weights for inspection.
"""
function bandit_world(action::Array{Int,1}, context::MyBanditContext)::Tuple{Float64,Array{Float64,1},Array{Float64,1}}

    K = length(context.tickers);

    # compute full preference weights -
    gamma = compute_preference_weights(context.sim_parameters, context.tickers,
        context.gm_t, context.lambda);

    # mask out excluded assets: set gamma to negative for excluded -
    for i in 1:K
        if action[i] == 0
            gamma[i] = -abs(gamma[i]) - 0.01; # force non-preferred
        end
    end

    # solve Cobb-Douglas allocation -
    problem = MyCobbDouglasChoiceProblem();
    problem.gamma = gamma;
    problem.prices = context.prices;
    problem.B = context.B;
    problem.epsilon = context.epsilon;
    (shares, _) = allocate_cobb_douglas(problem);

    # compute utility as reward -
    utility = evaluate_cobb_douglas(shares, gamma);

    return (utility, shares, gamma);
end

"""
    solve_bandit(bandit::MyEpsilonGreedyBanditModel, context::MyBanditContext) -> MyBanditResult

Run the epsilon-greedy combinatorial bandit to find the best asset subset.

Each arm is a binary vector of length K (which assets to include).
There are 2^K - 1 possible arms (excluding the empty set).

Exploration decays as: epsilon_t = t^(-1/3) * (K * log(t))^(1/3)
"""
function solve_bandit(bandit::MyEpsilonGreedyBanditModel, context::MyBanditContext)::MyBanditResult

    K = bandit.K;
    T = bandit.n_iterations;
    n_arms = 2^K - 1; # exclude empty set

    # encode arms as binary vectors -
    function arm_to_action(arm_idx::Int)::Array{Int,1}
        return [((arm_idx >> (i-1)) & 1) for i in 1:K];
    end

    # initialize reward estimates -
    mu = zeros(n_arms);      # average reward per arm
    counts = zeros(Int, n_arms); # how many times each arm was pulled

    # storage -
    reward_history = zeros(T);
    exploration_history = zeros(T);

    for t in 1:T

        # decaying exploration rate -
        ε_t = t > 1 ? t^(-1.0/3.0) * (K * log(t))^(1.0/3.0) : 1.0;
        ε_t = clamp(ε_t, 0.0, 1.0);
        exploration_history[t] = ε_t;

        # epsilon-greedy selection -
        if rand() < ε_t
            # explore: random arm -
            arm = rand(1:n_arms);
        else
            # exploit: best arm so far -
            arm = argmax(mu);
        end

        # pull arm: run the world function -
        action = arm_to_action(arm);
        (utility, _, _) = bandit_world(action, context);

        # update reward estimate (weighted online average) -
        counts[arm] += 1;
        lr = bandit.alpha > 0 ? bandit.alpha : 1.0 / counts[arm];
        mu[arm] += lr * (utility - mu[arm]);

        reward_history[t] = utility;
    end

    # best arm at convergence -
    best_arm = argmax(mu);
    best_action = arm_to_action(best_arm);
    (best_utility, _, _) = bandit_world(best_action, context);

    # package result -
    result = MyBanditResult();
    result.best_action = best_action;
    result.best_utility = best_utility;
    result.reward_history = reward_history;
    result.exploration_history = exploration_history;
    result.arm_means = mu;

    return result;
end

"""
    compute_regret(reward_history::Array{Float64,1}) -> Array{Float64,1}

Compute cumulative regret: the difference between the best-in-hindsight
reward and the actual reward at each step.
"""
function compute_regret(reward_history::Array{Float64,1})::Array{Float64,1}

    best = maximum(reward_history);
    regret = cumsum(best .- reward_history);
    return regret;
end

"""
    backtest_bandit(scenario::MyBacktestScenario, tickers, sim_params;
        B₀, offset, N_short, N_long, GAIN, N_growth, n_bandit_iters) -> MyBacktestResult

Run the bandit portfolio selector across all paths in a scenario.
On each path: run the bandit to select the best asset subset, then run
a simple rebalancing strategy using only the selected assets with Cobb-Douglas allocation.
"""
function backtest_bandit(scenario::MyBacktestScenario, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}};
    B₀::Float64 = 10000.0, offset::Int = 84,
    N_short::Int = 21, N_long::Int = 63,
    GAIN::Float64 = 10.0, N_growth::Int = 10,
    n_bandit_iters::Int = 200)::MyBacktestResult

    Δt = 1.0 / 252.0;
    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);

    for p in 1:n_paths

        # extract this path -
        mkt = scenario.market_paths[p, :];

        # compute EMAs and lambda -
        ema_s = compute_ema(mkt; window=N_short);
        ema_l = compute_ema(mkt; window=N_long);
        λ = compute_lambda(ema_s, ema_l; G=GAIN);
        λ[1:offset] .= 0.0;

        # compute market growth EMA -
        gm_raw = compute_market_growth(mkt; Δt=Δt);
        gm_e = compute_ema(gm_raw; window=N_growth);

        # run bandit at the start to select asset subset -
        prices_at_offset = [scenario.price_paths[p, offset, k] for k in 1:K];
        bandit_ctx = build(MyBanditContext, (
            tickers = tickers, sim_parameters = sim_params,
            prices = prices_at_offset, B = B₀,
            gm_t = gm_e[min(offset, length(gm_e))],
            lambda = λ[offset], epsilon = 0.1
        ));

        bandit_model = build(MyEpsilonGreedyBanditModel, (
            K = K, n_iterations = n_bandit_iters, alpha = 0.1
        ));

        bandit_result = solve_bandit(bandit_model, bandit_ctx);
        selected = bandit_result.best_action;

        # build price matrix -
        pmatrix = zeros(n_steps, K + 1);
        pmatrix[:, 1] = 1:n_steps;
        for k in 1:K
            pmatrix[:, k+1] = scenario.price_paths[p, :, k];
        end

        # run Cobb-Douglas engine with bandit-selected assets -
        # (non-selected assets get forced to epsilon via modified SIM params)
        modified_sim = copy(sim_params);
        for (k, ticker) in enumerate(tickers)
            if selected[k] == 0
                # force non-preferred by setting alpha very negative -
                (_, βᵢ, σᵢ) = sim_params[ticker];
                modified_sim[ticker] = (-10.0, βᵢ, σᵢ);
            end
        end

        ctx = build(MyRebalancingContextModel, (
            B = B₀, tickers = tickers, marketdata = pmatrix,
            marketfactor = gm_e, sim_parameters = modified_sim,
            lambda = 0.0, Δt = Δt, epsilon = 0.1
        ));

        rules = build(MyTriggerRules, (
            max_drawdown = 0.15, max_turnover = 0.50,
            rebalance_schedule = ones(Int, n_trading)
        ));

        results = run_rebalancing_engine(ctx, rules, λ; offset=offset, allocator=:cobb_douglas);
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
    result.strategy_label = "Bandit Selector";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;

    return result;
end

# --- Session 4: Sentiment Generation -------------------------------------------

"""
    generate_synthetic_sentiment(market_prices::Array{Float64,1};
        noise_σ::Float64 = 0.15, smoothing::Int = 5,
        seed::Int = -1) -> Array{Float64,1}

Generate synthetic sentiment scores correlated with market regime.
Sentiment is derived from short-term market returns + noise, smoothed and
mapped to [-1, 1] via tanh.

Bearish markets produce negative sentiment; bullish produce positive.
"""
function generate_synthetic_sentiment(market_prices::Array{Float64,1};
    noise_σ::Float64 = 0.15, smoothing::Int = 5,
    seed::Int = -1)::Array{Float64,1}

    if seed > 0
        Random.seed!(seed);
    end

    T = length(market_prices);
    sentiment = zeros(T);

    # compute 5-day return as base signal -
    for t in 6:T
        ret_5d = (market_prices[t] / market_prices[t-5] - 1.0);
        raw = ret_5d * 10.0 + noise_σ * randn(); # scale + noise
        sentiment[t] = tanh(raw);
    end

    # smooth with simple moving average -
    smoothed = copy(sentiment);
    for t in (smoothing+1):T
        smoothed[t] = mean(sentiment[(t-smoothing+1):t]);
    end

    return smoothed;
end

# --- Session 4: Escalation Checking --------------------------------------------

"""
    check_escalation_triggers(day::Int, ctx::MyProductionContext,
        wealth::Float64, peak_wealth::Float64, sentiment::Float64,
        current_action::Array{Int,1}, prev_action::Array{Int,1}) -> Array{MyEscalationEvent,1}

Check all escalation triggers and return any events that fired.

Triggers:
1. Drawdown exceeds threshold → critical
2. Sentiment crash (below threshold) → warning
3. Bandit churn (too many assets changed) → warning
"""
function check_escalation_triggers(day::Int, ctx::MyProductionContext,
    wealth::Float64, peak_wealth::Float64, sentiment::Float64,
    current_action::Array{Int,1}, prev_action::Array{Int,1})::Array{MyEscalationEvent,1}

    events = MyEscalationEvent[];

    # 1. Drawdown trigger -
    drawdown = peak_wealth > 0 ? (peak_wealth - wealth) / peak_wealth : 0.0;
    if drawdown > ctx.max_drawdown
        evt = MyEscalationEvent();
        evt.day = day;
        evt.trigger_type = "drawdown";
        evt.severity = :critical;
        evt.message = "Drawdown $(round(drawdown*100, digits=1))% exceeds limit $(round(ctx.max_drawdown*100, digits=1))%";
        evt.recommended_action = "De-risk to cash. Notify investment committee.";
        push!(events, evt);
    end

    # 2. Sentiment crash -
    if sentiment < ctx.sentiment_threshold
        evt = MyEscalationEvent();
        evt.day = day;
        evt.trigger_type = "sentiment_crash";
        evt.severity = :warning;
        evt.message = "Sentiment $(round(sentiment, digits=3)) below threshold $(ctx.sentiment_threshold)";
        evt.recommended_action = "Override lambda to $(ctx.sentiment_override_lambda). Review positions.";
        push!(events, evt);
    end

    # 3. Bandit churn -
    n_changes = sum(abs.(current_action .- prev_action));
    if n_changes > ctx.max_bandit_churn
        evt = MyEscalationEvent();
        evt.day = day;
        evt.trigger_type = "bandit_churn";
        evt.severity = :warning;
        evt.message = "Bandit changed $(n_changes) assets (limit: $(ctx.max_bandit_churn))";
        evt.recommended_action = "Hold previous allocation. Review bandit state.";
        push!(events, evt);
    end

    return events;
end

# --- Session 4: Production Simulation ------------------------------------------

"""
    run_production_simulation(ctx::MyProductionContext,
        market_prices::Array{Float64,1}, price_matrix::Array{Float64,2},
        sentiment_series::Array{Float64,1}, lambda_series::Array{Float64,1},
        gm_ema::Array{Float64,1};
        n_days::Int = 60, offset::Int = 84,
        n_bandit_iters::Int = 100) -> (results::Array{MyProductionDayResult,1}, events::Array{MyEscalationEvent,1})

Run a full production simulation: bandit selects assets, Cobb-Douglas allocates,
sentiment monitoring overrides lambda, escalation triggers fire when conditions are met.
"""
function run_production_simulation(ctx::MyProductionContext,
    market_prices::Array{Float64,1}, price_matrix::Array{Float64,2},
    sentiment_series::Array{Float64,1}, lambda_series::Array{Float64,1},
    gm_ema::Array{Float64,1};
    n_days::Int = 60, offset::Int = 84,
    n_bandit_iters::Int = 100)::Tuple{Array{MyProductionDayResult,1},Array{MyEscalationEvent,1}}

    K = length(ctx.tickers);
    Δt = 1.0 / 252.0;

    results = MyProductionDayResult[];
    all_events = MyEscalationEvent[];

    # initial state -
    current_shares = zeros(K);
    current_cash = ctx.B₀;
    peak_wealth = ctx.B₀;
    prev_bandit_action = ones(Int, K); # start with all assets

    for d in 1:n_days

        actual_day = offset + d;
        day_result = MyProductionDayResult();
        day_result.day = d;

        # --- Step 1: Read sentiment ---
        sent = sentiment_series[min(actual_day, length(sentiment_series))];
        day_result.sentiment = sent;

        # --- Step 2: Determine effective lambda ---
        λ_base = lambda_series[min(actual_day, length(lambda_series))];
        if sent < ctx.sentiment_threshold
            λ_eff = ctx.sentiment_override_lambda; # override: increase risk-aversion
        else
            λ_eff = λ_base;
        end
        day_result.lambda = λ_eff;

        # --- Step 3: Run bandit to select assets ---
        prices_now = [price_matrix[actual_day, k + 1] for k in 1:K];
        gm_t = gm_ema[min(actual_day, length(gm_ema))];

        bandit_ctx = build(MyBanditContext, (
            tickers = ctx.tickers, sim_parameters = ctx.sim_parameters,
            prices = prices_now, B = current_cash + sum(current_shares[k] * prices_now[k] for k in 1:K),
            gm_t = gm_t, lambda = λ_eff, epsilon = ctx.epsilon
        ));

        bandit_model = build(MyEpsilonGreedyBanditModel, (
            K = K, n_iterations = n_bandit_iters, alpha = 0.1
        ));

        bandit_result = solve_bandit(bandit_model, bandit_ctx);
        current_bandit_action = bandit_result.best_action;
        day_result.bandit_action = current_bandit_action;

        # --- Step 4: Check escalation triggers ---
        current_wealth = current_cash + sum(current_shares[k] * prices_now[k] for k in 1:K);
        peak_wealth = max(peak_wealth, current_wealth);

        events = check_escalation_triggers(d, ctx, current_wealth, peak_wealth,
            sent, current_bandit_action, prev_bandit_action);

        day_result.escalated = length(events) > 0;
        append!(all_events, events);

        # --- Step 5: Decide and execute ---
        has_critical = any(e.severity == :critical for e in events);
        has_churn = any(e.trigger_type == "bandit_churn" for e in events);

        if has_critical
            # de-risk to cash -
            current_shares = zeros(K);
            current_cash = current_wealth;
            day_result.rebalanced = true;
            day_result.gamma = zeros(K);
        elseif has_churn
            # hold previous allocation (ignore bandit recommendation) -
            day_result.rebalanced = false;
            current_bandit_action = prev_bandit_action;
            day_result.gamma = zeros(K);
        else
            # normal operation: allocate via Cobb-Douglas with bandit-selected assets -
            modified_sim = copy(ctx.sim_parameters);
            for (k, ticker) in enumerate(ctx.tickers)
                if current_bandit_action[k] == 0
                    (_, βᵢ, σᵢ) = ctx.sim_parameters[ticker];
                    modified_sim[ticker] = (-10.0, βᵢ, σᵢ);
                end
            end

            gamma = compute_preference_weights(modified_sim, ctx.tickers, gm_t, λ_eff);
            day_result.gamma = gamma;

            problem = MyCobbDouglasChoiceProblem();
            problem.gamma = gamma;
            problem.prices = prices_now;
            problem.B = current_wealth;
            problem.epsilon = ctx.epsilon;
            (new_shares, new_cash) = allocate_cobb_douglas(problem);

            # apply turnover cap -
            if d > 1
                old_value = sum(current_shares[k] * prices_now[k] for k in 1:K);
                trade_value = sum(abs(new_shares[k] - current_shares[k]) * prices_now[k] for k in 1:K);
                turnover = old_value > 0 ? trade_value / old_value : 0.0;
                if turnover > ctx.max_turnover
                    scale = ctx.max_turnover / turnover;
                    for k in 1:K
                        new_shares[k] = current_shares[k] + scale * (new_shares[k] - current_shares[k]);
                    end
                end
            end

            current_shares = new_shares;
            current_cash = new_cash;
            day_result.rebalanced = true;
        end

        # --- Step 6: Record state ---
        day_result.shares = copy(current_shares);
        day_result.cash = current_cash;
        day_result.wealth = current_cash + sum(current_shares[k] * prices_now[k] for k in 1:K);
        prev_bandit_action = current_bandit_action;

        push!(results, day_result);
    end

    return (results, all_events);
end

# --- Session 4: Dashboard Metrics -----------------------------------------------

"""
    compute_dashboard_metrics(results::Array{MyProductionDayResult,1},
        events::Array{MyEscalationEvent,1}) -> Dict{String, Any}

Compute aggregate metrics for the operational dashboard.
"""
function compute_dashboard_metrics(results::Array{MyProductionDayResult,1},
    events::Array{MyEscalationEvent,1})::Dict{String,Any}

    n_days = length(results);
    K = length(results[1].shares);

    # wealth series -
    wealth = [r.wealth for r in results];

    # rebalance frequency -
    n_rebalances = sum(r.rebalanced for r in results);

    # bandit churn: how often does the selected subset change? -
    n_bandit_changes = 0;
    for d in 2:n_days
        if results[d].bandit_action != results[d-1].bandit_action
            n_bandit_changes += 1;
        end
    end

    # escalation frequency -
    n_escalations = length(events);
    n_critical = sum(e.severity == :critical for e in events);
    n_warning = sum(e.severity == :warning for e in events);

    # drawdown -
    peak = accumulate(max, wealth);
    max_dd = maximum((peak .- wealth) ./ peak);

    # return metrics -
    return Dict(
        "n_days" => n_days,
        "final_wealth" => wealth[end],
        "max_drawdown" => max_dd,
        "n_rebalances" => n_rebalances,
        "n_bandit_changes" => n_bandit_changes,
        "n_escalations" => n_escalations,
        "n_critical" => n_critical,
        "n_warning" => n_warning,
        "avg_sentiment" => mean(r.sentiment for r in results),
        "wealth_series" => wealth
    );
end

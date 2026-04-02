# --- Signal Computation ----------------------------------------------------------

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

### Arguments
- `sim_parameters` — SIM params (alpha_i, beta_i, sigma_i) per ticker
- `tickers` — asset ticker names
- `gm_t` — expected market growth rate at time t
- `lambda` — sentiment/risk-aversion parameter

### Returns
- `gamma` — preference exponents, one per asset, in (-1, 1)
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

# --- Utility-Based Allocation Functions -----------------------------------------

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

# --- Utility Evaluation Functions -----------------------------------------------

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

# --- Rebalancing Engine ---------------------------------------------------------

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

# --- Portfolio Metrics ----------------------------------------------------------

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

"""
    compute_drawdown(returns::Array{Float64,1}) -> Float64

Compute the maximum drawdown from a return series.
"""
function compute_drawdown(returns::Array{Float64,1})::Float64

    # compute cumulative wealth -
    wealth = cumprod(1.0 .+ returns);
    peak = accumulate(max, wealth);
    drawdowns = (peak .- wealth) ./ peak;

    # return -
    return maximum(drawdowns);
end

"""
    compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1}) -> Float64

Compute portfolio turnover between two weight vectors.
"""
function compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1})::Float64
    return sum(abs.(w_new .- w_old));
end

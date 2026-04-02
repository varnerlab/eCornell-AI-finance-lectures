"""
    compute_ema(prices::Array{Float64,1}; window::Int = 21) -> Array{Float64,1}

Compute the exponential moving average of a price series.
Uses smoothing factor α = 2/(window + 1).
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

Compute the sentiment/risk-aversion signal λₜ from short and long EMA crossover.

λₜ = -G × (EMA_short_t / EMA_long_t - 1)

Positive λ → bearish (short < long), negative λ → bullish (short > long).
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

Compute the log growth rate of a market price series: gₘ = (1/Δt) × log(Sₜ/Sₜ₋₁).
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
    allocate_shares(t::Int, context::MyRebalancingContextModel) -> MyRebalancingResult

Allocate shares using the SIM-based risk-adjusted allocation model.
Follows the TradeBot pattern: compute preference weights γᵢ from SIM parameters,
then allocate budget proportionally to preferred assets.

g_hat_i = αᵢ / βᵢ^λ + (βᵢ / βᵢ^λ) × gₘ
γᵢ = tanh(g_hat_i)
"""
function allocate_shares(t::Int, context::MyRebalancingContextModel)::MyRebalancingResult

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
    gamma = zeros(K);
    for i in 1:K
        (αᵢ, βᵢ, σᵢ) = sim_params[tickers[i]];
        RF = max(abs(βᵢ)^λ, 1e-8); # risk factor, clamped
        g_hat = αᵢ / RF + (βᵢ / RF) * gm[min(t, length(gm))];
        gamma[i] = tanh(g_hat);
    end

    # separate preferred (γ > 0) and non-preferred (γ ≤ 0) -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate budget -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to minimum ε shares -
    for i in non_preferred
        price = marketdata[t, i + 1]; # column i+1 (col 1 = day index)
        shares[i] = ε;
        remaining_B -= ε * price;
    end

    # preferred: proportional allocation -
    if length(preferred) > 0 && remaining_B > 0
        gamma_bar = sum(gamma[preferred]);
        for i in preferred
            price = marketdata[t, i + 1];
            shares[i] = (gamma[i] / gamma_bar) * (remaining_B / price);
        end
        cash = 0.0;
    else
        cash = remaining_B;
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
        lambda_series::Array{Float64,1}; offset::Int = 84) -> Dict{Int, MyRebalancingResult}

Run the AI rebalancing engine over the trading period.

Follows the TradeBot daily loop:
1. Check rebalance schedule and trigger rules
2. If rebalance: liquidate, update budget and lambda, re-allocate
3. If hold: propagate prior positions
4. If drawdown exceeded: de-risk to cash

Returns a dictionary mapping day index → MyRebalancingResult.
"""
function run_rebalancing_engine(context::MyRebalancingContextModel, rules::MyTriggerRules,
    lambda_series::Array{Float64,1}; offset::Int = 84)::Dict{Int,MyRebalancingResult}

    # setup -
    tickers = context.tickers;
    marketdata = context.marketdata;
    K = length(tickers);
    schedule = rules.rebalance_schedule;
    T = length(schedule);

    # results storage -
    results = Dict{Int,MyRebalancingResult}();

    # initial allocation at offset -
    results[0] = allocate_shares(offset, context);

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
                # rebalance -
                ctx = deepcopy(context);
                ctx.B = liquidation_value;
                ctx.lambda = lambda_series[min(actual_day, length(lambda_series))];

                new_result = allocate_shares(actual_day, ctx);

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

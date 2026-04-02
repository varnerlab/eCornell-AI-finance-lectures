# --- Include Session 3 bandit functions (without re-importing Session 2) --------
# Session 2 types/functions already loaded via Types.jl import chain.
# We need the bandit_world and solve_bandit functions from Session 3.

include(joinpath(_PATH_TO_SESSION3, "src", "Compute.jl"));

# --- Sentiment Generation -------------------------------------------------------

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

# --- Escalation Checking -------------------------------------------------------

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

# --- Production Simulation ------------------------------------------------------

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

# --- Dashboard Metrics ----------------------------------------------------------

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

# --- Import Session 2 types and functions ---
# We need the rebalancing engine and Cobb-Douglas allocator from Session 2
const _PATH_TO_SESSION2 = joinpath(dirname(dirname(_ROOT)), "session-2", "L2");

# Include Session 2 types and compute (for the rebalancing engine) -
include(joinpath(_PATH_TO_SESSION2, "src", "Types.jl"));
include(joinpath(_PATH_TO_SESSION2, "src", "Factory.jl"));
include(joinpath(_PATH_TO_SESSION2, "src", "Compute.jl"));

# --- Training Data Generation ---------------------------------------------------

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

# --- Scenario Generation --------------------------------------------------------

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

# --- Backtesting Functions ------------------------------------------------------

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

# --- Bandit Functions -----------------------------------------------------------

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

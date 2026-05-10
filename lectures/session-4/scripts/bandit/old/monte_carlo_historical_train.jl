#!/usr/bin/env julia
# monte_carlo_historical_train.jl
#
# Multi-seed sweep of the historical-training pipeline (the one in
# `historical_train_dqn.jl`). For each seed, train the regime-conditional
# featurized DQN on 2014-2024 with realized 21-day forward returns, train
# the regime-blind sparse bandit on the same signal, deploy both at
# TEST_DATE (2025-01-02), forward CD walk through 2026-04-22, and record
# the realized metrics. Aggregate across seeds and print a summary table
# with the deterministic Full-Universe baseline as a reference row.
#
# Usage:
#   cd lectures/session-4
#   julia --project=. scripts/bandit/old/monte_carlo_historical_train.jl                            # defaults: 20 seeds
#   julia --project=. scripts/bandit/old/monte_carlo_historical_train.jl 30 5000 21 5000           # 30 seeds, full settings
#   julia --project=. scripts/bandit/old/monte_carlo_historical_train.jl 20 3000 21 3000           # 20 seeds, faster training
#
# Args (positional):
#   1) N_SEEDS       (default 20)
#   2) EPISODES      DQN episodes per seed (default 5000)
#   3) HORIZON       reward horizon in trading days (default 21)
#   4) BANDIT_ITERS  sparse-bandit iterations per seed (default 5000)
#
# Output:
#   1) Per-seed progress lines + summary stats table to stdout
#   2) Raw per-seed records saved to scripts/monte_carlo_historical_train_results.jld2
#
# Runtime: roughly 5-10 minutes per seed at the default settings (DQN training
# dominates). 20 seeds is ~2-3 hours; 30 seeds is ~3-5 hours. Lower EPISODES
# to trim runtime at the cost of less-converged DQNs.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using eCornellAIFinance
using Dates, DataFrames, Flux, JLD2, Printf, PrettyTables, Random, Statistics

# ===== CONFIG =====
const N_SEEDS_DEFAULT          = 20
const SEED_BASE                = 1000
const TRAINING_EPISODES_DEFAULT = 5_000
const FORWARD_HORIZON_DEFAULT   = 21
const BANDIT_ITERS_DEFAULT      = 5_000

const TRAIN_END_DATE      = "2024-12-31"
const TEST_DATE           = "2025-01-02"
const WARMUP_OFFSET       = 84

const B0                  = 10_000.0
const Δt                  = 1.0 / 252.0
const L_short             = 21
const L_long              = 63
const L_growth            = 10
const GAIN                = 10.0
const BANDIT_GM_WINDOW    = 63
const BANDIT_EPSILON      = 0.1
const TRIGGER_MAX_DRAWDOWN = 0.15
const TRIGGER_MAX_TURNOVER = 0.50

const K_BASKET_DQN        = 32             # apples-to-apples: same as bandit
const K_BASKET_BANDIT     = 32             # apples-to-apples: same as DQN; sparse Dict scales fine, only the algorithm differs
const BANDIT_ALPHA        = 0.1

const HIDDEN              = 256
const BUFFER_CAPACITY     = 50_000
const WARMUP_BUFFER       = 1_000
const MINIBATCH           = 128
const TARGET_SYNC         = 200
const DISCOUNT            = 0.95f0
const LR                  = 5.0f-4
const EPS_FLOOR           = 0.05f0

# ===== HELPERS (copies of the ones in historical_train_dqn.jl) =====

mutable struct MyReplayBuffer
    states::Vector{Vector{Float32}}
    actions::Vector{Int}
    rewards::Vector{Float32}
    next_states::Vector{Vector{Float32}}
    dones::Vector{Bool}
    capacity::Int
end
MyReplayBuffer(cap::Int) = MyReplayBuffer(
    Vector{Vector{Float32}}(), Int[], Float32[],
    Vector{Vector{Float32}}(), Bool[], cap)

function push_transition!(buf::MyReplayBuffer, s, a, r, s′, d)
    if length(buf.states) >= buf.capacity
        popfirst!(buf.states); popfirst!(buf.actions)
        popfirst!(buf.rewards); popfirst!(buf.next_states); popfirst!(buf.dones)
    end
    push!(buf.states, s); push!(buf.actions, a); push!(buf.rewards, r)
    push!(buf.next_states, s′); push!(buf.dones, d)
end

make_qnet_featurized(state_dim::Int, action_dim::Int, hidden::Int) = Chain(
    Dense(state_dim, hidden, relu),
    Dense(hidden, hidden, relu),
    Dense(hidden, action_dim),
)

function select_action_featurized(qnet, state::Vector{Float32}, ε::Real, K::Int)
    available = findall(==(0.0f0), @view state[1:K])
    isempty(available) && error("no available actions")
    if rand() <= ε
        return rand(available)
    else
        Q = vec(qnet(reshape(state, :, 1)))
        @inbounds for i in 1:K
            state[i] > 0.5f0 && (Q[i] = -Inf32)
        end
        return argmax(Q)
    end
end

function dqn_train_step!(main, target, opt_state, buf::MyReplayBuffer,
        B::Int, γ_disc::Float32, K::Int)
    n = length(buf.states); n < B && return 0.0f0
    idx = rand(1:n, B)
    S_mat  = reduce(hcat, buf.states[idx])
    Sn_mat = reduce(hcat, buf.next_states[idx])
    rewards = buf.rewards[idx]
    dones   = Float32.(buf.dones[idx])
    actions = buf.actions[idx]
    Qn = target(Sn_mat)
    @inbounds for j in 1:B, i in 1:K
        Sn_mat[i, j] > 0.5f0 && (Qn[i, j] = -1.0f30)
    end
    max_Qn = vec(maximum(Qn, dims = 1))
    targets = rewards .+ γ_disc .* (1.0f0 .- dones) .* max_Qn
    A_onehot = zeros(Float32, K, B)
    @inbounds for j in 1:B
        A_onehot[actions[j], j] = 1.0f0
    end
    _, grads = Flux.withgradient(main) do m
        Q = m(S_mat)
        Q_taken = vec(sum(Q .* A_onehot, dims = 1))
        Flux.mse(Q_taken, targets)
    end
    Flux.update!(opt_state, main, grads[1])
    return 0.0f0
end

build_state_featurized(mask::Vector{Float32}, γ::Vector{Float32}) = vcat(mask, γ)

function cd_allocate_basket_at_day(basket_indices::Vector{Int}, day::Int, env)
    γ_all = env.γ_matrix[day, :]
    γ_S   = γ_all[basket_indices]
    p_S   = env.train_price_matrix[day, basket_indices]
    K_b   = length(basket_indices)
    n     = zeros(Float64, K_b)
    has_neg = any(γ_S .< 0.0)
    if !has_neg
        γ_sum = sum(γ_S)
        γ_sum > 0.0 || return n
        for j in 1:K_b
            n[j] = (γ_S[j] / γ_sum) * (B0 / p_S[j])
        end
    else
        ε = BANDIT_EPSILON
        B̄ = B0
        γ_sum_pos = 0.0
        for j in 1:K_b
            if γ_S[j] < 0.0
                B̄ -= ε * p_S[j]
                n[j] = ε
            else
                γ_sum_pos += γ_S[j]
            end
        end
        if γ_sum_pos > 0.0
            for j in 1:K_b
                if γ_S[j] >= 0.0
                    n[j] = (γ_S[j] / γ_sum_pos) * (B̄ / p_S[j])
                end
            end
        end
    end
    return n
end

function realized_basket_return(basket_indices::Vector{Int}, day::Int,
        horizon::Int, env)::Float32
    n = cd_allocate_basket_at_day(basket_indices, day, env)
    p_d   = env.train_price_matrix[day, basket_indices]
    p_d_h = env.train_price_matrix[day + horizon, basket_indices]
    W_d   = sum(n .* p_d)
    W_dh  = sum(n .* p_d_h)
    (W_d <= 0.0 || W_dh <= 0.0) && return 0.0f0
    return Float32(log(W_dh / W_d))
end

function realized_picker_world(state::Vector{Float32}, action::Int, day::Int,
        env, K::Int, K_basket::Int, horizon::Int)
    s_next = copy(state)
    s_next[action] = 1.0f0
    basket_size = Int(round(sum(@view s_next[1:K])))
    done = (basket_size >= K_basket)
    if done
        basket_indices = findall(==(1.0f0), @view s_next[1:K])
        r = realized_basket_return(basket_indices, day, horizon, env)
    else
        r = 0.0f0
    end
    return (s_next, r, done)
end

# ===== TRAINING =====

function train_dqn_historical(env;
        episodes::Int, horizon::Int, K_basket::Int = K_BASKET_DQN,
        hidden::Int = HIDDEN, lr::Float32 = LR,
        buffer_cap::Int = BUFFER_CAPACITY, warmup::Int = WARMUP_BUFFER,
        batch_size::Int = MINIBATCH, sync_freq::Int = TARGET_SYNC,
        γ_disc::Float32 = DISCOUNT, ε_floor::Float32 = EPS_FLOOR)
    K = env.K
    state_dim = 2 * K
    main   = make_qnet_featurized(state_dim, K, hidden)
    target = make_qnet_featurized(state_dim, K, hidden)
    Flux.loadmodel!(target, Flux.state(main))
    opt_state = Flux.setup(Adam(lr), main)
    buf = MyReplayBuffer(buffer_cap)
    rewards_per_ep = Float32[]
    global_step = 0
    for _ in 1:episodes
        d = rand(env.train_offset:env.train_last)
        γ_d = Float32.(env.γ_matrix[d, :])
        s = build_state_featurized(zeros(Float32, K), γ_d)
        ep_return = 0.0f0
        while true
            global_step += 1; t = global_step
            ε_raw = min(1.0f0, Float32(t)^(-1/3) * (Float32(K) * log(t + 1))^(1/3))
            ε = max(ε_floor, ε_raw)
            a = select_action_featurized(main, s, ε, K)
            s′, r, done = realized_picker_world(s, a, d, env, K, K_basket, horizon)
            push_transition!(buf, s, a, r, s′, done)
            ep_return += r
            if length(buf.states) >= warmup
                dqn_train_step!(main, target, opt_state, buf, batch_size, γ_disc, K)
                global_step % sync_freq == 0 && Flux.loadmodel!(target, Flux.state(main))
            end
            s = s′
            done && break
        end
        push!(rewards_per_ep, ep_return)
    end
    return main, rewards_per_ep
end

function train_bandit_historical(env;
        K_basket::Int, n_iters::Int, horizon::Int,
        ε_floor::Float64 = Float64(EPS_FLOOR), α::Float64 = BANDIT_ALPHA)
    K = env.K
    arm_mean  = Dict{Vector{Int},Float64}()
    arm_count = Dict{Vector{Int},Int}()
    rewards   = zeros(Float64, n_iters)
    random_basket() = sort!(randperm(K)[1:K_basket])
    for t in 1:n_iters
        ε = max(ε_floor, t > 1 ? min(1.0, t^(-1.0/3.0) * (K_basket * log(t))^(1.0/3.0)) : 1.0)
        basket = if rand() < ε || isempty(arm_mean)
            random_basket()
        else
            argmax(arm_mean)
        end
        d = rand(env.train_offset:env.train_last)
        r = Float64(realized_basket_return(basket, d, horizon, env))
        c = get(arm_count, basket, 0) + 1
        arm_count[basket] = c
        old = get(arm_mean, basket, 0.0)
        lr = α > 0.0 ? α : 1.0 / c
        arm_mean[basket] = old + lr * (r - old)
        rewards[t] = r
    end
    best_basket = argmax(arm_mean)
    best_mean   = arm_mean[best_basket]
    return (best_basket = best_basket, best_mean = best_mean,
            n_unique_arms = length(arm_mean))
end

function deploy_dqn(env, dqn_main; K_basket::Int = K_BASKET_DQN)
    K = env.K
    γ_test = Float32.(env.γ_test)
    s = build_state_featurized(zeros(Float32, K), γ_test)
    picked = Int[]
    for _ in 1:K_basket
        a = select_action_featurized(dqn_main, s, 0.0f0, K)
        push!(picked, a)
        s[a] = 1.0f0
    end
    return env.my_tickers[picked], picked
end

"""
    random_picker(env; K_basket) -> Vector{String}

Pure baseline: pick K_basket tickers uniformly at random from the K-ticker
universe. No training, no learning, no regime-conditioning. If the trained
DQN is producing essentially-random baskets (as the diagnostic suggested),
this should land at roughly the DQN's median performance.
"""
function random_picker(env; K_basket::Int = K_BASKET_DQN)
    K = env.K
    idx = sort!(randperm(K)[1:K_basket])
    return env.my_tickers[idx], idx
end

# ===== METRICS + FORWARD WALK =====

function realized_metrics(W::Vector{Float64}, g_f::Float64)
    ret    = diff(W) ./ W[1:end-1]
    peak   = accumulate(max, W)
    max_dd = maximum((peak .- W) ./ peak)
    vol    = std(ret) * sqrt(252)
    ann_ret = log(W[end] / W[1]) * (252.0 / (length(W) - 1))
    sharpe  = vol > 1e-6 ? (ann_ret - g_f) / vol : 0.0
    return (W_T = W[end], W_T_over_W0 = W[end] / B0, max_dd = max_dd,
            ann_ret = ann_ret, sharpe = sharpe)
end

function fwd_cd(basket::Vector{String}, rules, env)
    sel_idx = findall(in(basket), env.my_tickers)
    sel_sim = Dict(t => env.sim_params[t] for t in basket)
    sel_p   = env.forward_price_matrix[:, vcat([1], sel_idx .+ 1)]
    ctx = build(MyRebalancingContextModel, (
        B = B0, tickers = basket, marketdata = sel_p,
        marketfactor = env.forward_gm_ema, sim_parameters = sel_sim,
        lambda = env.forward_lambda[1], Δt = Δt, epsilon = 0.1,
    ))
    res = run_rebalancing_engine(ctx, rules, env.forward_lambda;
        offset = 1, allocator = :cobb_douglas)
    return compute_wealth_series(res, sel_p, basket; offset = 1)
end

function full_universe_metrics(env)
    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN, max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1)
    ))
    ctx = build(MyRebalancingContextModel, (
        B = B0, tickers = env.my_tickers, marketdata = env.forward_price_matrix,
        marketfactor = env.forward_gm_ema, sim_parameters = env.sim_params,
        lambda = env.forward_lambda[1], Δt = Δt, epsilon = 0.1,
    ))
    res = run_rebalancing_engine(ctx, rules, env.forward_lambda;
        offset = 1, allocator = :cobb_douglas)
    W = compute_wealth_series(res, env.forward_price_matrix, env.my_tickers; offset = 1)
    return realized_metrics(W, env.g_f)
end

# ===== LOADER =====

function load_environment(; horizon::Int)
    calib = MySIMCalibration()
    sim_tickers = calib["tickers"]::Vector{String}
    α_vec = calib["alpha"]::Vector{Float64}
    β_vec = calib["beta"]::Vector{Float64}
    σ_vec = calib["sigma_eps"]::Vector{Float64}
    sim_full = Dict{String,Tuple{Float64,Float64,Float64}}(
        sim_tickers[i] => (α_vec[i], β_vec[i], σ_vec[i]) for i in eachindex(sim_tickers))
    g_f = 0.045

    train_spy = MyTrainingMarketDataSet()["dataset"]["SPY"]
    test_spy  = MyExtendedTestingMarketDataSet()["dataset"]["SPY"]
    spy_full  = vcat(train_spy, test_spy)
    sort!(spy_full, :timestamp); unique!(spy_full, :timestamp)

    test_target = Date(TEST_DATE)
    test_idx_full = findfirst(==(test_target), Date.(spy_full.timestamp))
    isnothing(test_idx_full) && error("TEST_DATE=$(TEST_DATE) not in SPY history")
    train_end_target = Date(TRAIN_END_DATE)
    train_end_idx = findlast(d -> Date(d) <= train_end_target, spy_full.timestamp)
    isnothing(train_end_idx) && error("TRAIN_END_DATE not found")

    train_test_ds = MyTrainingMarketDataSet()["dataset"]
    test_ds = MyExtendedTestingMarketDataSet()["dataset"]
    full_dates = Date.(spy_full.timestamp)
    target_dates_fwd = full_dates[test_idx_full:end]
    target_dates_train = full_dates[1:train_end_idx]

    keep = String[]
    for t in sim_tickers
        haskey(train_test_ds, t) && haskey(test_ds, t) || continue
        df_train = train_test_ds[t]; df_train_dates = Set(Date.(df_train.timestamp))
        df_test  = test_ds[t];       df_test_dates  = Set(Date.(df_test.timestamp))
        ok_train = all(d -> d in df_train_dates, target_dates_train)
        ok_train || continue
        ok_fwd   = all(d -> d in df_test_dates,  target_dates_fwd)
        ok_fwd && push!(keep, t)
    end
    my_tickers = sort(keep)
    sim_params = Dict(t => sim_full[t] for t in my_tickers)
    K = length(my_tickers)

    n_full = length(full_dates)
    full_price_matrix = zeros(n_full, K)
    for (j, t) in enumerate(my_tickers)
        df_train = train_test_ds[t]; tdates_train = Date.(df_train.timestamp)
        df_test  = test_ds[t];       tdates_test  = Date.(df_test.timestamp)
        for (i, d) in enumerate(full_dates)
            idx_train = findfirst(==(d), tdates_train)
            if !isnothing(idx_train)
                full_price_matrix[i, j] = df_train.close[idx_train]
            else
                idx_test = findfirst(==(d), tdates_test)
                if !isnothing(idx_test)
                    full_price_matrix[i, j] = df_test.close[idx_test]
                else
                    full_price_matrix[i, j] = NaN
                end
            end
        end
    end

    spy_close = Float64.(spy_full.close)
    ema_s_full = compute_ema(spy_close; window = L_short)
    ema_l_full = compute_ema(spy_close; window = L_long)
    λ_full     = compute_lambda(ema_s_full, ema_l_full; G = GAIN)
    gm_full    = compute_market_growth(spy_close; Δt = Δt)
    gm_ema_picker_full = compute_ema(gm_full; window = BANDIT_GM_WINDOW)
    gm_ema_engine_full = compute_ema(gm_full; window = L_growth)
    gm_ema_picker_aligned = vcat([gm_ema_picker_full[1]], gm_ema_picker_full)
    gm_ema_engine_aligned = vcat([gm_ema_engine_full[1]], gm_ema_engine_full)

    train_offset = WARMUP_OFFSET + 1
    train_last   = train_end_idx - horizon
    train_last >= train_offset || error("training window too short for horizon $(horizon)")
    γ_matrix = zeros(n_full, K)
    for d in train_offset:train_last
        gm_d = gm_ema_picker_aligned[d]
        λ_d  = λ_full[d]
        γ_d  = compute_preference_weights(sim_params, my_tickers, gm_d, λ_d)
        γ_matrix[d, :] = γ_d
    end

    test_idx = test_idx_full
    γ_test = compute_preference_weights(sim_params, my_tickers,
        gm_ema_picker_aligned[test_idx], λ_full[test_idx])

    n_fwd = n_full - test_idx + 1
    forward_dates  = full_dates[test_idx:end]
    forward_lambda = λ_full[test_idx:end]
    forward_gm_ema = gm_ema_engine_aligned[test_idx:end]
    forward_price_matrix = zeros(n_fwd, K + 1)
    forward_price_matrix[:, 1] = 1:n_fwd
    forward_price_matrix[:, 2:end] = full_price_matrix[test_idx:end, :]

    return (my_tickers = my_tickers, sim_params = sim_params, K = K, g_f = g_f,
            train_price_matrix = full_price_matrix, train_dates = full_dates,
            train_lambda = λ_full, train_gm_ema = gm_ema_picker_aligned,
            γ_matrix = γ_matrix,
            train_offset = train_offset, train_last = train_last,
            test_decision_index = test_idx, γ_test = γ_test,
            forward_dates = forward_dates,
            forward_price_matrix = forward_price_matrix,
            forward_lambda = forward_lambda, forward_gm_ema = forward_gm_ema,
            n_fwd = n_fwd, horizon = horizon)
end

# ===== ONE SEED =====

"""
    one_run(seed, env, episodes, horizon, bandit_iters) -> NamedTuple

Run one full historical-training pipeline at the given seed: train DQN +
bandit on 2014-2024, deploy both at TEST_DATE, forward CD walk through
2026-04-22, return realized metrics + chosen baskets.
"""
function one_run(seed::Int, env, episodes::Int, horizon::Int, bandit_iters::Int)
    Random.seed!(seed)

    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN, max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1)
    ))

    # --- DQN ---
    dqn_main, _ = train_dqn_historical(env;
        episodes = episodes, horizon = horizon, K_basket = K_BASKET_DQN)
    dqn_basket, _ = deploy_dqn(env, dqn_main; K_basket = K_BASKET_DQN)
    W_dqn = fwd_cd(dqn_basket, rules, env)
    m_dqn = realized_metrics(W_dqn, env.g_f)

    # --- Bandit ---
    bres = train_bandit_historical(env;
        K_basket = K_BASKET_BANDIT, n_iters = bandit_iters, horizon = horizon)
    bandit_basket = env.my_tickers[bres.best_basket]
    W_bnd = fwd_cd(bandit_basket, rules, env)
    m_bnd = realized_metrics(W_bnd, env.g_f)

    # --- Pure random baseline (no training, no learning) ---
    rnd_basket, _ = random_picker(env; K_basket = K_BASKET_DQN)
    W_rnd = fwd_cd(rnd_basket, rules, env)
    m_rnd = realized_metrics(W_rnd, env.g_f)

    return (
        seed = seed,
        dqn_basket = dqn_basket, dqn_W_T_over_W0 = m_dqn.W_T_over_W0,
        dqn_max_dd = m_dqn.max_dd, dqn_ann_ret = m_dqn.ann_ret, dqn_sharpe = m_dqn.sharpe,
        bandit_basket = bandit_basket, bandit_best_mean = bres.best_mean,
        bandit_W_T_over_W0 = m_bnd.W_T_over_W0, bandit_max_dd = m_bnd.max_dd,
        bandit_ann_ret = m_bnd.ann_ret, bandit_sharpe = m_bnd.sharpe,
        bandit_n_unique_arms = bres.n_unique_arms,
        random_basket = rnd_basket,
        random_W_T_over_W0 = m_rnd.W_T_over_W0, random_max_dd = m_rnd.max_dd,
        random_ann_ret = m_rnd.ann_ret, random_sharpe = m_rnd.sharpe,
    )
end

# ===== AGGREGATE =====

function build_summary_table(results, full_metrics)
    metrics = [
        ("W_T / W_0",     :W_T_over_W0, x -> round(x, digits = 3)),
        ("Ann. return %", :ann_ret,     x -> round(x * 100, digits = 2)),
        ("Max DD %",      :max_dd,      x -> round(x * 100, digits = 1)),
        ("Sharpe",        :sharpe,      x -> round(x, digits = 3)),
    ]
    rows = Any[]
    for (label, base_key, fmt) in metrics
        for picker in ("DQN", "Bandit", "Random")
            prefix = lowercase(picker)
            key = Symbol("$(prefix)_$(base_key)")
            vals = [getproperty(r, key) for r in results]
            push!(rows, (
                Metric = label, Picker = picker,
                Min    = fmt(minimum(vals)),
                Q25    = fmt(quantile(vals, 0.25)),
                Median = fmt(median(vals)),
                Mean   = fmt(mean(vals)),
                Q75    = fmt(quantile(vals, 0.75)),
                Max    = fmt(maximum(vals)),
                StdDev = fmt(std(vals)),
            ))
        end
        full_val = fmt(getproperty(full_metrics, base_key))
        zero_std = fmt(0.0)
        push!(rows, (
            Metric = label, Picker = "FullCD",
            Min = full_val, Q25 = full_val, Median = full_val,
            Mean = full_val, Q75 = full_val, Max = full_val,
            StdDev = zero_std,
        ))
    end
    return DataFrame(rows)
end

function head_to_head(results, full_metrics)
    n = length(results)
    dqn_wins = count(r -> r.dqn_W_T_over_W0    >  r.bandit_W_T_over_W0, results)
    bnd_wins = count(r -> r.bandit_W_T_over_W0 >  r.dqn_W_T_over_W0,    results)
    ties     = n - dqn_wins - bnd_wins
    full_W   = full_metrics.W_T_over_W0
    dqn_beats_full = count(r -> r.dqn_W_T_over_W0    > full_W, results)
    bnd_beats_full = count(r -> r.bandit_W_T_over_W0 > full_W, results)
    rnd_beats_full = count(r -> r.random_W_T_over_W0 > full_W, results)
    dqn_beats_random = count(r -> r.dqn_W_T_over_W0    > r.random_W_T_over_W0, results)
    bnd_beats_random = count(r -> r.bandit_W_T_over_W0 > r.random_W_T_over_W0, results)
    dqn_trip = count(r -> r.dqn_max_dd    >= TRIGGER_MAX_DRAWDOWN, results)
    bnd_trip = count(r -> r.bandit_max_dd >= TRIGGER_MAX_DRAWDOWN, results)
    rnd_trip = count(r -> r.random_max_dd >= TRIGGER_MAX_DRAWDOWN, results)
    return (n = n, dqn_wins = dqn_wins, bandit_wins = bnd_wins, ties = ties,
            dqn_beats_full = dqn_beats_full, bandit_beats_full = bnd_beats_full,
            random_beats_full = rnd_beats_full,
            dqn_beats_random = dqn_beats_random, bandit_beats_random = bnd_beats_random,
            dqn_trips = dqn_trip, bandit_trips = bnd_trip, random_trips = rnd_trip,
            full_W_T_over_W0 = full_W)
end

# ===== MAIN =====

function main(N_SEEDS::Int, episodes::Int, horizon::Int, bandit_iters::Int)
    println("Loading environment ...")
    env = load_environment(; horizon = horizon)
    @printf("  K = %d tickers; train window indices [%d, %d]\n",
        env.K, env.train_offset, env.train_last)
    @printf("  test deploy day = %s, forward window = %d days\n", TEST_DATE, env.n_fwd)
    @printf("  reward horizon = %d trading days\n", horizon)
    @printf("  DQN: episodes = %d, K_basket = %d\n", episodes, K_BASKET_DQN)
    @printf("  Bandit: iters = %d, K_basket = %d\n", bandit_iters, K_BASKET_BANDIT)
    println()

    println("Running Full-Universe CD baseline (deterministic) ...")
    full_metrics = full_universe_metrics(env)
    @printf("  W_T/W0 = %.3f  ann_ret = %+6.2f%%  max_dd = %5.2f%%  sharpe = %+6.3f\n",
        full_metrics.W_T_over_W0, full_metrics.ann_ret*100,
        full_metrics.max_dd*100, full_metrics.sharpe)
    println()

    @printf("Sweeping %d seeds ...\n", N_SEEDS)
    println()

    results = NamedTuple[]
    t_start = time()
    for i in 1:N_SEEDS
        seed = SEED_BASE + i
        r = one_run(seed, env, episodes, horizon, bandit_iters)
        push!(results, r)
        elapsed = time() - t_start
        rate = i / elapsed
        eta = (N_SEEDS - i) / rate
        @printf("  seed %3d/%d (#%d):  DQN W_T/W0=%.3f  Bandit W_T/W0=%.3f  Random W_T/W0=%.3f  | elapsed=%5.0fs  ETA=%5.0fs\n",
            i, N_SEEDS, seed, r.dqn_W_T_over_W0, r.bandit_W_T_over_W0, r.random_W_T_over_W0,
            elapsed, eta)
        flush(stdout)
    end

    println()
    println("=" ^ 80)
    @printf("Summary across %d seeds (FullCD row is deterministic, std=0)\n", N_SEEDS)
    println("=" ^ 80)
    df = build_summary_table(results, full_metrics)
    pretty_table(df; backend = :text,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        table_format = TextTableFormat(borders = text_table_borders__compact))

    h2h = head_to_head(results, full_metrics)
    println()
    println("Head-to-head (W_T comparison):")
    @printf("  DQN beats Bandit                : %d / %d (%.0f%%)\n", h2h.dqn_wins, h2h.n, 100*h2h.dqn_wins/h2h.n)
    @printf("  Bandit beats DQN                : %d / %d (%.0f%%)\n", h2h.bandit_wins, h2h.n, 100*h2h.bandit_wins/h2h.n)
    @printf("  Ties                            : %d / %d\n", h2h.ties, h2h.n)
    @printf("  DQN beats Random                : %d / %d (%.0f%%)\n", h2h.dqn_beats_random, h2h.n, 100*h2h.dqn_beats_random/h2h.n)
    @printf("  Bandit beats Random             : %d / %d (%.0f%%)\n", h2h.bandit_beats_random, h2h.n, 100*h2h.bandit_beats_random/h2h.n)
    @printf("  DQN beats Full-Universe         : %d / %d (%.0f%%)   [Full = %.3f]\n", h2h.dqn_beats_full, h2h.n, 100*h2h.dqn_beats_full/h2h.n, h2h.full_W_T_over_W0)
    @printf("  Bandit beats Full-Universe      : %d / %d (%.0f%%)\n", h2h.bandit_beats_full, h2h.n, 100*h2h.bandit_beats_full/h2h.n)
    @printf("  Random beats Full-Universe      : %d / %d (%.0f%%)\n", h2h.random_beats_full, h2h.n, 100*h2h.random_beats_full/h2h.n)
    println()
    println("Drawdown trigger trips (Max DD >= $(TRIGGER_MAX_DRAWDOWN * 100)%):")
    @printf("  DQN baskets that tripped     : %d / %d (%.0f%%)\n", h2h.dqn_trips, h2h.n, 100*h2h.dqn_trips/h2h.n)
    @printf("  Bandit baskets that tripped  : %d / %d (%.0f%%)\n", h2h.bandit_trips, h2h.n, 100*h2h.bandit_trips/h2h.n)
    @printf("  Random baskets that tripped  : %d / %d (%.0f%%)\n", h2h.random_trips, h2h.n, 100*h2h.random_trips/h2h.n)
    @printf("  Full-Universe tripped        : %s\n", (full_metrics.max_dd >= TRIGGER_MAX_DRAWDOWN ? "yes" : "no"))

    out_path = joinpath(@__DIR__, "monte_carlo_historical_train_results.jld2")
    save(out_path, Dict(
        "results"      => results,
        "summary_df"   => df,
        "head_to_head" => h2h,
        "full_metrics" => full_metrics,
        "config"       => Dict(
            "N_SEEDS"            => N_SEEDS, "SEED_BASE" => SEED_BASE,
            "EPISODES"           => episodes, "FORWARD_HORIZON" => horizon,
            "BANDIT_ITERS"       => bandit_iters,
            "K_BASKET_DQN"       => K_BASKET_DQN,
            "K_BASKET_BANDIT"    => K_BASKET_BANDIT,
            "HIDDEN"             => HIDDEN,
            "TRAIN_END_DATE"     => TRAIN_END_DATE,
            "TEST_DATE"          => TEST_DATE,
            "WARMUP_OFFSET"      => WARMUP_OFFSET,
            "TRIGGER_MAX_DRAWDOWN" => TRIGGER_MAX_DRAWDOWN,
        ),
    ))
    println()
    @printf("Per-seed records + summary saved to: %s\n", out_path)
end

# ===== ENTRY =====

let
    N_SEEDS      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : N_SEEDS_DEFAULT
    episodes     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : TRAINING_EPISODES_DEFAULT
    horizon      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : FORWARD_HORIZON_DEFAULT
    bandit_iters = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : BANDIT_ITERS_DEFAULT
    main(N_SEEDS, episodes, horizon, bandit_iters)
end

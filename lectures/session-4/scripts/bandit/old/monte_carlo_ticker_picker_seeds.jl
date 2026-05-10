#!/usr/bin/env julia
# Monte Carlo seed sweep for the ticker-picker DQN vs sparse-bandit comparison.
#
# Reproduces Task 1 + the static-basket portion of Task 2 from the notebook
# `eCornell-AI-Finance-S4-Example-Optional-TickerPickerDQN-May-2026.ipynb` for
# multiple random seeds and reports the distribution of realized metrics so we
# can separate seed-dependent performance noise from structural claims.
#
# Usage:
#   cd lectures/session-4
#   julia --project=. scripts/bandit/old/monte_carlo_ticker_picker_seeds.jl            # default 20 seeds
#   julia --project=. scripts/bandit/old/monte_carlo_ticker_picker_seeds.jl 50         # override N_SEEDS
#
# Output:
#   1) Per-seed progress lines + summary stats table to stdout
#   2) Raw per-seed records saved to scripts/monte_carlo_ticker_picker_results.jld2
#      so the user can re-aggregate or plot offline without rerunning the sweep.
#
# Runtime: roughly 30-60 seconds per seed on a modern laptop (no GPU). 20 seeds
# is ~10-20 minutes; 50 seeds is ~25-50 minutes. Reduce EPISODES or K below to
# trim runtime at the cost of less-converged DQNs.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using eCornellAIFinance
using Dates, DataFrames, Flux, JLD2, Printf, PrettyTables, Random, Statistics

# ===== CONFIG (mirrors the notebook constants) =====
const N_SEEDS_DEFAULT     = 100
const SEED_BASE           = 1000

const B0                  = 10_000.0
const Δt                  = 1.0 / 252.0
const L_short             = 21
const L_long              = 63
const L_growth            = 10
const GAIN                = 10.0
const BANDIT_DECISION_DATE = "2025-01-02"
const BANDIT_GM_WINDOW    = 63
const BANDIT_EPSILON      = 0.1
const TRIGGER_MAX_DRAWDOWN = 0.15
const TRIGGER_MAX_TURNOVER = 0.50

const K_BASKET            = 8

const HIDDEN              = 128
const BUFFER_CAPACITY     = 8_000
const WARMUP              = 400
const MINIBATCH           = 128
const TARGET_SYNC         = 100
const DISCOUNT            = 0.95f0
const LR                  = 5.0f-4
const EPISODES            = 1600
const EPS_FLOOR           = 0.05f0
const BANDIT_ITERS        = EPISODES * K_BASKET
const BANDIT_ALPHA        = 0.1

# ===== HELPERS (mirror the notebook's Implementation cells) =====

"""
    compute_basket_utility(mask, ctx) -> Float64

Signed log Cobb-Douglas utility on a basket mask. See notebook docs for the
overflow-resistant Ũ = sign(U) · log|U| convention.
"""
function compute_basket_utility(mask, ctx)
    action = Int.(mask .> 0.5f0)
    sum(action) == 0 && return 0.0
    bandit_ctx = build(MyBanditContext, (
        tickers = ctx.tickers, sim_parameters = ctx.sim_parameters,
        prices = ctx.prices, B = ctx.B,
        gm_t = ctx.gm_t, lambda = ctx.lambda, epsilon = ctx.epsilon,
    ))
    (_, n, γ) = eCornellAIFinance.bandit_world(action, bandit_ctx)
    S = findall(==(1), action)
    sign_U = any(γ[i] < 0.0 for i in S) ? -1.0 : 1.0
    log_abs_U = 0.0
    @inbounds for s in S
        n[s] > 0.0 && (log_abs_U += γ[s] * log(n[s]))
    end
    return sign_U * log_abs_U
end

function ticker_picker_world(state, action::Int, ctx)
    s_next = copy(state)
    s_next[action] = 1.0f0
    basket_size = Int(round(sum(s_next)))
    done = (basket_size >= ctx.K_basket)
    r = done ? Float32(compute_basket_utility(s_next, ctx)) : 0.0f0
    return (s_next, r, done)
end

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

make_qnet(K::Int, H::Int) = Chain(
    Dense(K, H, relu),
    Dense(H, H, relu),
    Dense(H, K),
)

function select_action(qnet, state::Vector{Float32}, ε::Real, K::Int)
    available = findall(==(0.0f0), state)
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

function train_dqn(ctx; episodes, hidden, lr, buffer_cap, warmup,
        batch_size, sync_freq, γ_disc, ε_floor)
    K = length(ctx.tickers)
    main   = make_qnet(K, hidden)
    target = make_qnet(K, hidden)
    Flux.loadmodel!(target, Flux.state(main))
    opt_state = Flux.setup(Adam(lr), main)
    buf = MyReplayBuffer(buffer_cap)
    global_step = 0
    for _ in 1:episodes
        s = zeros(Float32, K)
        while true
            global_step += 1; t = global_step
            ε_raw = min(1.0f0, Float32(t)^(-1/3) * (Float32(K) * log(t + 1))^(1/3))
            ε = max(ε_floor, ε_raw)
            a = select_action(main, s, ε, K)
            s′, r, done = ticker_picker_world(s, a, ctx)
            push_transition!(buf, s, a, r, s′, done)
            if length(buf.states) >= warmup
                dqn_train_step!(main, target, opt_state, buf, batch_size, γ_disc, K)
                global_step % sync_freq == 0 && Flux.loadmodel!(target, Flux.state(main))
            end
            s = s′
            done && break
        end
    end
    return main
end

function solve_bandit_sparse(K::Int, K_basket::Int, n_iters::Int, ctx;
        ε_floor::Float64 = 0.05, α::Float64 = 0.1)
    arm_mean  = Dict{Vector{Int},Float64}()
    arm_count = Dict{Vector{Int},Int}()
    random_basket() = sort!(randperm(K)[1:K_basket])
    for t in 1:n_iters
        ε = max(ε_floor, t > 1 ? min(1.0, t^(-1.0/3.0) * (K_basket * log(t))^(1.0/3.0)) : 1.0)
        basket = if rand() < ε || isempty(arm_mean)
            random_basket()
        else
            argmax(arm_mean)
        end
        mask = zeros(Float32, K)
        for i in basket
            mask[i] = 1.0f0
        end
        U = compute_basket_utility(mask, ctx)
        c = get(arm_count, basket, 0) + 1
        arm_count[basket] = c
        old = get(arm_mean, basket, 0.0)
        lr = α > 0.0 ? α : 1.0 / c
        arm_mean[basket] = old + lr * (U - old)
    end
    best_basket = argmax(arm_mean)
    best_action = zeros(Int, K)
    for i in best_basket
        best_action[i] = 1
    end
    best_mask = Float32.(best_action)
    best_utility = compute_basket_utility(best_mask, ctx)
    return (best_basket = best_basket, best_utility = best_utility,
            n_unique_arms = length(arm_mean))
end

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

function fwd_cd(basket::Vector{String}, rules, my_tickers, sim_params,
        forward_price_matrix, forward_lambda, forward_gm_ema)
    sel_idx = findall(in(basket), my_tickers)
    sel_sim = Dict(t => sim_params[t] for t in basket)
    sel_p   = forward_price_matrix[:, vcat([1], sel_idx .+ 1)]
    ctx = build(MyRebalancingContextModel, (
        B = B0, tickers = basket, marketdata = sel_p,
        marketfactor = forward_gm_ema, sim_parameters = sel_sim,
        lambda = forward_lambda[1], Δt = Δt, epsilon = 0.1,
    ))
    res = run_rebalancing_engine(ctx, rules, forward_lambda; offset = 1, allocator = :cobb_douglas)
    return compute_wealth_series(res, sel_p, basket; offset = 1)
end

"""
    compute_full_universe_metrics(env) -> NamedTuple

Run the daily Cobb-Douglas rebalancing engine on the full K-ticker universe
(no picker, no basket selection) and return realized_metrics. This is the
deterministic baseline; γ is computed from frozen SIM and the realized
forward path, so there is no random seed dependence.
"""
function compute_full_universe_metrics(env)
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

# ===== LOADER (one-shot, identical to notebook's loader cell) =====

function load_environment()
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
    spy_full  = vcat(train_spy, test_spy); sort!(spy_full, :timestamp); unique!(spy_full, :timestamp)
    target = Date(BANDIT_DECISION_DATE)
    t_idx = findfirst(==(target), Date.(spy_full.timestamp))
    isnothing(t_idx) && error("BANDIT_DECISION_DATE=$(BANDIT_DECISION_DATE) not in SPY history")

    spy_history = spy_full.close[1:t_idx]
    λ_real      = compute_lambda(
        compute_ema(spy_history; window = L_short),
        compute_ema(spy_history; window = L_long); G = GAIN)
    gm_ema_real = compute_ema(
        compute_market_growth(spy_history; Δt = Δt); window = BANDIT_GM_WINDOW)
    bandit_λ    = λ_real[end]
    bandit_gm_t = gm_ema_real[end]

    test_ds = MyExtendedTestingMarketDataSet()["dataset"]
    fwd_dates_full = spy_full.timestamp[t_idx:end]
    target_dates_fwd = Date.(fwd_dates_full)
    n_fwd = length(fwd_dates_full)

    keep = String[]
    for t in sim_tickers
        haskey(test_ds, t) || continue
        df = test_ds[t]
        df_dates = Date.(df.timestamp)
        ok_decision = !isnothing(findfirst(==(target), df_dates))
        ok_decision || continue
        ok_fwd = all(d -> !isnothing(findfirst(==(d), df_dates)), target_dates_fwd)
        ok_fwd && push!(keep, t)
    end
    my_tickers = sort(keep)
    sim_params = Dict(t => sim_full[t] for t in my_tickers)
    K = length(my_tickers)

    bandit_prices = zeros(K)
    for (k, t) in enumerate(my_tickers)
        df = test_ds[t]
        i  = findfirst(==(target), Date.(df.timestamp))
        bandit_prices[k] = df.close[i]
    end

    λ_full      = compute_lambda(
        compute_ema(spy_full.close; window = L_short),
        compute_ema(spy_full.close; window = L_long); G = GAIN)
    gm_full     = compute_market_growth(spy_full.close; Δt = Δt)
    gm_ema_full = compute_ema(gm_full; window = L_growth)

    forward_lambda = λ_full[t_idx:end]
    forward_gm_ema = gm_ema_full[t_idx-1:end]

    forward_price_matrix = zeros(n_fwd, K + 1)
    forward_price_matrix[:, 1] = 1:n_fwd
    for (k, t) in enumerate(my_tickers)
        df = test_ds[t]; df_dates = Date.(df.timestamp)
        for (day, d) in enumerate(target_dates_fwd)
            i = findfirst(==(d), df_dates)
            forward_price_matrix[day, k + 1] = df.close[i]
        end
    end

    return (my_tickers = my_tickers, sim_params = sim_params, g_f = g_f, K = K,
            bandit_λ = bandit_λ, bandit_gm_t = bandit_gm_t, bandit_prices = bandit_prices,
            forward_price_matrix = forward_price_matrix,
            forward_lambda = forward_lambda, forward_gm_ema = forward_gm_ema,
            n_fwd = n_fwd)
end

# ===== ONE SEED =====

"""
    one_run(seed, env) -> NamedTuple

Run one full pipeline at the given seed: train DQN, run sparse bandit, walk
both baskets forward through the Cobb-Douglas engine, return realized metrics
plus the chosen baskets and day-1 utilities.
"""
function one_run(seed::Int, env)
    Random.seed!(seed)

    picker_ctx = (tickers = env.my_tickers, sim_parameters = env.sim_params,
                  prices = env.bandit_prices, B = B0,
                  gm_t = env.bandit_gm_t, lambda = env.bandit_λ,
                  epsilon = BANDIT_EPSILON, K_basket = K_BASKET)

    # --- DQN: train, then greedy-rollout one basket ---
    dqn_main = train_dqn(picker_ctx;
        episodes = EPISODES, hidden = HIDDEN, lr = LR,
        buffer_cap = BUFFER_CAPACITY, warmup = WARMUP, batch_size = MINIBATCH,
        sync_freq = TARGET_SYNC, γ_disc = DISCOUNT, ε_floor = EPS_FLOOR)
    s_g = zeros(Float32, env.K); picked = Int[]
    for _ in 1:K_BASKET
        a = select_action(dqn_main, s_g, 0.0f0, env.K)
        push!(picked, a); s_g[a] = 1.0f0
    end
    dqn_basket = env.my_tickers[picked]
    dqn_U = compute_basket_utility(s_g, picker_ctx)

    # --- Bandit: sparse-Dict ε-greedy at matched training-sample budget ---
    bres = solve_bandit_sparse(env.K, K_BASKET, BANDIT_ITERS, picker_ctx;
        ε_floor = Float64(EPS_FLOOR), α = BANDIT_ALPHA)
    bandit_basket = env.my_tickers[bres.best_basket]
    bandit_U = bres.best_utility

    # --- Forward Cobb-Douglas walk for each picker ---
    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN, max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1)
    ))
    W_dqn = fwd_cd(dqn_basket, rules, env.my_tickers, env.sim_params,
                   env.forward_price_matrix, env.forward_lambda, env.forward_gm_ema)
    W_bnd = fwd_cd(bandit_basket, rules, env.my_tickers, env.sim_params,
                   env.forward_price_matrix, env.forward_lambda, env.forward_gm_ema)
    m_dqn = realized_metrics(W_dqn, env.g_f)
    m_bnd = realized_metrics(W_bnd, env.g_f)

    return (
        seed = seed,
        dqn_basket = dqn_basket, dqn_U = dqn_U,
        dqn_W_T_over_W0 = m_dqn.W_T_over_W0, dqn_max_dd = m_dqn.max_dd,
        dqn_ann_ret = m_dqn.ann_ret, dqn_sharpe = m_dqn.sharpe,
        bandit_basket = bandit_basket, bandit_U = bandit_U,
        bandit_W_T_over_W0 = m_bnd.W_T_over_W0, bandit_max_dd = m_bnd.max_dd,
        bandit_ann_ret = m_bnd.ann_ret, bandit_sharpe = m_bnd.sharpe,
    )
end

# ===== AGGREGATE & PRINT =====

function build_summary_table(results, full_metrics)
    metrics = [
        ("W_T / W_0",     :W_T_over_W0, x -> round(x, digits = 3)),
        ("Ann. return %", :ann_ret,     x -> round(x * 100, digits = 2)),
        ("Max DD %",      :max_dd,      x -> round(x * 100, digits = 1)),
        ("Sharpe",        :sharpe,      x -> round(x, digits = 3)),
        ("Day-1 U_tilde", :U,           x -> round(x, digits = 3)),
    ]

    rows = Any[]
    for (label, base_key, fmt) in metrics
        for picker in ("DQN", "Bandit")
            prefix = lowercase(picker)
            key = Symbol("$(prefix)_$(base_key)")
            vals = [getproperty(r, key) for r in results]
            push!(rows, (
                Metric = label,
                Picker = picker,
                Min    = fmt(minimum(vals)),
                Q25    = fmt(quantile(vals, 0.25)),
                Median = fmt(median(vals)),
                Mean   = fmt(mean(vals)),
                Q75    = fmt(quantile(vals, 0.75)),
                Max    = fmt(maximum(vals)),
                StdDev = fmt(std(vals)),
            ))
        end
        # --- Full-Universe baseline row (deterministic; same value in every column) ---
        # Day-1 Ũ does not apply to Full-Universe (it does not pick a basket), so we
        # skip that row for FullCD rather than report a misleading value.
        if base_key !== :U
            full_val = fmt(getproperty(full_metrics, base_key))
            zero_std = fmt(0.0)
            push!(rows, (
                Metric = label,
                Picker = "FullCD",
                Min    = full_val, Q25 = full_val, Median = full_val,
                Mean   = full_val, Q75 = full_val, Max    = full_val,
                StdDev = zero_std,
            ))
        end
    end
    return DataFrame(rows)
end

function head_to_head(results, full_metrics)
    n = length(results)
    dqn_wins = count(r -> r.dqn_W_T_over_W0  >  r.bandit_W_T_over_W0, results)
    bnd_wins = count(r -> r.bandit_W_T_over_W0 > r.dqn_W_T_over_W0,  results)
    ties     = n - dqn_wins - bnd_wins
    dqn_trip = count(r -> r.dqn_max_dd    >= TRIGGER_MAX_DRAWDOWN, results)
    bnd_trip = count(r -> r.bandit_max_dd >= TRIGGER_MAX_DRAWDOWN, results)
    full_W   = full_metrics.W_T_over_W0
    dqn_beats_full = count(r -> r.dqn_W_T_over_W0    > full_W, results)
    bnd_beats_full = count(r -> r.bandit_W_T_over_W0 > full_W, results)
    return (n = n, dqn_wins = dqn_wins, bandit_wins = bnd_wins, ties = ties,
            dqn_trips = dqn_trip, bandit_trips = bnd_trip,
            dqn_beats_full = dqn_beats_full, bandit_beats_full = bnd_beats_full,
            full_W_T_over_W0 = full_W)
end

# ===== MAIN =====

function main(N_SEEDS::Int = N_SEEDS_DEFAULT)
    println("Loading universe + forward arrays ...")
    env = load_environment()
    @printf "  K = %d tickers, decision day = %s, forward window = %d trading days\n" env.K BANDIT_DECISION_DATE env.n_fwd
    @printf "  Day-1 inputs: lambda = %.3f, gm_t = %.3f /yr\n" env.bandit_λ env.bandit_gm_t
    println()

    # --- Deterministic Full-Universe baseline (one run, no seed dependence) ---
    println("Running Full-Universe CD baseline (deterministic, no seed dependence) ...")
    full_metrics = compute_full_universe_metrics(env)
    @printf "  W_T/W0 = %.3f  ann_ret = %+6.2f%%  max_dd = %5.2f%%  sharpe = %+6.3f\n" full_metrics.W_T_over_W0 full_metrics.ann_ret*100 full_metrics.max_dd*100 full_metrics.sharpe
    println()

    @printf "Sweeping %d seeds, K_BASKET = %d, EPISODES = %d, BANDIT_ITERS = %d\n" N_SEEDS K_BASKET EPISODES BANDIT_ITERS
    println()

    results = NamedTuple[]
    t_start = time()
    for i in 1:N_SEEDS
        seed = SEED_BASE + i
        r = one_run(seed, env)
        push!(results, r)
        elapsed = time() - t_start
        rate = i / elapsed
        eta = (N_SEEDS - i) / rate
        @printf("  seed %3d/%d (#%d):  DQN W_T/W0=%.3f  U=%+.3f  |  Bandit W_T/W0=%.3f  U=%+.3f  |  elapsed=%5.0fs  ETA=%5.0fs\n",
            i, N_SEEDS, seed,
            r.dqn_W_T_over_W0, r.dqn_U,
            r.bandit_W_T_over_W0, r.bandit_U,
            elapsed, eta)
        flush(stdout)
    end

    println()
    println("=" ^ 80)
    @printf "Summary across %d seeds (FullCD row is deterministic, std=0)\n" N_SEEDS
    println("=" ^ 80)
    df = build_summary_table(results, full_metrics)
    pretty_table(df; backend = :text,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        table_format = TextTableFormat(borders = text_table_borders__compact))

    h2h = head_to_head(results, full_metrics)
    println()
    println("Head-to-head (W_T comparison):")
    @printf "  DQN beats Bandit                : %d / %d (%.0f%%)\n" h2h.dqn_wins h2h.n 100*h2h.dqn_wins/h2h.n
    @printf "  Bandit beats DQN                : %d / %d (%.0f%%)\n" h2h.bandit_wins h2h.n 100*h2h.bandit_wins/h2h.n
    @printf "  Ties                            : %d / %d\n" h2h.ties h2h.n
    @printf "  DQN beats Full-Universe         : %d / %d (%.0f%%)   [Full = %.3f]\n" h2h.dqn_beats_full h2h.n 100*h2h.dqn_beats_full/h2h.n h2h.full_W_T_over_W0
    @printf "  Bandit beats Full-Universe      : %d / %d (%.0f%%)\n" h2h.bandit_beats_full h2h.n 100*h2h.bandit_beats_full/h2h.n
    println()
    println("Drawdown trigger trips (Max DD >= $(TRIGGER_MAX_DRAWDOWN * 100)%):")
    @printf "  DQN baskets that tripped     : %d / %d (%.0f%%)\n" h2h.dqn_trips h2h.n 100*h2h.dqn_trips/h2h.n
    @printf "  Bandit baskets that tripped  : %d / %d (%.0f%%)\n" h2h.bandit_trips h2h.n 100*h2h.bandit_trips/h2h.n
    @printf "  Full-Universe tripped        : %s\n" (full_metrics.max_dd >= TRIGGER_MAX_DRAWDOWN ? "yes" : "no")

    out_path = joinpath(@__DIR__, "monte_carlo_ticker_picker_results.jld2")
    save(out_path, Dict(
        "results"        => results,
        "summary_df"     => df,
        "head_to_head"   => h2h,
        "full_metrics"   => full_metrics,
        "config"         => Dict(
            "N_SEEDS" => N_SEEDS, "SEED_BASE" => SEED_BASE,
            "K_BASKET" => K_BASKET, "EPISODES" => EPISODES,
            "BANDIT_ITERS" => BANDIT_ITERS, "HIDDEN" => HIDDEN,
            "BANDIT_DECISION_DATE" => BANDIT_DECISION_DATE,
            "TRIGGER_MAX_DRAWDOWN" => TRIGGER_MAX_DRAWDOWN,
        ),
    ))
    println()
    @printf "Per-seed records + summary saved to: %s\n" out_path
end

# ===== ENTRY =====

let
    N_SEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : N_SEEDS_DEFAULT
    main(N_SEEDS)
end

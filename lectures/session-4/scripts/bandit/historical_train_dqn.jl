#!/usr/bin/env julia
# historical_train_dqn.jl
#
# Train the ticker-picker DQN on 2014-2024 historical data with realized
# 21-day forward returns as the reward, then freeze the network and deploy
# it on the 2025-2026 forward window. This is the walk-forward, supervised-
# style alternative to the single-decision-day notebook setup.
#
# Why this is different from the notebook + monte_carlo_ticker_picker_seeds.jl:
#
# - The notebook trains the DQN on a SINGLE decision day with the day-1
#   signed log Cobb-Douglas utility as the reward. That reward is bounded,
#   regime-dependent, and weakly correlated with realized forward returns
#   (Goodhart's law: the optimized metric is not the test metric).
#
# - This script trains on MANY decision days sampled from 2014-2024, and
#   the reward at each terminal step is the realized 21-trading-day
#   buy-and-hold log return of the constructed basket starting from that
#   sampled day. The picker is now optimizing the actual quantity we test on.
#
# - The state is FEATURIZED: state = [basket_mask; γ_today] of size 2K so
#   the network can condition picks on the current regime. Without γ in the
#   state, the network has no idea what day it is making a decision for.
#
# Usage:
#   cd lectures/session-4
#   julia --project=. scripts/bandit/historical_train_dqn.jl                # default config
#   julia --project=. scripts/bandit/historical_train_dqn.jl 5000 21        # episodes, horizon
#
# Output:
#   1) Training progress lines
#   2) Test-day deployment scorecard (DQN basket vs Full-Universe baseline) on the
#      2025-2026 forward window, walked through the daily Cobb-Douglas engine
#   3) Trained model + metadata saved to scripts/historical_train_dqn_results.jld2

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using eCornellAIFinance
using Dates, DataFrames, Flux, JLD2, Printf, PrettyTables, Random, Statistics

# ===== CONFIG =====
const TRAINING_EPISODES_DEFAULT = 5_000     # total DQN episodes during training
const FORWARD_HORIZON_DEFAULT   = 21        # trading days the reward looks ahead
const TRAIN_END_DATE            = "2024-12-31"
const TEST_DATE                 = "2025-01-02"   # deploy day
const WARMUP_OFFSET             = 84             # skip the first ~quarter of training history for EMA warmup

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
const K_BASKET_BANDIT     = 32             # apples-to-apples: same as DQN
const BANDIT_ITERS_DEFAULT = 5_000          # bandit iterations during historical training
const BANDIT_ALPHA        = 0.1             # bandit learning rate for arm-mean updates

# DQN hyperparameters
const HIDDEN              = 256        # wider than the notebook's 128 to handle the 2K-input featurized state
const BUFFER_CAPACITY     = 50_000
const WARMUP_BUFFER       = 1_000
const MINIBATCH           = 128
const TARGET_SYNC         = 200
const DISCOUNT            = 0.95f0
const LR                  = 5.0f-4
const EPS_FLOOR           = 0.05f0

const SEED_DEFAULT        = 2026             # override on the CLI to test seed sensitivity

# ===== HELPERS =====

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

"""
    make_qnet_featurized(state_dim, action_dim, hidden) -> Flux.Chain

Q-network for the featurized MDP. Input is the concatenated state
[basket_mask (size K); γ_today (size K)] of length 2K. Output is one
Q-value per ticker, length K.
"""
make_qnet_featurized(state_dim::Int, action_dim::Int, hidden::Int) = Chain(
    Dense(state_dim, hidden, relu),
    Dense(hidden, hidden, relu),
    Dense(hidden, action_dim),
)

"""
    select_action_featurized(qnet, state, ε, K) -> Int

ε-greedy action selection. The basket-mask occupies state[1:K]; we mask
out indices where the mask is 1 (already picked) before argmax.
"""
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

"""
    build_state_featurized(mask::Vector{Float32}, γ::Vector{Float32}) -> Vector{Float32}

Concatenate [basket_mask; γ_today] into a single state vector of length 2K.
The mask is the picker's current basket; γ_today is the day's preference
weights for all K tickers under the SIM regime lens.
"""
build_state_featurized(mask::Vector{Float32}, γ::Vector{Float32}) = vcat(mask, γ)

"""
    cd_allocate_basket_at_day(basket_indices, day, env) -> (n::Vector{Float64}, S_indices_in_basket::Vector{Int})

Run the canonical INFORMS-form Cobb-Douglas allocation on a basket of ticker
indices at the given calendar day index, using budget B0 and the day's γ
values. Returns share counts (length = number of tickers in the basket).
"""
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

"""
    realized_basket_return(basket_indices, day, horizon, env) -> Float32

Buy-and-hold realized log return of the Cobb-Douglas-allocated basket from
`day` to `day + horizon`, both indices into the training price matrix.
This is the reward signal for the featurized DQN: the actual realized
return on a known historical 21-day window. No SIM, no expected utility,
no Goodhart.

We use buy-and-hold (not daily rebalancing) inside the 21-day window for
training speed. The qualitative reward signal is the same; daily rebalancing
adds at most a few bps over a 21-day horizon at typical turnover.
"""
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

"""
    realized_picker_world(state, action, day, env, K, K_basket, horizon)
        -> (s_next, r, done)

One MDP step in the featurized historical-training MDP. State is
[mask; γ_today] of length 2K. Action adds the chosen ticker to the mask.
Intermediate reward is 0; terminal reward (when basket is full) is the
realized 21-day forward log return of the constructed basket.
"""
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

# ===== LOADER =====

"""
    load_environment(; horizon) -> NamedTuple

Build the training + test environment. Returns:

- `my_tickers::Vector{String}`: the K-ticker universe (intersection of SIM
  calibration and full forward OHLC coverage, same filter as the notebook)
- `sim_params::Dict`: per-ticker (α, β, σ_ε)
- `K::Int`: number of tickers
- `train_price_matrix::Matrix{Float64}`: price matrix on the 2014-2024 timeline,
  shape (N_train, K). Column j is the close price for `my_tickers[j]`.
- `train_dates::Vector{Date}`: aligned with rows of train_price_matrix
- `train_lambda::Vector{Float64}`: λ_t at each training day
- `train_gm_ema::Vector{Float64}`: gm_t (BANDIT_GM_WINDOW EMA) at each training day
- `γ_matrix::Matrix{Float64}`: shape (N_train, K). Pre-computed γ values for every
  (day, ticker) pair using the day's λ and gm_t. Cached so we don't recompute
  during training.
- `train_offset::Int`: index of first usable training day (after EMA warmup)
- `train_last::Int`: last usable training day (such that day + horizon stays in 2024)
- `test_decision_index::Int`: index of TEST_DATE in the COMBINED 2014-2026 timeline
- Forward arrays for the 2025-2026 deployment (same shape/semantics as the notebook):
  `forward_price_matrix`, `forward_lambda`, `forward_gm_ema`, `forward_dates`, `n_fwd`
- `g_f::Float64`: risk-free rate
"""
function load_environment(; horizon::Int = FORWARD_HORIZON_DEFAULT)
    # --- Load SIM calibration ---
    calib = MySIMCalibration()
    sim_tickers = calib["tickers"]::Vector{String}
    α_vec = calib["alpha"]::Vector{Float64}
    β_vec = calib["beta"]::Vector{Float64}
    σ_vec = calib["sigma_eps"]::Vector{Float64}
    sim_full = Dict{String,Tuple{Float64,Float64,Float64}}(
        sim_tickers[i] => (α_vec[i], β_vec[i], σ_vec[i]) for i in eachindex(sim_tickers))
    g_f = 0.045

    # --- Load OHLC for SPY across the full timeline ---
    train_spy = MyTrainingMarketDataSet()["dataset"]["SPY"]
    test_spy  = MyExtendedTestingMarketDataSet()["dataset"]["SPY"]
    spy_full  = vcat(train_spy, test_spy)
    sort!(spy_full, :timestamp); unique!(spy_full, :timestamp)

    # --- Build the COMBINED date timeline ---
    test_target = Date(TEST_DATE)
    test_idx_full = findfirst(==(test_target), Date.(spy_full.timestamp))
    isnothing(test_idx_full) && error("TEST_DATE=$(TEST_DATE) not in SPY history")

    # --- 2014-2024 training window in the combined timeline ---
    train_end_target = Date(TRAIN_END_DATE)
    train_end_idx = findlast(d -> Date(d) <= train_end_target, spy_full.timestamp)
    isnothing(train_end_idx) && error("TRAIN_END_DATE=$(TRAIN_END_DATE) not found")

    # --- Filter universe: SIM tickers with full OHLC coverage on every forward day
    # AND every training day. ---
    train_test_ds = MyTrainingMarketDataSet()["dataset"]
    test_ds = MyExtendedTestingMarketDataSet()["dataset"]

    # all dates in the combined window 2014-2026 we need coverage for
    full_dates = Date.(spy_full.timestamp)
    target_dates_fwd = full_dates[test_idx_full:end]
    target_dates_train = full_dates[1:train_end_idx]

    keep = String[]
    for t in sim_tickers
        # Need 2014-2024 coverage (training) AND 2025-2026 coverage (deployment).
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

    # --- Build the COMBINED price matrix indexed [day, ticker] over 2014-2026 ---
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
                    full_price_matrix[i, j] = NaN  # should not happen given filter
                end
            end
        end
    end

    # --- λ_t and gm_t time series across the full timeline ---
    spy_close = Float64.(spy_full.close)
    ema_s_full = compute_ema(spy_close; window = L_short)
    ema_l_full = compute_ema(spy_close; window = L_long)
    λ_full     = compute_lambda(ema_s_full, ema_l_full; G = GAIN)
    gm_full    = compute_market_growth(spy_close; Δt = Δt)
    gm_ema_picker_full = compute_ema(gm_full; window = BANDIT_GM_WINDOW)
    gm_ema_engine_full = compute_ema(gm_full; window = L_growth)

    # The compute_market_growth result has length n_full - 1; pad the front so the
    # gm_ema arrays line up with the price matrix indices.
    gm_ema_picker_aligned = vcat([gm_ema_picker_full[1]], gm_ema_picker_full)
    gm_ema_engine_aligned = vcat([gm_ema_engine_full[1]], gm_ema_engine_full)

    # --- Pre-compute γ matrix [day, ticker] for the training portion ---
    train_offset = WARMUP_OFFSET + 1
    train_last   = train_end_idx - horizon  # need horizon-day forward window inside 2024
    train_last >= train_offset || error("training window too short for horizon $(horizon)")
    γ_matrix = zeros(n_full, K)
    for d in train_offset:train_last
        gm_d = gm_ema_picker_aligned[d]
        λ_d  = λ_full[d]
        γ_d  = compute_preference_weights(sim_params, my_tickers, gm_d, λ_d)
        γ_matrix[d, :] = γ_d
    end

    # --- Test/deploy decision-day γ + forward arrays for the 2025-2026 window ---
    test_idx = test_idx_full
    γ_test = compute_preference_weights(sim_params, my_tickers,
        gm_ema_picker_aligned[test_idx], λ_full[test_idx])

    # forward window matches the notebook
    n_fwd = n_full - test_idx + 1
    forward_dates  = full_dates[test_idx:end]
    forward_lambda = λ_full[test_idx:end]
    forward_gm_ema = gm_ema_engine_aligned[test_idx:end]
    # forward_price_matrix uses the engine's expected shape: column 1 is a day index,
    # columns 2..K+1 are the tickers
    forward_price_matrix = zeros(n_fwd, K + 1)
    forward_price_matrix[:, 1] = 1:n_fwd
    forward_price_matrix[:, 2:end] = full_price_matrix[test_idx:end, :]

    return (my_tickers = my_tickers, sim_params = sim_params, K = K, g_f = g_f,
            train_price_matrix = full_price_matrix,  # share storage; train_offset:train_last is the usable range
            train_dates = full_dates,
            train_lambda = λ_full, train_gm_ema = gm_ema_picker_aligned,
            γ_matrix = γ_matrix,
            train_offset = train_offset, train_last = train_last,
            test_decision_index = test_idx,
            γ_test = γ_test,
            forward_dates = forward_dates,
            forward_price_matrix = forward_price_matrix,
            forward_lambda = forward_lambda,
            forward_gm_ema = forward_gm_ema,
            n_fwd = n_fwd,
            horizon = horizon)
end

# ===== TRAIN =====

"""
    train_dqn_historical(env; episodes, horizon, ...) -> (main, history)

Train the featurized DQN on randomly-sampled historical decision days. Each
episode samples a day d ∈ [train_offset, train_last], builds a basket via
ε-greedy action selection on the day-d state, and on the terminal step
receives the realized 21-day buy-and-hold log return as the reward.

Returns the trained main network and a NamedTuple with per-episode rewards
and the days sampled.
"""
function train_dqn_historical(env;
        episodes::Int = TRAINING_EPISODES_DEFAULT,
        horizon::Int  = FORWARD_HORIZON_DEFAULT,
        K_basket::Int = K_BASKET_DQN,
        hidden::Int   = HIDDEN, lr::Float32 = LR,
        buffer_cap::Int = BUFFER_CAPACITY, warmup::Int = WARMUP_BUFFER,
        batch_size::Int = MINIBATCH, sync_freq::Int = TARGET_SYNC,
        γ_disc::Float32 = DISCOUNT, ε_floor::Float32 = EPS_FLOOR,
        report_every::Int = 200)

    K = env.K
    state_dim = 2 * K
    main   = make_qnet_featurized(state_dim, K, hidden)
    target = make_qnet_featurized(state_dim, K, hidden)
    Flux.loadmodel!(target, Flux.state(main))
    opt_state = Flux.setup(Adam(lr), main)
    buf = MyReplayBuffer(buffer_cap)

    rewards_per_ep = Float32[]
    days_sampled   = Int[]
    global_step = 0
    t_start = time()

    for ep in 1:episodes
        # --- Sample a training day uniformly within the usable training window ---
        d = rand(env.train_offset:env.train_last)
        push!(days_sampled, d)

        # --- Build the day-d state context ---
        γ_d = Float32.(env.γ_matrix[d, :])
        s = build_state_featurized(zeros(Float32, K), γ_d)

        # --- Run the basket-construction MDP ---
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

        if ep % report_every == 0 || ep == episodes
            recent = rewards_per_ep[max(1, end - report_every + 1):end]
            elapsed = time() - t_start
            @printf("  ep %5d/%d  recent mean reward = %+7.4f  buffer = %5d  elapsed = %5.0fs\n",
                ep, episodes, mean(recent), length(buf.states), elapsed)
            flush(stdout)
        end
    end

    return main, (rewards_per_ep = rewards_per_ep, days_sampled = days_sampled)
end

# ===== DEPLOY (Test) =====

"""
    deploy_dqn(env, dqn_main; K_basket) -> (basket_tickers, basket_indices)

Greedy rollout of the trained DQN at the TEST_DATE (deploy day). Returns
the basket indices and the corresponding ticker symbols.
"""
function deploy_dqn(env, dqn_main; K_basket::Int = K_BASKET_DQN)
    K = env.K
    γ_test = Float32.(env.γ_test)
    s = build_state_featurized(zeros(Float32, K), γ_test)
    picked = Int[]
    for _ in 1:K_basket
        a = select_action_featurized(dqn_main, s, 0.0f0, K)
        push!(picked, a)
        s[a] = 1.0f0  # update the mask portion of the state
    end
    return env.my_tickers[picked], picked
end

"""
    train_bandit_historical(env; K_basket, n_iters, horizon, ε_floor, α) -> NamedTuple

Sparse-Dict ε-greedy bandit trained on the same realized 21-day forward
return signal as the DQN. Each iteration samples a basket (random size-
K_basket subset on explore, best-mean-so-far on exploit) AND a random
training day, evaluates the realized 21-day forward return on that day,
and updates the running mean for that basket.

Unlike the DQN, the bandit has NO regime context. It picks one static
basket whose expected reward across the training-day distribution is
highest. At deploy time it has no way to condition on the deploy day's
regime; it just deploys the single best-mean basket.

Returns a NamedTuple with `best_basket::Vector{Int}` (ticker indices),
`best_mean::Float64` (its average training reward), `n_unique_arms::Int`,
and `reward_history::Vector{Float64}` (per-iteration rewards for plotting).
"""
function train_bandit_historical(env;
        K_basket::Int = K_BASKET_BANDIT,
        n_iters::Int  = BANDIT_ITERS_DEFAULT,
        horizon::Int  = FORWARD_HORIZON_DEFAULT,
        ε_floor::Float64 = Float64(EPS_FLOOR),
        α::Float64    = BANDIT_ALPHA,
        report_every::Int = 500)
    K = env.K
    arm_mean  = Dict{Vector{Int},Float64}()
    arm_count = Dict{Vector{Int},Int}()
    rewards   = zeros(Float64, n_iters)
    random_basket() = sort!(randperm(K)[1:K_basket])
    t_start = time()
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
        if t % report_every == 0 || t == n_iters
            recent = rewards[max(1, t - report_every + 1):t]
            elapsed = time() - t_start
            @printf("  iter %5d/%d  recent mean reward = %+7.4f  unique arms = %5d  elapsed = %4.0fs\n",
                t, n_iters, mean(recent), length(arm_mean), elapsed)
            flush(stdout)
        end
    end
    best_basket = argmax(arm_mean)
    best_mean   = arm_mean[best_basket]
    return (best_basket = best_basket, best_mean = best_mean,
            n_unique_arms = length(arm_mean), reward_history = rewards)
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

# ===== MAIN =====

function main(episodes::Int = TRAINING_EPISODES_DEFAULT,
              horizon::Int  = FORWARD_HORIZON_DEFAULT,
              bandit_iters::Int = BANDIT_ITERS_DEFAULT,
              seed::Int = SEED_DEFAULT)
    Random.seed!(seed)
    @printf("Seed: %d (override via 4th CLI arg to test seed sensitivity)\n", seed)

    println("Loading environment ...")
    env = load_environment(; horizon = horizon)
    @printf("  K = %d tickers; train window indices [%d, %d] (~%d usable days)\n",
        env.K, env.train_offset, env.train_last, env.train_last - env.train_offset + 1)
    @printf("  test deploy day = %s (timeline index %d), forward window = %d days\n",
        TEST_DATE, env.test_decision_index, env.n_fwd)
    @printf("  reward horizon = %d trading days\n", horizon)
    @printf("  DQN: episodes = %d, K_basket = %d (regime-conditional)\n",
        episodes, K_BASKET_DQN)
    @printf("  Bandit: iters = %d, K_basket = %d (regime-blind, single static basket)\n",
        bandit_iters, K_BASKET_BANDIT)
    println()

    # --- Train DQN on historical decision days ---
    println("Training DQN on historical decision days (this will take a while) ...")
    dqn_main, dqn_hist = train_dqn_historical(env;
        episodes = episodes, horizon = horizon, K_basket = K_BASKET_DQN)
    println()

    # --- Train sparse bandit on the same realized-return signal ---
    println("Training sparse bandit on historical decision days ...")
    bandit_res = train_bandit_historical(env;
        K_basket = K_BASKET_BANDIT, n_iters = bandit_iters, horizon = horizon)
    println()

    # --- Deploy both pickers at TEST_DATE ---
    println("Deploying frozen pickers at $(TEST_DATE) ...")
    dqn_basket, _ = deploy_dqn(env, dqn_main; K_basket = K_BASKET_DQN)
    bandit_basket = env.my_tickers[bandit_res.best_basket]
    println("  DQN basket    ($(length(dqn_basket))): $(dqn_basket)")
    println("  Bandit basket ($(length(bandit_basket))): $(bandit_basket)")
    @printf("  Bandit best-arm avg training reward = %+.4f over %d unique arms visited\n",
        bandit_res.best_mean, bandit_res.n_unique_arms)
    println()

    # --- Forward CD walk both baskets through 2025-2026 ---
    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN, max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1)
    ))
    W_dqn  = fwd_cd(dqn_basket,    rules, env)
    W_bnd  = fwd_cd(bandit_basket, rules, env)
    m_dqn  = realized_metrics(W_dqn, env.g_f)
    m_bnd  = realized_metrics(W_bnd, env.g_f)
    m_full = full_universe_metrics(env)

    df = DataFrame(
        "Strategy"        => ["DQN (historical, K=$(K_BASKET_DQN))",
                              "Bandit (historical, K=$(K_BASKET_BANDIT))",
                              "Full-Universe CD"],
        "Tickers"         => [length(dqn_basket), length(bandit_basket), env.K],
        "W_T (\$)"         => [round(m.W_T, digits = 0)         for m in (m_dqn, m_bnd, m_full)],
        "W_T / W_0"       => [round(m.W_T_over_W0, digits = 3)  for m in (m_dqn, m_bnd, m_full)],
        "Max DD (%)"      => [round(m.max_dd * 100, digits = 1) for m in (m_dqn, m_bnd, m_full)],
        "Ann. return (%)" => [round(m.ann_ret * 100, digits = 2) for m in (m_dqn, m_bnd, m_full)],
        "Sharpe"          => [round(m.sharpe, digits = 3)       for m in (m_dqn, m_bnd, m_full)],
    )
    println("=" ^ 80)
    @printf("Forward run: %s -> %s  (%d trading days)\n",
        Date(env.forward_dates[1]), Date(env.forward_dates[end]), env.n_fwd)
    println("=" ^ 80)
    pretty_table(df; backend = :text,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        table_format = TextTableFormat(borders = text_table_borders__compact))

    # --- Save ---
    out_path = joinpath(@__DIR__, "historical_train_dqn_results.jld2")
    save(out_path, Dict(
        "dqn_basket"            => dqn_basket,
        "dqn_metrics"           => m_dqn,
        "bandit_basket"         => bandit_basket,
        "bandit_metrics"        => m_bnd,
        "bandit_best_mean"      => bandit_res.best_mean,
        "bandit_n_unique_arms"  => bandit_res.n_unique_arms,
        "bandit_reward_history" => bandit_res.reward_history,
        "full_metrics"          => m_full,
        "dqn_training_rewards"  => dqn_hist.rewards_per_ep,
        "dqn_training_days"     => dqn_hist.days_sampled,
        "config"                => Dict(
            "EPISODES"            => episodes,
            "FORWARD_HORIZON"     => horizon,
            "K_BASKET_DQN"        => K_BASKET_DQN,
            "K_BASKET_BANDIT"     => K_BASKET_BANDIT,
            "BANDIT_ITERS"        => bandit_iters,
            "HIDDEN"              => HIDDEN,
            "TRAIN_END_DATE"      => TRAIN_END_DATE,
            "TEST_DATE"           => TEST_DATE,
            "WARMUP_OFFSET"       => WARMUP_OFFSET,
            "TRIGGER_MAX_DRAWDOWN"=> TRIGGER_MAX_DRAWDOWN,
            "SEED"                => seed,
        ),
    ))
    println()
    @printf("Trained models + metadata saved to: %s\n", out_path)
end

# ===== ENTRY =====
# Only run main() when this file is executed directly (`julia historical_train_dqn.jl`),
# not when it is `include`d as a library by another script.
if abspath(PROGRAM_FILE) == @__FILE__
    let
        episodes     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : TRAINING_EPISODES_DEFAULT
        horizon      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : FORWARD_HORIZON_DEFAULT
        bandit_iters = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : BANDIT_ITERS_DEFAULT
        seed         = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : SEED_DEFAULT
        main(episodes, horizon, bandit_iters, seed)
    end
end

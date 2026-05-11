# =====================================================================
# per_sector_bandit.jl
#
# Per-sector ε-greedy bandit picker (Option C).
#
# Trains S independent ε-greedy bandits, one per GICS sector. Each bandit
# picks q_s tickers from its sector universe N_s (where Σ q_s = K_BASKET).
# Reward is the basket's 21-day forward log return MINUS the equal-weight
# sector benchmark return over the same window — i.e. cross-sectional
# alpha within the sector, with macro and sector-beta variance stripped out.
#
# Why this design (vs. the unconstrained DQN/bandit in
# historical_train_dqn.jl + monte_carlo_historical_train.jl):
#
#   - Unconstrained action space C(413, 16) ≈ 1e27 was untrainable in 5000
#     episodes; the DQN training reward was flat.
#   - Per-sector action spaces are O(C(50, 2)) ≈ 1k arms — tractable.
#   - Within-sector returns share macro/sector beta; the residual variance
#     is exactly the cross-sectional picking signal we want to learn.
#
# Outputs `per_sector_bandit_results.jld2` with per-sector reward histories,
# winning baskets, and deploy metrics vs. random / full-universe.
# =====================================================================

using JLD2
using CSV
using DataFrames
using Random
using Statistics
using Printf
using PrettyTables
using eCornellAIFinance

include(joinpath(@__DIR__, "historical_train_dqn.jl"))

# ===== CONFIG =====

const K_BASKET_TOTAL    = 22   # 11 GICS sectors × q_s = 2 each; matches the K_COMPARE bake-off in compare_archetypes.jl
const ITERS_PER_ARM     = 50      # target iters per arm (× n_arms, capped)
const ITERS_MAX         = 5000
const ITERS_MIN         = 500
const ε_FLOOR           = 0.05
const SECTOR_BANDIT_SEED = 2026
const REPORT_EVERY      = 1000
const FORWARD_HORIZON   = 21
const TRIGGER_DD        = 0.15

const SECTOR_CSV = joinpath(@__DIR__, "..", "..", "..", "..", "code", "src", "data",
                            "sp500-sectors.csv")

# ===== SECTOR DATA =====

"""
    load_sector_map(tickers) -> (sector_of, sector_groups, dropped)

For each ticker in `tickers`, look up its GICS sector from the cached
S&P 500 constituents CSV. Returns:
- `sector_of::Dict{String,String}`: ticker -> sector
- `sector_groups::Dict{String,Vector{Int}}`: sector -> indices in `tickers`
- `dropped::Vector{String}`: tickers with no sector match (excluded)
"""
function load_sector_map(tickers::Vector{String})
    sec_df = CSV.read(SECTOR_CSV, DataFrame)
    lookup = Dict{String,String}(
        row.Symbol => row[Symbol("GICS Sector")] for row in eachrow(sec_df))
    sector_of = Dict{String,String}()
    dropped = String[]
    for t in tickers
        if haskey(lookup, t)
            sector_of[t] = lookup[t]
        else
            push!(dropped, t)
        end
    end
    sector_groups = Dict{String,Vector{Int}}()
    for (i, t) in enumerate(tickers)
        haskey(sector_of, t) || continue
        push!(get!(sector_groups, sector_of[t], Int[]), i)
    end
    return sector_of, sector_groups, dropped
end

"""
    assign_quotas(sector_groups, K_total) -> Dict{String,Int}

Equal-weight per sector with bonus to the largest. With 11 sectors and
K_total=16: each sector gets floor(K/S)=1, then the top remainder=5
sectors (by universe size) each get +1. Σ q_s = K_total exactly.
"""
function assign_quotas(sector_groups::Dict{String,Vector{Int}}, K_total::Int)
    sectors = collect(keys(sector_groups))
    S = length(sectors)
    base = K_total ÷ S
    remainder = K_total - base * S
    sorted_sectors = sort(sectors; by = s -> -length(sector_groups[s]))
    quotas = Dict{String,Int}()
    for (rank, s) in enumerate(sorted_sectors)
        quotas[s] = base + (rank <= remainder ? 1 : 0)
    end
    return quotas
end

# ===== REWARD =====

"""
    sector_ew_log_return(sector_indices, day, horizon, env) -> Float64

Equal-dollar-weighted log return of all sector members over
[day, day+horizon]. Used as the sector benchmark.
"""
function sector_ew_log_return(sector_indices::Vector{Int}, day::Int,
        horizon::Int, env)::Float64
    p_d  = env.train_price_matrix[day, sector_indices]
    p_dh = env.train_price_matrix[day + horizon, sector_indices]
    any(p_d .<= 0.0)  && return 0.0
    any(p_dh .<= 0.0) && return 0.0
    # equal-dollar at day d: $1/N_s in each name, so n_j = (1/N_s)/p_d[j]
    # W_d = 1; W_dh = (1/N_s) Σ p_dh / p_d
    W_dh = mean(p_dh ./ p_d)
    return log(W_dh)
end

"""
    sector_relative_reward(arm_full, sector_indices_full, day, horizon, env)

Cobb-Douglas-weighted log return of the picked basket minus the
equal-dollar-weighted log return of all sector members. Pure
cross-sectional alpha within the sector.
"""
function sector_relative_reward(arm_full::Vector{Int},
        sector_indices_full::Vector{Int}, day::Int, horizon::Int, env)
    r_basket = Float64(realized_basket_return(arm_full, day, horizon, env))
    r_sector = sector_ew_log_return(sector_indices_full, day, horizon, env)
    return r_basket - r_sector
end

# ===== BANDIT =====

"""
    sample_arm(rng, sector_indices_full, q, arm_mean, ε) -> Vector{Int}

ε-greedy sampler. With probability ε pick a fresh random q-subset; else
pick the arm with the highest mean reward seen so far. Arm is a sorted
vector of full-universe ticker indices.
"""
function sample_arm(rng::AbstractRNG, sector_indices_full::Vector{Int},
        q::Int, arm_mean::Dict{Vector{Int},Float64}, ε::Float64)
    if rand(rng) < ε || isempty(arm_mean)
        return sort(sample_without_replacement(rng, sector_indices_full, q))
    end
    best_arm = argmax(arm_mean)::Vector{Int}
    return best_arm
end

function sample_without_replacement(rng::AbstractRNG, pool::Vector{Int}, k::Int)
    k >= length(pool) && return copy(pool)
    return shuffle(rng, pool)[1:k]
end

"""
    train_sector_bandit(env, sector_indices_full, q, train_offset, train_last,
        horizon; iters, seed) -> NamedTuple

Train an ε-greedy bandit on a single sector. Returns best arm, mean reward
of best arm, full reward history, arm count, and number of unique arms
visited.
"""
function train_sector_bandit(env, sector_indices_full::Vector{Int}, q::Int,
        train_offset::Int, train_last::Int, horizon::Int;
        iters::Int, seed::Int)
    rng = MersenneTwister(seed)
    N_s = length(sector_indices_full)
    n_arms = binomial(N_s, q)
    arm_mean  = Dict{Vector{Int},Float64}()
    arm_count = Dict{Vector{Int},Int}()
    rewards   = zeros(Float64, iters)
    for t in 1:iters
        ε = max(ε_FLOOR,
            t > 1 ? min(1.0, t^(-1/3) * (n_arms * log(t))^(1/3)) : 1.0)
        arm = sample_arm(rng, sector_indices_full, q, arm_mean, ε)
        day = rand(rng, train_offset:train_last)
        r   = sector_relative_reward(arm, sector_indices_full, day, horizon, env)
        c   = get(arm_count, arm, 0) + 1
        m   = get(arm_mean, arm, 0.0)
        arm_mean[arm]  = m + (r - m) / c
        arm_count[arm] = c
        rewards[t] = r
    end
    best_arm   = argmax(arm_mean)
    best_mean  = arm_mean[best_arm]
    n_unique   = length(arm_mean)
    return (best_arm = best_arm, best_mean = best_mean,
            rewards = rewards, n_arms = n_arms, n_unique = n_unique)
end

# ===== RANDOM BASELINE (per-sector) =====

"""
    sector_random_arm(rng, sector_indices_full, q) -> Vector{Int}

Uniformly random q-subset of a sector — the random baseline for that
sub-basket. Returned sorted (so it's directly comparable as a key).
"""
function sector_random_arm(rng::AbstractRNG, sector_indices_full::Vector{Int},
        q::Int)
    return sort(sample_without_replacement(rng, sector_indices_full, q))
end

# ===== DEPLOY =====
# Reuses fwd_cd, realized_metrics, and full_universe_metrics from
# historical_train_dqn.jl (same B0, Δt, MyTriggerRules / MyRebalancingContextModel
# wiring). Don't re-implement the engine here.

function deploy_basket(env, basket_tickers::Vector{String}, rules)
    W = fwd_cd(basket_tickers, rules, env)
    return realized_metrics(W, env.g_f)
end

# ===== MAIN =====

function main()
    println("=" ^ 78)
    println("Per-sector ε-greedy bandit (Option C)")
    println("=" ^ 78)
    println()

    # --- Environment + sector map ---
    print("Loading environment ... ")
    env = load_environment(horizon = FORWARD_HORIZON)
    println("K = $(env.K), $(env.train_offset)..$(env.train_last) usable training days")

    print("Loading sector map ... ")
    _, sector_groups, dropped = load_sector_map(env.my_tickers)
    println("$(length(sector_groups)) sectors, $(length(dropped)) tickers dropped (no GICS match)")
    if !isempty(dropped)
        @printf("  dropped: %s\n", join(dropped, ", "))
    end

    # --- Quotas ---
    quotas = assign_quotas(sector_groups, K_BASKET_TOTAL)
    println("\nSector quotas (Σ q_s = $K_BASKET_TOTAL):")
    sectors_sorted = sort(collect(keys(sector_groups));
                          by = s -> (-quotas[s], -length(sector_groups[s]), s))
    for s in sectors_sorted
        @printf("  %-25s  N_s = %3d   q_s = %d\n",
                s, length(sector_groups[s]), quotas[s])
    end

    # --- Train per-sector bandits ---
    println("\n--- Training per-sector bandits ---")
    sector_results = Dict{String,NamedTuple}()
    sector_random_means = Dict{String,Float64}()
    rng_master = MersenneTwister(SECTOR_BANDIT_SEED)
    elapsed_total = @elapsed for s in sectors_sorted
        sector_idx = sector_groups[s]
        q = quotas[s]
        N_s = length(sector_idx)
        n_arms = binomial(N_s, q)
        iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
        seed = rand(rng_master, 1:10^9)
        @printf("  %-25s  N=%3d  q=%d  arms=%6d  iters=%5d  ... ",
                s, N_s, q, n_arms, iters)
        t0 = time()
        res = train_sector_bandit(env, sector_idx, q,
            env.train_offset, env.train_last, FORWARD_HORIZON;
            iters = iters, seed = seed)
        sector_results[s] = res
        # Random baseline mean reward (same n_iters)
        rng_rand = MersenneTwister(seed + 1)
        rand_rewards = zeros(Float64, iters)
        for t in 1:iters
            arm = sector_random_arm(rng_rand, sector_idx, q)
            day = rand(rng_rand, env.train_offset:env.train_last)
            rand_rewards[t] = sector_relative_reward(arm, sector_idx, day,
                FORWARD_HORIZON, env)
        end
        sector_random_means[s] = mean(rand_rewards)
        elapsed = time() - t0
        @printf("done in %5.1fs  best=%+.4f  random=%+.4f  edge=%+.4f\n",
                elapsed, res.best_mean, mean(rand_rewards),
                res.best_mean - mean(rand_rewards))
    end
    @printf("\nTotal training time: %.1fs\n", elapsed_total)

    # --- Build full basket ---
    full_basket_indices = Int[]
    for s in sectors_sorted
        append!(full_basket_indices, sector_results[s].best_arm)
    end
    full_basket_tickers = env.my_tickers[full_basket_indices]
    println("\nAssembled basket (size = $(length(full_basket_indices))):")
    println("  ", join(full_basket_tickers, ", "))

    # --- Deploy ---
    println("\n--- Deploy on 2025-2026 forward window ---")
    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN,
        max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1),
    ))

    print("  Per-sector bandit basket ... ")
    sb_metrics = deploy_basket(env, full_basket_tickers, rules)
    @printf("W_T/W_0=%.3f  ann=%+.2f%%  DD=%.1f%%  Sharpe=%+.3f\n",
            sb_metrics.W_T_over_W0, sb_metrics.ann_ret * 100,
            sb_metrics.max_dd * 100, sb_metrics.sharpe)

    print("  Random-per-sector basket ... ")
    rng_dep = MersenneTwister(SECTOR_BANDIT_SEED + 7)
    rand_basket = Int[]
    for s in sectors_sorted
        append!(rand_basket,
            sector_random_arm(rng_dep, sector_groups[s], quotas[s]))
    end
    rand_tickers = env.my_tickers[rand_basket]
    rs_metrics = deploy_basket(env, rand_tickers, rules)
    @printf("W_T/W_0=%.3f  ann=%+.2f%%  DD=%.1f%%  Sharpe=%+.3f\n",
            rs_metrics.W_T_over_W0, rs_metrics.ann_ret * 100,
            rs_metrics.max_dd * 100, rs_metrics.sharpe)

    print("  Full-universe ... ")
    fu_metrics = full_universe_metrics(env)
    @printf("W_T/W_0=%.3f  ann=%+.2f%%  DD=%.1f%%  Sharpe=%+.3f\n",
            fu_metrics.W_T_over_W0, fu_metrics.ann_ret * 100,
            fu_metrics.max_dd * 100, fu_metrics.sharpe)

    # --- Summary table ---
    println("\n--- Summary ---")
    summary_df = DataFrame(
        Strategy   = ["Sector-Bandit", "Random-per-sector", "Full-Universe"],
        K          = [length(full_basket_indices), length(rand_basket), env.K],
        W_T        = round.([sb_metrics.W_T, rs_metrics.W_T, fu_metrics.W_T];
                            digits = 1),
        W_T_over_W0 = round.([sb_metrics.W_T_over_W0, rs_metrics.W_T_over_W0,
                              fu_metrics.W_T_over_W0]; digits = 3),
        Max_DD_pct = round.([sb_metrics.max_dd, rs_metrics.max_dd,
                             fu_metrics.max_dd] .* 100; digits = 1),
        Ann_ret_pct = round.([sb_metrics.ann_ret, rs_metrics.ann_ret,
                              fu_metrics.ann_ret] .* 100; digits = 2),
        Sharpe     = round.([sb_metrics.sharpe, rs_metrics.sharpe,
                             fu_metrics.sharpe]; digits = 3),
    )
    pretty_table(summary_df; backend = :text,
        fit_table_in_display_horizontally = false)

    # --- Save ---
    outpath = joinpath(@__DIR__, "per_sector_bandit_results.jld2")
    JLD2.save(outpath, Dict(
        "config" => Dict(
            "K_BASKET_TOTAL"    => K_BASKET_TOTAL,
            "ITERS_PER_ARM"     => ITERS_PER_ARM,
            "ITERS_MAX"         => ITERS_MAX,
            "ITERS_MIN"         => ITERS_MIN,
            "ε_FLOOR"           => ε_FLOOR,
            "SECTOR_BANDIT_SEED" => SECTOR_BANDIT_SEED,
            "FORWARD_HORIZON"   => FORWARD_HORIZON,
            "TRIGGER_DD"        => TRIGGER_DD,
        ),
        "quotas"               => quotas,
        "sector_groups"        => sector_groups,
        "dropped_tickers"      => dropped,
        "sector_best_arms"     => Dict(s => sector_results[s].best_arm
                                       for s in keys(sector_results)),
        "sector_best_means"    => Dict(s => sector_results[s].best_mean
                                       for s in keys(sector_results)),
        "sector_reward_history" => Dict(s => sector_results[s].rewards
                                        for s in keys(sector_results)),
        "sector_n_arms"        => Dict(s => sector_results[s].n_arms
                                       for s in keys(sector_results)),
        "sector_n_unique"      => Dict(s => sector_results[s].n_unique
                                       for s in keys(sector_results)),
        "sector_random_means"  => sector_random_means,
        "full_basket_tickers"  => full_basket_tickers,
        "full_basket_indices"  => full_basket_indices,
        "sector_bandit_metrics" => sb_metrics,
        "random_per_sector_metrics" => rs_metrics,
        "full_universe_metrics" => fu_metrics,
    ))
    println("\nSaved $(outpath)")
end

# Only run main() when this file is executed directly, not when it is
# `include`d as a library by a Monte Carlo or notebook driver.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

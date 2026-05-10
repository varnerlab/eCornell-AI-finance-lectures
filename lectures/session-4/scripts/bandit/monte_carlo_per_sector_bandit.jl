# =====================================================================
# monte_carlo_per_sector_bandit.jl
#
# Multi-seed sweep of the per-sector bandit pipeline. For each seed,
# train the 11 sector bandits, build the assembled K=16 basket, deploy
# on the 2025-2026 forward window, and record the realized metrics.
# Compare against the random-per-sector control (same K, same sector
# quotas, random within-sector picks) and the deterministic Full-Universe
# baseline.
#
# Usage:
#   cd lectures/session-4
#   julia --project=. scripts/bandit/monte_carlo_per_sector_bandit.jl                # 30 seeds, default
#   julia --project=. scripts/bandit/monte_carlo_per_sector_bandit.jl 50             # 50 seeds
#
# Args (positional):
#   1) N_SEEDS  (default 30)
#
# Output:
#   - Per-seed progress + summary table to stdout
#   - Raw per-seed records to scripts/monte_carlo_per_sector_bandit_results.jld2
# =====================================================================

include(joinpath(@__DIR__, "per_sector_bandit.jl"))

const N_SEEDS_DEFAULT = 30
const MC_SEED_BASE    = 1000

# ===== PER-SEED RUN =====

"""
    run_one_seed(env, sector_groups, sectors_sorted, quotas, rules, seed)

Train all sector bandits with `seed`, deploy the assembled basket on the
forward window, and also deploy a random-per-sector control (same K,
same quotas, uniformly random within-sector picks). Returns a NamedTuple
of metrics + tickers.
"""
function run_one_seed(env, sector_groups::Dict{String,Vector{Int}},
        sectors_sorted::Vector{String}, quotas::Dict{String,Int},
        rules, seed::Int)
    # --- Train sector bandits ---
    rng_master = MersenneTwister(seed)
    sector_picks      = Dict{String,Vector{Int}}()
    sector_best_means = Dict{String,Float64}()
    for s in sectors_sorted
        sector_idx = sector_groups[s]
        q = quotas[s]
        N_s = length(sector_idx)
        n_arms = binomial(N_s, q)
        iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
        bandit_seed = rand(rng_master, 1:10^9)
        res = train_sector_bandit(env, sector_idx, q,
            env.train_offset, env.train_last, FORWARD_HORIZON;
            iters = iters, seed = bandit_seed)
        sector_picks[s]      = res.best_arm
        sector_best_means[s] = res.best_mean
    end

    # --- Assemble bandit basket ---
    bandit_basket_idx = Int[]
    for s in sectors_sorted
        append!(bandit_basket_idx, sector_picks[s])
    end
    bandit_tickers = env.my_tickers[bandit_basket_idx]

    # --- Random-per-sector control (different RNG, derived from same seed) ---
    rng_random = MersenneTwister(seed + 100_000)
    random_basket_idx = Int[]
    for s in sectors_sorted
        append!(random_basket_idx,
            sector_random_arm(rng_random, sector_groups[s], quotas[s]))
    end
    random_tickers = env.my_tickers[random_basket_idx]

    # --- Deploy ---
    sb = deploy_basket(env, bandit_tickers, rules)
    rs = deploy_basket(env, random_tickers, rules)

    return (sb = sb, rs = rs,
            bandit_tickers = bandit_tickers,
            random_tickers = random_tickers,
            sector_picks   = sector_picks,
            sector_best_means = sector_best_means)
end

# ===== AGGREGATION =====

function quantile_stats(x::AbstractVector)
    return (
        min    = minimum(x),
        q25    = quantile(x, 0.25),
        median = median(x),
        mean   = mean(x),
        q75    = quantile(x, 0.75),
        max    = maximum(x),
        std    = std(x),
    )
end

function summary_row(metric::String, strategy::String, vals::AbstractVector)
    s = quantile_stats(vals)
    return (Metric = metric, Strategy = strategy,
            Min    = round(s.min;    digits = 3),
            Q25    = round(s.q25;    digits = 3),
            Median = round(s.median; digits = 3),
            Mean   = round(s.mean;   digits = 3),
            Q75    = round(s.q75;    digits = 3),
            Max    = round(s.max;    digits = 3),
            Std    = round(s.std;    digits = 3))
end

function constant_row(metric::String, strategy::String, val::Real)
    v = round(val; digits = 3)
    return (Metric = metric, Strategy = strategy,
            Min = v, Q25 = v, Median = v, Mean = v,
            Q75 = v, Max = v, Std = 0.0)
end

# ===== MAIN =====

function main_mc(N_seeds::Int = N_SEEDS_DEFAULT)
    println("=" ^ 80)
    @printf("Monte Carlo: per-sector bandit, %d seeds\n", N_seeds)
    println("=" ^ 80)

    # --- Setup once ---
    print("Loading environment ... ")
    env = load_environment(horizon = FORWARD_HORIZON)
    println("K = $(env.K), $(env.train_offset)..$(env.train_last) usable training days")

    print("Loading sector map ... ")
    _, sector_groups, dropped = load_sector_map(env.my_tickers)
    println("$(length(sector_groups)) sectors, $(length(dropped)) tickers dropped (no GICS match)")

    quotas = assign_quotas(sector_groups, K_BASKET_TOTAL)
    sectors_sorted = sort(collect(keys(sector_groups));
        by = s -> (-quotas[s], -length(sector_groups[s]), s))

    println("\nSector quotas (Σ q_s = $K_BASKET_TOTAL):")
    for s in sectors_sorted
        @printf("  %-25s  N_s = %3d   q_s = %d\n",
                s, length(sector_groups[s]), quotas[s])
    end

    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN,
        max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1),
    ))

    # Full-universe baseline (deterministic — once)
    print("\nFull-universe baseline ... ")
    fu = full_universe_metrics(env)
    @printf("W_T/W_0=%.3f  ann=%+.2f%%  DD=%.1f%%  Sharpe=%+.3f\n",
            fu.W_T_over_W0, fu.ann_ret * 100, fu.max_dd * 100, fu.sharpe)

    # --- Per-seed sweep ---
    println("\n--- Per-seed sweep ---")
    records = Vector{NamedTuple}(undef, N_seeds)
    t_total = @elapsed for i in 1:N_seeds
        seed = MC_SEED_BASE + i
        t0 = time()
        r = run_one_seed(env, sector_groups, sectors_sorted, quotas, rules, seed)
        elapsed = time() - t0
        records[i] = (seed = seed, r...)
        @printf("  seed %3d (%5d): SB W_T=%.3f Sharpe=%+.3f DD=%4.1f%% | RND W_T=%.3f Sharpe=%+.3f | %4.1fs\n",
            i, seed, r.sb.W_T_over_W0, r.sb.sharpe, r.sb.max_dd * 100,
            r.rs.W_T_over_W0, r.rs.sharpe, elapsed)
    end
    @printf("\nTotal sweep time: %.1fs\n", t_total)

    # --- Extract metric vectors ---
    sb_W_T    = [r.sb.W_T_over_W0 for r in records]
    rs_W_T    = [r.rs.W_T_over_W0 for r in records]
    sb_sharpe = [r.sb.sharpe       for r in records]
    rs_sharpe = [r.rs.sharpe       for r in records]
    sb_dd     = [r.sb.max_dd * 100 for r in records]
    rs_dd     = [r.rs.max_dd * 100 for r in records]
    sb_ann    = [r.sb.ann_ret * 100 for r in records]
    rs_ann    = [r.rs.ann_ret * 100 for r in records]

    # --- Summary table ---
    println("\n--- Summary across $N_seeds seeds ---")
    rows = NamedTuple[]
    for (metric, sb_vals, rs_vals, fu_val) in [
            ("W_T / W_0",   sb_W_T,    rs_W_T,    fu.W_T_over_W0),
            ("Ann. ret %",  sb_ann,    rs_ann,    fu.ann_ret * 100),
            ("Max DD %",    sb_dd,     rs_dd,     fu.max_dd * 100),
            ("Sharpe",      sb_sharpe, rs_sharpe, fu.sharpe),
        ]
        push!(rows, summary_row(metric, "SectorBandit", sb_vals))
        push!(rows, summary_row(metric, "RandomBucket", rs_vals))
        push!(rows, constant_row(metric, "FullUniverse", fu_val))
    end
    df = DataFrame(rows)
    pretty_table(df; backend = :text,
        fit_table_in_display_horizontally = false)

    # --- Head-to-head ---
    sb_beats_rs = sum(sb_W_T .> rs_W_T)
    sb_beats_fu = sum(sb_W_T .> fu.W_T_over_W0)
    rs_beats_fu = sum(rs_W_T .> fu.W_T_over_W0)
    sb_beats_rs_sharpe = sum(sb_sharpe .> rs_sharpe)
    sb_beats_fu_sharpe = sum(sb_sharpe .> fu.sharpe)

    println("\nHead-to-head (W_T comparison):")
    @printf("  Sector-Bandit beats Random-per-sector : %d / %d (%.0f%%)\n",
            sb_beats_rs, N_seeds, 100 * sb_beats_rs / N_seeds)
    @printf("  Sector-Bandit beats Full-Universe     : %d / %d (%.0f%%)   [Full = %.3f]\n",
            sb_beats_fu, N_seeds, 100 * sb_beats_fu / N_seeds, fu.W_T_over_W0)
    @printf("  Random-per-sector beats Full-Universe : %d / %d (%.0f%%)\n",
            rs_beats_fu, N_seeds, 100 * rs_beats_fu / N_seeds)

    println("\nHead-to-head (Sharpe comparison):")
    @printf("  Sector-Bandit Sharpe > Random         : %d / %d (%.0f%%)\n",
            sb_beats_rs_sharpe, N_seeds, 100 * sb_beats_rs_sharpe / N_seeds)
    @printf("  Sector-Bandit Sharpe > Full-Universe  : %d / %d (%.0f%%)  [Full = %+.3f]\n",
            sb_beats_fu_sharpe, N_seeds, 100 * sb_beats_fu_sharpe / N_seeds, fu.sharpe)

    # --- Drawdown trip diagnostics ---
    sb_trips = sum(sb_dd .>= TRIGGER_MAX_DRAWDOWN * 100)
    rs_trips = sum(rs_dd .>= TRIGGER_MAX_DRAWDOWN * 100)
    println("\nDrawdown trigger trips (Max DD >= $(TRIGGER_MAX_DRAWDOWN * 100)%):")
    @printf("  Sector-Bandit baskets that tripped   : %d / %d (%.0f%%)\n",
            sb_trips, N_seeds, 100 * sb_trips / N_seeds)
    @printf("  Random-per-sector baskets tripped    : %d / %d (%.0f%%)\n",
            rs_trips, N_seeds, 100 * rs_trips / N_seeds)
    @printf("  Full-Universe tripped                : %s\n",
            fu.max_dd * 100 >= TRIGGER_MAX_DRAWDOWN * 100 ? "yes" : "no")

    # --- Save ---
    outpath = joinpath(@__DIR__, "monte_carlo_per_sector_bandit_results.jld2")
    JLD2.save(outpath, Dict(
        "config" => Dict(
            "N_SEEDS"              => N_seeds,
            "SEED_BASE"            => MC_SEED_BASE,
            "K_BASKET_TOTAL"       => K_BASKET_TOTAL,
            "ITERS_PER_ARM"        => ITERS_PER_ARM,
            "ITERS_MAX"            => ITERS_MAX,
            "ITERS_MIN"            => ITERS_MIN,
            "ε_FLOOR"              => ε_FLOOR,
            "FORWARD_HORIZON"      => FORWARD_HORIZON,
            "TRIGGER_MAX_DRAWDOWN" => TRIGGER_MAX_DRAWDOWN,
            "TRIGGER_MAX_TURNOVER" => TRIGGER_MAX_TURNOVER,
        ),
        "quotas"            => quotas,
        "sector_groups"     => sector_groups,
        "dropped_tickers"   => dropped,
        "seeds"             => [r.seed for r in records],
        "sb_W_T_over_W0"    => sb_W_T,
        "sb_sharpe"         => sb_sharpe,
        "sb_max_dd_pct"     => sb_dd,
        "sb_ann_ret_pct"    => sb_ann,
        "rs_W_T_over_W0"    => rs_W_T,
        "rs_sharpe"         => rs_sharpe,
        "rs_max_dd_pct"     => rs_dd,
        "rs_ann_ret_pct"    => rs_ann,
        "full_universe_metrics"   => fu,
        "per_seed_bandit_baskets" => [r.bandit_tickers for r in records],
        "per_seed_random_baskets" => [r.random_tickers for r in records],
    ))
    println("\nSaved $(outpath)")
end

# ===== ENTRY =====
if abspath(PROGRAM_FILE) == @__FILE__
    let
        n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : N_SEEDS_DEFAULT
        main_mc(n)
    end
end

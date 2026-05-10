# =====================================================================
# compare_archetypes.jl
#
# Bake-off between:
#   - 5 archetype baskets from lectures/session-1/interview.md (the
#     "Claude choices" — Conservative Income, Conservative Growth,
#     Balanced, Growth-Oriented, Aggressive Growth)
#   - The trained per-sector bandit (single seed 2026 + 30-seed MC stats)
#   - Random-per-sector control (single seed + MC stats)
#   - Full-Universe (deterministic, K=413)
#
# All baskets are walked through the same Cobb-Douglas rebalancing engine
# on the 2025-2026 forward window, with the same drawdown/turnover rules.
# =====================================================================

include(joinpath(@__DIR__, "per_sector_bandit.jl"))
include(joinpath(@__DIR__, "monte_carlo_per_sector_bandit.jl"))

# Bake-off K. Use 22 (uniform q_s=2 across all 11 sectors) to match the
# median archetype basket size — fairer than the per_sector_bandit default
# of 16. K_BASKET_TOTAL=16 is still used by per_sector_bandit.jl and
# monte_carlo_per_sector_bandit.jl when run directly; this override lives
# only in the bake-off.
const K_COMPARE = 22
const COMPARE_MC_SEEDS = 30

# --- Archetype ticker lists (parsed from session-1/interview.md, Step 9) ---
const ARCHETYPES = [
    ("ConservativeIncome",
        ["VZ","T","MCD","PG","KO","PEP","WMT","XOM","CVX","JPM",
         "BRK.B","JNJ","MRK","HON","UPS","AAPL","MSFT","APD","AMT","NEE"]),
    ("ConservativeGrowth",
        ["DIS","VZ","AMZN","HD","MCD","PG","COST","WMT","XOM","CVX",
         "JPM","BRK.B","V","JNJ","UNH","LLY","HON","UPS","AAPL","MSFT",
         "AVGO","SHW","AMT","NEE"]),
    ("Balanced",
        ["DIS","VZ","AMZN","HD","PG","COST","XOM","CVX","JPM","V",
         "BAC","JNJ","UNH","LLY","HON","CAT","AAPL","MSFT","NVDA",
         "SHW","AMT","NEE"]),
    ("GrowthOriented",
        ["DIS","NFLX","AMZN","HD","TJX","COST","CVX","JPM","V","MA",
         "UNH","LLY","ABBV","HON","CAT","AAPL","MSFT","NVDA","AVGO",
         "CRM","SHW","AMT"]),
    ("AggressiveGrowth",
        ["NFLX","AMZN","NKE","TSLA","COST","SLB","JPM","V","MA","LLY",
         "ABBV","CAT","AAPL","MSFT","NVDA","AVGO","CRM","ADBE","FCX","AMT"]),
]

const COMPARE_SEED   = 2026  # for the single-seed SB / RND lines
const MC_RESULTS_PATH = joinpath(@__DIR__,
    "monte_carlo_per_sector_bandit_results.jld2")

function deploy_archetype(env, tickers::Vector{String}, rules)
    kept    = [t for t in tickers if t in env.my_tickers]
    dropped = [t for t in tickers if !(t in env.my_tickers)]
    isempty(kept) && error("no archetype tickers found in universe")
    m = deploy_basket(env, kept, rules)
    return (metrics = m, kept = kept, dropped = dropped, K = length(kept))
end

function single_seed_sector_bandit(env, sector_groups, sectors_sorted, quotas,
        rules, seed::Int)
    rng_master = MersenneTwister(seed)
    bandit_idx = Int[]
    for s in sectors_sorted
        sector_idx = sector_groups[s]
        q = quotas[s]
        n_arms = binomial(length(sector_idx), q)
        iters = clamp(n_arms * ITERS_PER_ARM, ITERS_MIN, ITERS_MAX)
        bandit_seed = rand(rng_master, 1:10^9)
        res = train_sector_bandit(env, sector_idx, q,
            env.train_offset, env.train_last, FORWARD_HORIZON;
            iters = iters, seed = bandit_seed)
        append!(bandit_idx, res.best_arm)
    end
    bandit_tickers = env.my_tickers[bandit_idx]

    rng_random = MersenneTwister(seed + 100_000)
    random_idx = Int[]
    for s in sectors_sorted
        append!(random_idx,
            sector_random_arm(rng_random, sector_groups[s], quotas[s]))
    end
    random_tickers = env.my_tickers[random_idx]

    sb_metrics = deploy_basket(env, bandit_tickers, rules)
    rs_metrics = deploy_basket(env, random_tickers, rules)
    return (sb = sb_metrics, sb_tickers = bandit_tickers,
            rs = rs_metrics, rs_tickers = random_tickers)
end

function main_compare()
    println("=" ^ 80)
    println("Archetype bake-off: 5 Claude-curated baskets vs algorithmic baselines")
    println("=" ^ 80)

    print("Loading environment ... ")
    env = load_environment(horizon = FORWARD_HORIZON)
    println("K = $(env.K)")

    print("Loading sector map ... ")
    _, sector_groups, dropped_universe = load_sector_map(env.my_tickers)
    println("$(length(sector_groups)) sectors, $(length(dropped_universe)) tickers dropped")

    quotas = assign_quotas(sector_groups, K_COMPARE)
    sectors_sorted = sort(collect(keys(sector_groups));
        by = s -> (-quotas[s], -length(sector_groups[s]), s))
    @printf("\nBake-off K_COMPARE = %d (Σ q_s)\n", K_COMPARE)
    @printf("  per-sector quotas: %s\n",
            join(["$s=$(quotas[s])" for s in sectors_sorted], ", "))

    rules = build(MyTriggerRules, (
        max_drawdown = TRIGGER_MAX_DRAWDOWN,
        max_turnover = TRIGGER_MAX_TURNOVER,
        rebalance_schedule = ones(Int, env.n_fwd - 1),
    ))

    # --- Deploy 5 archetypes ---
    println("\n--- Deploying archetype baskets ---")
    arch_results = []
    for (name, tickers) in ARCHETYPES
        ar = deploy_archetype(env, tickers, rules)
        push!(arch_results, (name = name, ar...))
        @printf("  %-22s  K=%2d  W_T/W_0=%.3f  ann=%+6.2f%%  DD=%4.1f%%  Sharpe=%+.3f",
                name, ar.K, ar.metrics.W_T_over_W0,
                ar.metrics.ann_ret * 100, ar.metrics.max_dd * 100,
                ar.metrics.sharpe)
        if !isempty(ar.dropped)
            @printf("   (dropped: %s)", join(ar.dropped, ","))
        end
        println()
    end

    # --- Deterministic baseline ---
    println("\n--- Full-Universe (deterministic baseline, K = $(env.K)) ---")
    fu = full_universe_metrics(env)
    @printf("  W_T/W_0=%.3f  ann=%+6.2f%%  DD=%4.1f%%  Sharpe=%+.3f\n",
            fu.W_T_over_W0, fu.ann_ret * 100,
            fu.max_dd * 100, fu.sharpe)

    # Compute one bandit + random-per-sector run with COMPARE_SEED so the
    # JLD2 persists a sample basket for inspection. The single-seed numbers
    # are NOT used in the bake-off table — that's the MC-median's job.
    sb_rs = single_seed_sector_bandit(env, sector_groups, sectors_sorted,
        quotas, rules, COMPARE_SEED)

    # --- Inline MC at K_COMPARE for apples-to-apples distribution ---
    println("\n--- Inline MC sweep at K=$K_COMPARE ($COMPARE_MC_SEEDS seeds) ---")
    sb_W_T_mc = Float64[]; rs_W_T_mc = Float64[]
    sb_sharpe_mc = Float64[]; rs_sharpe_mc = Float64[]
    sb_dd_mc = Float64[]; rs_dd_mc = Float64[]
    t_mc = @elapsed for i in 1:COMPARE_MC_SEEDS
        seed = MC_SEED_BASE + i
        r = run_one_seed(env, sector_groups, sectors_sorted, quotas, rules, seed)
        push!(sb_W_T_mc, r.sb.W_T_over_W0)
        push!(rs_W_T_mc, r.rs.W_T_over_W0)
        push!(sb_sharpe_mc, r.sb.sharpe)
        push!(rs_sharpe_mc, r.rs.sharpe)
        push!(sb_dd_mc, r.sb.max_dd * 100)
        push!(rs_dd_mc, r.rs.max_dd * 100)
    end
    @printf("  done in %.1fs\n", t_mc)
    @printf("  Sector-Bandit       W_T median=%.3f  Sharpe median=%+.3f  DD median=%.1f%%\n",
            median(sb_W_T_mc), median(sb_sharpe_mc), median(sb_dd_mc))
    @printf("  Random-per-sector   W_T median=%.3f  Sharpe median=%+.3f  DD median=%.1f%%\n",
            median(rs_W_T_mc), median(rs_sharpe_mc), median(rs_dd_mc))
    sb_beats_rs = sum(sb_W_T_mc .> rs_W_T_mc)
    @printf("  Sector-Bandit beats Random-per-sector: %d / %d (%.0f%%)\n",
            sb_beats_rs, COMPARE_MC_SEEDS, 100 * sb_beats_rs / COMPARE_MC_SEEDS)

    # --- Combined ranking table sorted by Sharpe ---
    println("\n--- Bake-off (sorted by Sharpe, descending) ---")
    rows = []
    for r in arch_results
        push!(rows, (
            Strategy = r.name, K = r.K,
            W_T_over_W0 = round(r.metrics.W_T_over_W0; digits = 3),
            Ann_ret_pct = round(r.metrics.ann_ret * 100; digits = 2),
            Max_DD_pct  = round(r.metrics.max_dd * 100;  digits = 1),
            Sharpe      = round(r.metrics.sharpe;        digits = 3)))
    end
    # Note: the single-seed (rng=$COMPARE_SEED) numbers from above are
    # intentionally NOT in this table. Single seeds are noisy data points
    # in a 30-seed distribution; including them here would invite
    # cherry-picking. Use the MC-median rows for the stochastic strategies.
    push!(rows, (Strategy = "Full-Universe", K = env.K,
        W_T_over_W0 = round(fu.W_T_over_W0; digits = 3),
        Ann_ret_pct = round(fu.ann_ret * 100; digits = 2),
        Max_DD_pct  = round(fu.max_dd * 100;  digits = 1),
        Sharpe      = round(fu.sharpe;        digits = 3)))
    push!(rows, (Strategy = "Sector-Bandit (MC median)", K = K_COMPARE,
        W_T_over_W0 = round(median(sb_W_T_mc); digits = 3),
        Ann_ret_pct = round(median(log.(sb_W_T_mc)) * (252.0 / env.n_fwd) * 100; digits = 2),
        Max_DD_pct  = round(median(sb_dd_mc); digits = 1),
        Sharpe      = round(median(sb_sharpe_mc); digits = 3)))
    push!(rows, (Strategy = "Random-per-sector (MC median)", K = K_COMPARE,
        W_T_over_W0 = round(median(rs_W_T_mc); digits = 3),
        Ann_ret_pct = round(median(log.(rs_W_T_mc)) * (252.0 / env.n_fwd) * 100; digits = 2),
        Max_DD_pct  = round(median(rs_dd_mc); digits = 1),
        Sharpe      = round(median(rs_sharpe_mc); digits = 3)))
    sort!(rows; by = r -> -r.Sharpe)
    pretty_table(DataFrame(rows); backend = :text,
        fit_table_in_display_horizontally = false)

    # --- Where does each archetype sit in the bandit's Sharpe distribution? ---
    println("\n--- Archetype Sharpe vs bandit Sharpe distribution (K=$K_COMPARE, $COMPARE_MC_SEEDS seeds) ---")
    @printf("  Bandit Sharpe percentiles: min=%+.3f  Q25=%+.3f  median=%+.3f  Q75=%+.3f  max=%+.3f\n",
            minimum(sb_sharpe_mc), quantile(sb_sharpe_mc, 0.25),
            median(sb_sharpe_mc), quantile(sb_sharpe_mc, 0.75),
            maximum(sb_sharpe_mc))
    println()
    for r in arch_results
        # percentile rank: fraction of bandit seeds whose Sharpe is < archetype Sharpe
        pct = 100 * count(<(r.metrics.sharpe), sb_sharpe_mc) / length(sb_sharpe_mc)
        bandit_beats = 100 - pct
        @printf("  %-22s  Sharpe=%+.3f  →  beats %.0f%% of bandit seeds; bandit beats it %.0f%% of the time\n",
                r.name, r.metrics.sharpe, pct, bandit_beats)
    end

    # --- Save ---
    outpath = joinpath(@__DIR__, "compare_archetypes_results.jld2")
    JLD2.save(outpath, Dict(
        "K_COMPARE"             => K_COMPARE,
        "MC_N_SEEDS"            => COMPARE_MC_SEEDS,
        "MC_SEED_BASE"          => MC_SEED_BASE,
        "n_fwd"                 => env.n_fwd,
        "forward_first_date"    => string(env.forward_dates[1]),
        "forward_last_date"     => string(env.forward_dates[end]),
        "archetype_metrics"     => Dict(r.name => r.metrics for r in arch_results),
        "archetype_kept"        => Dict(r.name => r.kept    for r in arch_results),
        "archetype_dropped"     => Dict(r.name => r.dropped for r in arch_results),
        "sector_quotas"         => quotas,
        "sector_sizes"          => Dict(s => length(sector_groups[s]) for s in keys(sector_groups)),
        "sb_W_T_mc"             => sb_W_T_mc,
        "sb_sharpe_mc"          => sb_sharpe_mc,
        "sb_dd_mc"              => sb_dd_mc,
        "rs_W_T_mc"             => rs_W_T_mc,
        "rs_sharpe_mc"          => rs_sharpe_mc,
        "rs_dd_mc"              => rs_dd_mc,
        "single_seed"           => COMPARE_SEED,
        "sb_seed_metrics"       => sb_rs.sb,
        "sb_seed_tickers"       => sb_rs.sb_tickers,
        "rs_seed_metrics"       => sb_rs.rs,
        "rs_seed_tickers"       => sb_rs.rs_tickers,
        "full_universe_metrics" => fu,
    ))
    println("\nSaved $(outpath)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_compare()
end

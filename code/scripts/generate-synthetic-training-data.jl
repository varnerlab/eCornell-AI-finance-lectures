# generate-synthetic-training-data.jl
#
# Generates a frozen 20-year synthetic training dataset for all 424 tickers.
# Step 1: Generate candidate market index paths from the SPY surrogate
# Step 2: Score each candidate for realism (CAGR, drawdowns, jump clusters, kurtosis)
# Step 3: Select the best candidate (drops the unused first observation so that
#         G_market is aligned with prices: prices[t+1] = prices[t]·exp(G_market[t]·DT))
# Step 4: Generate per-ticker paths via the hybrid SIM construction:
#           g_i(t) = α_i + β_i · g_m(t) + ε_i(t)
#         where (α_i, β_i) come from real-data calibration (sim-calibration.jld2),
#         ε_i is a variance-corrected, copula-reordered HMM marginal draw, and the
#         variance correction preserves the per-ticker HMM marginal variance up to
#         a 10%-of-σ²_HMM idiosyncratic floor (β is clipped if it would breach it).
# Step 5: Save as the canonical training dataset
#
# Usage: julia --project=.. generate-synthetic-training-data.jl
#
# Output: ../src/data/synthetic-training-dataset.jld2

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using eCornellAIFinance
using DataFrames
using JLD2
using JumpHMM
using LinearAlgebra
using Random
using Statistics
using StatsBase

# --- Configuration ---
const T_YEARS = 20
const T_DAYS = T_YEARS * 252        # 5040 trading days
const N_CANDIDATES = 2000           # candidate paths to screen
const DT = 1.0 / 252.0
const SEED_CANDIDATES = 2026        # reproducible

# realism criteria for candidate selection -
const MIN_CAGR = 0.06               # min 6% annualized return
const MAX_CAGR = 0.09               # max 9% annualized return (target 7-8%)
const MIN_MAJOR_DRAWDOWNS = 3       # at least 3 drawdowns > 20%
const MAX_MAJOR_DRAWDOWNS = 8       # not too many
const MIN_JUMP_CLUSTERS = 3         # at least 3 distinct jump episodes
const MAX_FINAL_RETURN = 20.0       # price can't grow more than 20x
const MIN_FINAL_RETURN = 1.5        # must at least grow 50%

println("="^70)
println("  SYNTHETIC TRAINING DATA GENERATION")
println("  $(T_YEARS) years × 424 tickers")
println("="^70)

# --- Step 1: Load the SPY surrogate model ---
println("\nLoading market surrogate model...")
market_model = MyMarketSurrogateModel();

# --- Step 2: Generate and score candidate market paths ---
println("Generating $(N_CANDIDATES) candidate market paths ($(T_DAYS) days each)...")
Random.seed!(SEED_CANDIDATES);

struct CandidateScore
    idx::Int
    cagr::Float64
    n_drawdowns_20::Int      # drawdowns > 20%
    n_drawdowns_10::Int      # drawdowns > 10%
    max_drawdown::Float64
    n_jump_clusters::Int
    kurtosis::Float64
    final_return::Float64    # final_price / start_price
    score::Float64           # composite score
end

function count_drawdowns(prices::Vector{Float64}; threshold::Float64=0.20)
    peak = prices[1];
    in_drawdown = false;
    count = 0;
    for p ∈ prices
        peak = max(peak, p);
        dd = (peak - p) / peak;
        if dd >= threshold && !in_drawdown
            count += 1;
            in_drawdown = true;
        elseif dd < threshold * 0.5  # recovered past halfway
            in_drawdown = false;
        end
    end
    return count;
end

function count_jump_clusters(jumps::Vector{Bool}; gap::Int=20)
    # count distinct clusters of jumps separated by at least `gap` non-jump days
    clusters = 0;
    last_jump_day = -gap - 1;
    for (t, j) ∈ enumerate(jumps)
        if j
            if t - last_jump_day > gap
                clusters += 1;
            end
            last_jump_day = t;
        end
    end
    return clusters;
end

candidates = CandidateScore[];
sim_result = hmm_simulate(market_model, T_DAYS; n_paths=N_CANDIDATES);

for i ∈ 1:N_CANDIDATES

    path = sim_result.paths[i];
    G = Float64.(path.observations);  # excess growth rates

    # convert to prices (start at 100) -
    prices = zeros(T_DAYS);
    prices[1] = 100.0;
    for t ∈ 2:T_DAYS
        prices[t] = prices[t-1] * exp(G[t] * DT);
    end

    # metrics -
    final_return = prices[end] / prices[1];
    cagr = final_return^(1.0/T_YEARS) - 1.0;
    max_dd = maximum((accumulate(max, prices) .- prices) ./ accumulate(max, prices));
    n_dd_20 = count_drawdowns(prices; threshold=0.20);
    n_dd_10 = count_drawdowns(prices; threshold=0.10);
    n_jumps = count_jump_clusters(path.jumps; gap=20);
    kurt = kurtosis(G);

    # composite score: reward paths that are realistic -
    score = 0.0;

    # CAGR in range [4%, 14%] — penalize outside
    if MIN_CAGR <= cagr <= MAX_CAGR
        score += 10.0;
    else
        score -= 20.0 * abs(cagr - clamp(cagr, MIN_CAGR, MAX_CAGR));
    end

    # drawdowns: want 3-8 major ones
    if MIN_MAJOR_DRAWDOWNS <= n_dd_20 <= MAX_MAJOR_DRAWDOWNS
        score += 10.0 + 2.0 * n_dd_20;  # more is better (within range)
    else
        score -= 10.0;
    end

    # jump clusters: want at least 3
    if n_jumps >= MIN_JUMP_CLUSTERS
        score += 5.0 + 2.0 * min(n_jumps, 8);
    else
        score -= 15.0;
    end

    # kurtosis: want elevated (fat tails) but not insane
    if 3.0 <= kurt <= 30.0
        score += 5.0;
    end

    # final return: must be reasonable
    if MIN_FINAL_RETURN <= final_return <= MAX_FINAL_RETURN
        score += 5.0;
    else
        score -= 20.0;
    end

    push!(candidates, CandidateScore(i, cagr, n_dd_20, n_dd_10, max_dd, n_jumps, kurt, final_return, score));
end

# rank by score -
sort!(candidates, by=c -> c.score, rev=true);

println("\nTop 5 candidates:")
println("  Rank | CAGR  | DD>20% | DD>10% | MaxDD  | Jumps | Kurt  | Final | Score")
println("  " * "-"^75)
for (rank, c) ∈ enumerate(candidates[1:min(5, length(candidates))])
    println("  $(lpad(rank, 4)) | $(lpad(round(c.cagr*100, digits=1), 5))% | " *
            "$(lpad(c.n_drawdowns_20, 6)) | $(lpad(c.n_drawdowns_10, 6)) | " *
            "$(lpad(round(c.max_drawdown*100, digits=1), 5))% | " *
            "$(lpad(c.n_jump_clusters, 5)) | $(lpad(round(c.kurtosis, digits=1), 5)) | " *
            "$(lpad(round(c.final_return, digits=1), 5))x | $(round(c.score, digits=1))")
end

# --- Step 3: Select the best candidate ---
best = candidates[1];
println("\n★ Selected candidate #$(best.idx):")
println("  CAGR: $(round(best.cagr*100, digits=1))%")
println("  Drawdowns > 20%: $(best.n_drawdowns_20)")
println("  Drawdowns > 10%: $(best.n_drawdowns_10)")
println("  Max drawdown: $(round(best.max_drawdown*100, digits=1))%")
println("  Jump clusters: $(best.n_jump_clusters)")
println("  Excess kurtosis: $(round(best.kurtosis, digits=1))")
println("  Final return: $(round(best.final_return, digits=2))x")

# extract the selected market path -
# NOTE: drop the unused first observation. The HMM returns T_DAYS observations,
# but the very first one is never consumed by the price recursion (the price
# series seeds at $100 on day 1 and only steps forward starting on day 2).
# We trim it here so that prices[t+1] = prices[t] · exp(G_market[t]·DT) and
# the saved `market_returns` aligns one-for-one with growth rates that any
# downstream notebook computes from the saved prices.
selected_path = sim_result.paths[best.idx];
G_full = Float64.(selected_path.observations);
jumps_full = selected_path.jumps;

G_market = G_full[2:end];           # length T_DAYS - 1
market_jumps = jumps_full[2:end];    # length T_DAYS - 1
T_eff = length(G_market);            # = T_DAYS - 1
σ²_market_synth = var(G_market);

# convert to prices -
market_prices = zeros(T_DAYS);
market_prices[1] = 100.0;
for t ∈ 1:T_eff
    market_prices[t+1] = market_prices[t] * exp(G_market[t] * DT);
end

# --- Step 4: Generate per-ticker paths via the hybrid SIM construction ---
# For each ticker we compose
#     g_i(t) = α_i + β_i · g_m(t) + ε_i(t)
# where:
#   • (α_i, β_i) come from real-data calibration (sim-calibration.jld2)
#   • ε_i is the ticker's HMM marginal draw, scaled so the resulting g_i variance
#     matches the original HMM marginal variance up to a 10%-of-σ²_HMM floor on
#     the idiosyncratic share. If the loading β_i² · σ²_m would consume more
#     than 90% of σ²_HMM, β_i is clipped downward (sign preserved) to keep the
#     floor.
#   • the scaled ε_i is then copula rank-reordered to inject the cross-sectional
#     Student-t dependence structure.
#
# Properties: SIM β is recoverable by OLS, heavy tails / regime structure of the
# HMM marginal are preserved on the ε side, cross-sectional dependence preserved
# via the copula, marginal variance held close to the original HMM marginal.

println("\nLoading portfolio surrogate...")
portfolio = MyPortfolioSurrogateModel();
tickers = portfolio["tickers"];
marginals = portfolio["marginals"];
copula = portfolio["copula"];
K = length(tickers);

println("Loading SIM calibration table...")
calib = MySIMCalibration();
calib_tickers = calib["tickers"];
calib_alpha   = calib["alpha"];
calib_beta    = calib["beta"];
calib_r2      = calib["r_squared"];
calib_lookup = Dict{String, NamedTuple{(:α, :β, :r²), Tuple{Float64,Float64,Float64}}}();
for (i, t) ∈ enumerate(calib_tickers)
    calib_lookup[t] = (α = calib_alpha[i], β = calib_beta[i], r² = calib_r2[i]);
end
println("  $(length(calib_tickers)) calibrated tickers available")

# Tickers whose real-data SIM regression has R² above this threshold use the
# R²-preserving construction: ε is scaled so that the synthetic R² matches the
# real-data R² exactly. This recovers β AND R² for index ETFs (SPY, QQQ, SPYG)
# that practitioners expect to track the market by definition. SPY (R²=1.0)
# falls out as the limiting case σ²_ε = 0.
const R2_PRESERVE_THRESHOLD = 0.80

println("Generating $(K)-ticker paths via hybrid SIM ($(T_eff) growth-rate steps)...")
println("  α, β source: real S&P 500 VWAP regression on SPY (sim-calibration.jld2)")
println("  ε source: per-ticker HMM marginal draw (preserves tails, regimes)")
println("  Cross-sectional dependence: Student-t copula on ε")
println("  R²-preserving branch: tickers with real-data R² > $(R2_PRESERVE_THRESHOLD) target σ²_ε from real R²")

# sample copula uniforms -
# We sample T_DAYS uniforms and drop the first row to align with the trimmed
# growth-rate vectors (length T_eff = T_DAYS - 1).
U = JumpHMM.sample_dependence(copula, T_DAYS);  # T_DAYS × K

ticker_prices = Dict{String, Vector{Float64}}();
ticker_returns = Dict{String, Vector{Float64}}();
construction = Dict{String, String}();   # "r2-preserve", "hybrid", "hybrid-clipped", "fallback"

n_clipped = 0;
n_fallback = 0;
n_r2 = 0;
n_r2_inflate = 0;

for (j, ticker) ∈ enumerate(tickers)

    # --- 4a: pull α, β, R² from real-data calibration (fallback if missing) ---
    if haskey(calib_lookup, ticker)
        c = calib_lookup[ticker];
        α_i, β_i_raw, r²_real = c.α, c.β, c.r²;
        is_fallback = false;
    else
        α_i, β_i_raw, r²_real = (0.0, 1.0, 0.0);
        is_fallback = true;
        global n_fallback += 1;
        println("  [fallback] $(ticker): no calibration, using α=0, β=1");
    end

    # --- 4b: simulate the ticker's HMM marginal path and trim ---
    model_j = marginals[ticker];
    r_j = hmm_simulate(model_j, T_DAYS; n_paths=1);
    obs_full = Float64.(r_j.paths[1].observations);
    obs_j = obs_full[2:end];                # length T_eff
    σ²_HMM = var(obs_j);

    # --- 4c: choose construction branch based on real-data R² ---
    β_i = β_i_raw;
    local was_clipped = false;
    local branch = "hybrid";

    if !is_fallback && r²_real >= R2_PRESERVE_THRESHOLD

        # R²-PRESERVING BRANCH
        # Target σ²_ε so that synthetic R² = real-data R² exactly:
        #     R² = β² σ²_m / (β² σ²_m + σ²_ε)
        #   ⇒ σ²_ε = β² σ²_m · (1 - R²) / R²
        # SPY (R²=1.0) degenerates to σ²_ε = 0 automatically.
        if r²_real >= 1.0 - 1e-12
            σ²_ε_target = 0.0;
        else
            σ²_ε_target = β_i_raw^2 * σ²_market_synth * (1.0 - r²_real) / r²_real;
        end

        if σ²_ε_target > σ²_HMM
            global n_r2_inflate += 1;
            println("  [r2-inflate] $(ticker): target σ²_ε ($(round(σ²_ε_target, digits=2))) exceeds σ²_HMM ($(round(σ²_HMM, digits=2))); ε tails will inflate above HMM marginal");
        end

        # scale ε so that var(ε_scaled) = σ²_ε_target exactly
        ε_scaled = σ²_HMM > 0 && σ²_ε_target > 0 ?
                   sqrt(σ²_ε_target / σ²_HMM) .* obs_j :
                   zeros(T_eff);

        global n_r2 += 1;
        branch = "r2-preserve";
        println("  [r2-preserve] $(ticker): R²_real=$(round(r²_real, digits=3)), β=$(round(β_i, digits=3)), σ²_ε_target=$(round(σ²_ε_target, digits=3))");

    else

        # MARGINAL-PRESERVING (HYBRID) BRANCH
        # Target σ²_total = σ²_HMM via the variance correction with 10% floor
        # and β clipping fallback (see hybrid.md for the derivation).
        ratio = β_i_raw^2 * σ²_market_synth / σ²_HMM;
        if ratio > 0.90
            β_i = sign(β_i_raw) * sqrt(0.90 * σ²_HMM / σ²_market_synth);
            global n_clipped += 1;
            was_clipped = true;
            if n_clipped <= 10
                println("  [clip] $(ticker): β $(round(β_i_raw, digits=3)) → $(round(β_i, digits=3)) (would consume $(round(ratio*100, digits=1))% of σ²_HMM)");
            end
            s² = 0.10;
        else
            s² = 1.0 - ratio;
        end
        ε_scaled = sqrt(s²) .* obs_j;
        branch = is_fallback ? "fallback" : (was_clipped ? "hybrid-clipped" : "hybrid");
    end

    # --- 4d: copula rank-reorder the scaled ε ---
    sorted_eps = sort(ε_scaled);
    copula_ranks = ordinalrank(U[2:end, j]);   # length T_eff
    ε_reordered = sorted_eps[copula_ranks];

    # --- 4e: compose the SIM growth rates ---
    g_i = α_i .+ β_i .* G_market .+ ε_reordered;   # length T_eff

    construction[ticker] = branch;

    # --- 4f: convert to prices (length T_DAYS, seeded at $100) ---
    p_j = zeros(T_DAYS);
    p_j[1] = 100.0;
    for t ∈ 1:T_eff
        p_j[t+1] = p_j[t] * exp(g_i[t] * DT);
    end

    ticker_prices[ticker] = p_j;
    ticker_returns[ticker] = g_i;

    if j % 100 == 0 || j == K
        println("  [$(j)/$(K)] $(ticker) done");
    end
end

println("\nGeneration summary:")
println("  Tickers using real-data α,β: $(K - n_fallback)")
println("  Tickers via R²-preserving branch (R² > $(R2_PRESERVE_THRESHOLD)): $(n_r2)")
println("    of which target σ²_ε exceeded σ²_HMM (ε inflated): $(n_r2_inflate)")
println("  Tickers via marginal-preserving hybrid: $(K - n_r2 - n_fallback)")
println("    of which β clipped to floor: $(n_clipped)")
println("  Tickers using fallback (α=0, β=1, no calibration): $(n_fallback)")

# --- Step 5: Build and save the dataset ---
println("\nBuilding dataset...")

# create DataFrames matching the OHLC format (simplified: close prices only) -
# generate synthetic timestamps (trading days, no weekends)
using Dates
trading_dates = let
    start_date = Date(2000, 1, 3);
    dates = Date[];
    current = start_date;
    while length(dates) < T_DAYS
        if dayofweek(current) <= 5  # Mon-Fri
            push!(dates, current);
        end
        current += Day(1);
    end
    dates
end;

dataset = Dict{String, DataFrame}();
for ticker ∈ tickers
    p = ticker_prices[ticker];
    df = DataFrame(
        timestamp = trading_dates,
        close = p
    );
    dataset[ticker] = df;
end

# also save the market index -
market_df = DataFrame(
    timestamp = trading_dates,
    close = market_prices
);
dataset["MARKET"] = market_df;

# save -
output_path = joinpath(@__DIR__, "..", "src", "data", "synthetic-training-dataset.jld2");
println("Saving to: $(output_path)")
save(output_path, Dict(
    "dataset" => dataset,
    "tickers" => tickers,
    "market_ticker" => "MARKET",
    "n_tickers" => K,
    "n_days" => T_DAYS,
    "n_years" => T_YEARS,
    "selected_candidate" => Dict(
        "idx" => best.idx,
        "cagr" => best.cagr,
        "n_drawdowns_20" => best.n_drawdowns_20,
        "n_drawdowns_10" => best.n_drawdowns_10,
        "max_drawdown" => best.max_drawdown,
        "n_jump_clusters" => best.n_jump_clusters,
        "kurtosis" => best.kurtosis,
        "final_return" => best.final_return
    ),
    "market_prices" => market_prices,
    "market_jumps" => market_jumps,
    "market_returns" => G_market,
    "construction" => construction
));

println("\nDone! Saved $(K) tickers + MARKET index × $(T_DAYS) days ($(T_YEARS) years)")
println("="^70)

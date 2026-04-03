# generate-synthetic-training-data.jl
#
# Generates a frozen 20-year synthetic training dataset for all 424 tickers.
# Step 1: Generate candidate market index paths from the SPY surrogate
# Step 2: Score each candidate for realism (CAGR, drawdowns, jump clusters, kurtosis)
# Step 3: Select the best candidate
# Step 4: Generate correlated per-ticker paths using marginal surrogates + copula
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
selected_path = sim_result.paths[best.idx];
G_market = Float64.(selected_path.observations);
market_jumps = selected_path.jumps;

# convert to prices -
market_prices = zeros(T_DAYS);
market_prices[1] = 100.0;
for t ∈ 2:T_DAYS
    market_prices[t] = market_prices[t-1] * exp(G_market[t] * DT);
end

# --- Step 4: Generate per-ticker paths using SIM from market path ---
# Estimate SIM parameters (α, β, σ_ε) from real OHLC data, then use them
# to generate synthetic paths driven by the HMM market path.
# This preserves volatility clustering through the β·gₘ(t) term.

println("\nEstimating SIM parameters from real OHLC data...")
ohlc_data = MyTrainingMarketDataSet();
ohlc_dataset = ohlc_data["dataset"];

# compute real SPY market returns for SIM estimation -
spy_prices = Float64.(ohlc_dataset["SPY"][:, :close]);
T_ohlc = length(spy_prices);
real_market_returns = [(1.0/DT) * log(spy_prices[t]/spy_prices[t-1]) for t ∈ 2:T_ohlc];

# get tickers that have full history in OHLC data -
max_ohlc_days = maximum(nrow(ohlc_dataset[t]) for t ∈ keys(ohlc_dataset));
tickers = sort([t for t ∈ keys(ohlc_dataset) if nrow(ohlc_dataset[t]) == max_ohlc_days]);
K = length(tickers);
println("  $(K) tickers with full OHLC history")

# estimate SIM parameters for each ticker -
sim_params = Dict{String, NamedTuple{(:α, :β, :σ_ε), Tuple{Float64, Float64, Float64}}}();
for ticker ∈ tickers
    prices_t = Float64.(ohlc_dataset[ticker][:, :close]);
    asset_returns = [(1.0/DT) * log(prices_t[t]/prices_t[t-1]) for t ∈ 2:T_ohlc];
    est = estimate_sim(real_market_returns, asset_returns, ticker; δ=0.0, Δt=DT);
    sim_params[ticker] = (α=est.α, β=est.β, σ_ε=est.σ_ε);
end
println("  SIM parameters estimated for $(length(sim_params)) tickers")

# generate per-ticker paths via SIM: gᵢ(t) = αᵢ + βᵢ·gₘ(t) + εᵢ(t)
# where gₘ(t) is the HMM-generated market growth rate (temporal structure preserved)
println("\nGenerating $(K)-ticker paths via SIM from HMM market path...")
println("  Volatility clustering inherited through β·gₘ(t)")

ticker_prices = Dict{String, Vector{Float64}}();
ticker_returns = Dict{String, Vector{Float64}}();

for (j, ticker) ∈ enumerate(tickers)

    (αᵢ, βᵢ, σ_εᵢ) = sim_params[ticker];

    # generate returns via SIM: αᵢ + βᵢ·gₘ(t) + εᵢ(t) -
    g_ticker = zeros(T_DAYS);
    for t ∈ 1:T_DAYS
        g_ticker[t] = αᵢ + βᵢ * G_market[t] + σ_εᵢ * sqrt(DT) * randn();
    end

    # convert to prices -
    p_j = zeros(T_DAYS);
    p_j[1] = 100.0;
    for t ∈ 2:T_DAYS
        p_j[t] = p_j[t-1] * exp(g_ticker[t] * DT);
    end

    ticker_prices[ticker] = p_j;
    ticker_returns[ticker] = g_ticker;

    if j % 100 == 0 || j == K
        println("  [$(j)/$(K)] $(ticker) — β=$(round(βᵢ, digits=2))");
    end
end

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
    "sim_parameters" => sim_params
));

println("\nDone! Saved $(K) tickers + MARKET index × $(T_DAYS) days ($(T_YEARS) years)")
println("="^70)

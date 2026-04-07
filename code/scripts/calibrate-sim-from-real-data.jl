# calibrate-sim-from-real-data.jl
#
# Calibrate per-ticker Single Index Model (SIM) parameters (α, β, σ_ε) from
# real S&P 500 OHLC data, regressing each ticker's VWAP growth rate on SPY's
# VWAP growth rate over the full 2014-01-03 → 2024-12-31 window.
#
# Output: ../src/data/sim-calibration.jld2
# Loaded via: MySIMCalibration()
#
# Usage: julia --project=.. calibrate-sim-from-real-data.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using eCornellAIFinance
using DataFrames
using JLD2
using Statistics

# --- Configuration ---
const DT = 1.0 / 252.0
const MARKET_TICKER = "SPY"

println("="^70)
println("  SIM CALIBRATION FROM REAL S&P 500 DATA")
println("="^70)

# --- Step 1: Load the real OHLC training dataset ---
println("\nLoading real OHLC training dataset...")
raw = MyTrainingMarketDataSet();
dataset = raw["dataset"];   # Dict{String, DataFrame}, one entry per ticker
all_keys = collect(keys(dataset));
println("  $(length(all_keys)) tickers in file")

# --- Step 2: Identify tickers with full coverage ---
# "Full coverage" = matches the maximum number of trading days observed across the file.
ticker_lengths = Dict{String,Int}();
for k ∈ all_keys
    ticker_lengths[k] = nrow(dataset[k]);
end
max_T = maximum(values(ticker_lengths));
full_tickers = sort([k for k ∈ all_keys if ticker_lengths[k] == max_T]);
println("  Max trading-day count: $(max_T)")
println("  Tickers with full coverage: $(length(full_tickers))")

if !(MARKET_TICKER ∈ full_tickers)
    error("$(MARKET_TICKER) is not in the full-coverage set; cannot calibrate.")
end

# --- Step 3: Compute SPY growth rates from VWAP ---
spy_df = dataset[MARKET_TICKER];
spy_vwap = Float64.(spy_df[:, :volume_weighted_average_price]);
spy_g = [(1.0/DT) * log(spy_vwap[t]/spy_vwap[t-1]) for t ∈ 2:length(spy_vwap)];
σ_market = std(spy_g);
T_obs = length(spy_g);

# extract calibration window (first and last timestamps from SPY) -
spy_timestamps = spy_df[:, :timestamp];
window_start = string(spy_timestamps[1]);
window_end = string(spy_timestamps[end]);

println("\nMarket proxy: $(MARKET_TICKER)")
println("  Calibration window: $(window_start) → $(window_end)")
println("  Daily growth-rate observations: $(T_obs)")
println("  σ_market (annualized growth rate scale): $(round(σ_market, digits=4))")

# --- Step 4: Calibrate SIM for every full-coverage ticker ---
println("\nCalibrating SIM parameters for $(length(full_tickers)) tickers...")

tickers_out  = String[];
alpha_out    = Float64[];
beta_out     = Float64[];
sigma_eps_out = Float64[];
r2_out       = Float64[];

for (i, ticker) ∈ enumerate(full_tickers)

    df = dataset[ticker];
    vwap = Float64.(df[:, :volume_weighted_average_price]);

    # guard: VWAP must be strictly positive to take logs
    if any(vwap .<= 0.0)
        println("  [skip] $(ticker): non-positive VWAP encountered")
        continue
    end

    g = [(1.0/DT) * log(vwap[t]/vwap[t-1]) for t ∈ 2:length(vwap)];
    if length(g) != T_obs
        println("  [skip] $(ticker): length mismatch ($(length(g)) vs $(T_obs))")
        continue
    end

    est = estimate_sim(spy_g, g, ticker; δ=0.0, Δt=DT);

    push!(tickers_out, est.ticker);
    push!(alpha_out, est.α);
    push!(beta_out, est.β);
    push!(sigma_eps_out, est.σ_ε);
    push!(r2_out, est.r²);

    if i % 50 == 0 || i == length(full_tickers)
        println("  [$(i)/$(length(full_tickers))] $(ticker) done")
    end
end

println("\nCalibrated $(length(tickers_out)) tickers")

# --- Step 5: Quick sanity summary ---
println("\nSummary statistics across calibrated universe:")
println("  β  → mean=$(round(mean(beta_out), digits=3))  median=$(round(median(beta_out), digits=3))  min=$(round(minimum(beta_out), digits=3))  max=$(round(maximum(beta_out), digits=3))")
println("  α  → mean=$(round(mean(alpha_out), digits=4))  median=$(round(median(alpha_out), digits=4))")
println("  R² → mean=$(round(mean(r2_out), digits=3))    median=$(round(median(r2_out), digits=3))")

# spot-check a few well-known tickers if present -
for spot ∈ ("NVDA", "AAPL", "JNJ", "KO", "JPM", "TSLA")
    idx = findfirst(==(spot), tickers_out);
    if idx !== nothing
        println("  $(spot): α=$(round(alpha_out[idx], digits=4))  β=$(round(beta_out[idx], digits=3))  R²=$(round(r2_out[idx], digits=3))")
    end
end

# --- Step 6: Save to JLD2 ---
output_path = joinpath(@__DIR__, "..", "src", "data", "sim-calibration.jld2");
println("\nSaving calibration to: $(output_path)")
save(output_path, Dict(
    "tickers"      => tickers_out,
    "alpha"        => alpha_out,
    "beta"         => beta_out,
    "sigma_eps"    => sigma_eps_out,
    "r_squared"    => r2_out,
    "sigma_market" => σ_market,
    "n_obs"        => T_obs,
    "window_start" => window_start,
    "window_end"   => window_end,
    "market_ticker"=> MARKET_TICKER,
));

println("\nDone.")
println("="^70)

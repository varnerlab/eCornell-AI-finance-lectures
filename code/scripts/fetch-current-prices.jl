# fetch-current-prices.jl
#
# Build the "current prices" snapshot used by the buy-and-hold simulation in
# the BuildMinVariancePortfolio notebook (and downstream Sharpe / stress-test
# notebooks). For every ticker in the SIM calibration universe we pull the
# last available daily close from `MyTestingMarketDataSet()` (real S&P 500
# OHLC through 2025-12-31, sourced from Polygon — same provider as the
# training data the calibration was built on). Tickers missing from the
# testing dataset fall back to the last close from `MyTrainingMarketDataSet()`
# (2014-01-03 → 2024-12-31). The result is saved to a JLD2 cache that the
# notebook loads via `MyCurrentPrices()` — students get realistic share
# counts without any external network access.
#
# Re-run this script whenever a fresher OHLC snapshot is committed. There is
# no external HTTP, no API key, no rate limit, no Yahoo WAF.
#
# Source:    MyTestingMarketDataSet (last row, :close field) → MyTrainingMarketDataSet fallback
# Output:    ../src/data/current-prices.jld2
# Loader:    MyCurrentPrices()
#
# Usage: julia --project=.. fetch-current-prices.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using eCornellAIFinance
using DataFrames
using JLD2
using Dates

const SOURCE_LABEL = "polygon-ohlc-testing-dataset";

println("="^70)
println("  BUILD CURRENT PRICES SNAPSHOT")
println("="^70)

# --- Step 1: load the universe and the two OHLC sources ---

println("\nLoading SIM calibration universe...")
calib = MySIMCalibration();
tickers = calib["tickers"]::Vector{String};
println("  $(length(tickers)) tickers in universe")

println("Loading MyTestingMarketDataSet (primary, 2025 closes)...")
testing_ds  = MyTestingMarketDataSet()["dataset"];
println("  $(length(testing_ds)) tickers in testing dataset")

println("Loading MyTrainingMarketDataSet (fallback, 2024 closes)...")
training_ds = MyTrainingMarketDataSet()["dataset"];
println("  $(length(training_ds)) tickers in training dataset")

# --- Step 2: pull the last-row close for each calibration ticker ---

prices         = Vector{Float64}(undef, length(tickers));
sources        = Vector{String}(undef, length(tickers));
last_dates     = Vector{String}(undef, length(tickers));
n_testing      = 0;
n_training_fb  = 0;
n_missing      = 0;
fallback_list  = String[];
missing_list   = String[];

for (i, t) ∈ enumerate(tickers)
    if haskey(testing_ds, t) && nrow(testing_ds[t]) > 0
        df = testing_ds[t];
        prices[i]     = Float64(df[end, :close]);
        last_dates[i] = string(df[end, :timestamp]);
        sources[i]    = "testing";
        global n_testing += 1;
    elseif haskey(training_ds, t) && nrow(training_ds[t]) > 0
        df = training_ds[t];
        prices[i]     = Float64(df[end, :close]);
        last_dates[i] = string(df[end, :timestamp]);
        sources[i]    = "training_fallback";
        global n_training_fb += 1;
        push!(fallback_list, t);
    else
        prices[i]     = NaN;
        last_dates[i] = "";
        sources[i]    = "missing";
        global n_missing += 1;
        push!(missing_list, t);
    end
end

# --- Step 3: report ---

println("\nResolution summary:")
println("  Testing  (primary):    $(n_testing)/$(length(tickers))")
println("  Training (fallback):   $(n_training_fb)/$(length(tickers))")
println("  Missing:               $(n_missing)/$(length(tickers))")

if !isempty(fallback_list)
    println("\nTickers using training-data fallback (not in testing dataset):")
    for t in fallback_list
        println("  $(t)")
    end
end
if !isempty(missing_list)
    println("\nTickers with NO price (NaN saved):")
    for t in missing_list
        println("  $(t)")
    end
end

println("\nSpot check (well-known names):")
for spot in ("SPY", "AAPL", "MSFT", "NVDA", "JNJ", "TSLA", "JPM", "PG", "GS", "AMD", "BA")
    idx = findfirst(==(spot), tickers);
    if idx !== nothing
        println("  $(rpad(spot, 6)) \$$(rpad(round(prices[idx], digits=2), 8))  "
                * "[$(sources[idx]), as of $(last_dates[idx])]")
    end
end

# --- Step 4: save ---

output_path = joinpath(@__DIR__, "..", "src", "data", "current-prices.jld2");
fetched_at = string(Dates.now());

save(output_path, Dict(
    "tickers"       => tickers,
    "prices"        => prices,
    "sources"       => sources,
    "last_dates"    => last_dates,
    "fetched_at"    => fetched_at,
    "source_label"  => SOURCE_LABEL,
    "n_testing"     => n_testing,
    "n_fallback"    => n_training_fb,
    "n_missing"     => n_missing,
));

println("\nSaved to: $(output_path)")
println("  fetched_at:   $(fetched_at)")
println("  source_label: $(SOURCE_LABEL)")
println("="^70)

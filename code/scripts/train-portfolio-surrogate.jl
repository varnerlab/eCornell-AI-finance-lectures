# train-portfolio-surrogate.jl
#
# Pre-trains marginal JumpHMM models for every ticker in the S&P 500 dataset,
# then fits a Student-t copula on the joint returns. The resulting PortfolioModel
# can generate correlated multi-asset synthetic paths of any length.
#
# Usage: julia --project=.. train-portfolio-surrogate.jl
#
# Output: ../src/data/pretrained-portfolio-surrogate.jld2

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using eCornellAIFinance
using DataFrames
using JLD2
using Statistics
using JumpHMM: PortfolioModel, StudentTCopula, GaussianCopula

# --- Configuration ---
const N_STATES = 50
const NU = 5.0
const RF = 0.0
const DT = 1.0 / 252.0
const MIN_OBS_DAYS = 1000          # skip tickers with fewer days
const COPULA_TYPE = StudentTCopula  # or GaussianCopula

# tune grid — moderate for batch training
const EPSILON_RANGE = range(1e-4, 2.5e-2, length=10)
const LAMBDA_RANGE = range(10.0, 120.0, length=10)
const N_TUNE_PATHS = 100

# --- Load data ---
println("="^70)
println("  PORTFOLIO SURROGATE TRAINING")
println("="^70)

data = MyTrainingMarketDataSet();
dataset = data["dataset"];
all_tickers = sort(collect(keys(dataset)));
println("Total tickers in dataset: $(length(all_tickers))")

# filter by minimum length
valid_tickers = String[];
for t ∈ all_tickers
    if nrow(dataset[t]) >= MIN_OBS_DAYS
        push!(valid_tickers, t);
    end
end
println("Tickers with ≥ $(MIN_OBS_DAYS) days: $(length(valid_tickers))")

# find common date range (max overlap)
max_days = maximum(nrow(dataset[t]) for t ∈ valid_tickers);
full_tickers = [t for t ∈ valid_tickers if nrow(dataset[t]) == max_days];
println("Tickers with full $(max_days) days: $(length(full_tickers))")

# --- Build price matrix for full-length tickers ---
# use full_tickers for both marginal fitting and copula
println("\nBuilding price matrix for $(length(full_tickers)) tickers...")
price_matrix = zeros(max_days, length(full_tickers));
for (j, t) ∈ enumerate(full_tickers)
    price_matrix[:, j] = Float64.(dataset[t][:, :close]);
end

# --- Fit marginal models ---
println("\nFitting marginal JumpHMM models (N=$(N_STATES), ν=$(NU))...")
println("This will take ~$(round(length(full_tickers) * 10 / 60, digits=0)) minutes...\n")

marginals = Dict{String, JumpHiddenMarkovModel}();
failed_tickers = String[];
t_start = time();

for (j, ticker) ∈ enumerate(full_tickers)

    try
        prices_j = price_matrix[:, j];

        # fit -
        model = hmm_fit(JumpHiddenMarkovModel, prices_j; rf=RF, N=N_STATES, ν=NU, dt=DT);

        # tune -
        model = hmm_tune(model, prices_j;
            ϵ_range=EPSILON_RANGE, λ_range=LAMBDA_RANGE, n_paths=N_TUNE_PATHS);

        marginals[ticker] = model;

        if j % 50 == 0 || j == length(full_tickers)
            elapsed = round(time() - t_start, digits=0);
            rate = round(j / elapsed * 60, digits=1);
            eta = round((length(full_tickers) - j) / (j / elapsed) / 60, digits=1);
            println("  [$(j)/$(length(full_tickers))] $(ticker) ✓ " *
                    "(ε=$(round(model.jump.ϵ, digits=5)), λ=$(round(model.jump.λ, digits=1))) " *
                    "[$(elapsed)s elapsed, $(rate)/min, ETA $(eta)min]");
        end

    catch e
        push!(failed_tickers, ticker);
        if j % 50 == 0
            println("  [$(j)/$(length(full_tickers))] $(ticker) ✗ $(e)");
        end
    end
end

t_total = round(time() - t_start, digits=1);
println("\nMarginal fitting complete:")
println("  Succeeded: $(length(marginals))")
println("  Failed: $(length(failed_tickers))")
println("  Total time: $(t_total)s ($(round(t_total/60, digits=1)) min)")

if length(failed_tickers) > 0
    println("  Failed tickers: $(join(failed_tickers[1:min(20, length(failed_tickers))], ", "))")
end

# --- Fit Student-t copula ---
println("\nFitting $(COPULA_TYPE) copula on $(length(marginals)) tickers...")

# build returns matrix for tickers that succeeded -
fitted_tickers = sort(collect(keys(marginals)));
n_fitted = length(fitted_tickers);
ticker_indices = [findfirst(==(t), full_tickers) for t ∈ fitted_tickers];

returns_matrix = zeros(max_days - 1, n_fitted);
for (k, j) ∈ enumerate(ticker_indices)
    for t ∈ 2:max_days
        returns_matrix[t-1, k] = (1.0 / DT) * log(price_matrix[t, j] / price_matrix[t-1, j]);
    end
end

copula = JumpHMM.fit(COPULA_TYPE, returns_matrix);
println("  Copula fitted (ν = $(copula isa StudentTCopula ? copula.ν : "N/A"))")
println("  Correlation matrix size: $(size(copula.Σ))")

# --- Save ---
output_path = joinpath(@__DIR__, "..", "src", "data", "pretrained-portfolio-surrogate.jld2");
println("\nSaving to: $(output_path)")
save(output_path, Dict(
    "tickers" => fitted_tickers,
    "marginals" => marginals,
    "copula" => copula,
    "copula_type" => string(COPULA_TYPE),
    "n_states" => N_STATES,
    "nu" => NU,
    "rf" => RF,
    "dt" => DT,
    "n_training_days" => max_days,
    "n_tickers" => n_fitted,
    "failed_tickers" => failed_tickers
));

println("\nDone! Saved $(n_fitted) marginal models + $(COPULA_TYPE) copula.")
println("="^70)

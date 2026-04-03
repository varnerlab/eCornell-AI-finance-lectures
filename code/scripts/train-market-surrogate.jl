# train-market-surrogate.jl
#
# Pre-trains a JumpHMM market surrogate model on the full SPY price history
# (2014-2024) and saves the fitted model for use across all course sessions.
#
# Usage: julia --project=.. train-market-surrogate.jl
#
# Output: ../src/data/pretrained-jumphmm-market-surrogate.jld2

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using eCornellAIFinance
using JLD2
using Statistics

# --- Configuration ---
const TICKER = "SPY"                  # market index proxy
const N_STATES = 50                   # number of HMM states
const NU = 5.0                        # Student-t degrees of freedom
const RF = 0.0                        # no risk-free adjustment for now
const DT = 1.0 / 252.0               # daily time step

# tune grid — finer than notebook defaults for production quality
const EPSILON_RANGE = range(1e-4, 3.0e-2, length=15)
const LAMBDA_RANGE = range(5.0, 150.0, length=15)
const N_TUNE_PATHS = 200

# --- Load SPY prices ---
println("Loading $(TICKER) price data...")
data = MyTrainingMarketDataSet();
dataset = data["dataset"];
spy_df = dataset[TICKER];
prices = Float64.(spy_df[:, :close]);
println("  $(length(prices)) trading days: $(spy_df[1,:timestamp]) to $(spy_df[end,:timestamp])")
println("  Price range: \$$(round(prices[1], digits=2)) to \$$(round(prices[end], digits=2))")

# --- Fit the model ---
println("\nFitting JumpHMM (N=$(N_STATES) states, ν=$(NU))...")
t_fit = @elapsed begin
    model = hmm_fit(JumpHiddenMarkovModel, prices; rf=RF, N=N_STATES, ν=NU, dt=DT);
end
println("  Fit complete in $(round(t_fit, digits=1))s")

# --- Tune jump parameters ---
println("\nTuning jump parameters ($(length(EPSILON_RANGE))×$(length(LAMBDA_RANGE)) grid, $(N_TUNE_PATHS) paths per point)...")
t_tune = @elapsed begin
    model = hmm_tune(model, prices;
        ϵ_range=EPSILON_RANGE,
        λ_range=LAMBDA_RANGE,
        n_paths=N_TUNE_PATHS);
end
println("  Tune complete in $(round(t_tune, digits=1))s")
println("  Jump probability (ε): $(round(model.jump.ϵ, digits=6))")
println("  Jump duration mean (λ): $(round(model.jump.λ, digits=1))")

# --- Validate: quick simulation check ---
println("\nValidating: generating 10 sample paths...")
result = hmm_simulate(model, 252; n_paths=10);
sim_returns = Float64[];
for p in result.paths
    G = p.observations;
    push!(sim_returns, mean(G) * 252);  # annualized
end
println("  Simulated annualized returns: $(round.(sim_returns, digits=3))")

# --- Save ---
output_path = joinpath(@__DIR__, "..", "src", "data", "pretrained-jumphmm-market-surrogate.jld2");
println("\nSaving model to: $(output_path)")
save(output_path, Dict(
    "model" => model,
    "ticker" => TICKER,
    "n_states" => N_STATES,
    "nu" => NU,
    "rf" => RF,
    "dt" => DT,
    "n_training_days" => length(prices),
    "training_date_range" => string(spy_df[1,:timestamp]) * " to " * string(spy_df[end,:timestamp])
));
println("Done!")

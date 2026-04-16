# build-intro-cache.jl
#
# Run once (author-side) to cache the 5,000-path hybrid forward simulation for
# the S1 intro notebook. The intro notebook loads this cache and plots the
# cones figure directly — no heavy compute on the student's machine.
#
# Scenario: the "Balanced" archetype from interview.md (22 tickers, fully invested,
# R_target = 10%/yr, concentration cap = 20%, B₀ = $100,000). This matches Maya's
# client in the narrative opener.
#
# Output: data/intro-balanced-cones.jld2

include("Include.jl")

# --- Balanced archetype (from interview.md Step 9) ------------------------------------
const BALANCED_TICKERS = [
    "DIS", "VZ",
    "AMZN", "HD",
    "PG", "COST",
    "XOM", "CVX",
    "JPM", "V", "BAC",
    "JNJ", "UNH", "LLY",
    "HON", "CAT",
    "AAPL", "MSFT", "NVDA",
    "SHW",
    "AMT",
    "NEE",
];

# Balanced archetype defaults (interview.md Step 6/8; g_f from portfolio-config.toml default)
B₀           = 100_000.0;
g_f          = 0.045;
R_target     = 0.10;
max_weight   = 0.20;

# Simulation settings (same as the stress-test example so the cones match visually)
n_paths      = 5_000;
n_steps      = 252;
Δt           = 1.0 / 252.0;
seed         = 2026;

# --- Step 1: Load SIM parameter estimates for the Balanced subset --------------------
sim_data      = load_results(joinpath(_PATH_TO_DATA, "sim-parameter-estimates.jld2"));
all_estimates = sim_data["sim_estimates"];
est_lookup    = Dict(est.ticker => est for est ∈ all_estimates);
σ_m           = sim_data["sigma_market"];

my_tickers  = BALANCED_TICKERS;
N           = length(my_tickers);
missing_t   = [t for t ∈ my_tickers if !haskey(est_lookup, t)];
@assert isempty(missing_t) "Tickers not in SIM estimates: $(missing_t)"
sim_subset  = [est_lookup[t] for t ∈ my_tickers];

println("Universe: $(N) tickers (Balanced archetype)")
println("  $(my_tickers)")

# --- Step 2: Expected growth rates per ticker (SIM: E[g_i] = α_i + β_i · E[g_m]) -----
ds     = MySyntheticTrainingDataSet();
Eg_mkt = mean(ds["market_returns"]);
Eg_vec = [est.α + est.β * Eg_mkt for est ∈ sim_subset];

# --- Step 3: Solve the long-only min-var QP at R_target with the concentration cap ---
Σ         = build_sim_covariance(sim_subset, σ_m);
bounds    = hcat(zeros(N), fill(max_weight, N));
problem   = build(MyPortfolioAllocationProblem;
    μ = Eg_vec, Σ = Σ, bounds = bounds, R = R_target);
mv        = solve_minvariance(problem);
weights   = mv.weights;

println("Min-var solved: exp. growth = $(round(mv.expected_return*100, digits=2))%/yr, "
        * "vol = $(round(sqrt(mv.variance)/sqrt(252)*100, digits=2))%/yr")
println("Top weights (%):")
order = sortperm(weights, rev = true)
for i ∈ order[1:min(8, N)]
    println("  $(my_tickers[i])  $(round(weights[i]*100, digits=2))%")
end

# --- Step 4: Start prices from the January 2025 snapshot -----------------------------
snap        = MyCurrentPrices();
snap_lookup = Dict(snap["tickers"] .=> snap["prices"]);
start_prices = Dict(t => snap_lookup[t] for t ∈ my_tickers);

# --- Step 5: Generate the 5,000-path hybrid forward scenario -------------------------
market_model = MyMarketSurrogateModel();
portfolio    = MyPortfolioSurrogateModel();

println("\nGenerating $(n_paths)-path hybrid scenario ($(n_steps) days)...")
scen = generate_hybrid_scenario(market_model, portfolio, sim_data, my_tickers;
    n_paths = n_paths,
    n_steps = n_steps,
    Δt      = Δt,
    start_prices = start_prices,
    label   = "Maya-Intro-Balanced ($(n_paths)p × $(n_steps)d)",
    seed    = seed);
println("  scenario.price_paths shape:  $(size(scen.price_paths))")

# --- Step 6: Buy-and-hold backtest for the Min-Var allocation ------------------------
result_mv = backtest_buyhold(scen, my_tickers; B₀ = B₀, offset = 1, weights = weights);

final_wealth = result_mv.final_wealth;
max_dd       = result_mv.max_drawdowns;
println("\nDistributional summary (Min-Var, buy-and-hold, 5,000 paths):")
println("  Median terminal wealth: \$$(round(median(final_wealth), digits=0))")
println("  P5 terminal wealth:     \$$(round(quantile(final_wealth, 0.05), digits=0))")
println("  P95 terminal wealth:    \$$(round(quantile(final_wealth, 0.95), digits=0))")
println("  Median max drawdown:    $(round(median(max_dd)*100, digits=1))%")

# --- Step 7: Save the cache (wealth_paths is the load-bearing payload) ---------------
out = Dict{String, Any}(
    "archetype"          => "Balanced",
    "my_tickers"         => my_tickers,
    "allocation_weights" => weights,
    "B_0"                => B₀,
    "g_f"                => g_f,
    "R_target"           => R_target,
    "max_weight"         => max_weight,
    "n_paths"            => n_paths,
    "n_steps"            => n_steps,
    "dt"                 => Δt,
    "seed"               => seed,
    "sigma_market"       => σ_m,
    "wealth_paths"       => result_mv.wealth_paths,
    "final_wealth"       => final_wealth,
    "max_drawdowns"      => max_dd,
    "sharpe_ratios"      => result_mv.sharpe_ratios,
    "scenario_label"     => scen.label,
    "expected_return"    => mv.expected_return,
    "variance"           => mv.variance,
);

cache_path = joinpath(_PATH_TO_DATA, "intro-balanced-cones.jld2");
save_results(cache_path, out);
sz_mb = round(stat(cache_path).size / 1e6; digits = 1);
println("\nSaved: $(cache_path)  ($(sz_mb) MB)")

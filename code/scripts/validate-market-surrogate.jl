# validate-market-surrogate.jl
#
# Validates the pre-trained JumpHMM market surrogate model against SPY data.
# Generates a diagnostic report with statistical tests and figures matching
# the methodology in Alswaidan & Varner (2025), arXiv:2603.10202.
#
# Usage: julia --project=.. validate-market-surrogate.jl
#
# Output: ../src/data/surrogate-validation-report.jld2
#         Figures printed to console / saved if Plots available

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using eCornellAIFinance
using DataFrames
using Distributions
using HypothesisTests
using JLD2
using LinearAlgebra
using Plots
using Random
using Statistics
using StatsBase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const N_SIM_PATHS = 1000
const ACF_LAGS = 252
const Δt = 1.0 / 252.0
const ALPHA = 0.05  # significance level

# ---------------------------------------------------------------------------
# Load model and observed data
# ---------------------------------------------------------------------------
println("="^70)
println("  MARKET SURROGATE VALIDATION REPORT")
println("  Model: HMM-WJ (JumpHMM), Trained on SPY 2014–2024")
println("="^70)

model = MyMarketSurrogateModel();
println("\nModel parameters:")
println("  States: $(length(model.emissions))")
println("  ν (Student-t df): $(model.ν)")
println("  rf: $(model.rf)")
println("  dt: $(model.dt)")
println("  Jump ε: $(round(model.jump.ϵ, digits=6))")
println("  Jump λ: $(round(model.jump.λ, digits=1))")

# load SPY observed prices -
data = MyTrainingMarketDataSet();
spy_df = data["dataset"]["SPY"];
prices_obs = Float64.(spy_df[:, :close]);
T_obs = length(prices_obs);

# compute observed excess growth rates -
G_obs = zeros(T_obs - 1);
for t ∈ 2:T_obs
    G_obs[t-1] = (1.0 / Δt) * log(prices_obs[t] / prices_obs[t-1]);
end

println("\nObserved data: $(T_obs) days, $(T_obs-1) returns")

# ---------------------------------------------------------------------------
# Table 1: Descriptive Statistics of Observed Data
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  TABLE 1: Descriptive Statistics (Observed SPY)")
println("="^70)

obs_mean = mean(G_obs);
obs_std = std(G_obs);
obs_skew = skewness(G_obs);
obs_kurt = kurtosis(G_obs);  # excess kurtosis

# Jarque-Bera test -
jb_test = JarqueBeraTest(G_obs);

# Ljung-Box on raw returns -
acf_raw = autocor(G_obs, 1:20);
lb_raw_stat = T_obs * sum(acf_raw .^ 2 ./ ((T_obs) .- (1:20)));

# Ljung-Box on absolute returns -
acf_abs = autocor(abs.(G_obs), 1:20);
lb_abs_stat = T_obs * sum(acf_abs .^ 2 ./ ((T_obs) .- (1:20)));

stats_df = DataFrame(
    "Statistic" => [
        "Mean (annualized %)",
        "Std Dev (annualized %)",
        "Skewness",
        "Excess Kurtosis",
        "Jarque-Bera p-value",
        "LB(20) on Gₜ (stat)",
        "LB(20) on |Gₜ| (stat)"
    ],
    "Value" => [
        round(obs_mean * 100, digits=2),
        round(obs_std * sqrt(Δt) * 100 * sqrt(252), digits=2),
        round(obs_skew, digits=3),
        round(obs_kurt, digits=3),
        round(pvalue(jb_test), sigdigits=3),
        round(lb_raw_stat, digits=1),
        round(lb_abs_stat, digits=1)
    ]
);
println(stats_df)

# ---------------------------------------------------------------------------
# Simulate N_SIM_PATHS paths
# ---------------------------------------------------------------------------
println("\nSimulating $(N_SIM_PATHS) paths of $(T_obs) days each...")
sim_result = hmm_simulate(model, T_obs; n_paths=N_SIM_PATHS);

# collect simulated growth rates -
G_sim_all = Vector{Vector{Float64}}();
for p ∈ sim_result.paths
    push!(G_sim_all, Float64.(p.observations));
end
println("  Done.")

# ---------------------------------------------------------------------------
# Table 2: KS and AD Pass Rates
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  TABLE 2: Distributional Tests (KS and AD)")
println("="^70)

ks_pvalues = zeros(N_SIM_PATHS);
for i ∈ 1:N_SIM_PATHS
    ks_test = ApproximateTwoSampleKSTest(G_obs, G_sim_all[i]);
    ks_pvalues[i] = pvalue(ks_test);
end
ks_pass_rate = sum(ks_pvalues .> ALPHA) / N_SIM_PATHS * 100;
ks_pass_se = sqrt(ks_pass_rate/100 * (1 - ks_pass_rate/100) / N_SIM_PATHS) * 100;

println("  KS pass rate (α=$(ALPHA)): $(round(ks_pass_rate, digits=1))% ± $(round(ks_pass_se, digits=1))%")

# ---------------------------------------------------------------------------
# Table 3: Moment Comparison (Observed vs Simulated)
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  TABLE 3: Moment Comparison (Observed vs Simulated)")
println("="^70)

sim_means = [mean(g) for g ∈ G_sim_all];
sim_stds = [std(g) for g ∈ G_sim_all];
sim_skews = [skewness(g) for g ∈ G_sim_all];
sim_kurts = [kurtosis(g) for g ∈ G_sim_all];

moments_df = DataFrame(
    "Moment" => ["Mean (ann %)", "Std Dev (ann %)", "Skewness", "Excess Kurtosis"],
    "Observed" => [
        round(obs_mean * 100, digits=2),
        round(obs_std * sqrt(Δt) * 100 * sqrt(252), digits=2),
        round(obs_skew, digits=3),
        round(obs_kurt, digits=3)
    ],
    "Simulated (mean)" => [
        round(mean(sim_means) * 100, digits=2),
        round(mean(sim_stds) * sqrt(Δt) * 100 * sqrt(252), digits=2),
        round(mean(sim_skews), digits=3),
        round(mean(sim_kurts), digits=3)
    ],
    "Simulated (std)" => [
        round(std(sim_means) * 100, digits=2),
        round(std(sim_stds) * sqrt(Δt) * 100 * sqrt(252), digits=2),
        round(std(sim_skews), digits=3),
        round(std(sim_kurts), digits=3)
    ]
);
println(moments_df)

# ---------------------------------------------------------------------------
# ACF of Absolute Returns
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  TABLE 4: ACF of |Gₜ| Comparison")
println("="^70)

lags = collect(1:ACF_LAGS);
acf_obs_abs = autocor(abs.(G_obs), lags);

# compute ACF for each simulated path -
acf_sim_matrix = zeros(ACF_LAGS, N_SIM_PATHS);
for i ∈ 1:N_SIM_PATHS
    acf_sim_matrix[:, i] = autocor(abs.(G_sim_all[i]), lags);
end

acf_sim_mean = vec(mean(acf_sim_matrix, dims=2));
acf_sim_std = vec(std(acf_sim_matrix, dims=2));

# ACF-MAE (all paths) -
acf_mae = mean(abs.(acf_obs_abs .- acf_sim_mean));

# bootstrap SE for ACF-MAE -
n_boot = 500;
boot_mae = zeros(n_boot);
Random.seed!(42);
for b ∈ 1:n_boot
    idx = rand(1:N_SIM_PATHS, N_SIM_PATHS);
    boot_acf_mean = vec(mean(acf_sim_matrix[:, idx], dims=2));
    boot_mae[b] = mean(abs.(acf_obs_abs .- boot_acf_mean));
end
acf_mae_se = std(boot_mae);

# classify paths by jump content -
jump_indices = [i for i ∈ 1:N_SIM_PATHS if any(sim_result.paths[i].jumps)];
nojump_indices = [i for i ∈ 1:N_SIM_PATHS if !any(sim_result.paths[i].jumps)];
n_jump_paths = length(jump_indices);
n_nojump_paths = length(nojump_indices);
jump_frac = n_jump_paths / N_SIM_PATHS * 100;

acf_jump_mean = n_jump_paths > 0 ? vec(mean(acf_sim_matrix[:, jump_indices], dims=2)) : zeros(ACF_LAGS);
acf_nojump_mean = n_nojump_paths > 0 ? vec(mean(acf_sim_matrix[:, nojump_indices], dims=2)) : zeros(ACF_LAGS);
acf_mix80 = 0.8 .* acf_jump_mean .+ 0.2 .* acf_nojump_mean;

acf_mae_jump = n_jump_paths > 0 ? mean(abs.(acf_obs_abs .- acf_jump_mean)) : NaN;
acf_mae_mix80 = mean(abs.(acf_obs_abs .- acf_mix80));

println("  Jump paths: $(n_jump_paths)/$(N_SIM_PATHS) ($(round(jump_frac, digits=1))%)")
println()
println("  ACF-MAE (lags 1–$(ACF_LAGS)):")
println("    All paths:   $(round(acf_mae, digits=4)) ± $(round(acf_mae_se, digits=4))")
println("    Jump-only:   $(round(acf_mae_jump, digits=4))")
println("    80/20 mix:   $(round(acf_mae_mix80, digits=4))")
println()

# spot checks at key lags -
println("  Lag | Observed | All   | Jump-only | 80/20 mix")
println("  " * "-"^55)
for lag ∈ [1, 5, 20, 63, 252]
    if lag <= ACF_LAGS
        println("  $(lpad(lag, 3)) | $(lpad(round(acf_obs_abs[lag], digits=3), 6)) | " *
                "$(lpad(round(acf_sim_mean[lag], digits=3), 5)) | " *
                "$(lpad(round(acf_jump_mean[lag], digits=3), 9)) | " *
                "$(lpad(round(acf_mix80[lag], digits=3), 9))")
    end
end

# ---------------------------------------------------------------------------
# Quantile Coverage
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  TABLE 5: Quantile Coverage")
println("="^70)

quantiles_to_check = collect(0.01:0.01:0.99);
obs_quantiles = quantile(G_obs, quantiles_to_check);

# compute quantile envelope from simulated paths -
sim_quantile_matrix = zeros(length(quantiles_to_check), N_SIM_PATHS);
for i ∈ 1:N_SIM_PATHS
    sim_quantile_matrix[:, i] = quantile(G_sim_all[i], quantiles_to_check);
end

sim_q05 = [quantile(sim_quantile_matrix[q, :], 0.05) for q ∈ 1:length(quantiles_to_check)];
sim_q95 = [quantile(sim_quantile_matrix[q, :], 0.95) for q ∈ 1:length(quantiles_to_check)];

n_covered = sum((obs_quantiles .>= sim_q05) .& (obs_quantiles .<= sim_q95));
coverage = n_covered / length(quantiles_to_check) * 100;

println("  Quantile coverage (99 quantiles, [5th, 95th] envelope): $(round(coverage, digits=1))%")

# ---------------------------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  TABLE 6: Distance Metrics")
println("="^70)

# Wasserstein-1 (Earth Mover's Distance) -
w1_distances = zeros(N_SIM_PATHS);
for i ∈ 1:N_SIM_PATHS
    obs_sorted = sort(G_obs);
    sim_sorted = sort(G_sim_all[i]);
    # interpolate to common length -
    n_common = min(length(obs_sorted), length(sim_sorted));
    obs_interp = quantile(G_obs, range(0, 1, length=n_common));
    sim_interp = quantile(G_sim_all[i], range(0, 1, length=n_common));
    w1_distances[i] = mean(abs.(obs_interp .- sim_interp));
end

println("  Wasserstein-1: $(round(mean(w1_distances), digits=4)) ± $(round(std(w1_distances), digits=4))")

# Hellinger distance (histogram-based) -
nbins = 100;
h_obs = fit(Histogram, G_obs, nbins=nbins);
hellinger_distances = zeros(N_SIM_PATHS);
for i ∈ 1:N_SIM_PATHS
    h_sim = fit(Histogram, G_sim_all[i], h_obs.edges[1]);
    p_obs = h_obs.weights ./ sum(h_obs.weights);
    p_sim = h_sim.weights ./ sum(h_sim.weights);
    hellinger_distances[i] = sqrt(0.5 * sum((sqrt.(p_obs) .- sqrt.(p_sim)) .^ 2));
end

println("  Hellinger: $(round(mean(hellinger_distances), digits=4)) ± $(round(std(hellinger_distances), digits=4))")

# ---------------------------------------------------------------------------
# Generate Figures
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  GENERATING FIGURES")
println("="^70)

figs_dir = joinpath(@__DIR__, "..", "src", "data");

# Fig 1: KS p-value histogram -
p1 = histogram(ks_pvalues, bins=50, xlabel="KS p-value", ylabel="Count",
    title="KS Test p-values ($(N_SIM_PATHS) paths, pass=$(round(ks_pass_rate, digits=1))%)", label="",
    color=:steelblue, alpha=0.7, size=(600, 400));
vline!(p1, [ALPHA], label="α=$(ALPHA)", color=:red, lw=2, ls=:dash);
savefig(p1, joinpath(figs_dir, "fig-surrogate-ks-pvalues.png"));
println("  Saved: fig-surrogate-ks-pvalues.png")

# Fig 2: Density comparison (observed vs simulated overlay) -
# pool 10 simulated paths for a representative simulated density -
G_sim_pooled = vcat(G_sim_all[1:10]...);

p2 = histogram(G_obs, bins=150, normalize=:pdf, alpha=0.6, color=:red,
    label="Observed (SPY)", xlabel="Excess Growth Rate", ylabel="Density",
    title="Marginal Density: Observed vs Simulated", size=(700, 450),
    xlims=(quantile(G_obs, 0.002), quantile(G_obs, 0.998)));
histogram!(p2, G_sim_pooled, bins=150, normalize=:pdf, alpha=0.4, color=:steelblue,
    label="Simulated (10 paths pooled)");
savefig(p2, joinpath(figs_dir, "fig-surrogate-density-comparison.png"));
println("  Saved: fig-surrogate-density-comparison.png")

# Fig 3: ACF of |Gₜ| comparison — all paths, jump-only, and 80/20 mix -
acf_sim_10 = [quantile(acf_sim_matrix[l, :], 0.10) for l ∈ 1:ACF_LAGS];
acf_sim_90 = [quantile(acf_sim_matrix[l, :], 0.90) for l ∈ 1:ACF_LAGS];

p3 = plot(lags, acf_obs_abs, lw=2.5, color=:red, label="Observed (SPY)",
    xlabel="Lag (days)", ylabel="ACF",
    title="ACF of |Gₜ|: Observed vs Simulated (lags 1–$(ACF_LAGS))", size=(750, 450),
    legend=:topright);
plot!(p3, lags, acf_sim_90, fillrange=acf_sim_10, fillalpha=0.15, color=:steelblue,
    label="All paths [10th, 90th]");
plot!(p3, lags, acf_sim_mean, lw=1.5, color=:steelblue, ls=:dash, label="All paths (mean)");
plot!(p3, lags, acf_jump_mean, lw=2, color=:darkgreen, label="Jump-only ($(n_jump_paths) paths)");
plot!(p3, lags, acf_mix80, lw=2, color=:orange, ls=:dashdot, label="80/20 mix");
savefig(p3, joinpath(figs_dir, "fig-surrogate-acf-abs-returns.png"));
println("  Saved: fig-surrogate-acf-abs-returns.png")

# Fig 4: QQ plot (observed vs simulated quantiles) -
qq_probs = collect(range(0.005, 0.995, length=200));
qq_obs = quantile(G_obs, qq_probs);

n_qq_paths = min(100, N_SIM_PATHS);
qq_sim_matrix = zeros(length(qq_probs), n_qq_paths);
for i ∈ 1:n_qq_paths
    qq_sim_matrix[:, i] = quantile(G_sim_all[i], qq_probs);
end
qq_sim_median = vec(median(qq_sim_matrix, dims=2));
qq_sim_10 = [quantile(qq_sim_matrix[j, :], 0.10) for j ∈ 1:length(qq_probs)];
qq_sim_90 = [quantile(qq_sim_matrix[j, :], 0.90) for j ∈ 1:length(qq_probs)];

p4 = plot(qq_obs, qq_sim_90, fillrange=qq_sim_10, fillalpha=0.3, color=:steelblue,
    label="Simulated [10th, 90th]", xlabel="Observed Quantiles", ylabel="Simulated Quantiles",
    title="Q-Q Plot: Observed vs Simulated", size=(600, 600));
plot!(p4, qq_obs, qq_sim_median, lw=1.5, color=:steelblue, ls=:dash, label="Simulated median");
plot!(p4, qq_obs, qq_obs, lw=1, color=:black, ls=:dot, label="Perfect fit");
savefig(p4, joinpath(figs_dir, "fig-surrogate-qq-plot.png"));
println("  Saved: fig-surrogate-qq-plot.png")

# ---------------------------------------------------------------------------
# Save report data
# ---------------------------------------------------------------------------
report = Dict(
    "model_params" => Dict(
        "n_states" => length(model.emissions),
        "nu" => model.ν,
        "rf" => model.rf,
        "jump_epsilon" => model.jump.ϵ,
        "jump_lambda" => model.jump.λ
    ),
    "observed_stats" => Dict(
        "mean_ann_pct" => obs_mean * 100,
        "std_ann_pct" => obs_std * sqrt(Δt) * 100 * sqrt(252),
        "skewness" => obs_skew,
        "excess_kurtosis" => obs_kurt,
        "jb_pvalue" => pvalue(jb_test),
        "n_obs" => T_obs
    ),
    "ks_test" => Dict(
        "pass_rate_pct" => ks_pass_rate,
        "pass_rate_se" => ks_pass_se,
        "alpha" => ALPHA,
        "n_paths" => N_SIM_PATHS
    ),
    "moments_comparison" => moments_df,
    "acf_mae" => Dict(
        "all_paths" => acf_mae,
        "all_paths_se" => acf_mae_se,
        "jump_only" => acf_mae_jump,
        "mix_80_20" => acf_mae_mix80,
        "n_lags" => ACF_LAGS,
        "n_bootstrap" => n_boot,
        "n_jump_paths" => n_jump_paths,
        "jump_fraction_pct" => jump_frac
    ),
    "quantile_coverage_pct" => coverage,
    "wasserstein1" => Dict("mean" => mean(w1_distances), "std" => std(w1_distances)),
    "hellinger" => Dict("mean" => mean(hellinger_distances), "std" => std(hellinger_distances))
);

output_path = joinpath(figs_dir, "surrogate-validation-report.jld2");
save(output_path, report);
println("\n  Report data saved: surrogate-validation-report.jld2")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
println("\n" * "="^70)
println("  SUMMARY")
println("="^70)
println("  KS pass rate:       $(round(ks_pass_rate, digits=1))% ± $(round(ks_pass_se, digits=1))%")
println("  Excess kurtosis:    Obs=$(round(obs_kurt, digits=2)), Sim=$(round(mean(sim_kurts), digits=2)) ± $(round(std(sim_kurts), digits=2))")
println("  ACF-MAE (|Gₜ|):")
println("    All paths:        $(round(acf_mae, digits=4)) ± $(round(acf_mae_se, digits=4))")
println("    Jump-only:        $(round(acf_mae_jump, digits=4))  ($(n_jump_paths) paths, $(round(jump_frac, digits=1))%)")
println("    80/20 mix:        $(round(acf_mae_mix80, digits=4))")
println("  Quantile coverage:  $(round(coverage, digits=1))%")
println("  Wasserstein-1:      $(round(mean(w1_distances), digits=4)) ± $(round(std(w1_distances), digits=4))")
println("  Hellinger:          $(round(mean(hellinger_distances), digits=4)) ± $(round(std(hellinger_distances), digits=4))")
println("="^70)

using JSON

cells = []

add_md(src) = push!(cells, Dict("cell_type"=>"markdown", "metadata"=>Dict(), "source"=>[src], "id"=>string(hash(src), base=16)[1:12]))
add_code(src) = push!(cells, Dict("cell_type"=>"code", "metadata"=>Dict(), "source"=>[src], "execution_count"=>nothing, "outputs"=>[], "id"=>string(hash(src), base=16)[1:12]))

# ===== Title =====
add_md(raw"""# Example: SIM Parameter Estimation, Bootstrap Uncertainty, and the Maximum Sharpe Ratio Portfolio

In this example, we load the frozen synthetic training dataset (20 years of JumpHMM-generated market data), estimate Single Index Model parameters for a portfolio of assets via regularized OLS, bootstrap the sampling distribution to quantify parameter uncertainty, construct the SIM-derived covariance matrix, and solve for the tangency (maximum Sharpe ratio) portfolio.

> __Learning Objectives:__
>
> * **SIM Parameter Estimation**: Estimate $\alpha$, $\beta$, $\sigma_\varepsilon$ for each asset via regularized OLS regression on synthetic training data, and bootstrap the sampling distribution to quantify uncertainty with 95% confidence intervals
> * **SIM Covariance vs. Sample Covariance**: Construct the SIM-derived covariance matrix and compare its eigenstructure, condition number, and stability to the naive sample covariance
> * **Maximum Sharpe Ratio Portfolio**: Solve the SOCP to find the tangency portfolio, plot the Capital Market Line on the efficient frontier, and compare tangency vs. min-variance vs. equal-weight in a scorecard""")

add_md("## Setup, Data and Prerequisites\nWe begin by loading the `eCornellAIFinance` package and the frozen synthetic training dataset.")

add_code("include(\"Include.jl\");")

add_md("Load the 20-year synthetic training dataset and select a subset of tickers for our portfolio. We use 10 tickers spanning different sectors and risk profiles.")

add_code(raw"""# load the frozen synthetic training dataset -
training_data = MySyntheticTrainingDataSet();
dataset = training_data["dataset"];
market_prices = training_data["market_prices"];
T_total = training_data["n_days"];
Δt = 1.0 / 252.0;

# select a portfolio of tickers -
my_tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "JPM", "PG", "XOM", "BA", "GS", "AMD"];
K = length(my_tickers);

# compute market growth rates -
market_returns = [(1.0/Δt) * log(market_prices[t]/market_prices[t-1]) for t ∈ 2:T_total];
T = length(market_returns);

# compute per-ticker growth rates -
ticker_returns = Dict{String, Vector{Float64}}();
for ticker ∈ my_tickers
    prices = dataset[ticker][:, :close];
    g = [(1.0/Δt) * log(prices[t]/prices[t-1]) for t ∈ 2:T_total];
    ticker_returns[ticker] = g;
end

println("Training data: $(T) daily returns × $(K) tickers")
println("Market CAGR: $(round((market_prices[end]/market_prices[1])^(1.0/20) - 1, digits=3)*100)%")
println("Selected tickers: $(my_tickers)")""")

# ===== Task 1 =====
add_md(raw"""___
## Task 1: Estimate SIM Parameters and Bootstrap the Sampling Distribution
We estimate the Single Index Model parameters $(\alpha_i, \beta_i, \sigma_{\varepsilon,i})$ for each ticker via regularized OLS regression, then bootstrap the sampling distribution (1,000 replicates) to quantify uncertainty.

> __What are we going to do?__ For each of the $K$ tickers, we regress asset returns on market returns using `estimate_sim`, then run `bootstrap_sim` with 1,000 replicates to get 95% confidence intervals for $\alpha$ and $\beta$. We compare the bootstrap CIs to the theoretical standard errors.
>
> __What should you see?__ High-beta tickers (NVDA, AMD) will have $\beta > 1$ with tight confidence intervals (high R²). Defensive tickers (JNJ, PG) will have $\beta < 1$. The bootstrap CIs should closely match the theoretical CIs — confirming the Gaussian error assumption is reasonable.""")

add_code(raw"""# estimate SIM parameters for each ticker -
sim_estimates = MySIMParameterEstimate[];
for ticker ∈ my_tickers
    est = estimate_sim(market_returns, ticker_returns[ticker], ticker; δ=0.0, Δt=Δt);
    push!(sim_estimates, est);
end

# display results -
params_df = DataFrame(
    "Ticker" => [e.ticker for e ∈ sim_estimates],
    "α (ann bps)" => [round(e.α * 252 * 10000, digits=1) for e ∈ sim_estimates],
    "β" => [round(e.β, digits=3) for e ∈ sim_estimates],
    "σ_ε (ann %)" => [round(e.σ_ε * sqrt(252) * 100, digits=1) for e ∈ sim_estimates],
    "R²" => [round(e.r², digits=3) for e ∈ sim_estimates]
);

println("SIM Parameter Estimates:")
println("═"^60)
pretty_table(params_df, tf = tf_markdown)""")

add_md(raw"""**Bootstrap Deep Dive:** Let's run the full bootstrap for one ticker (NVDA) and compare the empirical sampling distribution to the theoretical prediction.

> __What should you see?__ The histograms of bootstrap $\hat{\alpha}$ and $\hat{\beta}$ should be approximately normal. The bootstrap mean should match the point estimate, and the bootstrap standard deviation should match the theoretical standard error.""")

add_code(raw"""# bootstrap NVDA -
bs_nvda = bootstrap_sim(market_returns, ticker_returns["NVDA"], "NVDA";
    δ=0.0, Δt=Δt, n_bootstrap=1000, seed=42);

est_nvda = bs_nvda["point_estimate"];
println("NVDA Bootstrap Results (1,000 replicates):")
println("═"^55)
println("  α: point = $(round(est_nvda.α, digits=6)), bootstrap mean = $(round(bs_nvda["alpha_mean"], digits=6))")
println("  β: point = $(round(est_nvda.β, digits=4)), bootstrap mean = $(round(bs_nvda["beta_mean"], digits=4))")
println()
println("  α 95% CI: ($(round(bs_nvda["alpha_ci_95"][1], digits=6)), $(round(bs_nvda["alpha_ci_95"][2], digits=6)))")
println("  β 95% CI: ($(round(bs_nvda["beta_ci_95"][1], digits=4)), $(round(bs_nvda["beta_ci_95"][2], digits=4)))")
println()
println("  Bootstrap SE(β): $(round(bs_nvda["beta_std"], digits=5))")
println("  Theoretical SE(β): $(round(bs_nvda["theoretical_se"][2], digits=5))")
println("  Ratio: $(round(bs_nvda["beta_std"] / bs_nvda["theoretical_se"][2], digits=3))")""")

add_md(raw"**Visualize:** Histograms of bootstrap $\hat{\alpha}$ and $\hat{\beta}$ for NVDA, with the point estimate and 95% CI marked.")

add_code(raw"""let
    p1 = histogram(bs_nvda["alpha_samples"], bins=50, alpha=0.7, color=:steelblue,
        xlabel="α̂", ylabel="Count", title="Bootstrap Distribution of α̂ (NVDA)", label="");
    vline!(p1, [est_nvda.α], lw=2, color=:red, label="Point estimate");
    vline!(p1, [bs_nvda["alpha_ci_95"]...], lw=1.5, ls=:dash, color=:orange, label="95% CI");

    p2 = histogram(bs_nvda["beta_samples"], bins=50, alpha=0.7, color=:steelblue,
        xlabel="β̂", ylabel="Count", title="Bootstrap Distribution of β̂ (NVDA)", label="");
    vline!(p2, [est_nvda.β], lw=2, color=:red, label="Point estimate");
    vline!(p2, [bs_nvda["beta_ci_95"]...], lw=1.5, ls=:dash, color=:orange, label="95% CI");

    plot(p1, p2, layout=(1,2), size=(1000, 400), legend=:topright)
end""")

add_md("**All Tickers:** Bootstrap all 10 tickers and display the comparison table — point estimates, bootstrap 95% CIs, and theoretical SEs.")

add_code(raw"""# bootstrap all tickers -
bootstrap_results = Dict{String, Dict{String,Any}}();
for ticker ∈ my_tickers
    bootstrap_results[ticker] = bootstrap_sim(market_returns, ticker_returns[ticker], ticker;
        δ=0.0, Δt=Δt, n_bootstrap=1000, seed=42);
end

# build comparison table -
comparison = DataFrame(
    "Ticker" => String[], "β̂" => Float64[], "β 95% CI" => String[],
    "Boot SE(β)" => Float64[], "Theory SE(β)" => Float64[], "R²" => Float64[]
);

for ticker ∈ my_tickers
    bs = bootstrap_results[ticker]; est = bs["point_estimate"];
    ci = bs["beta_ci_95"];
    push!(comparison, (ticker, round(est.β, digits=3),
        "[$(round(ci[1], digits=3)), $(round(ci[2], digits=3))]",
        round(bs["beta_std"], digits=4), round(bs["theoretical_se"][2], digits=4),
        round(est.r², digits=3)));
end

println("Bootstrap Comparison — All Tickers:")
println("═"^70)
pretty_table(comparison, tf = tf_markdown)""")

add_md(raw"""**Visualize:** Predicted vs. actual returns for a high-R² ticker (NVDA) and a low-R² ticker (JNJ).

> __What should you see?__ NVDA should cluster tightly around the x=y line (high R²). JNJ should show more scatter — its returns are less explained by the market factor alone.""")

add_code(raw"""let
    function plot_sim_fit(ticker)
        est = bootstrap_results[ticker]["point_estimate"];
        y = ticker_returns[ticker]; ŷ = est.α .+ est.β .* market_returns;
        lims = (min(minimum(ŷ), minimum(y)), max(maximum(ŷ), maximum(y)));
        scatter(ŷ, y, alpha=0.3, ms=2, color=:navy, label="",
            xlabel="Predicted gᵢ", ylabel="Actual gᵢ",
            title="$(ticker) (R²=$(round(est.r², digits=3)), β=$(round(est.β, digits=2)))");
        plot!([lims...], [lims...], lw=2, ls=:dash, color=:red, label="x = y")
    end
    p1 = plot_sim_fit("NVDA"); p2 = plot_sim_fit("JNJ");
    plot(p1, p2, layout=(1,2), size=(1000, 450))
end""")

# ===== Task 2 =====
add_md(raw"""___
## Task 2: Build the SIM Covariance Matrix and Compare to Sample Covariance
We construct the SIM-derived covariance matrix $\boldsymbol{\Sigma}^{\text{SIM}}$ and compare its properties to the naive sample covariance $\hat{\boldsymbol{\Sigma}}$.

> __What are we going to do?__ Build $\boldsymbol{\Sigma}^{\text{SIM}}$ from the estimated $\beta_i$ and $\sigma_{\varepsilon,i}$ values, compute the sample covariance for comparison, and examine eigenvalue spectra, condition numbers, and correlation heatmaps.
>
> __What should you see?__ The SIM covariance should have a much lower condition number (better conditioning for optimization) and a cleaner eigenvalue spectrum (one dominant market factor + idiosyncratic noise).""")

add_code(raw"""# compute market volatility -
σ_m = std(market_returns);
println("Market volatility (annualized): $(round(σ_m * sqrt(252) * 100, digits=1))%")

# build SIM covariance -
Σ_sim = build_sim_covariance(sim_estimates, σ_m; Δt=Δt);

# build sample covariance -
R_matrix = hcat([ticker_returns[t] for t ∈ my_tickers]...);
Σ_sample = cov(R_matrix);

# verify properties -
@assert issymmetric(Σ_sim) "SIM covariance must be symmetric"
@assert isposdef(Σ_sim) "SIM covariance must be positive definite"

# condition numbers -
κ_sim = cond(Σ_sim);
κ_sample = cond(Σ_sample);
println("\nCondition numbers:")
println("  Sample covariance: $(round(κ_sample, digits=1))")
println("  SIM covariance:    $(round(κ_sim, digits=1))")
println("  Improvement: $(round(κ_sample / κ_sim, digits=1))×")""")

add_md("**Visualize:** Side-by-side correlation heatmaps and eigenvalue spectra.")

add_code(raw"""let
    D_sim = diagm(1.0 ./ sqrt.(diag(Σ_sim)));
    C_sim = D_sim * Σ_sim * D_sim;
    D_samp = diagm(1.0 ./ sqrt.(diag(Σ_sample)));
    C_sample = D_samp * Σ_sample * D_samp;

    p1 = heatmap(C_sample, title="Sample Correlation", xticks=(1:K, my_tickers),
        yticks=(1:K, my_tickers), clims=(-1,1), color=:RdBu, xrotation=45);
    p2 = heatmap(C_sim, title="SIM Correlation", xticks=(1:K, my_tickers),
        yticks=(1:K, my_tickers), clims=(-1,1), color=:RdBu, xrotation=45);

    λ_sample = eigvals(Σ_sample) |> sort |> reverse;
    λ_sim = eigvals(Σ_sim) |> sort |> reverse;
    p3 = plot(1:K, λ_sample .* 10000, marker=:circle, label="Sample", lw=2,
        xlabel="Component", ylabel="Eigenvalue (×10⁴)", title="Eigenvalue Spectrum");
    plot!(p3, 1:K, λ_sim .* 10000, marker=:diamond, label="SIM", lw=2, ls=:dash);

    plot(p1, p2, p3, layout=@layout([a b; c{0.5h}]), size=(900, 700))
end""")

# ===== Task 3 =====
add_md(raw"""___
## Task 3: Solve for the Maximum Sharpe Ratio Portfolio
We solve the SOCP to find the tangency portfolio — the allocation that maximizes the Sharpe ratio — and compare it to the min-variance portfolio and an equal-weight benchmark.

> __What are we going to do?__ Build the Sharpe ratio problem from SIM parameters, solve via Clarabel (SOCP), compute the efficient frontier, plot the Capital Market Line through the tangency portfolio, and produce a comparison scorecard.
>
> __What should you see?__ The tangency portfolio will have higher return _and_ higher volatility than the min-variance portfolio, but a better Sharpe ratio. It will also be differently concentrated — loading more on high-alpha tickers rather than just low-variance ones.""")

add_code(raw"""# build the Sharpe ratio problem -
α_vec = [e.α for e ∈ sim_estimates];
β_vec = [e.β for e ∈ sim_estimates];
gm_mean = mean(market_returns);
rf = 0.05 / 252;

sharpe_problem = build(MySharpeRatioPortfolioChoiceProblem, (
    Σ = Σ_sim, risk_free_rate = rf, α = α_vec, β = β_vec,
    gₘ = gm_mean, bounds = hcat(zeros(K), ones(K))));
sharpe_result = solve_max_sharpe(sharpe_problem);

# min-variance for comparison -
μ_hat = vec(mean(R_matrix, dims=1));
minvar_problem = build(MyPortfolioAllocationProblem;
    μ = μ_hat, Σ = Σ_sim, bounds = hcat(zeros(K), ones(K)), R = rf);
minvar_result = solve_minvariance(minvar_problem);

# equal weight -
w_equal = fill(1.0/K, K);

# display weights -
weights_df = DataFrame("Ticker" => my_tickers,
    "Tangency (%)" => round.(sharpe_result["weights"] .* 100, digits=1),
    "Min-Var (%)" => round.(minvar_result.weights .* 100, digits=1),
    "Equal (%)" => round.(w_equal .* 100, digits=1));

println("Portfolio Weights Comparison:")
println("═"^55)
pretty_table(weights_df, tf = tf_markdown)
println("\nSharpe Ratio (tangency): $(round(sharpe_result["sharpe_ratio"], digits=3))")""")

add_md("**Visualize:** Side-by-side weight comparison — tangency vs. min-variance.")

add_code(raw"""let
    groupedbar(my_tickers,
        hcat(sharpe_result["weights"] .* 100, minvar_result.weights .* 100, w_equal .* 100),
        bar_position=:dodge, bar_width=0.25,
        label=["Tangency" "Min-Variance" "Equal-Weight"],
        color=[:coral :steelblue :gray60],
        ylabel="Weight (%)", xlabel="Ticker",
        title="Portfolio Weights: Tangency vs Min-Var vs Equal",
        size=(800, 450), legend=:topright)
end""")

add_md(raw"""**Efficient Frontier and Capital Market Line:** We sweep the target return to trace the efficient frontier, then plot the tangency portfolio and the CML.

> __What should you see?__ The CML is a straight line from the risk-free rate through the tangency portfolio (red star). Every point on the CML dominates the corresponding point on the frontier at the same risk level.""")

add_code(raw"""let
    R_sweep = range(0.0, stop=maximum(μ_hat)*0.95, length=100) |> collect;
    frontier_risk = Float64[]; frontier_return = Float64[];
    for R_i ∈ R_sweep
        try
            prob = build(MyPortfolioAllocationProblem;
                μ = μ_hat, Σ = Σ_sim, bounds = hcat(zeros(K), ones(K)), R = R_i);
            sol = solve_minvariance(prob);
            push!(frontier_risk, sqrt(sol.variance) * sqrt(252) * 100);
            push!(frontier_return, sol.expected_return * 252 * 100);
        catch; end
    end

    w_t = sharpe_result["weights"];
    tang_ret = dot(μ_hat, w_t) * 252 * 100;
    tang_vol = sqrt(dot(w_t, Σ_sim * w_t)) * sqrt(252) * 100;
    rf_ann = rf * 252 * 100;
    mv_ret = minvar_result.expected_return * 252 * 100;
    mv_vol = sqrt(minvar_result.variance) * sqrt(252) * 100;
    eq_ret = dot(μ_hat, w_equal) * 252 * 100;
    eq_vol = sqrt(dot(w_equal, Σ_sim * w_equal)) * sqrt(252) * 100;

    cml_x = range(0, stop=tang_vol*1.5, length=50);
    cml_slope = (tang_ret - rf_ann) / tang_vol;
    cml_y = rf_ann .+ cml_slope .* cml_x;

    p = plot(frontier_risk, frontier_return, lw=2, color=:steelblue, label="Efficient Frontier",
        xlabel="Volatility (annual %)", ylabel="Expected Return (annual %)",
        title="Efficient Frontier, Tangency Portfolio, and CML", legend=:topleft, size=(750, 500));
    plot!(p, collect(cml_x), collect(cml_y), lw=2, ls=:dash, color=:darkred, label="Capital Market Line");
    scatter!(p, [tang_vol], [tang_ret], marker=:star5, ms=14, color=:red, label="Tangency (max SR)");
    scatter!(p, [mv_vol], [mv_ret], marker=:circle, ms=8, color=:steelblue, label="Min-Variance");
    scatter!(p, [eq_vol], [eq_ret], marker=:diamond, ms=8, color=:green, label="Equal-Weight");
    scatter!(p, [0], [rf_ann], marker=:square, ms=6, color=:black, label="Risk-Free");
    p
end""")

add_md(raw"""**Scorecard:** Compare the three portfolios across key metrics.

> __What should you see?__ The tangency portfolio should have the highest Sharpe ratio but also the highest volatility. The min-variance portfolio should have the lowest volatility but a lower Sharpe. Equal-weight sits between them.""")

add_code(raw"""let
    R = R_matrix;
    w_t = sharpe_result["weights"]; w_mv = minvar_result.weights;
    tang_rets = R * w_t; mv_rets = R * w_mv; eq_rets = R * w_equal;

    function scorecard_row(name, rets)
        ret_ann = mean(rets) * 252 * 100;
        vol_ann = std(rets) * sqrt(252) * 100;
        sr = ret_ann / vol_ann;
        dd = compute_drawdown(rets) * 100;
        return (name, round(ret_ann, digits=2), round(vol_ann, digits=2),
                round(sr, digits=3), round(dd, digits=2))
    end

    sc = DataFrame([scorecard_row("Tangency", tang_rets),
                    scorecard_row("Min-Variance", mv_rets),
                    scorecard_row("Equal-Weight", eq_rets)],
        ["Portfolio", "Return (%)", "Vol (%)", "Sharpe", "MaxDD (%)"])

    println("═"^60)
    println("  PORTFOLIO SCORECARD")
    println("═"^60)
    pretty_table(sc, tf = tf_markdown)
end""")

# ===== Summary =====
add_md(raw"""___
## Summary

> __Key Takeaways:__
>
> * **SIM estimation from synthetic data** recovers meaningful parameters — high-beta tickers (NVDA, AMD) have $\beta > 1$ and high R², while defensive tickers (JNJ, PG) have $\beta < 1$. Bootstrap CIs match theoretical SEs, confirming the Gaussian error model is reasonable
> * **SIM covariance is better conditioned** than the sample covariance — fewer parameters ($2N+1$ vs. $N(N+1)/2$) means lower estimation noise and more stable portfolio weights, at the cost of assuming single-factor correlation structure
> * **The tangency portfolio maximizes risk-adjusted return** — it sits at the CML's tangent point on the efficient frontier, accepting more variance than min-variance but delivering a substantially better Sharpe ratio

### Disclaimer
This content is for educational purposes only and does not constitute investment advice. The examples use synthetic data and simplified models.""")

# Build notebook
nb = Dict(
    "nbformat" => 4, "nbformat_minor" => 5,
    "metadata" => Dict(
        "kernelspec" => Dict("display_name"=>"Julia 1.12", "language"=>"julia", "name"=>"julia-1.12"),
        "language_info" => Dict("name"=>"julia", "version"=>"1.12.5")),
    "cells" => cells)

outpath = joinpath(@__DIR__, "..", "..", "lectures", "session-1",
    "eCornell-AI-Finance-S1-Example-SIMSharpeRatio-May-2026.ipynb")
open(outpath, "w") do f
    JSON.print(f, nb, 1)
end
println("Created Example 3 with $(length(cells)) cells at: $(outpath)")

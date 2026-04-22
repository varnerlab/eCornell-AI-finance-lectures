# --- Session 1: Portfolio Optimization ------------------------------------------

"""
    solve_minvariance(problem::MyPortfolioAllocationProblem) -> MyPortfolioPerformanceResult

Solve the classical minimum-variance portfolio problem using JuMP.
Returns a `MyPortfolioPerformanceResult` with the optimal weights and portfolio statistics.
"""
function solve_minvariance(problem::MyPortfolioAllocationProblem)::MyPortfolioPerformanceResult

    # unpack -
    μ = problem.μ;
    Σ = problem.Σ;
    bounds = problem.bounds;
    R = problem.R;
    N = length(μ);

    # setup the JuMP model -
    model = Model(Ipopt.Optimizer);
    set_silent(model);

    # decision variables: portfolio weights
    @variable(model, bounds[i,1] <= w[i=1:N] <= bounds[i,2]);

    # constraints -
    @constraint(model, sum(w) == 1.0);                  # fully invested
    @constraint(model, dot(μ, w) >= R);                  # target return

    # objective: minimize portfolio variance -
    @objective(model, Min, dot(w, Σ * w));

    # solve -
    optimize!(model);

    # package results -
    result = MyPortfolioPerformanceResult();
    result.weights = value.(w);
    result.expected_return = dot(μ, result.weights);
    result.variance = dot(result.weights, Σ * result.weights);

    # return -
    return result;
end

# --- Shared Portfolio Metrics ---------------------------------------------------

"""
    compute_drawdown(returns::Array{Float64,1}) -> Float64

Compute the maximum drawdown from a return series.
"""
function compute_drawdown(returns::Array{Float64,1})::Float64

    # compute cumulative wealth -
    wealth = cumprod(1.0 .+ returns);
    peak = accumulate(max, wealth);
    drawdowns = (peak .- wealth) ./ peak;

    # return max drawdown -
    return maximum(drawdowns);
end

"""
    compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1}) -> Float64

Compute portfolio turnover between two weight vectors.
"""
function compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1})::Float64
    return sum(abs.(w_new .- w_old));
end

# --- Session 1: Single Index Model Estimation -----------------------------------

"""
    estimate_sim(market_returns::Array{Float64,1}, asset_returns::Array{Float64,1},
        ticker::String; δ::Float64 = 0.0) -> MySIMParameterEstimate

Estimate Single Index Model parameters (α, β, σ_ε) for one asset via regularized OLS regression.

    gᵢ(t) = αᵢ + βᵢ · gₘ(t) + εᵢ(t)

Inputs are **annualized growth rates** (per year); the course convention throughout
is `g(t) = (1/Δt)·log(p_t/p_{t-1})` with `Δt = 1/252`. The closed-form solution
is `θ̂ = (X'X + δI)⁻¹ X'y` where `X = [1 gₘ]` and `y = gᵢ`.

### Arguments
- `market_returns` — market index annualized growth rates (T × 1), units: 1/year
- `asset_returns` — asset annualized growth rates (T × 1), same length as `market_returns`, units: 1/year
- `ticker` — asset ticker name
- `δ` — ridge regularization parameter (0 = plain OLS, >0 = ridge regression)

### Returns
- `MySIMParameterEstimate` with fields `ticker`, `α` (1/year), `β` (dimensionless),
  `σ_ε` (sample std of annualized growth-rate residuals, 1/year), and `r²`.
"""
function estimate_sim(market_returns::Array{Float64,1}, asset_returns::Array{Float64,1},
    ticker::String; δ::Float64 = 0.0)::MySIMParameterEstimate

    # setup design matrix: X = [1 gₘ] -
    T = length(market_returns);
    X = hcat(ones(T), market_returns);
    y = asset_returns;
    p = 2; # number of parameters

    # regularized OLS: θ = (X'X + δI)^(-1) X'y -
    θ̂ = (X' * X + δ * I(p)) \ (X' * y);
    α_hat = θ̂[1];
    β_hat = θ̂[2];

    # residuals and error variance -
    # σ_ε is the sample std of the residuals in annualized growth-rate units (1/year).
    # No extra 1/Δt factor: y is already annualized, so residuals are too.
    ŷ = X * θ̂;
    residuals = y .- ŷ;
    σ_ε = sqrt(dot(residuals, residuals) / (T - p));

    # R-squared -
    SS_res = dot(residuals, residuals);
    SS_tot = dot(y .- mean(y), y .- mean(y));
    r² = 1.0 - SS_res / SS_tot;

    # build result -
    est = MySIMParameterEstimate();
    est.ticker = ticker;
    est.α = α_hat;
    est.β = β_hat;
    est.σ_ε = σ_ε;
    est.r² = r²;

    return est;
end

"""
    build_sim_covariance(sim_estimates::Array{MySIMParameterEstimate,1},
        σ_m::Float64) -> Array{Float64,2}

Construct the SIM-derived covariance matrix of annualized growth rates from
estimated parameters.

    Σᵢⱼ = βᵢ βⱼ σₘ²                      (off-diagonal)
    Σᵢᵢ = βᵢ² σₘ² + σ²_εᵢ                (diagonal)

All variances and the resulting matrix are in units of (1/year)², matching the
convention used by `estimate_sim` (inputs and outputs are annualized growth
rates, per year).

### Arguments
- `sim_estimates` — array of SIM parameter estimates (one per asset)
- `σ_m` — market annualized growth-rate standard deviation (1/year)

### Returns
- `Σ` — N × N covariance matrix of annualized growth rates (symmetric, positive definite)
"""
function build_sim_covariance(sim_estimates::Array{MySIMParameterEstimate,1},
    σ_m::Float64)::Array{Float64,2}

    N = length(sim_estimates);
    Σ = zeros(N, N);
    σ_m² = σ_m^2;

    for i ∈ 1:N
        βᵢ = sim_estimates[i].β;
        σ_εᵢ = sim_estimates[i].σ_ε;
        for j ∈ 1:N
            βⱼ = sim_estimates[j].β;
            if i == j
                Σ[i, j] = βᵢ^2 * σ_m² + σ_εᵢ^2;
            else
                Σ[i, j] = βᵢ * βⱼ * σ_m²;
            end
        end
    end

    return Σ;
end

"""
    bootstrap_sim(market_returns::Array{Float64,1}, asset_returns::Array{Float64,1},
        ticker::String; δ::Float64 = 0.0,
        n_bootstrap::Int = 1000, seed::Int = -1) -> Dict{String, Any}

Bootstrap the sampling distribution of SIM parameters (α, β) for one asset.
Inputs are annualized growth rates (1/year), matching `estimate_sim`.

Generates `n_bootstrap` synthetic datasets by resampling residuals from the fitted
model, re-estimates parameters on each, and returns the empirical distribution.

### Bootstrap Procedure
For each k = 1, ..., n_bootstrap:
1. Sample errors: ε⁽ᵏ⁾ ~ N(0, σ̂²·I) where σ̂² is the sample variance of the residuals
2. Create synthetic observations: y⁽ᵏ⁾ = X̂·θ̂ + ε⁽ᵏ⁾
3. Re-estimate: θ̂⁽ᵏ⁾ = (X̂'X̂ + δI)⁻¹ · X̂' · y⁽ᵏ⁾

### Returns
Dictionary with keys:
- `"point_estimate"` — MySIMParameterEstimate from original data
- `"alpha_samples"` — Vector{Float64} of bootstrap α estimates
- `"beta_samples"` — Vector{Float64} of bootstrap β estimates
- `"alpha_mean"`, `"alpha_std"` — bootstrap mean and std of α
- `"beta_mean"`, `"beta_std"` — bootstrap mean and std of β
- `"alpha_ci_95"` — (lower, upper) 95% confidence interval for α
- `"beta_ci_95"` — (lower, upper) 95% confidence interval for β
- `"theoretical_se"` — [SE(α), SE(β)] from analytical OLS formula
- `"theoretical_cov"` — 2×2 covariance matrix of (α̂, β̂)
- `"error_variance"` — sample variance of annualized growth-rate residuals (1/year²)
"""
function bootstrap_sim(market_returns::Array{Float64,1}, asset_returns::Array{Float64,1},
    ticker::String; δ::Float64 = 0.0,
    n_bootstrap::Int = 1000, seed::Int = -1)::Dict{String,Any}

    if seed > 0
        Random.seed!(seed);
    end

    # point estimate -
    est = estimate_sim(market_returns, asset_returns, ticker; δ=δ);

    # setup -
    T = length(market_returns);
    p = 2;
    X = hcat(ones(T), market_returns);
    θ̂ = [est.α, est.β];
    ŷ = X * θ̂;
    residuals = asset_returns .- ŷ;

    # error variance (sample variance of the residuals in annualized units) -
    σ̂² = dot(residuals, residuals) / (T - p);

    # theoretical OLS covariance of θ̂ -
    XtX_inv = inv(X' * X + δ * I(p));
    cov_θ = σ̂² * XtX_inv;
    se_theoretical = sqrt.(diag(cov_θ));

    # residual std used to draw bootstrap innovations -
    residual_σ = sqrt(σ̂²);

    # bootstrap loop -
    α_samples = zeros(n_bootstrap);
    β_samples = zeros(n_bootstrap);

    for k ∈ 1:n_bootstrap
        # generate synthetic errors -
        ε_k = residual_σ .* randn(T);

        # create synthetic observations -
        y_k = ŷ .+ ε_k;

        # re-estimate parameters -
        θ̂_k = XtX_inv * (X' * y_k);
        α_samples[k] = θ̂_k[1];
        β_samples[k] = θ̂_k[2];
    end

    # compute bootstrap statistics -
    z = 1.96;  # 95% CI
    α_mean = mean(α_samples);
    α_std = std(α_samples);
    β_mean = mean(β_samples);
    β_std = std(β_samples);

    results = Dict{String,Any}();
    results["point_estimate"] = est;
    results["alpha_samples"] = α_samples;
    results["beta_samples"] = β_samples;
    results["alpha_mean"] = α_mean;
    results["alpha_std"] = α_std;
    results["beta_mean"] = β_mean;
    results["beta_std"] = β_std;
    results["alpha_ci_95"] = (α_mean - z * α_std, α_mean + z * α_std);
    results["beta_ci_95"] = (β_mean - z * β_std, β_mean + z * β_std);
    results["theoretical_se"] = se_theoretical;
    results["theoretical_cov"] = cov_θ;
    results["n_bootstrap"] = n_bootstrap;
    results["error_variance"] = σ̂²;

    return results;
end

"""
    solve_max_sharpe(problem::MySharpeRatioPortfolioChoiceProblem) -> Dict{String, Any}

Solve the maximum Sharpe ratio portfolio problem via Second-Order Cone Programming (COSMO).

The Sharpe ratio SR = (E[gₚ] - gf) / σₚ is maximized subject to fully-invested
and long-only constraints using the SOCP reformulation.

### Returns
Dictionary with keys:
- `"weights"` — optimal portfolio weights
- `"sharpe_ratio"` — maximum Sharpe ratio
- `"expected_return"` — portfolio expected return (annualized)
- `"volatility"` — portfolio volatility (annualized)
- `"status"` — solver termination status
"""
function solve_max_sharpe(problem::MySharpeRatioPortfolioChoiceProblem)::Dict{String,Any}

    # unpack -
    Σ = problem.Σ;
    rf = problem.risk_free_rate;
    α = problem.α;
    β = problem.β;
    gₘ = problem.gₘ;
    bounds = problem.bounds;
    N = length(α);

    # excess return vector: c = α + β·gₘ - rf -
    c = α .+ β .* gₘ .- rf .* ones(N);

    # Cholesky decomposition: Σ = U'U -
    U = cholesky(Symmetric(Σ)).U;

    # estimate a reasonable τ (lower bound on achievable Sharpe) -
    # use equal-weight portfolio Sharpe as starting point
    w_eq = fill(1.0/N, N);
    τ = max(dot(c, w_eq) / norm(U * w_eq), 0.01);

    # setup SOCP via JuMP + COSMO -
    model = Model(Clarabel.Optimizer);
    set_attribute(model, "verbose", false);

    @variable(model, w[i=1:N] >= bounds[i,1]);
    for i ∈ 1:N
        @constraint(model, w[i] <= bounds[i,2]);
    end
    @constraint(model, sum(w) == 1.0);

    # SOC constraint: [c'w/τ; U*w] ∈ SecondOrderCone -
    @constraint(model, [dot(c, w) / τ; U * w] in SecondOrderCone());
    @constraint(model, dot(c, w) >= 0.0);

    # maximize excess return -
    @objective(model, Max, dot(c, w));

    optimize!(model);

    # extract results -
    w_opt = value.(w);
    sr_num = dot(c, w_opt);
    sr_den = norm(U * w_opt);
    sr_opt = sr_den > 1e-10 ? sr_num / sr_den : 0.0;

    results = Dict{String,Any}();
    results["weights"] = w_opt;
    results["sharpe_ratio"] = sr_opt;
    results["expected_return"] = dot(α .+ β .* gₘ, w_opt);
    results["volatility"] = sr_den;
    results["status"] = termination_status(model);

    return results;
end

# --- Session 2: Signal Computation ----------------------------------------------

"""
    compute_ema(prices::Array{Float64,1}; window::Int = 21) -> Array{Float64,1}

Compute the exponential moving average of a price series.
Uses smoothing factor alpha = 2/(window + 1).
"""
function compute_ema(prices::Array{Float64,1}; window::Int = 21)::Array{Float64,1}

    # setup -
    T = length(prices);
    ema = zeros(T);
    α = 2.0 / (window + 1.0);

    # initialize with first price -
    ema[1] = prices[1];

    # compute -
    for t in 2:T
        ema[t] = α * prices[t] + (1 - α) * ema[t - 1];
    end

    # return -
    return ema;
end

"""
    compute_lambda(short_ema::Array{Float64,1}, long_ema::Array{Float64,1};
        G::Float64 = 10.0) -> Array{Float64,1}

Compute the sentiment/risk-aversion signal lambda_t from short and long EMA crossover.

    lambda_t = -G * (EMA_short_t / EMA_long_t - 1)

Positive lambda -> bearish (short < long), negative lambda -> bullish (short > long).
"""
function compute_lambda(short_ema::Array{Float64,1}, long_ema::Array{Float64,1};
    G::Float64 = 10.0)::Array{Float64,1}

    # setup -
    T = length(short_ema);
    λ = zeros(T);

    # compute -
    for t in 1:T
        λ[t] = -G * (short_ema[t] / long_ema[t] - 1.0);
    end

    # return -
    return λ;
end

"""
    compute_market_growth(prices::Array{Float64,1}; Δt::Float64 = 1.0/252.0) -> Array{Float64,1}

Compute the log growth rate of a market price series: g_m = (1/Δt) * log(S_t/S_{t-1}).
Returns an array of length T-1.
"""
function compute_market_growth(prices::Array{Float64,1}; Δt::Float64 = 1.0/252.0)::Array{Float64,1}

    # setup -
    T = length(prices);
    gm = zeros(T - 1);

    # compute -
    for t in 2:T
        gm[t - 1] = (1.0 / Δt) * log(prices[t] / prices[t - 1]);
    end

    # return -
    return gm;
end

"""
    compute_preference_weights(sim_parameters, tickers, gm_t, lambda;
        news_t=nothing, nu_loadings=nothing) -> Array{Float64,1}

Compute Cobb-Douglas preference exponents gamma_i from SIM parameters and
sentiment, with an optional news factor extension.

Without news (default):

    gamma_i = tanh(alpha_i / beta_i^lambda + beta_i^(1-lambda) * E[g_m])

With news (`news_t` and `nu_loadings` both supplied, length `K`):

    gamma_i = tanh(alpha_i / beta_i^lambda + beta_i^(1-lambda) * E[g_m] + nu_i * news_i)

The news term sits outside the `beta_i^lambda` regime-lens wrapper because
news is a separate information channel: lambda interprets market exposure,
while `nu_i * news_i` adds information that did not enter the SIM regression.

### Arguments
- `sim_parameters` — `Dict{String,Tuple{Float64,Float64,Float64}}` of `(alpha, beta, sigma_eps)` per ticker
- `tickers` — column order
- `gm_t` — current expected market growth rate
- `lambda` — current sentiment/risk-aversion parameter

### Keyword arguments
- `news_t::Union{Nothing,Vector{Float64}} = nothing` — current-day news factor (length K)
- `nu_loadings::Union{Nothing,Vector{Float64}} = nothing` — per-asset news loadings (length K)

If either keyword is `nothing`, the news term is dropped and the function
reduces to the single-index form. This preserves backwards compatibility with
all existing callers.
"""
function compute_preference_weights(sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}},
    tickers::Array{String,1}, gm_t::Float64, lambda::Float64;
    news_t::Union{Nothing,Array{Float64,1}} = nothing,
    nu_loadings::Union{Nothing,Array{Float64,1}} = nothing)::Array{Float64,1}

    # setup -
    K = length(tickers);
    gamma = zeros(K);

    use_news = (news_t !== nothing) && (nu_loadings !== nothing);
    if use_news
        @assert length(news_t) == K "news_t length must equal number of tickers";
        @assert length(nu_loadings) == K "nu_loadings length must equal number of tickers";
    end

    # compute preference weights -
    for i in 1:K
        (αᵢ, βᵢ, _) = sim_parameters[tickers[i]];
        RF = max(abs(βᵢ)^lambda, 1e-8); # risk factor, clamped
        g_hat = αᵢ / RF + (βᵢ / RF) * gm_t;
        if use_news
            g_hat += nu_loadings[i] * news_t[i];
        end
        gamma[i] = tanh(g_hat);
    end

    # return -
    return gamma;
end

# --- Session 2: Utility-Based Allocation Functions ------------------------------

"""
    allocate_cobb_douglas(problem::MyCobbDouglasChoiceProblem) -> (shares::Array{Float64,1}, cash::Float64)

Solve the Cobb-Douglas utility maximization problem analytically.

    maximize   kappa(gamma) * prod(n_i^gamma_i)
    subject to B = sum(n_i * p_i)

Analytical solution for preferred assets (gamma_i > 0):

    n_i* = (gamma_i / sum_{j in S+} gamma_j) * (B_adj / p_i)

Non-preferred assets (gamma_i <= 0) receive epsilon shares.
"""
function allocate_cobb_douglas(problem::MyCobbDouglasChoiceProblem)::Tuple{Array{Float64,1},Float64}

    # unpack -
    gamma = problem.gamma;
    prices = problem.prices;
    B = problem.B;
    ε = problem.epsilon;
    K = length(gamma);

    # separate preferred (gamma > 0) and non-preferred (gamma <= 0) -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate budget -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to minimum epsilon shares -
    for i in non_preferred
        shares[i] = ε;
        remaining_B -= ε * prices[i];
    end

    # preferred: proportional allocation from analytical solution -
    cash = 0.0;
    if length(preferred) > 0 && remaining_B > 0
        gamma_bar = sum(gamma[preferred]);
        for i in preferred
            shares[i] = (gamma[i] / gamma_bar) * (remaining_B / prices[i]);
        end
    else
        cash = remaining_B;
    end

    # return -
    return (shares, cash);
end

"""
    allocate_ces(problem::MyCESChoiceProblem) -> (shares::Array{Float64,1}, cash::Float64)

Solve the CES utility maximization problem analytically.

    maximize   (sum gamma_i * n_i^rho)^(1/rho)
    subject to B = sum(n_i * p_i)

where rho = (eta - 1)/eta. Analytical solution for preferred assets:

    n_i* = (gamma_i^eta / p_i^eta) / sum_{j in S+}(gamma_j^eta / p_j^(eta-1)) * B_adj

As eta -> 1, this converges to Cobb-Douglas. As eta -> inf, converges to linear (all-in on best).
"""
function allocate_ces(problem::MyCESChoiceProblem)::Tuple{Array{Float64,1},Float64}

    # unpack -
    gamma = problem.gamma;
    prices = problem.prices;
    B = problem.B;
    ε = problem.epsilon;
    η = problem.eta;
    K = length(gamma);

    # separate preferred and non-preferred -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to epsilon -
    for i in non_preferred
        shares[i] = ε;
        remaining_B -= ε * prices[i];
    end

    # preferred: CES analytical solution -
    cash = 0.0;
    if length(preferred) > 0 && remaining_B > 0
        # weight_i = gamma_i^eta / p_i^eta (demand function from CES)
        weights = zeros(length(preferred));
        for (j, i) in enumerate(preferred)
            weights[j] = (gamma[i]^η) / (prices[i]^η);
        end
        denom = sum((gamma[preferred[j]]^η) / (prices[preferred[j]]^(η - 1)) for j in 1:length(preferred));
        for (j, i) in enumerate(preferred)
            shares[i] = (weights[j] / denom) * remaining_B;
        end
    else
        cash = remaining_B;
    end

    # return -
    return (shares, cash);
end

"""
    allocate_log_linear(problem::MyLogLinearChoiceProblem) -> (shares::Array{Float64,1}, cash::Float64)

Solve the log-linear utility maximization problem analytically.

    maximize   sum gamma_i * log(n_i)
    subject to B = sum(n_i * p_i)

Analytical solution for preferred assets (gamma_i > 0):

    n_i* = (gamma_i / sum_{j in S+} gamma_j) * (B_adj / p_i)

Note: This yields the same allocation as Cobb-Douglas — the log transform preserves
the optimum. The utility *values* differ, which matters for bandit reward signals.
"""
function allocate_log_linear(problem::MyLogLinearChoiceProblem)::Tuple{Array{Float64,1},Float64}

    # unpack -
    gamma = problem.gamma;
    prices = problem.prices;
    B = problem.B;
    ε = problem.epsilon;
    K = length(gamma);

    # separate preferred and non-preferred -
    preferred = findall(gamma .> 0);
    non_preferred = findall(gamma .<= 0);

    # allocate -
    shares = zeros(K);
    remaining_B = B;

    # non-preferred: pin to epsilon -
    for i in non_preferred
        shares[i] = ε;
        remaining_B -= ε * prices[i];
    end

    # preferred: proportional (same allocation as Cobb-Douglas) -
    cash = 0.0;
    if length(preferred) > 0 && remaining_B > 0
        gamma_bar = sum(gamma[preferred]);
        for i in preferred
            shares[i] = (gamma[i] / gamma_bar) * (remaining_B / prices[i]);
        end
    else
        cash = remaining_B;
    end

    # return -
    return (shares, cash);
end

# --- Session 2: Utility Evaluation Functions ------------------------------------

"""
    evaluate_cobb_douglas(shares::Array{Float64,1}, gamma::Array{Float64,1}) -> Float64

Compute Cobb-Douglas utility value: U = kappa * prod(n_i^gamma_i).
kappa = +1 if all gamma_i >= 0, else -1.
"""
function evaluate_cobb_douglas(shares::Array{Float64,1}, gamma::Array{Float64,1})::Float64

    κ = any(gamma .< 0) ? -1.0 : 1.0;
    U = κ;
    for i in 1:length(shares)
        if shares[i] > 0
            U *= shares[i]^gamma[i];
        end
    end

    return U;
end

"""
    evaluate_ces(shares::Array{Float64,1}, gamma::Array{Float64,1}; eta::Float64 = 2.0) -> Float64

Compute CES utility value: U = (sum gamma_i * n_i^rho)^(1/rho) where rho = (eta-1)/eta.
"""
function evaluate_ces(shares::Array{Float64,1}, gamma::Array{Float64,1}; eta::Float64 = 2.0)::Float64

    ρ = (eta - 1.0) / eta;
    preferred = findall(gamma .> 0);
    if length(preferred) == 0
        return 0.0;
    end
    inner = sum(gamma[i] * max(shares[i], 1e-8)^ρ for i in preferred);
    return inner > 0 ? inner^(1.0 / ρ) : 0.0;
end

"""
    evaluate_log_linear(shares::Array{Float64,1}, gamma::Array{Float64,1}) -> Float64

Compute log-linear utility value: U = sum gamma_i * log(n_i).
"""
function evaluate_log_linear(shares::Array{Float64,1}, gamma::Array{Float64,1})::Float64

    preferred = findall(gamma .> 0);
    if length(preferred) == 0
        return 0.0;
    end
    return sum(gamma[i] * log(max(shares[i], 1e-8)) for i in preferred);
end

# --- Session 2: Rebalancing Engine ----------------------------------------------

"""
    compute_adaptive_eta(λ::Float64; η_min::Float64 = 0.5, η_max::Float64 = 5.0) -> Float64

Compute sentiment-adaptive CES elasticity of substitution. Maps the current
sentiment signal λ to an elasticity η ∈ [η_min, η_max]:

    η(λ) = η_min + (η_max - η_min) / (1 + |λ|)

When neutral (λ ≈ 0), η → η_max (concentrate on best asset). When sentiment
is extreme (|λ| large), η → η_min (diversify, hedge bets).
"""
function compute_adaptive_eta(λ::Float64; η_min::Float64 = 0.5, η_max::Float64 = 5.0)::Float64
    return η_min + (η_max - η_min) / (1.0 + abs(λ));
end

"""
    allocate_shares(t::Int, context::MyRebalancingContextModel;
        allocator::Symbol = :cobb_douglas) -> MyRebalancingResult

Allocate shares at time step t using the specified utility function.
This is the bridge between the utility-based allocators and the rebalancing engine.

### Allocator options
- `:cobb_douglas` — Cobb-Douglas utility (default)
- `:ces` — CES utility with eta = 2.0
- `:log_linear` — Log-linear utility
"""
function allocate_shares(t::Int, context::MyRebalancingContextModel;
    allocator::Symbol = :cobb_douglas,
    eta::Float64 = 2.0,
    news_t::Union{Nothing,Array{Float64,1}} = nothing,
    nu_loadings::Union{Nothing,Array{Float64,1}} = nothing)::MyRebalancingResult

    # unpack -
    B = context.B;
    tickers = context.tickers;
    marketdata = context.marketdata;
    gm = context.marketfactor;
    sim_params = context.sim_parameters;
    λ = context.lambda;
    ε = context.epsilon;
    K = length(tickers);

    # compute preference weights -
    gm_t = gm[min(t, length(gm))];
    gamma = compute_preference_weights(sim_params, tickers, gm_t, λ;
        news_t = news_t, nu_loadings = nu_loadings);

    # get current prices -
    prices = [marketdata[t, i + 1] for i in 1:K];

    # dispatch to the chosen allocator -
    if allocator == :cobb_douglas
        problem = MyCobbDouglasChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices;
        problem.B = B;
        problem.epsilon = ε;
        (shares, cash) = allocate_cobb_douglas(problem);
    elseif allocator == :ces
        problem = MyCESChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices;
        problem.B = B;
        problem.epsilon = ε;
        problem.eta = eta;
        (shares, cash) = allocate_ces(problem);
    elseif allocator == :log_linear
        problem = MyLogLinearChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices;
        problem.B = B;
        problem.epsilon = ε;
        (shares, cash) = allocate_log_linear(problem);
    else
        error("Unknown allocator: $allocator. Use :cobb_douglas, :ces, or :log_linear.");
    end

    # package result -
    result = MyRebalancingResult();
    result.shares = shares;
    result.cash = cash;
    result.gamma = gamma;
    result.eta = (allocator == :ces) ? eta : 0.0;

    # return -
    return result;
end

"""
    run_rebalancing_engine(context::MyRebalancingContextModel, rules::MyTriggerRules,
        lambda_series::Array{Float64,1}; offset::Int = 84,
        allocator::Symbol = :cobb_douglas) -> Dict{Int, MyRebalancingResult}

Run the AI rebalancing engine over the trading period using the specified utility allocator.

Daily loop:
1. Check rebalance schedule and trigger rules
2. If rebalance: liquidate, update budget and lambda, re-allocate via utility maximization
3. If hold: propagate prior positions
4. If drawdown exceeded: de-risk to cash

When `adaptive_eta = true` and `allocator = :ces`, the elasticity of substitution
is recomputed each rebalance day via `compute_adaptive_eta(λ_t)` using `eta_bounds`.
"""
function run_rebalancing_engine(context::MyRebalancingContextModel, rules::MyTriggerRules,
    lambda_series::Array{Float64,1}; offset::Int = 84,
    allocator::Symbol = :cobb_douglas,
    eta::Float64 = 2.0,
    adaptive_eta::Bool = false,
    eta_bounds::Tuple{Float64,Float64} = (0.5, 5.0),
    cost_bps::Float64 = 0.0,
    news_paths::Union{Nothing,Array{Float64,2}} = nothing,
    nu_loadings::Union{Nothing,Array{Float64,1}} = nothing)::Dict{Int,MyRebalancingResult}

    # setup -
    tickers = context.tickers;
    marketdata = context.marketdata;
    K = length(tickers);
    schedule = rules.rebalance_schedule;
    T = length(schedule);
    cost_rate = cost_bps / 10_000.0;

    use_news = (news_paths !== nothing) && (nu_loadings !== nothing);
    if use_news
        @assert size(news_paths, 2) == K "news_paths must have K columns";
        @assert length(nu_loadings) == K "nu_loadings length must equal K";
    end

    # helper to slice the news factor for a given day, or nothing if news disabled -
    news_for_day = function(day_idx::Int)
        use_news || return nothing;
        d = min(day_idx, size(news_paths, 1));
        return news_paths[d, :];
    end

    # results storage -
    results = Dict{Int,MyRebalancingResult}();

    # initial allocation at offset -
    η_init = (adaptive_eta && allocator == :ces) ?
        compute_adaptive_eta(context.lambda; η_min = eta_bounds[1], η_max = eta_bounds[2]) : eta;
    results[0] = allocate_shares(offset, context; allocator = allocator, eta = η_init,
        news_t = news_for_day(offset), nu_loadings = use_news ? nu_loadings : nothing);

    # initial trade cost: all of B₀ is "traded" from cash into shares -
    if cost_rate > 0.0
        init_trade_value = sum(results[0].shares[i] * marketdata[offset, i + 1] for i in 1:K);
        init_cost = cost_rate * init_trade_value;
        results[0].cash -= init_cost;
    end

    # track peak wealth for drawdown -
    peak_wealth = context.B;

    # daily loop -
    for day in 1:T

        actual_day = offset + day;
        action = schedule[day];

        if action == 1
            # liquidate current position -
            prev = results[day - 1];
            liquidation_value = prev.cash;
            for i in 1:K
                price = marketdata[actual_day, i + 1];
                liquidation_value += prev.shares[i] * price;
            end

            # check drawdown trigger -
            peak_wealth = max(peak_wealth, liquidation_value);
            drawdown = (peak_wealth - liquidation_value) / peak_wealth;

            if drawdown > rules.max_drawdown
                # de-risk: move to cash. The liquidation itself is a trade
                # (sell all shares), so apply the bps cost to the full
                # liquidation_value before stashing in cash.
                derisk_result = MyRebalancingResult();
                derisk_result.shares = zeros(K);
                prev_shares_value = liquidation_value - prev.cash;
                derisk_cash = liquidation_value;
                if cost_rate > 0.0
                    derisk_cash -= cost_rate * prev_shares_value;
                end
                derisk_result.cash = derisk_cash;
                derisk_result.gamma = zeros(K);
                derisk_result.eta = 0.0;
                results[day] = derisk_result;
            else
                # rebalance via utility maximization -
                ctx = deepcopy(context);
                ctx.B = liquidation_value;
                ctx.lambda = lambda_series[min(actual_day, length(lambda_series))];

                η_t = (adaptive_eta && allocator == :ces) ?
                    compute_adaptive_eta(ctx.lambda; η_min = eta_bounds[1], η_max = eta_bounds[2]) : eta;
                new_result = allocate_shares(actual_day, ctx; allocator = allocator, eta = η_t,
                    news_t = news_for_day(actual_day), nu_loadings = use_news ? nu_loadings : nothing);

                # check turnover cap -
                if day > 0 && haskey(results, day - 1)
                    old_shares = results[day - 1].shares;
                    old_cash   = results[day - 1].cash;
                    new_shares = copy(new_result.shares);
                    new_cash   = new_result.cash;

                    # compute turnover as fraction of total portfolio value
                    # (shares + cash). Using only shares value as the denominator
                    # explodes when the portfolio is sitting in cash after an
                    # all-non-preferred day, silently capping the next allocation
                    # to near zero.
                    trade_value = sum(abs(new_shares[i] - old_shares[i]) * marketdata[actual_day, i + 1] for i in 1:K);
                    turnover_frac = liquidation_value > 0 ? trade_value / liquidation_value : 0.0;

                    if turnover_frac > rules.max_turnover
                        # cap: blend the full portfolio (shares AND cash) toward
                        # the new allocation by the same scale factor. Blending
                        # shares alone without updating cash breaks the budget
                        # invariant (total value no longer equals liquidation_value).
                        scale = rules.max_turnover / turnover_frac;
                        for i in 1:K
                            new_result.shares[i] = old_shares[i] + scale * (new_shares[i] - old_shares[i]);
                        end
                        new_result.cash = old_cash + scale * (new_cash - old_cash);
                    end

                    # apply per-trade bps cost on the realized (possibly capped)
                    # trade value, debiting from the new cash balance.
                    if cost_rate > 0.0
                        realized_trade = sum(abs(new_result.shares[i] - old_shares[i]) * marketdata[actual_day, i + 1] for i in 1:K);
                        new_result.cash -= cost_rate * realized_trade;
                    end
                end

                results[day] = new_result;
            end
        else
            # hold: propagate prior -
            results[day] = results[day - 1];
        end
    end

    # return -
    return results;
end

"""
    compute_wealth_series(results::Dict{Int, MyRebalancingResult},
        marketdata::Array{Float64,2}, tickers::Array{String,1};
        offset::Int = 84) -> Array{Float64,1}

Compute the total wealth at each time step from a rebalancing results dictionary.
"""
function compute_wealth_series(results::Dict{Int,MyRebalancingResult},
    marketdata::Array{Float64,2}, tickers::Array{String,1};
    offset::Int = 84)::Array{Float64,1}

    # setup -
    T = maximum(keys(results));
    K = length(tickers);
    wealth = zeros(T + 1);

    # compute -
    for day in 0:T
        actual_day = offset + day;
        r = results[day];
        total = r.cash;
        for i in 1:K
            total += r.shares[i] * marketdata[actual_day, i + 1];
        end
        wealth[day + 1] = total;
    end

    # return -
    return wealth;
end

# --- Session 3: Training Data Generation ----------------------------------------

"""
    generate_training_prices(; S₀, μ, σ, T, Δt, seed) -> Array{Float64,1}

Generate a synthetic "training" price path using GBM for fitting the HMM.
"""
function generate_training_prices(; S₀::Float64 = 100.0, μ::Float64 = 0.08,
    σ::Float64 = 0.18, T::Int = 1260, Δt::Float64 = 1.0/252.0,
    seed::Int = 42)::Array{Float64,1}

    Random.seed!(seed);
    prices = zeros(T);
    prices[1] = S₀;
    for t in 2:T
        z = randn();
        prices[t] = prices[t-1] * exp((μ - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * z);
    end

    return prices;
end

# --- Session 3: Scenario Generation ---------------------------------------------

"""
    generate_hmm_scenario(model::JumpHiddenMarkovModel, tickers, sim_params;
        n_paths, n_steps, P₀_market, start_prices, rf, Δt, label) -> MyBacktestScenario

Generate a backtest scenario: use the HMM to generate market index paths,
then use SIM to generate per-ticker paths from each market path.
"""
function generate_hmm_scenario(model::JumpHiddenMarkovModel, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}};
    n_paths::Int = 100, n_steps::Int = 252, P₀_market::Float64 = 100.0,
    start_prices::Dict{String,Float64} = Dict("LargeCap" => 150.0, "SmallCap" => 45.0,
        "International" => 80.0, "Bond" => 100.0, "Commodity" => 60.0),
    rf::Float64 = 0.05, Δt::Float64 = 1.0/252.0,
    label::String = "HMM")::MyBacktestScenario

    K = length(tickers);

    # storage -
    market_paths = zeros(n_paths, n_steps);
    price_paths = zeros(n_paths, n_steps, K);

    for p in 1:n_paths

        # generate market path via HMM -
        result = hmm_simulate(model, n_steps; n_paths=1);
        G_market = result.paths[1].observations; # excess growth rates

        # convert to prices -
        mkt_prices = JumpHMM.prices_from_growth_rates(G_market, P₀_market; rf=rf, dt=Δt);
        market_paths[p, :] = mkt_prices;

        # compute market growth for SIM -
        gm = zeros(n_steps);
        for t in 2:n_steps
            gm[t] = (1.0 / Δt) * log(mkt_prices[t] / mkt_prices[t-1]);
        end

        # generate per-ticker paths via SIM -
        for (k, ticker) in enumerate(tickers)
            (αᵢ, βᵢ, σᵢ) = sim_params[ticker];
            price_paths[p, 1, k] = start_prices[ticker];
            for t in 2:n_steps
                gᵢ = αᵢ + βᵢ * gm[t] * Δt + σᵢ * sqrt(Δt) * randn();
                price_paths[p, t, k] = price_paths[p, t-1, k] * exp(gᵢ);
            end
        end
    end

    return build(MyBacktestScenario, (
        label = label,
        price_paths = price_paths,
        market_paths = market_paths,
        n_paths = n_paths,
        n_steps = n_steps
    ));
end

"""
    generate_hybrid_scenario(market_model::JumpHiddenMarkovModel,
        portfolio_surrogate::Dict{String,Any},
        sim_calibration::Dict{String,Any},
        my_tickers::Vector{String};
        n_paths::Int = 500, n_steps::Int = 252, Δt::Float64 = 1.0/252.0,
        P₀_market::Float64 = 100.0,
        start_prices::Union{Nothing,Dict{String,Float64}} = nothing,
        r2_preserve_threshold::Float64 = 0.80,
        label::String = "Hybrid-SIM",
        seed::Union{Nothing,Int} = nothing) -> MyBacktestScenario

Generate a forward Monte Carlo scenario for a user-specified subset of tickers
via the hybrid Single Index Model construction used throughout the course. For
each of `n_paths` paths this function (1) simulates a market growth-rate path
from the JumpHMM market surrogate, (2) draws each ticker's idiosyncratic path
from its per-ticker HMM marginal, (3) scales the idiosyncratic series using
either the R²-preserving branch (for tickers with real-data `r² ≥ r2_preserve_threshold`)
or the marginal-preserving variance correction, (4) copula rank-reorders the
scaled innovations using the portfolio surrogate's Student-t copula, and
(5) composes SIM growth rates `g_i = α_i + β_i · g_m + ε_i` to build price
series seeded at `start_prices` (default `\$100` each).

All inputs and outputs are in annualized growth-rate units (1/year). The
market variance `σ²_m_synth = sim_calibration["sigma_market"]^2` is used
throughout, matching the variance `build_sim_covariance` assembles, so the QP
covariance and the forward simulator share the same market volatility
scaffolding.

### Arguments
- `market_model` — pre-trained JumpHMM on the market index (e.g. SPY)
- `portfolio_surrogate` — Dict from `MyPortfolioSurrogateModel()` with keys
  `"tickers"` (ordered universe for copula column indexing), `"marginals"`
  (per-ticker HMM models), `"copula"` (Student-t copula)
- `sim_calibration` — Dict from `MySIMCalibration()` with keys `"tickers"`,
  `"alpha"`, `"beta"`, `"sigma_eps"`, `"r_squared"`, `"sigma_market"`
- `my_tickers` — subset of tickers to simulate; every entry must exist in
  both `sim_calibration["tickers"]` and `portfolio_surrogate["marginals"]`
- `n_paths` — number of Monte Carlo paths (default 500)
- `n_steps` — horizon in trading days (default 252 = 1 year)
- `Δt` — time step in years (default 1/252)
- `P₀_market` — market index starting price (default 100.0)
- `start_prices` — per-ticker starting prices; if `nothing`, all tickers start
  at `\$100`
- `r2_preserve_threshold` — R² cutoff for the R²-preserving branch
  (default 0.80 — captures SPY, QQQ, SPYG in the S&P 500 universe)
- `label` — scenario label stored in the returned `MyBacktestScenario`
- `seed` — if supplied, calls `Random.seed!(seed)` before the batched
  `hmm_simulate` for reproducibility (side effect on the global RNG)

### Returns
- `MyBacktestScenario` with `price_paths` of shape `(n_paths, n_steps, K)`
  and `market_paths` of shape `(n_paths, n_steps)`.

### Notes
- **Copula column indexing:** the Student-t copula is fitted on the full
  surrogate universe, so samples must be indexed by each ticker's position in
  `portfolio_surrogate["tickers"]`, not by its position in `my_tickers`.
- **Index-ETF corner case:** tickers with `r² ≈ 1` (e.g. SPY) land in the
  R²-preserving branch with `σ²_ε_target = 0`, which makes the composition
  deterministic: `g = α + β·g_m`. This is the intended behavior — SPY tracks
  itself by definition.
- **Missing tickers** throw `ArgumentError` listing the offenders; there is
  no silent fallback to `α = 0, β = 1`.
"""
function generate_hybrid_scenario(market_model::JumpHiddenMarkovModel,
    portfolio_surrogate::Dict{String,Any},
    sim_calibration::Dict{String,Any},
    my_tickers::Vector{String};
    n_paths::Int = 500, n_steps::Int = 252, Δt::Float64 = 1.0/252.0,
    P₀_market::Float64 = 100.0,
    start_prices::Union{Nothing,Dict{String,Float64}} = nothing,
    r2_preserve_threshold::Float64 = 0.80,
    label::String = "Hybrid-SIM",
    seed::Union{Nothing,Int} = nothing)::MyBacktestScenario

    # --- Preflight: unpack calibration into a lookup ---
    calib_tickers = sim_calibration["tickers"]::Vector{String};
    calib_alpha   = sim_calibration["alpha"]::Vector{Float64};
    calib_beta    = sim_calibration["beta"]::Vector{Float64};
    calib_r2      = sim_calibration["r_squared"]::Vector{Float64};
    calib_lookup = Dict{String, NamedTuple{(:α, :β, :r²), Tuple{Float64,Float64,Float64}}}();
    for (i, t) ∈ enumerate(calib_tickers)
        calib_lookup[t] = (α = calib_alpha[i], β = calib_beta[i], r² = calib_r2[i]);
    end

    marginals = portfolio_surrogate["marginals"];
    copula    = portfolio_surrogate["copula"];
    surrogate_tickers = portfolio_surrogate["tickers"]::Vector{String};
    col_idx = Dict{String, Int}();
    for (i, t) ∈ enumerate(surrogate_tickers)
        col_idx[t] = i;
    end

    # --- Validate: every user ticker must be in both lookups ---
    missing_calib = String[];
    missing_marg  = String[];
    for t ∈ my_tickers
        haskey(calib_lookup, t) || push!(missing_calib, t);
        haskey(marginals,    t) || push!(missing_marg,  t);
    end
    if !isempty(missing_calib) || !isempty(missing_marg)
        msg = "generate_hybrid_scenario: missing tickers";
        if !isempty(missing_calib)
            msg *= "\n  not in sim_calibration: $(missing_calib)";
        end
        if !isempty(missing_marg)
            msg *= "\n  not in portfolio_surrogate[\"marginals\"]: $(missing_marg)";
        end
        msg *= "\n  source of truth: MySIMCalibration()[\"tickers\"]";
        throw(ArgumentError(msg));
    end

    # --- Shared scale ---
    σ²_m_synth = (sim_calibration["sigma_market"]::Float64)^2;

    # --- Per-ticker starting prices ---
    K = length(my_tickers);
    p0 = Float64[
        (start_prices === nothing) ? 100.0 : get(start_prices, t, 100.0)
        for t ∈ my_tickers
    ];

    # --- Reproducibility: seed the RNG before the batched market simulation ---
    if seed !== nothing
        Random.seed!(seed);
    end

    # --- Batched market simulation ---
    sim_result = hmm_simulate(market_model, n_steps; n_paths = n_paths);

    # --- Output tensors ---
    market_paths = zeros(n_paths, n_steps);
    price_paths  = zeros(n_paths, n_steps, K);

    for p ∈ 1:n_paths

        # full observations length n_steps; drop index 1 so G_market aligns
        # with the one-period price recursion (same convention as the hybrid
        # generator script).
        G_full = Float64.(sim_result.paths[p].observations);
        G_market = G_full[2:end];                          # length n_steps - 1
        T_eff = length(G_market);

        # build market price path seeded at P₀_market
        mkt = zeros(n_steps);
        mkt[1] = P₀_market;
        for t ∈ 1:T_eff
            mkt[t+1] = mkt[t] * exp(G_market[t] * Δt);
        end
        market_paths[p, :] = mkt;

        # copula uniforms for this path — one draw of length n_steps, columns
        # indexed by the full surrogate universe
        U = JumpHMM.sample_dependence(copula, n_steps);

        for (k, ticker) ∈ enumerate(my_tickers)

            # draw the ticker's HMM marginal path, trim, and demean.
            # The HMM marginal carries the ticker's full mean growth rate;
            # demeaning ensures ε has E[ε]=0 so the SIM intercept α_i is
            # the sole source of the level — no double-counting.
            r_j = hmm_simulate(marginals[ticker], n_steps; n_paths = 1);
            obs_full = Float64.(r_j.paths[1].observations);
            obs_j = obs_full[2:end] .- mean(obs_full[2:end]); # zero-mean residuals
            σ²_HMM = var(obs_j);

            # pull α, β, R² for this ticker
            c = calib_lookup[ticker];
            α_i     = c.α;
            β_i_raw = c.β;
            r²_real = c.r²;

            # choose construction branch
            β_i = β_i_raw;
            if r²_real >= r2_preserve_threshold
                # R²-PRESERVING BRANCH: scale ε so synthetic R² = real R² exactly
                if r²_real >= 1.0 - 1e-12
                    σ²_ε_target = 0.0;
                else
                    σ²_ε_target = β_i_raw^2 * σ²_m_synth * (1.0 - r²_real) / r²_real;
                end
                ε_scaled = (σ²_HMM > 0.0 && σ²_ε_target > 0.0) ?
                           sqrt(σ²_ε_target / σ²_HMM) .* obs_j :
                           zeros(T_eff);
            else
                # MARGINAL-PRESERVING BRANCH: preserve σ²_HMM with 10% floor
                ratio = β_i_raw^2 * σ²_m_synth / σ²_HMM;
                if ratio > 0.90
                    β_i = sign(β_i_raw) * sqrt(0.90 * σ²_HMM / σ²_m_synth);
                    s² = 0.10;
                else
                    s² = 1.0 - ratio;
                end
                ε_scaled = sqrt(s²) .* obs_j;
            end

            # copula rank-reorder the scaled ε (index copula column by ticker's
            # position in the full surrogate universe, not in my_tickers)
            sorted_eps   = sort(ε_scaled);
            copula_ranks = ordinalrank(U[2:end, col_idx[ticker]]);
            ε_reordered  = sorted_eps[copula_ranks];

            # compose SIM growth rates
            g_i = α_i .+ β_i .* G_market .+ ε_reordered;    # length T_eff

            # convert to prices
            price_paths[p, 1, k] = p0[k];
            for t ∈ 1:T_eff
                price_paths[p, t+1, k] = price_paths[p, t, k] * exp(g_i[t] * Δt);
            end
        end
    end

    return build(MyBacktestScenario, (
        label = label,
        price_paths = price_paths,
        market_paths = market_paths,
        n_paths = n_paths,
        n_steps = n_steps
    ));
end

# --- Session 3: Backtesting Functions -------------------------------------------

"""
    backtest_engine(scenario::MyBacktestScenario, tickers, sim_params, rules_params;
        B₀, offset, N_short, N_long, GAIN, N_growth) -> MyBacktestResult

Run the Cobb-Douglas rebalancing engine from Session 2 across all paths in a scenario.
"""
function backtest_engine(scenario::MyBacktestScenario, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
    rules_params::NamedTuple;
    B₀::Float64 = 10000.0, offset::Int = 84,
    N_short::Int = 21, N_long::Int = 63,
    GAIN::Float64 = 10.0, N_growth::Int = 10)::MyBacktestResult

    Δt = 1.0 / 252.0;
    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);
    wealth_paths = zeros(n_trading + 1, n_paths);

    for p in 1:n_paths

        # extract this path's market prices -
        mkt = scenario.market_paths[p, :];

        # compute EMAs and lambda -
        ema_s = compute_ema(mkt; window=N_short);
        ema_l = compute_ema(mkt; window=N_long);
        λ = compute_lambda(ema_s, ema_l; G=GAIN);
        λ[1:offset] .= 0.0;

        # compute market growth EMA -
        gm_raw = compute_market_growth(mkt; Δt=Δt);
        gm_e = compute_ema(gm_raw; window=N_growth);

        # build price matrix (day_index, ticker_1, ..., ticker_K) -
        pmatrix = zeros(n_steps, K + 1);
        pmatrix[:, 1] = 1:n_steps;
        for k in 1:K
            pmatrix[:, k+1] = scenario.price_paths[p, :, k];
        end

        # build context -
        ctx = build(MyRebalancingContextModel, (
            B = B₀, tickers = tickers, marketdata = pmatrix,
            marketfactor = gm_e, sim_parameters = sim_params,
            lambda = 0.0, Δt = Δt, epsilon = 0.1
        ));

        # build rules -
        rules = build(MyTriggerRules, (
            max_drawdown = rules_params.max_drawdown,
            max_turnover = rules_params.max_turnover,
            rebalance_schedule = ones(Int, n_trading)
        ));

        # run engine with Cobb-Douglas allocator -
        results = run_rebalancing_engine(ctx, rules, λ; offset=offset, allocator=:cobb_douglas);
        wealth = compute_wealth_series(results, pmatrix, tickers; offset=offset);
        wealth_paths[:, p] .= wealth;

        # metrics -
        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "Cobb-Douglas Engine";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;
    result.wealth_paths = wealth_paths;

    return result;
end

"""
    backtest_buyhold(scenario::MyBacktestScenario, tickers;
        B₀, offset, weights) -> MyBacktestResult

Run a buy-and-hold strategy across all paths in the scenario. If `weights` is
`nothing` (the default) each asset receives an equal 1/K share of the initial
budget `B₀` — backward-compatible with earlier callers. Otherwise `weights`
must be a vector of length `K` summing to one; each asset then receives
`weights[k] · B₀` of the initial budget and the position is held without
rebalancing until the end of the scenario.
"""
function backtest_buyhold(scenario::MyBacktestScenario, tickers::Array{String,1};
    B₀::Float64 = 10000.0, offset::Int = 84,
    weights::Union{Nothing,Vector{Float64}} = nothing)::MyBacktestResult

    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    # resolve and validate the weight vector once, outside the path loop
    w = weights === nothing ? fill(1.0/K, K) : weights;
    @assert length(w) == K "weights length ($(length(w))) must equal number of tickers ($(K))"
    @assert isapprox(sum(w), 1.0; atol=1e-6) "weights must sum to 1 (got $(sum(w)))"

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);
    wealth_paths = zeros(n_trading + 1, n_paths);

    for p in 1:n_paths

        # buy at offset with the (possibly custom) weight vector -
        shares = [(B₀ * w[k]) / scenario.price_paths[p, offset, k] for k in 1:K];

        wealth = zeros(n_trading + 1);
        for d in 0:n_trading
            day = offset + d;
            wealth[d+1] = sum(shares[k] * scenario.price_paths[p, day, k] for k in 1:K);
        end
        wealth_paths[:, p] .= wealth;

        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "Buy-and-Hold";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;
    result.wealth_paths = wealth_paths;

    return result;
end

"""
    backtest_buyhold_market(scenario::MyBacktestScenario;
        B₀::Float64 = 10000.0, offset::Int = 1) -> MyBacktestResult

Run a buy-and-hold strategy on the **market index** time series stored in
`scenario.market_paths`, treating it as a single risky asset. For each path
the entire initial budget `B₀` is allocated to the market on day `offset`,
and the wealth trajectory is held to the end of the scenario without
rebalancing. The result has the same `MyBacktestResult` shape as
`backtest_buyhold` so that downstream scorecard code can compare a tickers
portfolio against a market benchmark uniformly.

This is the standard "did my optimization beat just buying the market"
benchmark. The market path itself is whatever the scenario's underlying
market HMM produced (in the eCornell course this is the SPY surrogate
fitted on real 2014–2024 OHLC data via `MyMarketSurrogateModel`).

### Returns
- `MyBacktestResult` with `strategy_label = "Market Buy-and-Hold"`,
  `final_wealth`, `max_drawdowns`, `sharpe_ratios` arrays of length `n_paths`.
"""
function backtest_buyhold_market(scenario::MyBacktestScenario;
    B₀::Float64 = 10000.0, offset::Int = 1)::MyBacktestResult

    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    n_trading = n_steps - offset;

    final_wealth  = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);
    wealth_paths = zeros(n_trading + 1, n_paths);

    for p in 1:n_paths

        # Buy the market at day `offset` with the entire budget
        shares = B₀ / scenario.market_paths[p, offset];

        wealth = zeros(n_trading + 1);
        for d in 0:n_trading
            day = offset + d;
            wealth[d+1] = shares * scenario.market_paths[p, day];
        end
        wealth_paths[:, p] .= wealth;

        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "Market Buy-and-Hold";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;
    result.wealth_paths = wealth_paths;

    return result;
end

"""
    compute_cvar(wealth::AbstractVector{<:Real}; α::Float64 = 0.05) -> Float64

Conditional Value-at-Risk (Expected Shortfall) of a wealth distribution at
confidence level `α`. Returns the **mean of the bottom α-fraction** of the
input — i.e. the average outcome across the worst `α·n` paths. With
`α = 0.05` and 5000 paths this is the average wealth across the worst 250.

The convention is **wealth-based**, not loss-based: lower values are worse,
so the function averages the smallest values. To use it on a loss vector
(where larger is worse), pass `-losses` and negate the result, or use
`mean(sort(losses, rev=true)[1:floor(Int, α*n)])` directly.

For small tails the estimate has a non-trivial standard error
(`std(tail) / sqrt(n_tail)`); the caller should report or display that SE
alongside the point estimate when `α·n` is in the low hundreds.

### Arguments
- `wealth` — vector of outcomes; lower is worse
- `α` — tail fraction in `(0, 1)` (default `0.05`)

### Returns
- `Float64` — mean of the worst `floor(α·n)` outcomes (at least 1).
"""
function compute_cvar(wealth::AbstractVector{<:Real}; α::Float64 = 0.05)::Float64
    n = length(wealth);
    n > 0 || throw(ArgumentError("wealth must be non-empty"));
    (0.0 < α < 1.0) || throw(ArgumentError("α must be in (0, 1), got $(α)"));

    n_tail = max(1, floor(Int, α * n));
    sorted = sort(collect(wealth));
    return mean(sorted[1:n_tail]);
end

# --- Session 3: Bandit Functions ------------------------------------------------

"""
    bandit_world(action::Array{Int,1}, context::MyBanditContext) -> (utility::Float64, shares::Array{Float64,1}, gamma::Array{Float64,1})

The "world" function for the combinatorial bandit. Given a binary action vector
(1 = include asset, 0 = exclude), computes the Cobb-Douglas utility-maximizing
allocation over the included assets.

Returns the Cobb-Douglas utility value as the reward signal, plus the share allocation
and preference weights for inspection.
"""
function bandit_world(action::Array{Int,1}, context::MyBanditContext)::Tuple{Float64,Array{Float64,1},Array{Float64,1}}

    K = length(context.tickers);

    # compute full preference weights -
    gamma = compute_preference_weights(context.sim_parameters, context.tickers,
        context.gm_t, context.lambda);

    # mask out excluded assets: set gamma to negative for excluded -
    for i in 1:K
        if action[i] == 0
            gamma[i] = -abs(gamma[i]) - 0.01; # force non-preferred
        end
    end

    # solve Cobb-Douglas allocation -
    problem = MyCobbDouglasChoiceProblem();
    problem.gamma = gamma;
    problem.prices = context.prices;
    problem.B = context.B;
    problem.epsilon = context.epsilon;
    (shares, _) = allocate_cobb_douglas(problem);

    # compute utility as reward -
    utility = evaluate_cobb_douglas(shares, gamma);

    return (utility, shares, gamma);
end

"""
    solve_bandit(bandit::MyEpsilonGreedyBanditModel, context::MyBanditContext) -> MyBanditResult

Run the epsilon-greedy combinatorial bandit to find the best asset subset.

Each arm is a binary vector of length K (which assets to include).
There are 2^K - 1 possible arms (excluding the empty set).

Exploration decays as: epsilon_t = t^(-1/3) * (K * log(t))^(1/3)
"""
function solve_bandit(bandit::MyEpsilonGreedyBanditModel, context::MyBanditContext)::MyBanditResult

    K = bandit.K;
    T = bandit.n_iterations;
    n_arms = 2^K - 1; # exclude empty set

    # encode arms as binary vectors -
    function arm_to_action(arm_idx::Int)::Array{Int,1}
        return [((arm_idx >> (i-1)) & 1) for i in 1:K];
    end

    # initialize reward estimates -
    mu = zeros(n_arms);      # average reward per arm
    counts = zeros(Int, n_arms); # how many times each arm was pulled

    # storage -
    reward_history = zeros(T);
    exploration_history = zeros(T);

    for t in 1:T

        # decaying exploration rate -
        ε_t = t > 1 ? t^(-1.0/3.0) * (K * log(t))^(1.0/3.0) : 1.0;
        ε_t = clamp(ε_t, 0.0, 1.0);
        exploration_history[t] = ε_t;

        # epsilon-greedy selection -
        if rand() < ε_t
            # explore: random arm -
            arm = rand(1:n_arms);
        else
            # exploit: best arm so far -
            arm = argmax(mu);
        end

        # pull arm: run the world function -
        action = arm_to_action(arm);
        (utility, _, _) = bandit_world(action, context);

        # update reward estimate (weighted online average) -
        counts[arm] += 1;
        lr = bandit.alpha > 0 ? bandit.alpha : 1.0 / counts[arm];
        mu[arm] += lr * (utility - mu[arm]);

        reward_history[t] = utility;
    end

    # best arm at convergence -
    best_arm = argmax(mu);
    best_action = arm_to_action(best_arm);
    (best_utility, _, _) = bandit_world(best_action, context);

    # package result -
    result = MyBanditResult();
    result.best_action = best_action;
    result.best_utility = best_utility;
    result.reward_history = reward_history;
    result.exploration_history = exploration_history;
    result.arm_means = mu;

    return result;
end

"""
    compute_regret(reward_history::Array{Float64,1}) -> Array{Float64,1}

Compute cumulative regret: the difference between the best-in-hindsight
reward and the actual reward at each step.
"""
function compute_regret(reward_history::Array{Float64,1})::Array{Float64,1}

    best = maximum(reward_history);
    regret = cumsum(best .- reward_history);
    return regret;
end

"""
    backtest_bandit(scenario::MyBacktestScenario, tickers, sim_params;
        B₀, offset, N_short, N_long, GAIN, N_growth, n_bandit_iters) -> MyBacktestResult

Run the bandit portfolio selector across all paths in a scenario.
On each path: run the bandit to select the best asset subset, then run
a simple rebalancing strategy using only the selected assets with Cobb-Douglas allocation.
"""
function backtest_bandit(scenario::MyBacktestScenario, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}};
    B₀::Float64 = 10000.0, offset::Int = 84,
    N_short::Int = 21, N_long::Int = 63,
    GAIN::Float64 = 10.0, N_growth::Int = 10,
    n_bandit_iters::Int = 200)::MyBacktestResult

    Δt = 1.0 / 252.0;
    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);
    wealth_paths = zeros(n_trading + 1, n_paths);

    for p in 1:n_paths

        # extract this path -
        mkt = scenario.market_paths[p, :];

        # compute EMAs and lambda -
        ema_s = compute_ema(mkt; window=N_short);
        ema_l = compute_ema(mkt; window=N_long);
        λ = compute_lambda(ema_s, ema_l; G=GAIN);
        λ[1:offset] .= 0.0;

        # compute market growth EMA -
        gm_raw = compute_market_growth(mkt; Δt=Δt);
        gm_e = compute_ema(gm_raw; window=N_growth);

        # run bandit at the start to select asset subset -
        prices_at_offset = [scenario.price_paths[p, offset, k] for k in 1:K];
        bandit_ctx = build(MyBanditContext, (
            tickers = tickers, sim_parameters = sim_params,
            prices = prices_at_offset, B = B₀,
            gm_t = gm_e[min(offset, length(gm_e))],
            lambda = λ[offset], epsilon = 0.1
        ));

        bandit_model = build(MyEpsilonGreedyBanditModel, (
            K = K, n_iterations = n_bandit_iters, alpha = 0.1
        ));

        bandit_result = solve_bandit(bandit_model, bandit_ctx);
        selected = bandit_result.best_action;

        # build price matrix -
        pmatrix = zeros(n_steps, K + 1);
        pmatrix[:, 1] = 1:n_steps;
        for k in 1:K
            pmatrix[:, k+1] = scenario.price_paths[p, :, k];
        end

        # run Cobb-Douglas engine with bandit-selected assets -
        # (non-selected assets get forced to epsilon via modified SIM params)
        modified_sim = copy(sim_params);
        for (k, ticker) in enumerate(tickers)
            if selected[k] == 0
                # force non-preferred by setting alpha very negative -
                (_, βᵢ, σᵢ) = sim_params[ticker];
                modified_sim[ticker] = (-10.0, βᵢ, σᵢ);
            end
        end

        ctx = build(MyRebalancingContextModel, (
            B = B₀, tickers = tickers, marketdata = pmatrix,
            marketfactor = gm_e, sim_parameters = modified_sim,
            lambda = 0.0, Δt = Δt, epsilon = 0.1
        ));

        rules = build(MyTriggerRules, (
            max_drawdown = 0.15, max_turnover = 0.50,
            rebalance_schedule = ones(Int, n_trading)
        ));

        results = run_rebalancing_engine(ctx, rules, λ; offset=offset, allocator=:cobb_douglas);
        wealth = compute_wealth_series(results, pmatrix, tickers; offset=offset);
        wealth_paths[:, p] .= wealth;

        # metrics -
        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);

        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "Bandit Selector";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;
    result.wealth_paths = wealth_paths;

    return result;
end

# --- Session 3: Eta-Bandit Functions --------------------------------------------

"""
    classify_regime(λ::Float64; θ::Float64 = 0.5) -> Symbol

Classify the current sentiment into a regime bin. Returns `:bearish` if
λ > θ, `:bullish` if λ < -θ, `:neutral` otherwise.
"""
function classify_regime(λ::Float64; θ::Float64 = 0.5)::Symbol
    if λ > θ
        return :bearish;
    elseif λ < -θ
        return :bullish;
    else
        return :neutral;
    end
end

"""
    eta_bandit_world(η::Float64, context::MyRebalancingContextModel, t::Int) -> Float64

The "world" function for the eta-bandit. Given a candidate CES elasticity η,
computes the CES utility-maximizing allocation at time step t using the context's
current preference weights and prices. Returns the CES utility as the reward signal.
"""
function eta_bandit_world(η::Float64, context::MyRebalancingContextModel, t::Int)::Float64

    # compute preference weights from current context -
    gm_t = context.marketfactor[min(t, length(context.marketfactor))];
    gamma = compute_preference_weights(context.sim_parameters, context.tickers, gm_t, context.lambda);

    K = length(context.tickers);
    prices = [context.marketdata[t, i + 1] for i in 1:K];

    # solve CES allocation -
    problem = MyCESChoiceProblem();
    problem.gamma = gamma;
    problem.prices = prices;
    problem.B = context.B;
    problem.epsilon = context.epsilon;
    problem.eta = η;
    (shares, _) = allocate_ces(problem);

    # evaluate CES utility as reward -
    utility = evaluate_ces(shares, gamma; eta = η);

    return utility;
end

"""
    solve_eta_bandit(bandit::MyEtaBanditModel, context::MyRebalancingContextModel,
        lambda_series::Array{Float64,1}, time_indices::Array{Int,1}) -> MyEtaBanditResult

Run per-regime epsilon-greedy over the eta grid. For each time index:
classify the regime from lambda_series, select an η arm (epsilon-greedy
within that regime's bandit state), call eta_bandit_world, and update
the regime's arm mean.
"""
function solve_eta_bandit(bandit::MyEtaBanditModel, context::MyRebalancingContextModel,
    lambda_series::Array{Float64,1}, time_indices::Array{Int,1})::MyEtaBanditResult

    η_grid = bandit.eta_grid;
    K_arms = length(η_grid);
    θ = bandit.lambda_threshold;
    T = length(time_indices);

    # per-regime state -
    regimes = [:bearish, :neutral, :bullish];
    arm_means = Dict(r => zeros(K_arms) for r in regimes);
    arm_counts = Dict(r => zeros(Int, K_arms) for r in regimes);
    regime_rounds = Dict(r => 0 for r in regimes);

    # storage -
    reward_history = zeros(T);
    exploration_history = zeros(T);

    for (iter, t) in enumerate(time_indices)

        # classify regime -
        λ_t = lambda_series[min(t, length(lambda_series))];
        regime = classify_regime(λ_t; θ = θ);

        # update context lambda -
        ctx = deepcopy(context);
        ctx.lambda = λ_t;

        # regime-local round count for epsilon decay -
        regime_rounds[regime] += 1;
        t_local = regime_rounds[regime];

        # decaying epsilon -
        ε_t = t_local > 1 ? t_local^(-1.0/3.0) * (K_arms * log(t_local))^(1.0/3.0) : 1.0;
        ε_t = clamp(ε_t, 0.0, 1.0);
        exploration_history[iter] = ε_t;

        # epsilon-greedy arm selection -
        if rand() < ε_t
            arm = rand(1:K_arms);
        else
            arm = argmax(arm_means[regime]);
        end

        # pull arm -
        η_chosen = η_grid[arm];
        utility = eta_bandit_world(η_chosen, ctx, t);

        # update -
        arm_counts[regime][arm] += 1;
        lr = bandit.alpha > 0 ? bandit.alpha : 1.0 / arm_counts[regime][arm];
        arm_means[regime][arm] += lr * (utility - arm_means[regime][arm]);

        reward_history[iter] = utility;
    end

    # extract best η per regime -
    best_eta = Dict{Symbol,Float64}();
    for r in regimes
        if sum(arm_counts[r]) > 0
            best_eta[r] = η_grid[argmax(arm_means[r])];
        else
            best_eta[r] = η_grid[div(K_arms, 2) + 1]; # default to middle
        end
    end

    # package result -
    result = MyEtaBanditResult();
    result.best_eta_per_regime = best_eta;
    result.arm_means_per_regime = arm_means;
    result.arm_counts_per_regime = arm_counts;
    result.reward_history = reward_history;
    result.exploration_history = exploration_history;

    return result;
end

"""
    backtest_eta_bandit(scenario::MyBacktestScenario, tickers::Array{String,1},
        sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
        eta_map::Dict{Symbol,Float64};
        B₀, offset, L_short, L_long, GAIN, L_growth, lambda_threshold, epsilon) -> MyBacktestResult

Run the rebalancing engine with CES allocator across all paths, where η is looked
up per regime from eta_map based on the current λ. Pattern follows backtest_bandit
but uses :ces allocation with regime-dependent η.
"""
function backtest_eta_bandit(scenario::MyBacktestScenario, tickers::Array{String,1},
    sim_params::Dict{String,Tuple{Float64,Float64,Float64}},
    eta_map::Dict{Symbol,Float64};
    B₀::Float64 = 10000.0, offset::Int = 84,
    L_short::Int = 21, L_long::Int = 63,
    GAIN::Float64 = 10.0, L_growth::Int = 10,
    lambda_threshold::Float64 = 0.5,
    epsilon::Float64 = 0.1)::MyBacktestResult

    Δt = 1.0 / 252.0;
    n_paths = scenario.n_paths;
    n_steps = scenario.n_steps;
    K = length(tickers);
    n_trading = n_steps - offset;

    final_wealth = zeros(n_paths);
    max_drawdowns = zeros(n_paths);
    sharpe_ratios = zeros(n_paths);
    wealth_paths = zeros(n_trading + 1, n_paths);

    for p in 1:n_paths

        # extract path -
        mkt = scenario.market_paths[p, :];
        ema_s = compute_ema(mkt; window = L_short);
        ema_l = compute_ema(mkt; window = L_long);
        λ = compute_lambda(ema_s, ema_l; G = GAIN);
        λ[1:offset] .= 0.0;

        gm_raw = compute_market_growth(mkt; Δt = Δt);
        gm_e = compute_ema(gm_raw; window = L_growth);

        # build price matrix -
        pmatrix = zeros(n_steps, K + 1);
        pmatrix[:, 1] = 1:n_steps;
        for k in 1:K
            pmatrix[:, k + 1] = scenario.price_paths[p, :, k];
        end

        # build context -
        ctx = build(MyRebalancingContextModel, (
            B = B₀, tickers = tickers, marketdata = pmatrix,
            marketfactor = gm_e, sim_parameters = sim_params,
            lambda = 0.0, Δt = Δt, epsilon = epsilon
        ));

        rules = build(MyTriggerRules, (
            max_drawdown = 0.15, max_turnover = 0.50,
            rebalance_schedule = ones(Int, n_trading)
        ));

        # custom engine loop with regime-dependent η -
        results = Dict{Int,MyRebalancingResult}();

        # initial allocation -
        λ_init = λ[offset];
        regime_init = classify_regime(λ_init; θ = lambda_threshold);
        η_init = eta_map[regime_init];
        ctx.lambda = λ_init;
        results[0] = allocate_shares(offset, ctx; allocator = :ces, eta = η_init);

        peak_wealth = B₀;

        for day in 1:n_trading
            actual_day = offset + day;

            # liquidate -
            prev = results[day - 1];
            liquidation_value = prev.cash;
            for i in 1:K
                liquidation_value += prev.shares[i] * pmatrix[actual_day, i + 1];
            end

            peak_wealth = max(peak_wealth, liquidation_value);
            drawdown = peak_wealth > 0 ? (peak_wealth - liquidation_value) / peak_wealth : 0.0;

            if drawdown > rules.max_drawdown
                derisk_result = MyRebalancingResult();
                derisk_result.shares = zeros(K);
                derisk_result.cash = liquidation_value;
                derisk_result.gamma = zeros(K);
                derisk_result.eta = 0.0;
                results[day] = derisk_result;
            else
                ctx_day = deepcopy(ctx);
                ctx_day.B = liquidation_value;
                ctx_day.lambda = λ[min(actual_day, length(λ))];

                regime = classify_regime(ctx_day.lambda; θ = lambda_threshold);
                η_day = eta_map[regime];

                new_result = allocate_shares(actual_day, ctx_day; allocator = :ces, eta = η_day);

                # turnover cap -
                old_shares = results[day - 1].shares;
                trade_value = sum(abs(new_result.shares[i] - old_shares[i]) * pmatrix[actual_day, i + 1] for i in 1:K);
                turnover_frac = liquidation_value > 0 ? trade_value / liquidation_value : 0.0;
                if turnover_frac > rules.max_turnover
                    scale = rules.max_turnover / turnover_frac;
                    old_cash = results[day - 1].cash;
                    new_cash = new_result.cash;
                    for i in 1:K
                        new_result.shares[i] = old_shares[i] + scale * (new_result.shares[i] - old_shares[i]);
                    end
                    new_result.cash = old_cash + scale * (new_cash - old_cash);
                end

                results[day] = new_result;
            end
        end

        # compute wealth and metrics -
        wealth = compute_wealth_series(results, pmatrix, tickers; offset = offset);
        wealth_paths[:, p] .= wealth;
        final_wealth[p] = wealth[end];
        returns = diff(wealth) ./ wealth[1:end-1];
        peak = accumulate(max, wealth);
        max_drawdowns[p] = maximum((peak .- wealth) ./ peak);
        vol = std(returns) * sqrt(252);
        mean_ret = (wealth[end] / wealth[1] - 1.0);
        sharpe_ratios[p] = vol > 0 ? mean_ret / vol : 0.0;
    end

    result = MyBacktestResult();
    result.scenario_label = scenario.label;
    result.strategy_label = "Eta-Bandit CES";
    result.final_wealth = final_wealth;
    result.max_drawdowns = max_drawdowns;
    result.sharpe_ratios = sharpe_ratios;
    result.wealth_paths = wealth_paths;

    return result;
end

"""
    build_compliance_config(; concentration_cap::Float64 = 0.40,
        drawdown_gate::Float64 = 0.15, turnover_limit::Float64 = 0.50,
        position_size_limit::Float64 = 5000.0) -> Dict{String,Float64}

Package compliance parameters into a dictionary for Session 4's production system.
Trades within these limits auto-execute; trades exceeding them queue for human review.
"""
function build_compliance_config(;
    concentration_cap::Float64 = 0.40,
    drawdown_gate::Float64 = 0.15,
    turnover_limit::Float64 = 0.50,
    position_size_limit::Float64 = 5000.0)::Dict{String,Float64}

    return Dict{String,Float64}(
        "concentration_cap" => concentration_cap,
        "drawdown_gate" => drawdown_gate,
        "turnover_limit" => turnover_limit,
        "position_size_limit" => position_size_limit,
    );
end

# --- Session 3: EWLS Functions --------------------------------------------------

"""
    ewls_init(α₀::Float64, β₀::Float64, σ_ε₀::Float64;
        half_life::Float64 = 63.0, prior_weight::Float64 = 63.0) -> MyEWLSState

Initialize an EWLS state from pre-calibrated SIM parameters. The decay factor
is η = 2^(−1/half_life) so data from `half_life` days ago receives half the
weight of today's observation. `prior_weight` seeds the sufficient statistics
as if `prior_weight` observations consistent with (α₀, β₀) had already been
seen — this anchors the estimates to the calibrated values until enough new
data accumulates.
"""
function ewls_init(α₀::Float64, β₀::Float64, σ_ε₀::Float64;
    half_life::Float64 = 63.0, prior_weight::Float64 = 63.0)::MyEWLSState

    η = 2.0^(-1.0 / half_life);

    # seed sufficient statistics from the prior. We assume the prior
    # was estimated from data with zero-mean market growth (centered),
    # so Swx ≈ 0 and Swxx ≈ prior_weight · σ_m² with a small σ_m.
    # A simpler and more robust initialization: set the sums so that
    # the WLS formulas recover (α₀, β₀) exactly.
    #   β = (Sw·Swxy − Swx·Swy) / (Sw·Swxx − Swx²)
    #   α = (Swy − β·Swx) / Sw
    # With Swx = 0 this simplifies to β = Swxy/Swxx and α = Swy/Sw.
    Sw   = prior_weight;
    Swx  = 0.0;
    Swy  = prior_weight * α₀;      # so α = Swy/Sw = α₀
    Swxx = prior_weight * 1.0;      # unit variance placeholder
    Swxy = prior_weight * β₀;       # so β = Swxy/Swxx = β₀
    Swyy = prior_weight * (α₀^2 + β₀^2 + σ_ε₀^2); # consistent with Var(y)

    state = MyEWLSState();
    state.Sw   = Sw;
    state.Swx  = Swx;
    state.Swy  = Swy;
    state.Swxx = Swxx;
    state.Swxy = Swxy;
    state.Swyy = Swyy;
    state.η    = η;
    state.α    = α₀;
    state.β    = β₀;
    state.σ_ε  = σ_ε₀;

    return state;
end

"""
    ewls_update!(state::MyEWLSState, g_i::Float64, g_m::Float64) -> Tuple{Float64,Float64,Float64}

Update the EWLS state with a new observation (g_i, g_m) where g_i is the asset's
CCGR and g_m is the market's CCGR. Decays all running sums by η, adds the new
observation with unit weight, and recomputes (α̂, β̂, σ̂_ε). Returns the updated
estimates as a tuple (α, β, σ_ε).
"""
function ewls_update!(state::MyEWLSState, g_i::Float64, g_m::Float64)::Tuple{Float64,Float64,Float64}

    η = state.η;

    # decay and accumulate -
    state.Sw   = η * state.Sw   + 1.0;
    state.Swx  = η * state.Swx  + g_m;
    state.Swy  = η * state.Swy  + g_i;
    state.Swxx = η * state.Swxx + g_m * g_m;
    state.Swxy = η * state.Swxy + g_i * g_m;
    state.Swyy = η * state.Swyy + g_i * g_i;

    # solve weighted least squares: g_i = α + β��g_m + ε -
    denom = state.Sw * state.Swxx - state.Swx^2;
    if abs(denom) > 1e-12
        state.β = (state.Sw * state.Swxy - state.Swx * state.Swy) / denom;
        state.α = (state.Swy - state.β * state.Swx) / state.Sw;

        # weighted residual variance: E[ε²] = E[y²] - 2β·E[xy] - 2α·E[y] + β²·E[x²] + 2αβ·E[x] + α²
        mse = (state.Swyy - 2.0 * state.β * state.Swxy - 2.0 * state.α * state.Swy +
               state.β^2 * state.Swxx + 2.0 * state.α * state.β * state.Swx +
               state.α^2 * state.Sw) / state.Sw;
        state.σ_ε = sqrt(max(mse, 0.0));
    end

    return (state.α, state.β, state.σ_ε);
end

"""
    replay_engine_ewls(price_matrix::Array{Float64,2}, market_prices::Array{Float64,1},
        tickers::Array{String,1}, sim_params_init::Dict{String,Tuple{Float64,Float64,Float64}},
        rules_params::NamedTuple;
        B₀::Float64 = 10000.0, offset::Int = 63,
        half_life::Float64 = 63.0, prior_weight::Float64 = 63.0,
        N_short::Int = 21, N_long::Int = 63, GAIN::Float64 = 10.0,
        N_growth::Int = 10, cost_bps::Float64 = 0.0,
        epsilon::Float64 = 0.1) -> NamedTuple

Run the Cobb-Douglas rebalancing engine day-by-day over a single path of real
market data, interleaving EWLS updates of SIM parameters with allocation
decisions. The first `offset` days are warm-up (EWLS accumulates data, no
trading). Trading runs from day `offset+1` to the end of the price matrix.

### Arguments
- `price_matrix` — (n_days, K+1) matrix: col 1 = day index, cols 2:K+1 = close prices per ticker
- `market_prices` — (n_days,) vector of market (SPY) close prices
- `tickers` — ticker symbols matching cols 2:K+1
- `sim_params_init` — pre-calibrated SIM parameters Dict(ticker => (α, β, σ_ε))
- `rules_params` — NamedTuple with `max_drawdown`, `max_turnover`

### Returns
Named tuple with fields:
- `results::Dict{Int,MyRebalancingResult}` — per-day allocation results (keyed 0:n_trading)
- `param_history::Dict{String,Vector{Tuple{Float64,Float64,Float64}}}` — (α,β,σ_ε) snapshots per ticker per trading day
- `wealth::Array{Float64,1}` — wealth series over the trading period
"""
function replay_engine_ewls(price_matrix::Array{Float64,2}, market_prices::Array{Float64,1},
    tickers::Array{String,1}, sim_params_init::Dict{String,Tuple{Float64,Float64,Float64}},
    rules_params::NamedTuple;
    B₀::Float64 = 10000.0, offset::Int = 63,
    half_life::Float64 = 63.0, prior_weight::Float64 = 63.0,
    N_short::Int = 21, N_long::Int = 63, GAIN::Float64 = 10.0,
    N_growth::Int = 10, cost_bps::Float64 = 0.0,
    epsilon::Float64 = 0.1)

    Δt = 1.0 / 252.0;
    K = length(tickers);
    n_days = size(price_matrix, 1);
    n_trading = n_days - offset;
    cost_rate = cost_bps / 10_000.0;

    # initialize EWLS states from pre-calibrated parameters -
    ewls_states = Dict{String,MyEWLSState}();
    for ticker in tickers
        (α₀, β₀, σ₀) = sim_params_init[ticker];
        ewls_states[ticker] = ewls_init(α₀, β₀, σ₀; half_life = half_life, prior_weight = prior_weight);
    end

    # compute market growth rates (CCGR) -
    gm_raw = compute_market_growth(market_prices; Δt = Δt);

    # compute per-ticker growth rates -
    ticker_growth = Dict{String,Array{Float64,1}}();
    for (k, ticker) in enumerate(tickers)
        prices_k = price_matrix[:, k + 1];
        gi = zeros(n_days - 1);
        for t in 2:n_days
            gi[t - 1] = (1.0 / Δt) * log(prices_k[t] / prices_k[t - 1]);
        end
        ticker_growth[ticker] = gi;
    end

    # warm-up: feed first offset days into EWLS without trading -
    for t in 1:min(offset - 1, length(gm_raw))
        gm_t = gm_raw[t];
        for ticker in tickers
            gi_t = ticker_growth[ticker][t];
            ewls_update!(ewls_states[ticker], gi_t, gm_t);
        end
    end

    # compute EMAs and lambda over full series -
    ema_s = compute_ema(market_prices; window = N_short);
    ema_l = compute_ema(market_prices; window = N_long);
    λ_series = compute_lambda(ema_s, ema_l; G = GAIN);
    λ_series[1:offset] .= 0.0;

    # compute market growth EMA -
    gm_ema = compute_ema(gm_raw; window = N_growth);

    # build the rebalancing context template -
    context = build(MyRebalancingContextModel, (
        B = B₀, tickers = tickers, marketdata = price_matrix,
        marketfactor = gm_ema, sim_parameters = sim_params_init,
        lambda = 0.0, Δt = Δt, epsilon = epsilon
    ));

    # build trigger rules -
    rules = build(MyTriggerRules, (
        max_drawdown = rules_params.max_drawdown,
        max_turnover = rules_params.max_turnover,
        rebalance_schedule = ones(Int, n_trading)
    ));

    # storage -
    results = Dict{Int,MyRebalancingResult}();
    param_history = Dict{String,Vector{Tuple{Float64,Float64,Float64}}}();
    for ticker in tickers
        param_history[ticker] = Tuple{Float64,Float64,Float64}[];
    end

    # --- initial allocation at offset ---
    sim_params_current = Dict{String,Tuple{Float64,Float64,Float64}}(
        ticker => (ewls_states[ticker].α, ewls_states[ticker].β, ewls_states[ticker].σ_ε)
        for ticker in tickers
    );
    context.sim_parameters = sim_params_current;
    results[0] = allocate_shares(offset, context; allocator = :cobb_douglas);

    if cost_rate > 0.0
        init_trade_value = sum(results[0].shares[i] * price_matrix[offset, i + 1] for i in 1:K);
        results[0].cash -= cost_rate * init_trade_value;
    end

    # record initial params -
    for ticker in tickers
        push!(param_history[ticker], sim_params_current[ticker]);
    end

    # track peak wealth for drawdown -
    peak_wealth = B₀;

    # --- daily trading loop ---
    for day in 1:n_trading

        actual_day = offset + day;

        # step 1: update EWLS with today's data -
        if actual_day - 1 <= length(gm_raw)
            gm_t = gm_raw[actual_day - 1];
            for ticker in tickers
                gi_t = ticker_growth[ticker][actual_day - 1];
                ewls_update!(ewls_states[ticker], gi_t, gm_t);
            end
        end

        # step 2: extract updated SIM params -
        for ticker in tickers
            sim_params_current[ticker] = (ewls_states[ticker].α, ewls_states[ticker].β, ewls_states[ticker].σ_ε);
        end

        # record params -
        for ticker in tickers
            push!(param_history[ticker], sim_params_current[ticker]);
        end

        # step 3: liquidate and check drawdown -
        prev = results[day - 1];
        liquidation_value = prev.cash;
        for i in 1:K
            liquidation_value += prev.shares[i] * price_matrix[actual_day, i + 1];
        end

        peak_wealth = max(peak_wealth, liquidation_value);
        drawdown = peak_wealth > 0.0 ? (peak_wealth - liquidation_value) / peak_wealth : 0.0;

        if drawdown > rules.max_drawdown
            # de-risk to cash -
            derisk_result = MyRebalancingResult();
            derisk_result.shares = zeros(K);
            prev_shares_value = liquidation_value - prev.cash;
            derisk_cash = liquidation_value;
            if cost_rate > 0.0
                derisk_cash -= cost_rate * prev_shares_value;
            end
            derisk_result.cash = derisk_cash;
            derisk_result.gamma = zeros(K);
            results[day] = derisk_result;
        else
            # step 4: rebalance with updated params -
            ctx = deepcopy(context);
            ctx.B = liquidation_value;
            ctx.lambda = λ_series[min(actual_day, length(λ_series))];
            ctx.sim_parameters = sim_params_current;

            new_result = allocate_shares(actual_day, ctx; allocator = :cobb_douglas);

            # turnover cap -
            if day > 0 && haskey(results, day - 1)
                old_shares = results[day - 1].shares;
                old_cash = results[day - 1].cash;
                new_shares = copy(new_result.shares);
                new_cash = new_result.cash;

                trade_value = sum(abs(new_shares[i] - old_shares[i]) * price_matrix[actual_day, i + 1] for i in 1:K);
                turnover_frac = liquidation_value > 0 ? trade_value / liquidation_value : 0.0;

                if turnover_frac > rules.max_turnover
                    scale = rules.max_turnover / turnover_frac;
                    for i in 1:K
                        new_result.shares[i] = old_shares[i] + scale * (new_shares[i] - old_shares[i]);
                    end
                    new_result.cash = old_cash + scale * (new_cash - old_cash);
                end

                if cost_rate > 0.0
                    realized_trade = sum(abs(new_result.shares[i] - old_shares[i]) * price_matrix[actual_day, i + 1] for i in 1:K);
                    new_result.cash -= cost_rate * realized_trade;
                end
            end

            results[day] = new_result;
        end
    end

    # compute wealth series -
    wealth = compute_wealth_series(results, price_matrix, tickers; offset = offset);

    return (results = results, param_history = param_history, wealth = wealth);
end

# --- Session 4: Sentiment Generation -------------------------------------------

"""
    generate_synthetic_sentiment(market_prices::Array{Float64,1};
        noise_σ::Float64 = 0.15, smoothing::Int = 5,
        seed::Int = -1) -> Array{Float64,1}

Generate synthetic sentiment scores correlated with market regime.
Sentiment is derived from short-term market returns + noise, smoothed and
mapped to [-1, 1] via tanh.

Bearish markets produce negative sentiment; bullish produce positive.
"""
function generate_synthetic_sentiment(market_prices::Array{Float64,1};
    noise_σ::Float64 = 0.15, smoothing::Int = 5,
    seed::Int = -1)::Array{Float64,1}

    if seed > 0
        Random.seed!(seed);
    end

    T = length(market_prices);
    sentiment = zeros(T);

    # compute 5-day return as base signal -
    for t in 6:T
        ret_5d = (market_prices[t] / market_prices[t-5] - 1.0);
        raw = ret_5d * 10.0 + noise_σ * randn(); # scale + noise
        sentiment[t] = tanh(raw);
    end

    # smooth with simple moving average -
    smoothed = copy(sentiment);
    for t in (smoothing+1):T
        smoothed[t] = mean(sentiment[(t-smoothing+1):t]);
    end

    return smoothed;
end

# --- Session 4: Escalation Checking --------------------------------------------

"""
    check_escalation_triggers(day::Int, ctx::MyProductionContext,
        wealth::Float64, peak_wealth::Float64, sentiment::Float64,
        current_action::Array{Int,1}, prev_action::Array{Int,1}) -> Array{MyEscalationEvent,1}

Check all escalation triggers and return any events that fired.

Triggers:
1. Drawdown exceeds threshold → critical
2. Sentiment crash (below threshold) → warning
3. Bandit churn (too many assets changed) → warning
"""
function check_escalation_triggers(day::Int, ctx::MyProductionContext,
    wealth::Float64, peak_wealth::Float64, sentiment::Float64,
    current_action::Array{Int,1}, prev_action::Array{Int,1})::Array{MyEscalationEvent,1}

    events = MyEscalationEvent[];

    # 1. Drawdown trigger -
    drawdown = peak_wealth > 0 ? (peak_wealth - wealth) / peak_wealth : 0.0;
    if drawdown > ctx.max_drawdown
        evt = MyEscalationEvent();
        evt.day = day;
        evt.trigger_type = "drawdown";
        evt.severity = :critical;
        evt.message = "Drawdown $(round(drawdown*100, digits=1))% exceeds limit $(round(ctx.max_drawdown*100, digits=1))%";
        evt.recommended_action = "De-risk to cash. Notify investment committee.";
        push!(events, evt);
    end

    # 2. Sentiment crash -
    if sentiment < ctx.sentiment_threshold
        evt = MyEscalationEvent();
        evt.day = day;
        evt.trigger_type = "sentiment_crash";
        evt.severity = :warning;
        evt.message = "Sentiment $(round(sentiment, digits=3)) below threshold $(ctx.sentiment_threshold)";
        evt.recommended_action = "Override lambda to $(ctx.sentiment_override_lambda). Review positions.";
        push!(events, evt);
    end

    # 3. Bandit churn -
    n_changes = sum(abs.(current_action .- prev_action));
    if n_changes > ctx.max_bandit_churn
        evt = MyEscalationEvent();
        evt.day = day;
        evt.trigger_type = "bandit_churn";
        evt.severity = :warning;
        evt.message = "Bandit changed $(n_changes) assets (limit: $(ctx.max_bandit_churn))";
        evt.recommended_action = "Hold previous allocation. Review bandit state.";
        push!(events, evt);
    end

    return events;
end

# --- Session 4: Production Simulation ------------------------------------------

"""
    run_production_simulation(ctx::MyProductionContext,
        market_prices::Array{Float64,1}, price_matrix::Array{Float64,2},
        sentiment_series::Array{Float64,1}, lambda_series::Array{Float64,1},
        gm_ema::Array{Float64,1};
        n_days::Int = 60, offset::Int = 84,
        n_bandit_iters::Int = 100) -> (results::Array{MyProductionDayResult,1}, events::Array{MyEscalationEvent,1})

Run a full production simulation: bandit selects assets, Cobb-Douglas allocates,
sentiment monitoring overrides lambda, escalation triggers fire when conditions are met.
"""
function run_production_simulation(ctx::MyProductionContext,
    market_prices::Array{Float64,1}, price_matrix::Array{Float64,2},
    sentiment_series::Array{Float64,1}, lambda_series::Array{Float64,1},
    gm_ema::Array{Float64,1};
    n_days::Int = 60, offset::Int = 84,
    n_bandit_iters::Int = 100)::Tuple{Array{MyProductionDayResult,1},Array{MyEscalationEvent,1}}

    K = length(ctx.tickers);
    Δt = 1.0 / 252.0;

    results = MyProductionDayResult[];
    all_events = MyEscalationEvent[];

    # initial state -
    current_shares = zeros(K);
    current_cash = ctx.B₀;
    peak_wealth = ctx.B₀;
    prev_bandit_action = ones(Int, K); # start with all assets

    for d in 1:n_days

        actual_day = offset + d;
        day_result = MyProductionDayResult();
        day_result.day = d;

        # --- Step 1: Read sentiment ---
        sent = sentiment_series[min(actual_day, length(sentiment_series))];
        day_result.sentiment = sent;

        # --- Step 2: Determine effective lambda ---
        λ_base = lambda_series[min(actual_day, length(lambda_series))];
        if sent < ctx.sentiment_threshold
            λ_eff = ctx.sentiment_override_lambda; # override: increase risk-aversion
        else
            λ_eff = λ_base;
        end
        day_result.lambda = λ_eff;

        # --- Step 3: Run bandit to select assets ---
        prices_now = [price_matrix[actual_day, k + 1] for k in 1:K];
        gm_t = gm_ema[min(actual_day, length(gm_ema))];

        bandit_ctx = build(MyBanditContext, (
            tickers = ctx.tickers, sim_parameters = ctx.sim_parameters,
            prices = prices_now, B = current_cash + sum(current_shares[k] * prices_now[k] for k in 1:K),
            gm_t = gm_t, lambda = λ_eff, epsilon = ctx.epsilon
        ));

        bandit_model = build(MyEpsilonGreedyBanditModel, (
            K = K, n_iterations = n_bandit_iters, alpha = 0.1
        ));

        bandit_result = solve_bandit(bandit_model, bandit_ctx);
        current_bandit_action = bandit_result.best_action;
        day_result.bandit_action = current_bandit_action;

        # --- Step 4: Check escalation triggers ---
        current_wealth = current_cash + sum(current_shares[k] * prices_now[k] for k in 1:K);
        peak_wealth = max(peak_wealth, current_wealth);

        events = check_escalation_triggers(d, ctx, current_wealth, peak_wealth,
            sent, current_bandit_action, prev_bandit_action);

        day_result.escalated = length(events) > 0;
        append!(all_events, events);

        # --- Step 5: Decide and execute ---
        has_critical = any(e.severity == :critical for e in events);
        has_churn = any(e.trigger_type == "bandit_churn" for e in events);

        if has_critical
            # de-risk to cash -
            current_shares = zeros(K);
            current_cash = current_wealth;
            day_result.rebalanced = true;
            day_result.gamma = zeros(K);
        elseif has_churn
            # hold previous allocation (ignore bandit recommendation) -
            day_result.rebalanced = false;
            current_bandit_action = prev_bandit_action;
            day_result.gamma = zeros(K);
        else
            # normal operation: allocate via Cobb-Douglas with bandit-selected assets -
            modified_sim = copy(ctx.sim_parameters);
            for (k, ticker) in enumerate(ctx.tickers)
                if current_bandit_action[k] == 0
                    (_, βᵢ, σᵢ) = ctx.sim_parameters[ticker];
                    modified_sim[ticker] = (-10.0, βᵢ, σᵢ);
                end
            end

            gamma = compute_preference_weights(modified_sim, ctx.tickers, gm_t, λ_eff);
            day_result.gamma = gamma;

            problem = MyCobbDouglasChoiceProblem();
            problem.gamma = gamma;
            problem.prices = prices_now;
            problem.B = current_wealth;
            problem.epsilon = ctx.epsilon;
            (new_shares, new_cash) = allocate_cobb_douglas(problem);

            # apply turnover cap -
            if d > 1
                old_value = sum(current_shares[k] * prices_now[k] for k in 1:K);
                trade_value = sum(abs(new_shares[k] - current_shares[k]) * prices_now[k] for k in 1:K);
                turnover = old_value > 0 ? trade_value / old_value : 0.0;
                if turnover > ctx.max_turnover
                    scale = ctx.max_turnover / turnover;
                    for k in 1:K
                        new_shares[k] = current_shares[k] + scale * (new_shares[k] - current_shares[k]);
                    end
                end
            end

            current_shares = new_shares;
            current_cash = new_cash;
            day_result.rebalanced = true;
        end

        # --- Step 6: Record state ---
        day_result.shares = copy(current_shares);
        day_result.cash = current_cash;
        day_result.wealth = current_cash + sum(current_shares[k] * prices_now[k] for k in 1:K);
        prev_bandit_action = current_bandit_action;

        push!(results, day_result);
    end

    return (results, all_events);
end

# --- Session 4: Dashboard Metrics -----------------------------------------------

"""
    compute_dashboard_metrics(results::Array{MyProductionDayResult,1},
        events::Array{MyEscalationEvent,1}) -> Dict{String, Any}

Compute aggregate metrics for the operational dashboard.
"""
function compute_dashboard_metrics(results::Array{MyProductionDayResult,1},
    events::Array{MyEscalationEvent,1})::Dict{String,Any}

    n_days = length(results);
    K = length(results[1].shares);

    # wealth series -
    wealth = [r.wealth for r in results];

    # rebalance frequency -
    n_rebalances = sum(r.rebalanced for r in results);

    # bandit churn: how often does the selected subset change? -
    n_bandit_changes = 0;
    for d in 2:n_days
        if results[d].bandit_action != results[d-1].bandit_action
            n_bandit_changes += 1;
        end
    end

    # escalation frequency -
    n_escalations = length(events);
    n_critical = sum(e.severity == :critical for e in events; init=0);
    n_warning = sum(e.severity == :warning for e in events; init=0);

    # drawdown -
    peak = accumulate(max, wealth);
    max_dd = maximum((peak .- wealth) ./ peak);

    # return metrics -
    return Dict(
        "n_days" => n_days,
        "final_wealth" => wealth[end],
        "max_drawdown" => max_dd,
        "n_rebalances" => n_rebalances,
        "n_bandit_changes" => n_bandit_changes,
        "n_escalations" => n_escalations,
        "n_critical" => n_critical,
        "n_warning" => n_warning,
        "avg_sentiment" => mean(r.sentiment for r in results),
        "wealth_series" => wealth
    );
end

# --- Session 4: Live Production Functions ---------------------------------------

"""
    compute_live_sentiment(prices::Array{Float64,1}; lookback::Int = 5) -> Float64

Compute a single sentiment score from the tail of a real price series. Uses the
same formula as `generate_synthetic_sentiment` (5-day return mapped through tanh)
but returns a scalar from the most recent data, not a full series.

Returns 0.0 if insufficient data (fewer than `lookback + 1` prices).
"""
function compute_live_sentiment(prices::Array{Float64,1}; lookback::Int = 5)::Float64

    T = length(prices);
    if T < lookback + 1
        return 0.0;
    end

    ret = prices[end] / prices[end - lookback] - 1.0;
    return tanh(10.0 * ret);
end

"""
    compute_position_drawdown(equity_history::Array{Float64,1}) -> Float64

Compute the current drawdown from peak of an equity time series. Returns 0.0
if the series is at or above its all-time high.
"""
function compute_position_drawdown(equity_history::Array{Float64,1})::Float64

    length(equity_history) == 0 && return 0.0;
    peak = maximum(equity_history);
    peak <= 0.0 && return 0.0;
    return (peak - equity_history[end]) / peak;
end

"""
    run_production_step(ctx::MyProductionContext, ewls_states::Dict{String,MyEWLSState},
        price_matrix::Array{Float64,2}, market_prices::Array{Float64,1},
        tickers::Array{String,1}, day_idx::Int;
        n_bandit_iters::Int = 100, prev_action::Union{Nothing,Array{Int,1}} = nothing,
        peak_wealth::Float64 = 0.0, current_shares::Array{Float64,1} = Float64[],
        current_cash::Float64 = 0.0,
        N_short::Int = 21, N_long::Int = 63, GAIN::Float64 = 10.0,
        N_growth::Int = 10) -> Tuple{MyLiveProductionDayResult,Array{MyEscalationEvent,1}}

Execute a single production step: read sentiment, run bandit, check escalation
triggers, allocate via Cobb-Douglas. This is the single-day version of
`run_production_simulation`, designed for live execution where each day is
processed independently.

### Arguments
- `ctx` — production context (tickers, SIM params, risk limits)
- `ewls_states` — current EWLS states per ticker (mutated in place)
- `price_matrix` — historical price matrix (n_days, K+1)
- `market_prices` — historical market (SPY) prices
- `tickers` — ticker symbols
- `day_idx` — index into price_matrix for "today"
- `n_bandit_iters` — number of bandit iterations
- `prev_action` — previous day's bandit action (for churn detection)
- `peak_wealth` — peak portfolio value seen so far
- `current_shares` — current share holdings
- `current_cash` — current cash balance
"""
function run_production_step(ctx::MyProductionContext, ewls_states::Dict{String,MyEWLSState},
    price_matrix::Array{Float64,2}, market_prices::Array{Float64,1},
    tickers::Array{String,1}, day_idx::Int;
    n_bandit_iters::Int = 100, prev_action::Union{Nothing,Array{Int,1}} = nothing,
    peak_wealth::Float64 = 0.0, current_shares::Array{Float64,1} = Float64[],
    current_cash::Float64 = 0.0,
    N_short::Int = 21, N_long::Int = 63, GAIN::Float64 = 10.0,
    N_growth::Int = 10)::Tuple{MyLiveProductionDayResult,Array{MyEscalationEvent,1}}

    Δt = 1.0 / 252.0;
    K = length(tickers);

    if isempty(current_shares)
        current_shares = zeros(K);
    end
    if prev_action === nothing
        prev_action = ones(Int, K);
    end

    # --- Step 1: Update EWLS with today's data ---
    if day_idx >= 2 && day_idx <= length(market_prices)
        gm_t = (1.0 / Δt) * log(market_prices[day_idx] / market_prices[day_idx - 1]);
        for (k, ticker) in enumerate(tickers)
            gi_t = (1.0 / Δt) * log(price_matrix[day_idx, k + 1] / price_matrix[day_idx - 1, k + 1]);
            ewls_update!(ewls_states[ticker], gi_t, gm_t);
        end
    end

    # extract current EWLS params -
    sim_params_current = Dict{String,Tuple{Float64,Float64,Float64}}(
        ticker => (ewls_states[ticker].α, ewls_states[ticker].β, ewls_states[ticker].σ_ε)
        for ticker in tickers
    );

    # --- Step 2: Compute sentiment from real prices ---
    sentiment = compute_live_sentiment(market_prices[1:min(day_idx, length(market_prices))]);

    # --- Step 3: Compute effective lambda ---
    ema_s = compute_ema(market_prices[1:min(day_idx, length(market_prices))]; window = N_short);
    ema_l = compute_ema(market_prices[1:min(day_idx, length(market_prices))]; window = N_long);
    λ_series = compute_lambda(ema_s, ema_l; G = GAIN);
    λ_base = λ_series[end];

    λ_eff = sentiment < ctx.sentiment_threshold ? ctx.sentiment_override_lambda : λ_base;

    # --- Step 4: Compute current wealth ---
    wealth = current_cash;
    for i in 1:K
        wealth += current_shares[i] * price_matrix[day_idx, i + 1];
    end
    peak_wealth = max(peak_wealth, wealth);

    # --- Step 5: Run bandit to select assets ---
    gm_raw = compute_market_growth(market_prices[1:min(day_idx, length(market_prices))]; Δt = Δt);
    gm_ema = compute_ema(gm_raw; window = N_growth);
    gm_current = gm_ema[end];

    prices_now = [price_matrix[day_idx, k + 1] for k in 1:K];
    bandit_ctx = MyBanditContext();
    bandit_ctx.tickers = tickers;
    bandit_ctx.sim_parameters = sim_params_current;
    bandit_ctx.prices = prices_now;
    bandit_ctx.B = wealth;
    bandit_ctx.lambda = λ_eff;
    bandit_ctx.gm_t = gm_current;
    bandit_ctx.epsilon = ctx.epsilon;

    bandit_model = MyEpsilonGreedyBanditModel();
    bandit_model.K = K;
    bandit_model.n_iterations = n_bandit_iters;
    bandit_model.alpha = 0.1;

    bandit_result = solve_bandit(bandit_model, bandit_ctx);
    bandit_action = bandit_result.best_action;

    # --- Step 6: Check escalation triggers ---
    events = check_escalation_triggers(day_idx, ctx, wealth, peak_wealth,
        sentiment, bandit_action, prev_action);

    has_critical = any(e.severity == :critical for e in events);
    has_churn = any(e.trigger_type == "bandit_churn" for e in events);
    escalated = !isempty(events);

    # --- Step 7: Allocate ---
    rebalanced = true;
    if has_critical
        # de-risk to cash -
        target_shares = zeros(K);
        target_cash = wealth;
        gamma = zeros(K);
    elseif has_churn
        # hold previous allocation -
        target_shares = copy(current_shares);
        target_cash = current_cash;
        gamma = zeros(K);
        rebalanced = false;
    else
        # apply bandit: excluded assets get penalized SIM params -
        modified_params = copy(sim_params_current);
        for (k, ticker) in enumerate(tickers)
            if bandit_action[k] == 0
                (_, β_k, σ_k) = modified_params[ticker];
                modified_params[ticker] = (-10.0, β_k, σ_k);
            end
        end

        gamma = compute_preference_weights(modified_params, tickers, gm_current, λ_eff);

        problem = MyCobbDouglasChoiceProblem();
        problem.gamma = gamma;
        problem.prices = prices_now;
        problem.B = wealth;
        problem.epsilon = ctx.epsilon;
        (target_shares, target_cash) = allocate_cobb_douglas(problem);

        # apply turnover cap -
        trade_value = sum(abs(target_shares[i] - current_shares[i]) * prices_now[i] for i in 1:K);
        turnover_frac = wealth > 0 ? trade_value / wealth : 0.0;
        if turnover_frac > ctx.max_turnover
            scale = ctx.max_turnover / turnover_frac;
            for i in 1:K
                target_shares[i] = current_shares[i] + scale * (target_shares[i] - current_shares[i]);
            end
            target_cash = current_cash + scale * (target_cash - current_cash);
        end
    end

    # --- Step 8: Package result ---
    result = MyLiveProductionDayResult();
    result.day = day_idx;
    result.timestamp = "";
    result.shares = target_shares;
    result.cash = target_cash;
    result.wealth = wealth;
    result.gamma = gamma;
    result.bandit_action = bandit_action;
    result.sentiment = sentiment;
    result.lambda = λ_eff;
    result.rebalanced = rebalanced;
    result.escalated = escalated;
    result.ewls_params = sim_params_current;
    result.order_ids = String[];

    return (result, events);
end

"""
    apply_stress_scenario(scenario::MyStressScenario, current_prices::Array{Float64,1},
        current_shares::Array{Float64,1}, cash::Float64, ctx::MyProductionContext,
        peak_wealth::Float64, prev_action::Array{Int,1},
        tickers::Array{String,1}) -> MyStressResult

Apply a hypothetical shock to the current portfolio and evaluate whether the
escalation system would respond correctly. Computes stressed portfolio value,
drawdown, and which triggers would fire.

For tickers without explicit per-ticker shocks in `scenario.ticker_shocks`, the
market-level shock `scenario.market_shock` is applied directly to the price.
"""
function apply_stress_scenario(scenario::MyStressScenario, current_prices::Array{Float64,1},
    current_shares::Array{Float64,1}, cash::Float64, ctx::MyProductionContext,
    peak_wealth::Float64, prev_action::Array{Int,1},
    tickers::Array{String,1})::MyStressResult

    K = length(tickers);

    # apply shocks to prices -
    stressed_prices = copy(current_prices);
    for k in 1:K
        ticker = tickers[k];
        if haskey(scenario.ticker_shocks, ticker)
            stressed_prices[k] *= (1.0 + scenario.ticker_shocks[ticker]);
        else
            stressed_prices[k] *= (1.0 + scenario.market_shock);
        end
    end

    # compute stressed wealth -
    stressed_wealth = cash;
    for k in 1:K
        stressed_wealth += current_shares[k] * stressed_prices[k];
    end

    # compute drawdown from peak -
    drawdown = peak_wealth > 0.0 ? (peak_wealth - stressed_wealth) / peak_wealth : 0.0;

    # compute stressed sentiment (assume market shock affects returns) -
    stressed_sentiment = tanh(10.0 * scenario.market_shock);

    # check escalation triggers -
    triggers = check_escalation_triggers(0, ctx, stressed_wealth, peak_wealth,
        stressed_sentiment, prev_action, prev_action);

    has_critical = any(e.severity == :critical for e in triggers);
    would_derisk = has_critical;

    # capital preserved: if de-risk, we liquidate at stressed prices -
    capital_preserved = would_derisk ? stressed_wealth : stressed_wealth;

    result = MyStressResult();
    result.scenario_label = scenario.label;
    result.stressed_wealth = stressed_wealth;
    result.drawdown = drawdown;
    result.triggers_fired = triggers;
    result.would_derisk = would_derisk;
    result.capital_preserved = capital_preserved;

    return result;
end

# --- Session 4: News Sentiment Pipeline -----------------------------------------

"""
    simulate_news_corpus(prices, tickers, scenario; seed) -> MyNewsCorpus

Generate a synthetic news corpus and a news-shocked price matrix for the given
scenario. For each ticker on each trading day, draw a Poisson-distributed
number of items at rate `scenario.arrival_intensity`. Each item gets a
sentiment score `s ~ N(scenario.sentiment_mean, scenario.sentiment_sd)` clipped
to `[-1, +1]` and a templated stub text (filled with realistic copy by
[`generate_news_text!`](@ref)).

The asymmetric shock applied to ticker `i` on day `t` is the sum over items
mentioning `(i, t)` of

    Δ = -κ_neg · max(-s, 0) + κ_pos · max(s, 0)

The shocked price path compounds the original returns and applies the news
shock multiplicatively `shock_lag` days after each item's publication:

    P_shocked[t, i] = P_shocked[t-1, i] · (P[t, i] / P[t-1, i]) · exp(Δ_{i, t - shock_lag})

With `shock_lag = 0` the shock hits the same day (contemporaneous); with
`shock_lag = 1` (the pedagogical default used in the S4 example) news
published on day `t` moves the price on day `t+1`, mirroring the typical
real-world event study where an 8-K or earnings release after the close
drives the following session's return. The lag also makes `news_factor[t]`
genuinely predictive of `g_{t+1}` so the news-extended γ has out-of-sample
alpha to chase.

The returned `MyNewsCorpus` carries the items, the `T × K` `news_factor`
(daily mean of `true_score` over items mentioning each ticker; zero if none),
the shocked price matrix, and the seed for reproducibility. After calling
[`score_news_with_claude!`](@ref), recompute the factor with
[`aggregate_news_factor`](@ref) using `:claude_score` to get the
LLM-estimated signal.

### Arguments
- `prices::Array{Float64,2}` — `T × K` baseline price matrix (no day-index column)
- `tickers::Array{String,1}` — length-`K` ticker labels (column order)
- `scenario::MyNewsScenario` — controls shock magnitudes and arrival rates

### Keyword arguments
- `seed::Int = -1` — RNG seed; if non-positive, RNG is left as-is
- `shock_lag::Int = 0` — days between publication and price reaction

### Returns
- `MyNewsCorpus` with `items`, `tickers`, `scenario`, `news_factor`,
  `shocked_prices`, and `seed` populated.
"""
function simulate_news_corpus(prices::Array{Float64,2}, tickers::Array{String,1},
    scenario::MyNewsScenario; seed::Int = -1, shock_lag::Int = 0)::MyNewsCorpus

    if seed > 0
        Random.seed!(seed);
    end

    (T, K) = size(prices);
    @assert length(tickers) == K "tickers length must equal number of price columns";

    # generate items per ticker per day -
    items = MyNewsItem[];
    arrival = Distributions.Poisson(scenario.arrival_intensity);
    for i in 1:K, t in 1:T
        n_items = rand(arrival);
        for _ in 1:n_items
            s = scenario.sentiment_mean + scenario.sentiment_sd * randn();
            s = clamp(s, -1.0, 1.0);
            item = MyNewsItem();
            item.ticker = tickers[i];
            item.publication_day = t;
            item.text = "[stub] $(tickers[i]) news, day $t, planted sentiment $(round(s, digits=3))";
            item.true_score = s;
            item.claude_score = NaN;
            item.source = "synthetic_news";
            push!(items, item);
        end
    end

    # aggregate true sentiment per (ticker, day) into the news factor -
    news_factor = aggregate_news_factor(items, T, tickers; use_score = :true_score);

    # apply asymmetric shock to the price path -
    κp = scenario.kappa_pos;
    κn = scenario.kappa_neg;
    shocked = zeros(T, K);
    shocked[1, :] .= prices[1, :];
    for i in 1:K, t in 2:T
        # items whose publication predates this return by `shock_lag` days
        src_day = t - shock_lag;
        Δ = 0.0;
        if src_day >= 1
            day_items = filter(it -> it.ticker == tickers[i] && it.publication_day == src_day, items);
            for it in day_items
                s = it.true_score;
                Δ += -κn * max(-s, 0.0) + κp * max(s, 0.0);
            end
        end
        raw_growth = prices[t, i] / prices[t-1, i];
        shocked[t, i] = shocked[t-1, i] * raw_growth * exp(Δ);
    end

    # package -
    corpus = MyNewsCorpus();
    corpus.items = items;
    corpus.tickers = tickers;
    corpus.scenario = scenario;
    corpus.news_factor = news_factor;
    corpus.shocked_prices = shocked;
    corpus.seed = seed;

    return corpus;
end

"""
    aggregate_news_factor(items, T, tickers; use_score) -> Matrix{Float64}

Build the `T × K` daily news-factor matrix by averaging item scores per
`(publication_day, ticker)`. Days without items get zero (interpreted as "no
news today"). The score field is selected by `use_score`:

- `:true_score` — the planted sentiment used to generate the price shock
- `:claude_score` — the sentiment Claude extracted from the item text
- `:raw_score` — synonym for `:true_score`

### Arguments
- `items::Array{MyNewsItem,1}` — corpus items
- `T::Int` — number of trading days (rows)
- `tickers::Array{String,1}` — column order

### Keyword arguments
- `use_score::Symbol = :claude_score` — which score field to aggregate
"""
function aggregate_news_factor(items::Array{MyNewsItem,1}, T::Int,
    tickers::Array{String,1}; use_score::Symbol = :claude_score)::Array{Float64,2}

    K = length(tickers);
    ticker_idx = Dict(t => i for (i, t) in enumerate(tickers));

    sums = zeros(T, K);
    counts = zeros(Int, T, K);
    for it in items
        haskey(ticker_idx, it.ticker) || continue;
        i = ticker_idx[it.ticker];
        t = it.publication_day;
        (1 <= t <= T) || continue;
        s = if use_score === :claude_score
            it.claude_score;
        elseif use_score === :true_score || use_score === :raw_score
            it.true_score;
        else
            error("Unknown score field: $use_score");
        end
        isnan(s) && continue; # skip unscored items
        sums[t, i] += s;
        counts[t, i] += 1;
    end

    factor = zeros(T, K);
    for t in 1:T, i in 1:K
        factor[t, i] = counts[t, i] > 0 ? sums[t, i] / counts[t, i] : 0.0;
    end

    return factor;
end

# --- Session 4: Claude API Integration ------------------------------------------
# All API calls are gated behind `live=true`. When `live=false` (the default),
# `generate_news_text!` is a no-op (items keep the templated stub text from
# `simulate_news_corpus`) and `score_news_with_claude!` simulates Claude scoring
# by adding small Gaussian noise to `true_score`. This lets the notebook run
# end-to-end without an API key while preserving the live-mode code path.

"""
    _sentiment_bucket(s) -> String

Map a sentiment score in `[-1, +1]` to a discrete tone label used in headline
generation prompts. Bucketing avoids asking the LLM to encode a real number;
the round-trip noise is then recovered by [`score_news_with_claude!`](@ref).
"""
function _sentiment_bucket(s::Float64)::String
    if s <= -0.6
        return "strongly negative";
    elseif s <= -0.2
        return "mildly negative";
    elseif s < 0.2
        return "neutral";
    elseif s < 0.6
        return "mildly positive";
    else
        return "strongly positive";
    end
end

"""
    _call_claude(prompt; api_key, model, max_tokens) -> String

Make a single POST to the Anthropic Messages API and return the text payload.
Errors propagate up. Used by [`generate_news_text!`](@ref) and
[`score_news_with_claude!`](@ref).
"""
function _call_claude(prompt::String; api_key::String, model::String,
    max_tokens::Int)::String

    url = "https://api.anthropic.com/v1/messages";
    headers = [
        "x-api-key" => api_key,
        "anthropic-version" => "2023-06-01",
        "content-type" => "application/json",
    ];
    body = JSON.json(Dict(
        "model" => model,
        "max_tokens" => max_tokens,
        "messages" => [Dict("role" => "user", "content" => prompt)],
    ));
    response = HTTP.request("POST", url, headers, body);
    parsed = JSON.parse(String(response.body));
    return String(parsed["content"][1]["text"]);
end

"""
    generate_news_text!(corpus; live, api_key, model, rate_limit_seconds) -> MyNewsCorpus

Replace each item's templated stub text with a Claude-generated single-line
headline whose tone matches the item's `true_score`. The score is bucketed
into one of five tone labels (strongly negative / mildly negative / neutral /
mildly positive / strongly positive) and the LLM writes a plausible headline
in that tone.

In cached/stub mode (`live=false`, the default) the corpus is returned
unchanged — items keep the stub text written by [`simulate_news_corpus`](@ref).
This lets the notebook execute end-to-end without an API key.

In live mode, makes one HTTP call per item to the Anthropic Messages API. For
~400 items at the default 0-second rate limit, regeneration takes
2-3 minutes and costs roughly \$1 on Sonnet 4.6.

### Arguments
- `corpus::MyNewsCorpus` — corpus to populate (mutated in place)

### Keyword arguments
- `live::Bool = false` — call Claude when true; no-op when false
- `api_key::String = ""` — Anthropic API key; if empty, reads `ENV["ANTHROPIC_API_KEY"]`
- `model::String = "claude-sonnet-4-6"` — Anthropic model id
- `rate_limit_seconds::Float64 = 0.0` — sleep between requests
"""
function generate_news_text!(corpus::MyNewsCorpus;
    live::Bool = false,
    api_key::String = "",
    model::String = "claude-sonnet-4-6",
    rate_limit_seconds::Float64 = 0.0)::MyNewsCorpus

    if !live
        return corpus;
    end

    if isempty(api_key)
        api_key = get(ENV, "ANTHROPIC_API_KEY", "");
    end
    isempty(api_key) && error("generate_news_text! live=true requires ANTHROPIC_API_KEY in ENV or api_key kwarg");

    for item in corpus.items
        bucket = _sentiment_bucket(item.true_score);
        prompt = string(
            "Write a single-line financial news headline (max 100 characters) ",
            "about ticker $(item.ticker). The tone should be $bucket. ",
            "Output only the headline text, no preamble, no quotes."
        );
        text = _call_claude(prompt; api_key = api_key, model = model, max_tokens = 100);
        item.text = strip(text);
        item.source = "synthetic_news_claude";
        rate_limit_seconds > 0.0 && sleep(rate_limit_seconds);
    end

    return corpus;
end

"""
    score_news_with_claude!(corpus; live, api_key, model, cached_noise_sd, seed,
        rate_limit_seconds) -> MyNewsCorpus

Fill `item.claude_score` for each item with Claude's extracted sentiment in
`[-1, +1]`.

In cached/stub mode (`live=false`, the default), `claude_score = true_score +
N(0, cached_noise_sd)` clipped to `[-1, +1]`. The Gaussian noise simulates
imperfect LLM scoring so the downstream regression sees a realistic
signal-to-noise ratio without an API call. With `cached_noise_sd = 0.0`, the
LLM is treated as oracle (perfect knowledge case).

In live mode, makes one HTTP call per item asking Claude to score the
headline. Unparseable replies become `NaN` and are skipped by
[`aggregate_news_factor`](@ref).

### Arguments
- `corpus::MyNewsCorpus` — corpus whose items get `claude_score` filled (mutated)

### Keyword arguments
- `live::Bool = false` — call Claude when true; otherwise stub with Gaussian noise
- `api_key::String = ""` — Anthropic API key; defaults to `ENV["ANTHROPIC_API_KEY"]`
- `model::String = "claude-sonnet-4-6"` — Anthropic model id
- `cached_noise_sd::Float64 = 0.10` — stub-mode scoring noise standard deviation
- `seed::Int = -1` — RNG seed for stub-mode noise; non-positive leaves RNG as-is
- `rate_limit_seconds::Float64 = 0.0` — sleep between live-mode requests
"""
function score_news_with_claude!(corpus::MyNewsCorpus;
    live::Bool = false,
    api_key::String = "",
    model::String = "claude-sonnet-4-6",
    cached_noise_sd::Float64 = 0.10,
    seed::Int = -1,
    rate_limit_seconds::Float64 = 0.0)::MyNewsCorpus

    if !live
        if seed > 0
            Random.seed!(seed);
        end
        for item in corpus.items
            noisy = item.true_score + cached_noise_sd * randn();
            item.claude_score = clamp(noisy, -1.0, 1.0);
        end
        return corpus;
    end

    if isempty(api_key)
        api_key = get(ENV, "ANTHROPIC_API_KEY", "");
    end
    isempty(api_key) && error("score_news_with_claude! live=true requires ANTHROPIC_API_KEY in ENV or api_key kwarg");

    for item in corpus.items
        prompt = string(
            "Read this financial news headline and assign a sentiment score in [-1.0, +1.0] ",
            "for the mentioned company. -1 = strongly bearish, 0 = neutral, ",
            "+1 = strongly bullish. Output only the number, no explanation.\n\n",
            "Headline: $(item.text)\nScore:"
        );
        reply = _call_claude(prompt; api_key = api_key, model = model, max_tokens = 10);
        s = try
            parse(Float64, strip(reply));
        catch
            NaN;
        end
        item.claude_score = isnan(s) ? NaN : clamp(s, -1.0, 1.0);
        rate_limit_seconds > 0.0 && sleep(rate_limit_seconds);
    end

    return corpus;
end

# --- Session 4: SIM-with-news Loading Estimator ---------------------------------

"""
    estimate_sim_with_news(growth_rates, market_growth, news_factor, tickers)
        -> (sim_parameters, nu_loadings)

Estimate the SIM-with-news model per asset by ordinary least squares,

    g_{i,t} = α_i + β_i · g_{m,t} + ν_i · news_{i,t} + ε_{i,t}

The three input matrices must share the same `T` rows; align them at the
caller (typically by trimming the first day so growth rates of length `T-1`
match a corresponding `T-1`-row slice of the news factor).

### Arguments
- `growth_rates::Array{Float64,2}` — `T × K` per-asset CCGR growth rates
- `market_growth::Array{Float64,1}` — length-`T` market growth rate series
- `news_factor::Array{Float64,2}` — `T × K` daily aggregated news per ticker
- `tickers::Array{String,1}` — length-`K` ticker labels (column order)

### Returns
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` —
  `(α_i, β_i, σ_ε,i)` per ticker, drop-in replacement for the SIM dict the
  rebalancing engine consumes
- `nu_loadings::Array{Float64,1}` — length-`K` news loadings in column order
"""
function estimate_sim_with_news(growth_rates::Array{Float64,2},
    market_growth::Array{Float64,1}, news_factor::Array{Float64,2},
    tickers::Array{String,1})

    (T, K) = size(growth_rates);
    @assert length(market_growth) == T "market_growth length must match growth_rates rows";
    @assert size(news_factor) == (T, K) "news_factor shape must match growth_rates";
    @assert length(tickers) == K "tickers length must match growth_rates columns";

    sim_params = Dict{String,Tuple{Float64,Float64,Float64}}();
    nu_loadings = zeros(K);

    for i in 1:K
        # design matrix [1, g_m, news_i] -
        X = hcat(ones(T), market_growth, news_factor[:, i]);
        y = growth_rates[:, i];
        coeffs = (X' * X) \ (X' * y);
        residuals = y .- X * coeffs;
        dof = max(T - 3, 1);
        σ_ε = sqrt(sum(residuals .^ 2) / dof);
        sim_params[tickers[i]] = (coeffs[1], coeffs[2], σ_ε);
        nu_loadings[i] = coeffs[3];
    end

    return (sim_params, nu_loadings);
end

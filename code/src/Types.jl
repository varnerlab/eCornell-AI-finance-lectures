# --- Session 1: Single Index Model Types ----------------------------------------

"""
    MySIMParameterEstimate

Holds the estimated Single Index Model parameters for one asset.

### Fields
- `ticker::String` — asset ticker name
- `α::Float64` — Jensen's alpha (firm-specific excess return)
- `β::Float64` — market beta (sensitivity to market factor)
- `σ_ε::Float64` — residual standard deviation (idiosyncratic risk)
- `r²::Float64` — R-squared of the regression fit
"""
mutable struct MySIMParameterEstimate

    # data -
    ticker::String
    α::Float64
    β::Float64
    σ_ε::Float64
    r²::Float64

    # constructor -
    MySIMParameterEstimate() = new();
end

"""
    MySharpeRatioPortfolioChoiceProblem

Holds the data needed to solve the maximum Sharpe ratio portfolio problem via SOCP.

### Fields
- `Σ::Array{Float64,2}` — covariance matrix (from SIM or sample)
- `risk_free_rate::Float64` — risk-free rate (annualized)
- `α::Array{Float64,1}` — firm-specific alphas from SIM
- `β::Array{Float64,1}` — market betas from SIM
- `gₘ::Float64` — expected market growth rate
- `bounds::Array{Float64,2}` — lower/upper weight bounds (N × 2)
"""
mutable struct MySharpeRatioPortfolioChoiceProblem

    # data -
    Σ::Array{Float64,2}
    risk_free_rate::Float64
    α::Array{Float64,1}
    β::Array{Float64,1}
    gₘ::Float64
    bounds::Array{Float64,2}

    # constructor -
    MySharpeRatioPortfolioChoiceProblem() = new();
end

# --- Session 1: Portfolio Optimization Types ------------------------------------

"""
    MyPortfolioAllocationProblem

Holds the data needed to solve a portfolio allocation problem.

### Fields
- `μ::Array{Float64,1}` — expected return vector (N × 1)
- `Σ::Array{Float64,2}` — covariance matrix (N × N)
- `bounds::Array{Float64,2}` — lower/upper weight bounds (N × 2)
- `R::Float64` — target return (for mean-variance problems)
"""
mutable struct MyPortfolioAllocationProblem

    # data -
    μ::Array{Float64,1}
    Σ::Array{Float64,2}
    bounds::Array{Float64,2}
    R::Float64

    # constructor -
    MyPortfolioAllocationProblem() = new();
end

"""
    MyPortfolioPerformanceResult

Holds the results of a portfolio backtest or evaluation.

### Fields
- `weights::Array{Float64,1}` — optimal portfolio weights
- `expected_return::Float64` — portfolio expected return
- `variance::Float64` — portfolio variance
- `drawdown::Float64` — maximum drawdown over the evaluation period
- `turnover::Float64` — total turnover
- `trading_cost::Float64` — estimated trading cost
"""
mutable struct MyPortfolioPerformanceResult

    # data -
    weights::Array{Float64,1}
    expected_return::Float64
    variance::Float64
    drawdown::Float64
    turnover::Float64
    trading_cost::Float64

    # constructor -
    MyPortfolioPerformanceResult() = new();
end

# --- Session 2: Utility Choice Problem Types -----------------------------------

"""
    MyCobbDouglasChoiceProblem

Budget-constrained Cobb-Douglas utility maximization problem.

    maximize   kappa(gamma) * prod(n_i^gamma_i)
    subject to B = sum(n_i * p_i)
               n_i >= epsilon  for non-preferred assets

### Fields
- `gamma::Array{Float64,1}` — preference exponents per asset (from SIM + sentiment)
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `epsilon::Float64` — minimum share floor for non-preferred assets
"""
mutable struct MyCobbDouglasChoiceProblem

    # data -
    gamma::Array{Float64,1}
    prices::Array{Float64,1}
    B::Float64
    epsilon::Float64

    # constructor -
    MyCobbDouglasChoiceProblem() = new();
end

"""
    MyCESChoiceProblem

Budget-constrained CES (Constant Elasticity of Substitution) utility maximization.

    U(n) = (sum gamma_i * n_i^rho)^(1/rho)

where rho = (sigma - 1)/sigma and sigma is the elasticity of substitution.
As sigma -> 1, CES -> Cobb-Douglas. As sigma -> inf, CES -> linear.

### Fields
- `gamma::Array{Float64,1}` — preference weights per asset
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `epsilon::Float64` — minimum share floor for non-preferred assets
- `sigma::Float64` — elasticity of substitution (sigma > 0, sigma != 1)
"""
mutable struct MyCESChoiceProblem

    # data -
    gamma::Array{Float64,1}
    prices::Array{Float64,1}
    B::Float64
    epsilon::Float64
    sigma::Float64

    # constructor -
    MyCESChoiceProblem() = new();
end

"""
    MyLogLinearChoiceProblem

Budget-constrained log-linear (weighted log) utility maximization.

    U(n) = sum gamma_i * log(n_i)

Equivalent to Cobb-Douglas with equal exponent scaling (Nash bargaining solution).

### Fields
- `gamma::Array{Float64,1}` — preference weights per asset
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `epsilon::Float64` — minimum share floor for non-preferred assets
"""
mutable struct MyLogLinearChoiceProblem

    # data -
    gamma::Array{Float64,1}
    prices::Array{Float64,1}
    B::Float64
    epsilon::Float64

    # constructor -
    MyLogLinearChoiceProblem() = new();
end

# --- Session 2: Rebalancing Engine Types ----------------------------------------

"""
    MyRebalancingContextModel

Holds the context needed by the AI rebalancing engine at each decision point.

### Fields
- `B::Float64` — current budget (cash + liquidation value)
- `tickers::Array{String,1}` — asset ticker names
- `marketdata::Array{Float64,2}` — price matrix (T x N+1), column 1 = day index
- `marketfactor::Array{Float64,1}` — EMA-smoothed market growth series
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params (alpha_i, beta_i, sigma_i) per ticker
- `lambda::Float64` — current sentiment/risk-aversion parameter
- `Δt::Float64` — time step in years (1/252 for daily)
- `epsilon::Float64` — minimum share floor for non-preferred assets
"""
mutable struct MyRebalancingContextModel

    # data -
    B::Float64
    tickers::Array{String,1}
    marketdata::Array{Float64,2}
    marketfactor::Array{Float64,1}
    sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}
    lambda::Float64
    Δt::Float64
    epsilon::Float64

    # constructor -
    MyRebalancingContextModel() = new();
end

"""
    MyTriggerRules

Defines the decision rules that govern when the rebalancing engine acts.

### Fields
- `max_drawdown::Float64` — drawdown threshold to trigger de-risk (e.g., 0.10 = 10%)
- `max_turnover::Float64` — maximum turnover per rebalance (e.g., 0.50 = 50%)
- `rebalance_schedule::Array{Int,1}` — binary schedule: 1 = rebalance, 0 = hold
"""
mutable struct MyTriggerRules

    # data -
    max_drawdown::Float64
    max_turnover::Float64
    rebalance_schedule::Array{Int,1}

    # constructor -
    MyTriggerRules() = new();
end

"""
    MyRebalancingResult

Holds the state of the portfolio at a single time step.

### Fields
- `shares::Array{Float64,1}` — number of shares held per asset
- `cash::Float64` — unallocated cash
- `gamma::Array{Float64,1}` — preference weights at this step
"""
mutable struct MyRebalancingResult

    # data -
    shares::Array{Float64,1}
    cash::Float64
    gamma::Array{Float64,1}

    # constructor -
    MyRebalancingResult() = new();
end

# --- Session 3: Backtest Types --------------------------------------------------

"""
    MyBacktestScenario

Describes a single backtest scenario with metadata about market conditions.

### Fields
- `label::String` — human-readable scenario name (e.g., "Normal", "Crisis")
- `price_paths::Array{Float64,3}` — synthetic price paths (n_paths x T x N_assets)
- `market_paths::Array{Float64,2}` — synthetic market index paths (n_paths x T)
- `n_paths::Int` — number of Monte Carlo paths
- `n_steps::Int` — number of trading days per path
"""
mutable struct MyBacktestScenario

    # data -
    label::String
    price_paths::Array{Float64,3}
    market_paths::Array{Float64,2}
    n_paths::Int
    n_steps::Int

    # constructor -
    MyBacktestScenario() = new();
end

"""
    MyBacktestResult

Holds the results of backtesting a strategy across multiple paths.

### Fields
- `scenario_label::String` — which scenario was tested
- `strategy_label::String` — which strategy was tested
- `final_wealth::Array{Float64,1}` — final wealth for each path
- `max_drawdowns::Array{Float64,1}` — max drawdown for each path
- `sharpe_ratios::Array{Float64,1}` — Sharpe ratio for each path
"""
mutable struct MyBacktestResult

    # data -
    scenario_label::String
    strategy_label::String
    final_wealth::Array{Float64,1}
    max_drawdowns::Array{Float64,1}
    sharpe_ratios::Array{Float64,1}

    # constructor -
    MyBacktestResult() = new();
end

"""
    MyValidationReport

Holds pass/fail criteria and results for strategy validation.

### Fields
- `strategy_label::String` — which strategy
- `criteria::Dict{String,Float64}` — threshold values (e.g., "min_sharpe" => 0.3)
- `actuals::Dict{String,Float64}` — actual median values across paths
- `passed::Dict{String,Bool}` — pass/fail for each criterion
"""
mutable struct MyValidationReport

    # data -
    strategy_label::String
    criteria::Dict{String,Float64}
    actuals::Dict{String,Float64}
    passed::Dict{String,Bool}

    # constructor -
    MyValidationReport() = new();
end

# --- Session 3: Bandit Types ----------------------------------------------------

"""
    MyBanditContext

Static context for the bandit portfolio selection problem.
The bandit chooses *which assets* to include; the Cobb-Douglas allocator decides *how many shares*.

### Fields
- `tickers::Array{String,1}` — asset ticker names
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params per ticker
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `gm_t::Float64` — current expected market growth
- `lambda::Float64` — current sentiment/risk-aversion
- `epsilon::Float64` — minimum share floor
"""
mutable struct MyBanditContext

    # data -
    tickers::Array{String,1}
    sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}
    prices::Array{Float64,1}
    B::Float64
    gm_t::Float64
    lambda::Float64
    epsilon::Float64

    # constructor -
    MyBanditContext() = new();
end

"""
    MyEpsilonGreedyBanditModel

Parameters for the epsilon-greedy combinatorial bandit.

### Fields
- `K::Int` — number of assets (2^K arms = all subsets)
- `n_iterations::Int` — number of bandit rounds
- `alpha::Float64` — learning rate for reward averaging
"""
mutable struct MyEpsilonGreedyBanditModel

    # data -
    K::Int
    n_iterations::Int
    alpha::Float64

    # constructor -
    MyEpsilonGreedyBanditModel() = new();
end

"""
    MyBanditResult

Output from a bandit run: the best action (asset subset) and convergence data.

### Fields
- `best_action::Array{Int,1}` — binary vector: 1 = include asset, 0 = exclude
- `best_utility::Float64` — utility of the best action
- `reward_history::Array{Float64,1}` — reward at each iteration
- `exploration_history::Array{Float64,1}` — exploration probability at each iteration
- `arm_means::Array{Float64,1}` — average reward per arm at convergence
"""
mutable struct MyBanditResult

    # data -
    best_action::Array{Int,1}
    best_utility::Float64
    reward_history::Array{Float64,1}
    exploration_history::Array{Float64,1}
    arm_means::Array{Float64,1}

    # constructor -
    MyBanditResult() = new();
end

# --- Session 3: EWLS Types -----------------------------------------------------

"""
    MyEWLSState

Running state for Exponentially Weighted Least Squares (EWLS) estimation of the
Single Index Model parameters (α, β, σ_ε). Each observation is weighted by
η^(T−t) where η = 2^(−1/half_life), so data from `half_life` days ago receives
half the weight of today's observation.

### Fields
- `Sw::Float64` — sum of weights
- `Swx::Float64` — weighted sum of market growth rates
- `Swy::Float64` — weighted sum of asset growth rates
- `Swxx::Float64` — weighted sum of squared market growth rates
- `Swxy::Float64` — weighted cross-product of asset and market growth rates
- `Swyy::Float64` — weighted sum of squared asset growth rates
- `η::Float64` — per-step decay factor, 2^(−1/half_life)
- `α::Float64` — current Jensen's alpha estimate
- `β::Float64` — current market beta estimate
- `σ_ε::Float64` — current idiosyncratic volatility estimate
"""
mutable struct MyEWLSState

    # sufficient statistics -
    Sw::Float64
    Swx::Float64
    Swy::Float64
    Swxx::Float64
    Swxy::Float64
    Swyy::Float64

    # decay factor -
    η::Float64

    # current estimates -
    α::Float64
    β::Float64
    σ_ε::Float64

    # constructor -
    MyEWLSState() = new();
end

# --- Session 4: Production Types ------------------------------------------------

"""
    MySentimentSignal

A single sentiment observation — synthetic or from an external source.

### Fields
- `score::Float64` — sentiment score in [-1, 1] (negative = bearish, positive = bullish)
- `source::String` — where it came from (e.g., "synthetic", "news", "filings")
- `day::Int` — trading day index
"""
mutable struct MySentimentSignal

    # data -
    score::Float64
    source::String
    day::Int

    # constructor -
    MySentimentSignal() = new();
end

"""
    MyEscalationEvent

Records when the production system flags a condition for human review.

### Fields
- `day::Int` — when the escalation occurred
- `trigger_type::String` — what caused it (e.g., "drawdown", "sentiment_crash", "bandit_churn")
- `severity::Symbol` — :warning or :critical
- `message::String` — human-readable description
- `recommended_action::String` — what the system suggests
"""
mutable struct MyEscalationEvent

    # data -
    day::Int
    trigger_type::String
    severity::Symbol
    message::String
    recommended_action::String

    # constructor -
    MyEscalationEvent() = new();
end

"""
    MyProductionDayResult

The state of the production portfolio system at a single trading day.

### Fields
- `day::Int` — trading day index
- `shares::Array{Float64,1}` — shares held per asset
- `cash::Float64` — unallocated cash
- `wealth::Float64` — total portfolio value
- `gamma::Array{Float64,1}` — preference weights
- `bandit_action::Array{Int,1}` — bandit's selected asset subset
- `sentiment::Float64` — sentiment score for this day
- `lambda::Float64` — effective lambda after sentiment override
- `rebalanced::Bool` — did we rebalance today?
- `escalated::Bool` — did an escalation trigger fire?
"""
mutable struct MyProductionDayResult

    # data -
    day::Int
    shares::Array{Float64,1}
    cash::Float64
    wealth::Float64
    gamma::Array{Float64,1}
    bandit_action::Array{Int,1}
    sentiment::Float64
    lambda::Float64
    rebalanced::Bool
    escalated::Bool

    # constructor -
    MyProductionDayResult() = new();
end

"""
    MyProductionContext

Holds the full context for a production simulation run.

### Fields
- `tickers::Array{String,1}` — asset ticker names
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params per ticker
- `B₀::Float64` — initial budget
- `epsilon::Float64` — minimum share floor
- `max_drawdown::Float64` — drawdown threshold for de-risking
- `max_turnover::Float64` — turnover cap
- `sentiment_threshold::Float64` — sentiment level that triggers lambda override
- `sentiment_override_lambda::Float64` — lambda value to use when sentiment is below threshold
- `max_bandit_churn::Int` — max assets the bandit can switch in one day before escalation
"""
mutable struct MyProductionContext

    # data -
    tickers::Array{String,1}
    sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}
    B₀::Float64
    epsilon::Float64
    max_drawdown::Float64
    max_turnover::Float64
    sentiment_threshold::Float64
    sentiment_override_lambda::Float64
    max_bandit_churn::Int

    # constructor -
    MyProductionContext() = new();
end

# --- Session 4: Live Production Types -------------------------------------------

"""
    MyLiveProductionDayResult

The state of the live production portfolio system at a single trading day. Extends
`MyProductionDayResult` with real-execution metadata: wall-clock timestamp, EWLS
parameter snapshots, and Alpaca order IDs for audit trail.

### Fields
- `day::Int` — production day index (1, 2, ...)
- `timestamp::String` — wall-clock time of execution (ISO 8601)
- `shares::Array{Float64,1}` — shares held per asset after execution
- `cash::Float64` — unallocated cash
- `wealth::Float64` — total portfolio value (shares + cash)
- `gamma::Array{Float64,1}` — Cobb-Douglas preference weights
- `bandit_action::Array{Int,1}` — bandit's selected asset subset (binary)
- `sentiment::Float64` — sentiment score for this day
- `lambda::Float64` — effective lambda after sentiment override
- `rebalanced::Bool` — did we execute trades today?
- `escalated::Bool` — did an escalation trigger fire?
- `ewls_params::Dict{String,Tuple{Float64,Float64,Float64}}` — EWLS (α, β, σ_ε) snapshot
- `order_ids::Array{String,1}` — Alpaca order IDs for audit trail
"""
mutable struct MyLiveProductionDayResult

    # core state (same as MyProductionDayResult) -
    day::Int
    timestamp::String
    shares::Array{Float64,1}
    cash::Float64
    wealth::Float64
    gamma::Array{Float64,1}
    bandit_action::Array{Int,1}
    sentiment::Float64
    lambda::Float64
    rebalanced::Bool
    escalated::Bool

    # live-execution metadata -
    ewls_params::Dict{String,Tuple{Float64,Float64,Float64}}
    order_ids::Array{String,1}

    # constructor -
    MyLiveProductionDayResult() = new();
end

"""
    MyStressScenario

A hypothetical shock scenario for what-if analysis on a live portfolio.

### Fields
- `label::String` — human-readable scenario name (e.g., "Market -20%")
- `market_shock::Float64` — proportional shock to market prices (e.g., -0.20 for -20%)
- `ticker_shocks::Dict{String,Float64}` — per-ticker overrides (empty = use market_shock × β)
"""
mutable struct MyStressScenario

    # data -
    label::String
    market_shock::Float64
    ticker_shocks::Dict{String,Float64}

    # constructor -
    MyStressScenario() = new();
end

"""
    MyStressResult

The result of applying a stress scenario to a live portfolio.

### Fields
- `scenario_label::String` — which scenario was applied
- `stressed_wealth::Float64` — portfolio value after shock
- `drawdown::Float64` — drawdown from peak after shock
- `triggers_fired::Array{MyEscalationEvent,1}` — escalation events from the shock
- `would_derisk::Bool` — would the system de-risk to cash?
- `capital_preserved::Float64` — estimated capital after de-risking (or stressed wealth if no de-risk)
"""
mutable struct MyStressResult

    # data -
    scenario_label::String
    stressed_wealth::Float64
    drawdown::Float64
    triggers_fired::Array{MyEscalationEvent,1}
    would_derisk::Bool
    capital_preserved::Float64

    # constructor -
    MyStressResult() = new();
end

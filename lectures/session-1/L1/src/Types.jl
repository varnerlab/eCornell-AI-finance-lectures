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

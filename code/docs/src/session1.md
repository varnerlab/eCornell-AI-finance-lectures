# Session 1: Portfolio Optimization

Minimum-variance portfolio construction and stress testing using classical Markowitz optimization.

## Types

```@docs
MyPortfolioAllocationProblem
MyPortfolioPerformanceResult
```

## Factory Methods

```@docs
build(::Type{MyPortfolioAllocationProblem}; μ::Array{Float64,1}, Σ::Array{Float64,2}, bounds::Array{Float64,2}, R::Float64)
```

## Functions

```@docs
solve_minvariance
compute_drawdown
compute_turnover
```

## File I/O

```@docs
load_price_data
save_results
load_results
```

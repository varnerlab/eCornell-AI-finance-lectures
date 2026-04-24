# Session 1: Portfolio Optimization

Single Index Model (SIM) parameter estimation with bootstrap uncertainty
quantification, minimum-variance portfolio construction, and maximum Sharpe
ratio portfolio selection via classical Markowitz optimization.

## Types

### Single Index Model

```@docs
MySIMParameterEstimate
MySharpeRatioPortfolioChoiceProblem
```

### Portfolio Allocation

```@docs
MyPortfolioAllocationProblem
MyPortfolioPerformanceResult
```

## Factory Methods

```@docs
build(::Type{MySIMParameterEstimate}, ::NamedTuple)
build(::Type{MySharpeRatioPortfolioChoiceProblem}, ::NamedTuple)
build(::Type{MyPortfolioAllocationProblem}; μ::Array{Float64,1}, Σ::Array{Float64,2}, bounds::Array{Float64,2}, R::Float64)
```

## Single Index Model Estimation

```@docs
estimate_sim
bootstrap_sim
build_sim_covariance
```

## Portfolio Optimization

```@docs
solve_minvariance
solve_max_sharpe
```

## Performance Metrics

```@docs
compute_drawdown
compute_turnover
```

## Market Data Loaders

```@docs
MyTrainingMarketDataSet
MyTestingMarketDataSet
MyExtendedTestingMarketDataSet
MyDeploymentMarketDataSet
MyMarketSurrogateModel
MyPortfolioSurrogateModel
MySyntheticTrainingDataSet
MySIMCalibration
MyCurrentPrices
```

## File I/O

```@docs
load_price_data
save_results
load_results
```

## HMM Aliases

The `eCornellAIFinance` module re-exports a handful of
[JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) functions under short
aliases so that session notebooks can call them without qualifying the
namespace. For details see the
[JumpHMM.jl documentation](https://varnerlab.github.io/JumpHMM.jl/).

| Alias | Target |
|:------|:-------|
| `hmm_fit` | `JumpHMM.fit` |
| `hmm_tune` | `JumpHMM.tune` |
| `hmm_simulate` | `JumpHMM.simulate` |
| `hmm_validate` | `JumpHMM.validate` |
| `JumpHiddenMarkovModel` | `JumpHMM.JumpHiddenMarkovModel` |

# Session 3: HMM Backtesting & Bandits

Monte Carlo backtesting using Hidden Markov Model-generated scenarios, validation framework, and combinatorial bandit asset selection.

## Types

### Backtesting

```@docs
MyBacktestScenario
MyBacktestResult
MyValidationReport
```

### Bandits

```@docs
MyBanditContext
MyEpsilonGreedyBanditModel
MyBanditResult
```

## Factory Methods

```@docs
build(::Type{MyBacktestScenario}, ::NamedTuple)
build(::Type{MyValidationReport}; strategy_label::String, criteria::Dict{String,Float64}, actuals::Dict{String,Float64})
build(::Type{MyBanditContext}, ::NamedTuple)
build(::Type{MyEpsilonGreedyBanditModel}, ::NamedTuple)
build(::Type{MySigmaBanditModel}, ::NamedTuple)
```

## Data Generation

```@docs
generate_training_prices
generate_hmm_scenario
generate_hybrid_scenario
```

## Backtesting

```@docs
backtest_engine
backtest_buyhold
backtest_buyhold_market
backtest_bandit
compute_cvar
```

## Bandit Functions

```@docs
bandit_world
solve_bandit
compute_regret
```

## Sigma-Bandit (CES Elasticity Learning)

### Types

```@docs
MySigmaBanditModel
MySigmaBanditResult
```

### Functions

```@docs
classify_regime
sigma_bandit_world
solve_sigma_bandit
backtest_sigma_bandit
build_compliance_config
```

## EWLS (Exponentially Weighted Least Squares)

### Types

```@docs
MyEWLSState
```

### Functions

```@docs
ewls_init
ewls_update!
replay_engine_ewls
```

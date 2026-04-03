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
```

## Data Generation

```@docs
generate_training_prices
generate_hmm_scenario
```

## Backtesting

```@docs
backtest_engine
backtest_buyhold
backtest_bandit
```

## Bandit Functions

```@docs
bandit_world
solve_bandit
compute_regret
```

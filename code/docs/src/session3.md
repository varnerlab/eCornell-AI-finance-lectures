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
build(::Type{MyEtaBanditModel}, ::NamedTuple)
```

## Data Generation

```@docs
generate_training_prices
generate_hmm_scenario
generate_hybrid_scenario
generate_drifted_hybrid_scenario
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

## Eta-Bandit (CES Elasticity Learning)

### Types

```@docs
MyEtaBanditModel
MyEtaBanditResult
```

### Functions

```@docs
classify_regime
eta_bandit_world
solve_eta_bandit
solve_eta_bandit_multipath
backtest_eta_bandit
build_compliance_config
```

## REINFORCE Policy (Policy-Gradient Elasticity Learning)

### Types

```@docs
MyREINFORCEPolicy
MyValueBaseline
MyREINFORCETrainer
MyPolicyTrainingResult
```

### Factory Methods

```@docs
build(::Type{MyREINFORCEPolicy}, ::NamedTuple)
build(::Type{MyValueBaseline}, ::NamedTuple)
build(::Type{MyREINFORCETrainer}, ::NamedTuple)
```

### Functions

```@docs
build_policy_state
sample_action
policy_log_pdf
policy_eta
evaluate_baseline
compute_returns
train_reinforce!
rollout_policy_episode
backtest_eta_policy
backtest_eta_heuristic
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

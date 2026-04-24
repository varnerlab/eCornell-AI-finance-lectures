# Session 2: AI Rebalancing Engine

Sentiment-driven utility maximization with Cobb-Douglas, CES, and log-linear allocators, wrapped in a daily rebalancing engine.

## Types

### Utility Choice Problems

```@docs
MyCobbDouglasChoiceProblem
MyCESChoiceProblem
MyLogLinearChoiceProblem
```

### Rebalancing Engine

```@docs
MyRebalancingContextModel
MyTriggerRules
MyRebalancingResult
```

## Factory Methods

```@docs
build(::Type{MyCobbDouglasChoiceProblem}, ::NamedTuple)
build(::Type{MyCESChoiceProblem}, ::NamedTuple)
build(::Type{MyLogLinearChoiceProblem}, ::NamedTuple)
build(::Type{MyRebalancingContextModel}, ::NamedTuple)
build(::Type{MyTriggerRules}, ::NamedTuple)
```

## Signal Computation

```@docs
compute_ema
compute_lambda
compute_market_growth
compute_preference_weights
compute_adaptive_eta
```

## Utility-Based Allocation

```@docs
allocate_cobb_douglas
allocate_ces
allocate_log_linear
```

## Utility Evaluation

```@docs
evaluate_cobb_douglas
evaluate_ces
evaluate_log_linear
```

## Rebalancing Engine

```@docs
allocate_shares
run_rebalancing_engine
compute_wealth_series
```

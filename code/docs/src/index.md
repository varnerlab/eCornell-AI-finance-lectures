# eCornell AI Finance

API documentation for the `eCornellAIFinance.jl` package — the shared code library for the eCornell AI in Finance work sessions (May 2026).

## Sessions

| Session | Topic | Key Functions |
|:--------|:------|:--------------|
| 1 | Portfolio Optimization | [`solve_minvariance`](@ref), [`compute_drawdown`](@ref), [`compute_turnover`](@ref) |
| 2 | AI Rebalancing Engine | [`allocate_cobb_douglas`](@ref), [`run_rebalancing_engine`](@ref), [`compute_preference_weights`](@ref) |
| 3 | HMM Backtesting & Bandits | [`generate_hmm_scenario`](@ref), [`backtest_engine`](@ref), [`solve_bandit`](@ref) |
| 4 | Production Operations | [`run_production_simulation`](@ref), [`check_escalation_triggers`](@ref), [`compute_dashboard_metrics`](@ref) |

## Installation

This package is included as a local dependency in each session's `Project.toml`. When you run `include("Include.jl")` in a session notebook, the package is automatically loaded via `using eCornellAIFinance`.

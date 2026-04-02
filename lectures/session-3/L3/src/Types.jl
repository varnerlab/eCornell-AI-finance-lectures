"""
    MyBacktestScenario

Describes a single backtest scenario with metadata about market conditions.

### Fields
- `label::String` — human-readable scenario name (e.g., "Normal", "Crisis")
- `price_paths::Array{Float64,3}` — synthetic price paths (n_paths × T × N_assets)
- `market_paths::Array{Float64,2}` — synthetic market index paths (n_paths × T)
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

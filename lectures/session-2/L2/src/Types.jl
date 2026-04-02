"""
    MyRebalancingContextModel

Holds the context needed by the AI rebalancing engine at each decision point.

### Fields
- `B::Float64` — current budget (cash + liquidation value)
- `tickers::Array{String,1}` — asset ticker names
- `marketdata::Array{Float64,2}` — price matrix (T × N+1), column 1 = day index
- `marketfactor::Array{Float64,1}` — EMA-smoothed market growth series
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params (αᵢ, βᵢ, σᵢ) per ticker
- `lambda::Float64` — current sentiment/risk-aversion from EMA crossover
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

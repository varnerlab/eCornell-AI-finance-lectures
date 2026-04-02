# --- Import Session 2 and Session 3 types and functions ---
const _PATH_TO_SESSION2 = joinpath(dirname(dirname(_ROOT)), "session-2", "L2");
const _PATH_TO_SESSION3 = joinpath(dirname(dirname(_ROOT)), "session-3", "L3");

# Session 2: Cobb-Douglas allocator, rebalancing engine -
include(joinpath(_PATH_TO_SESSION2, "src", "Types.jl"));
include(joinpath(_PATH_TO_SESSION2, "src", "Factory.jl"));
include(joinpath(_PATH_TO_SESSION2, "src", "Compute.jl"));

# Session 3: Backtest and bandit types -
include(joinpath(_PATH_TO_SESSION3, "src", "Types.jl"));
include(joinpath(_PATH_TO_SESSION3, "src", "Factory.jl"));

# --- Production Types -----------------------------------------------------------

"""
    MySentimentSignal

A single sentiment observation — synthetic or from an external source.

### Fields
- `score::Float64` — sentiment score in [-1, 1] (negative = bearish, positive = bullish)
- `source::String` — where it came from (e.g., "synthetic", "news", "filings")
- `day::Int` — trading day index
"""
mutable struct MySentimentSignal

    # data -
    score::Float64
    source::String
    day::Int

    # constructor -
    MySentimentSignal() = new();
end

"""
    MyEscalationEvent

Records when the production system flags a condition for human review.

### Fields
- `day::Int` — when the escalation occurred
- `trigger_type::String` — what caused it (e.g., "drawdown", "sentiment_crash", "bandit_churn")
- `severity::Symbol` — :warning or :critical
- `message::String` — human-readable description
- `recommended_action::String` — what the system suggests
"""
mutable struct MyEscalationEvent

    # data -
    day::Int
    trigger_type::String
    severity::Symbol
    message::String
    recommended_action::String

    # constructor -
    MyEscalationEvent() = new();
end

"""
    MyProductionDayResult

The state of the production portfolio system at a single trading day.

### Fields
- `day::Int` — trading day index
- `shares::Array{Float64,1}` — shares held per asset
- `cash::Float64` — unallocated cash
- `wealth::Float64` — total portfolio value
- `gamma::Array{Float64,1}` — preference weights
- `bandit_action::Array{Int,1}` — bandit's selected asset subset
- `sentiment::Float64` — sentiment score for this day
- `lambda::Float64` — effective lambda after sentiment override
- `rebalanced::Bool` — did we rebalance today?
- `escalated::Bool` — did an escalation trigger fire?
"""
mutable struct MyProductionDayResult

    # data -
    day::Int
    shares::Array{Float64,1}
    cash::Float64
    wealth::Float64
    gamma::Array{Float64,1}
    bandit_action::Array{Int,1}
    sentiment::Float64
    lambda::Float64
    rebalanced::Bool
    escalated::Bool

    # constructor -
    MyProductionDayResult() = new();
end

"""
    MyProductionContext

Holds the full context for a production simulation run.

### Fields
- `tickers::Array{String,1}` — asset ticker names
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params per ticker
- `B₀::Float64` — initial budget
- `epsilon::Float64` — minimum share floor
- `max_drawdown::Float64` — drawdown threshold for de-risking
- `max_turnover::Float64` — turnover cap
- `sentiment_threshold::Float64` — sentiment level that triggers lambda override
- `sentiment_override_lambda::Float64` — lambda value to use when sentiment is below threshold
- `max_bandit_churn::Int` — max assets the bandit can switch in one day before escalation
"""
mutable struct MyProductionContext

    # data -
    tickers::Array{String,1}
    sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}
    B₀::Float64
    epsilon::Float64
    max_drawdown::Float64
    max_turnover::Float64
    sentiment_threshold::Float64
    sentiment_override_lambda::Float64
    max_bandit_churn::Int

    # constructor -
    MyProductionContext() = new();
end

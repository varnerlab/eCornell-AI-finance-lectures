# --- Utility Choice Problem Types ------------------------------------------------

"""
    MyCobbDouglasChoiceProblem

Budget-constrained Cobb-Douglas utility maximization problem.

    maximize   kappa(gamma) * prod(n_i^gamma_i)
    subject to B = sum(n_i * p_i)
               n_i >= epsilon  for non-preferred assets

### Fields
- `gamma::Array{Float64,1}` — preference exponents per asset (from SIM + sentiment)
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `epsilon::Float64` — minimum share floor for non-preferred assets
"""
mutable struct MyCobbDouglasChoiceProblem

    # data -
    gamma::Array{Float64,1}
    prices::Array{Float64,1}
    B::Float64
    epsilon::Float64

    # constructor -
    MyCobbDouglasChoiceProblem() = new();
end

"""
    MyCESChoiceProblem

Budget-constrained CES (Constant Elasticity of Substitution) utility maximization.

    U(n) = (sum gamma_i * n_i^rho)^(1/rho)

where rho = (sigma - 1)/sigma and sigma is the elasticity of substitution.
As sigma -> 1, CES -> Cobb-Douglas. As sigma -> inf, CES -> linear.

### Fields
- `gamma::Array{Float64,1}` — preference weights per asset
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `epsilon::Float64` — minimum share floor for non-preferred assets
- `sigma::Float64` — elasticity of substitution (sigma > 0, sigma != 1)
"""
mutable struct MyCESChoiceProblem

    # data -
    gamma::Array{Float64,1}
    prices::Array{Float64,1}
    B::Float64
    epsilon::Float64
    sigma::Float64

    # constructor -
    MyCESChoiceProblem() = new();
end

"""
    MyLogLinearChoiceProblem

Budget-constrained log-linear (weighted log) utility maximization.

    U(n) = sum gamma_i * log(n_i)

Equivalent to Cobb-Douglas with equal exponent scaling (Nash bargaining solution).

### Fields
- `gamma::Array{Float64,1}` — preference weights per asset
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `epsilon::Float64` — minimum share floor for non-preferred assets
"""
mutable struct MyLogLinearChoiceProblem

    # data -
    gamma::Array{Float64,1}
    prices::Array{Float64,1}
    B::Float64
    epsilon::Float64

    # constructor -
    MyLogLinearChoiceProblem() = new();
end

# --- Rebalancing Engine Types ---------------------------------------------------

"""
    MyRebalancingContextModel

Holds the context needed by the AI rebalancing engine at each decision point.

### Fields
- `B::Float64` — current budget (cash + liquidation value)
- `tickers::Array{String,1}` — asset ticker names
- `marketdata::Array{Float64,2}` — price matrix (T x N+1), column 1 = day index
- `marketfactor::Array{Float64,1}` — EMA-smoothed market growth series
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params (alpha_i, beta_i, sigma_i) per ticker
- `lambda::Float64` — current sentiment/risk-aversion parameter
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

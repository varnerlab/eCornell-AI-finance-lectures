# --- Backtest Types --------------------------------------------------------------

"""
    MyBacktestScenario

Describes a single backtest scenario with metadata about market conditions.

### Fields
- `label::String` — human-readable scenario name (e.g., "Normal", "Crisis")
- `price_paths::Array{Float64,3}` — synthetic price paths (n_paths x T x N_assets)
- `market_paths::Array{Float64,2}` — synthetic market index paths (n_paths x T)
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

# --- Bandit Types ---------------------------------------------------------------

"""
    MyBanditContext

Static context for the bandit portfolio selection problem.
The bandit chooses *which assets* to include; the Cobb-Douglas allocator decides *how many shares*.

### Fields
- `tickers::Array{String,1}` — asset ticker names
- `sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}` — SIM params per ticker
- `prices::Array{Float64,1}` — current share prices
- `B::Float64` — total budget
- `gm_t::Float64` — current expected market growth
- `lambda::Float64` — current sentiment/risk-aversion
- `epsilon::Float64` — minimum share floor
"""
mutable struct MyBanditContext

    # data -
    tickers::Array{String,1}
    sim_parameters::Dict{String,Tuple{Float64,Float64,Float64}}
    prices::Array{Float64,1}
    B::Float64
    gm_t::Float64
    lambda::Float64
    epsilon::Float64

    # constructor -
    MyBanditContext() = new();
end

"""
    MyEpsilonGreedyBanditModel

Parameters for the epsilon-greedy combinatorial bandit.

### Fields
- `K::Int` — number of assets (2^K arms = all subsets)
- `n_iterations::Int` — number of bandit rounds
- `alpha::Float64` — learning rate for reward averaging
"""
mutable struct MyEpsilonGreedyBanditModel

    # data -
    K::Int
    n_iterations::Int
    alpha::Float64

    # constructor -
    MyEpsilonGreedyBanditModel() = new();
end

"""
    MyBanditResult

Output from a bandit run: the best action (asset subset) and convergence data.

### Fields
- `best_action::Array{Int,1}` — binary vector: 1 = include asset, 0 = exclude
- `best_utility::Float64` — utility of the best action
- `reward_history::Array{Float64,1}` — reward at each iteration
- `exploration_history::Array{Float64,1}` — exploration probability at each iteration
- `arm_means::Array{Float64,1}` — average reward per arm at convergence
"""
mutable struct MyBanditResult

    # data -
    best_action::Array{Int,1}
    best_utility::Float64
    reward_history::Array{Float64,1}
    exploration_history::Array{Float64,1}
    arm_means::Array{Float64,1}

    # constructor -
    MyBanditResult() = new();
end

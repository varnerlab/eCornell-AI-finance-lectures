# --- Session 1: Portfolio Builders -----------------------------------------------

"""
    build(type::Type{MyPortfolioAllocationProblem};
        μ::Array{Float64,1}, Σ::Array{Float64,2},
        bounds::Array{Float64,2}, R::Float64) -> MyPortfolioAllocationProblem

Build a `MyPortfolioAllocationProblem` instance from the given data.
"""
function build(type::Type{MyPortfolioAllocationProblem};
    μ::Array{Float64,1}, Σ::Array{Float64,2},
    bounds::Array{Float64,2}, R::Float64)::MyPortfolioAllocationProblem

    # build -
    problem = MyPortfolioAllocationProblem();
    problem.μ = μ;
    problem.Σ = Σ;
    problem.bounds = bounds;
    problem.R = R;

    # return -
    return problem;
end

# --- Session 2: Utility Problem Builders ----------------------------------------

"""
    build(type::Type{MyCobbDouglasChoiceProblem}, data::NamedTuple) -> MyCobbDouglasChoiceProblem

Build a Cobb-Douglas choice problem from a named tuple.
Expected fields: gamma, prices, B, epsilon.
"""
function build(type::Type{MyCobbDouglasChoiceProblem}, data::NamedTuple)::MyCobbDouglasChoiceProblem

    # build -
    problem = MyCobbDouglasChoiceProblem();
    problem.gamma = data.gamma;
    problem.prices = data.prices;
    problem.B = data.B;
    problem.epsilon = data.epsilon;

    # return -
    return problem;
end

"""
    build(type::Type{MyCESChoiceProblem}, data::NamedTuple) -> MyCESChoiceProblem

Build a CES choice problem from a named tuple.
Expected fields: gamma, prices, B, epsilon, sigma.
"""
function build(type::Type{MyCESChoiceProblem}, data::NamedTuple)::MyCESChoiceProblem

    # build -
    problem = MyCESChoiceProblem();
    problem.gamma = data.gamma;
    problem.prices = data.prices;
    problem.B = data.B;
    problem.epsilon = data.epsilon;
    problem.sigma = data.sigma;

    # return -
    return problem;
end

"""
    build(type::Type{MyLogLinearChoiceProblem}, data::NamedTuple) -> MyLogLinearChoiceProblem

Build a log-linear choice problem from a named tuple.
Expected fields: gamma, prices, B, epsilon.
"""
function build(type::Type{MyLogLinearChoiceProblem}, data::NamedTuple)::MyLogLinearChoiceProblem

    # build -
    problem = MyLogLinearChoiceProblem();
    problem.gamma = data.gamma;
    problem.prices = data.prices;
    problem.B = data.B;
    problem.epsilon = data.epsilon;

    # return -
    return problem;
end

# --- Session 2: Rebalancing Engine Builders -------------------------------------

"""
    build(type::Type{MyRebalancingContextModel}, data::NamedTuple) -> MyRebalancingContextModel

Build a rebalancing context model from a named tuple of parameters.
"""
function build(type::Type{MyRebalancingContextModel}, data::NamedTuple)::MyRebalancingContextModel

    # build -
    model = MyRebalancingContextModel();
    model.B = data.B;
    model.tickers = data.tickers;
    model.marketdata = data.marketdata;
    model.marketfactor = data.marketfactor;
    model.sim_parameters = data.sim_parameters;
    model.lambda = data.lambda;
    model.Δt = data.Δt;
    model.epsilon = data.epsilon;

    # return -
    return model;
end

"""
    build(type::Type{MyTriggerRules}, data::NamedTuple) -> MyTriggerRules

Build trigger rules from a named tuple.
"""
function build(type::Type{MyTriggerRules}, data::NamedTuple)::MyTriggerRules

    # build -
    rules = MyTriggerRules();
    rules.max_drawdown = data.max_drawdown;
    rules.max_turnover = data.max_turnover;
    rules.rebalance_schedule = data.rebalance_schedule;

    # return -
    return rules;
end

# --- Session 3: Backtest Builders -----------------------------------------------

"""
    build(type::Type{MyBacktestScenario}, data::NamedTuple) -> MyBacktestScenario
"""
function build(type::Type{MyBacktestScenario}, data::NamedTuple)::MyBacktestScenario

    scenario = MyBacktestScenario();
    scenario.label = data.label;
    scenario.price_paths = data.price_paths;
    scenario.market_paths = data.market_paths;
    scenario.n_paths = data.n_paths;
    scenario.n_steps = data.n_steps;

    return scenario;
end

"""
    build(type::Type{MyValidationReport}; strategy_label, criteria, actuals) -> MyValidationReport
"""
function build(type::Type{MyValidationReport};
    strategy_label::String, criteria::Dict{String,Float64},
    actuals::Dict{String,Float64})::MyValidationReport

    report = MyValidationReport();
    report.strategy_label = strategy_label;
    report.criteria = criteria;
    report.actuals = actuals;

    # evaluate pass/fail -
    report.passed = Dict{String,Bool}();
    for (key, threshold) in criteria
        if haskey(actuals, key)
            if startswith(key, "max_")
                report.passed[key] = actuals[key] <= threshold;
            else
                report.passed[key] = actuals[key] >= threshold;
            end
        else
            report.passed[key] = false;
        end
    end

    return report;
end

# --- Session 3: Bandit Builders -------------------------------------------------

"""
    build(type::Type{MyBanditContext}, data::NamedTuple) -> MyBanditContext
"""
function build(type::Type{MyBanditContext}, data::NamedTuple)::MyBanditContext

    ctx = MyBanditContext();
    ctx.tickers = data.tickers;
    ctx.sim_parameters = data.sim_parameters;
    ctx.prices = data.prices;
    ctx.B = data.B;
    ctx.gm_t = data.gm_t;
    ctx.lambda = data.lambda;
    ctx.epsilon = data.epsilon;

    return ctx;
end

"""
    build(type::Type{MyEpsilonGreedyBanditModel}, data::NamedTuple) -> MyEpsilonGreedyBanditModel
"""
function build(type::Type{MyEpsilonGreedyBanditModel}, data::NamedTuple)::MyEpsilonGreedyBanditModel

    model = MyEpsilonGreedyBanditModel();
    model.K = data.K;
    model.n_iterations = data.n_iterations;
    model.alpha = data.alpha;

    return model;
end

# --- Session 4: Production Builders ---------------------------------------------

"""
    build(type::Type{MySentimentSignal}, data::NamedTuple) -> MySentimentSignal
"""
function build(type::Type{MySentimentSignal}, data::NamedTuple)::MySentimentSignal

    signal = MySentimentSignal();
    signal.score = data.score;
    signal.source = data.source;
    signal.day = data.day;

    return signal;
end

"""
    build(type::Type{MyProductionContext}, data::NamedTuple) -> MyProductionContext
"""
function build(type::Type{MyProductionContext}, data::NamedTuple)::MyProductionContext

    ctx = MyProductionContext();
    ctx.tickers = data.tickers;
    ctx.sim_parameters = data.sim_parameters;
    ctx.B₀ = data.B₀;
    ctx.epsilon = data.epsilon;
    ctx.max_drawdown = data.max_drawdown;
    ctx.max_turnover = data.max_turnover;
    ctx.sentiment_threshold = data.sentiment_threshold;
    ctx.sentiment_override_lambda = data.sentiment_override_lambda;
    ctx.max_bandit_churn = data.max_bandit_churn;

    return ctx;
end

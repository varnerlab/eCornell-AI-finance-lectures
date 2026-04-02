# --- Utility Problem Builders ---------------------------------------------------

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

# --- Rebalancing Engine Builders ------------------------------------------------

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

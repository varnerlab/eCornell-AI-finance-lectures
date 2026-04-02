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

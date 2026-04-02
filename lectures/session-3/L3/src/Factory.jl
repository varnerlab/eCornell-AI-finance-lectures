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

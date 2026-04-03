"""
    load_price_data(path::String) -> DataFrame

Load a CSV file of historical price data and return as a DataFrame.
"""
function load_price_data(path::String)::DataFrame
    return CSV.read(path, DataFrame);
end

"""
    save_results(path::String, data::Dict)

Save results to a JLD2 file.
"""
function save_results(path::String, data::Dict)
    save(path, data);
end

"""
    load_results(path::String) -> Dict

Load results from a JLD2 file.
"""
function load_results(path::String)::Dict
    return load(path);
end

"""
    save_production_results(path::String, results, events)

Save production simulation results and escalation events to a JLD2 file.
"""
function save_production_results(path::String, results::Array{MyProductionDayResult,1},
    events::Array{MyEscalationEvent,1})
    save(path, Dict("results" => results, "events" => events));
end

"""
    load_production_results(path::String) -> (results, events)

Load production simulation results from a JLD2 file.
"""
function load_production_results(path::String)
    data = load(path);
    return (data["results"], data["events"]);
end

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

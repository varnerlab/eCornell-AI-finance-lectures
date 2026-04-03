# --- Private helpers ------------------------------------------------------------
function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

# --- Market Data Accessors ------------------------------------------------------

"""
    MyTrainingMarketDataSet() -> Dict{String, Any}

Load the S&P 500 Daily OHLC training dataset as a dictionary.
Data provided by [Polygon.io](https://polygon.io/) covering January 3, 2014 to December 31, 2024.

Each key is a ticker symbol; each value is a DataFrame with columns:
`timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `transactions`.
"""
MyTrainingMarketDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"));

"""
    MyTestingMarketDataSet() -> Dict{String, Any}

Load the S&P 500 Daily OHLC testing dataset as a dictionary.
Data provided by [Polygon.io](https://polygon.io/) covering January 2, 2025 to December 31, 2025
(updated periodically).

Each key is a ticker symbol; each value is a DataFrame with columns:
`timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `transactions`.
"""
MyTestingMarketDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2"));

# --- General File I/O -----------------------------------------------------------

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

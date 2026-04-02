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

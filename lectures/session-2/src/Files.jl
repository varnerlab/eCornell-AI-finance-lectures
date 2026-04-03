"""
    load_session1_data(path::String) -> Dict

Load the baseline portfolio and synthetic returns from Session 1.
"""
function load_session1_data(path::String)::Dict
    return load(path);
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

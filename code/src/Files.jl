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

"""
    MyMarketSurrogateModel() -> JumpHiddenMarkovModel

Load the pre-trained JumpHMM market surrogate model fitted on SPY (2014–2024).
Use `hmm_simulate(model, T; n_paths=N)` to generate synthetic market index paths of any length.

The model captures regime persistence, volatility clustering, and fat tails from 11 years
of daily S&P 500 data. Per-ticker paths can be generated via the Single Index Model (SIM).
"""
function MyMarketSurrogateModel()::JumpHiddenMarkovModel
    data = _jld2(joinpath(_PATH_TO_DATA, "pretrained-jumphmm-market-surrogate.jld2"));
    return data["model"];
end

"""
    MyPortfolioSurrogateModel() -> Dict{String, Any}

Load the pre-trained portfolio surrogate: marginal JumpHMM models for ~424 S&P 500 tickers
plus a Student-t copula for cross-asset dependence. Both fitted on daily data from 2014–2024.

### Returns
Dictionary with keys:
- `"tickers"` — sorted vector of ticker symbols
- `"marginals"` — Dict{String, JumpHiddenMarkovModel} of per-ticker models
- `"copula"` — StudentTCopula with fitted correlation matrix and ν
- `"n_tickers"` — number of fitted tickers
"""
MyPortfolioSurrogateModel() = _jld2(joinpath(_PATH_TO_DATA, "pretrained-portfolio-surrogate.jld2"));

"""
    MySyntheticTrainingDataSet() -> Dict{String, Any}

Load the frozen 20-year synthetic training dataset. Generated from the pre-trained
JumpHMM portfolio surrogate with a curated market path (realistic CAGR, multiple
drawdowns, jump clusters). Use this for SIM parameter estimation, covariance
computation, and baseline analysis across all sessions.

### Returns
Dictionary with keys:
- `"dataset"` — Dict{String, DataFrame} with 424 tickers + "MARKET" index, each with `timestamp` and `close` columns
- `"tickers"` — sorted vector of ticker symbols (excludes "MARKET")
- `"market_ticker"` — "MARKET"
- `"market_prices"` — market index price series
- `"market_returns"` — market excess growth rates
- `"market_jumps"` — Bool vector of jump indicators
- `"selected_candidate"` — Dict with CAGR, drawdown counts, kurtosis of the selected path
- `"n_days"`, `"n_years"` — 5040 days, 20 years
"""
MySyntheticTrainingDataSet() = _jld2(joinpath(_PATH_TO_DATA, "synthetic-training-dataset.jld2"));

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

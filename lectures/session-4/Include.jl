# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");
const _PATH_TO_DATA_S1 = joinpath(_ROOT, "..", "session-1", "data");

# ensure our own data directory exists for notebook writes -
isdir(_PATH_TO_DATA) || mkpath(_PATH_TO_DATA);

# check: do we need to install packages?
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    using Pkg;
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load the local package -
using eCornellAIFinance

# load session-specific external packages -
using Alpaca
using Colors
using DataFrames
using Dates
using JLD2
using Plots
using PrettyTables
using Random
using Statistics
using TOML
using StatsPlots

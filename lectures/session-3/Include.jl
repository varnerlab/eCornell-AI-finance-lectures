# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_DATA_S1 = joinpath(_ROOT, "..", "session-1", "data");
const _PATH_TO_DATA_S2 = joinpath(_ROOT, "..", "session-2", "data");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

# check: do we need to install packages?
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    using Pkg;
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load the local package -
using eCornellAIFinance

# load session-specific external packages -
using Colors
using HypothesisTests
using Plots
using PrettyTables
using Random
using Statistics
using StatsPlots

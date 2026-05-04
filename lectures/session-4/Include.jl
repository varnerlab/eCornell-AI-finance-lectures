# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");
const _PATH_TO_DATA_S1 = joinpath(_ROOT, "..", "session-1", "data");
const _PATH_TO_DATA_S2 = joinpath(_ROOT, "..", "session-2", "data");
const _PATH_TO_DATA_S3 = joinpath(_ROOT, "..", "session-3", "data");
const _PATH_TO_CONFIG = joinpath(_ROOT, "config");
const _PATH_TO_QUEUE = joinpath(_PATH_TO_DATA, "queue");
const _PATH_TO_DECISIONS = joinpath(_PATH_TO_DATA, "decisions");
const _PATH_TO_TICKETS = joinpath(_PATH_TO_DATA, "tickets");
const _PATH_TO_TAPE = joinpath(_PATH_TO_DATA, "intraday-tape");
const _PATH_TO_NEWS = joinpath(_PATH_TO_DATA, "news");
for d in (_PATH_TO_QUEUE, _PATH_TO_DECISIONS, _PATH_TO_TICKETS, _PATH_TO_TAPE, _PATH_TO_NEWS)
    isdir(d) || mkpath(d);
end

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
using JSON
using Plots
using PrettyTables
using Random
using Statistics
using TOML
using StatsPlots

# setup paths -
const _ROOT = pwd();
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

# check: do we need to install packages?
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    using Pkg;
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using CSV
using Colors
using DataFrames
using Distributions
using FileIO
using JLD2
using JuMP
using JumpHMM
using LinearAlgebra
using Plots
using PrettyTables
using Random
using Statistics
using StatsBase
using StatsPlots

# resolve name collision: JumpHMM.fit vs Distributions.fit -
const hmm_fit = JumpHMM.fit;
const hmm_tune = JumpHMM.tune;
const hmm_simulate = JumpHMM.simulate;
const hmm_validate = JumpHMM.validate;

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Files.jl"));

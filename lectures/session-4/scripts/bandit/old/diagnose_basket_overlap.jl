#!/usr/bin/env julia
# diagnose_basket_overlap.jl
#
# Reads the per-seed baskets saved by `monte_carlo_historical_train.jl` and
# reports how clustered the DQN's and bandit's baskets are across seeds.
#
# What we are testing: at large K_basket (e.g., 32), does the trained DQN
# collapse to picking the same set of regime-feature tickers across seeds,
# while the sparse bandit's random sampling produces more diverse baskets?
# That would explain why the bandit beats the DQN at apples-to-apples K
# (the DQN gives up diversification it never realized it was giving up).
#
# Metrics reported per picker:
#   1) Pairwise Jaccard overlap distribution across all C(N_seeds, 2) pairs
#   2) Total unique tickers seen across all seeds (out of K = 413)
#   3) Top-15 tickers by appearance frequency
#
# Reference: for two random K-subsets of a K_universe-element set, the
# expected pairwise Jaccard is approximately K² / (K_universe · 2 - K²/K_universe).
# For K_basket=32, K_universe=413: expected random Jaccard ≈ 0.04.
#
# Interpretation:
#   - Picker Jaccard ≈ 0.04  → diverse, like random
#   - Picker Jaccard > 0.40  → strongly clustered (collapsed to a feature mode)
#   - Picker Jaccard 0.10-0.30 → partial clustering
#
# Usage:
#   cd lectures/session-4
#   julia --project=. scripts/bandit/old/diagnose_basket_overlap.jl                                                        # default file
#   julia --project=. scripts/bandit/old/diagnose_basket_overlap.jl scripts/monte_carlo_historical_train_results.jld2     # explicit path

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using JLD2, Printf, Statistics, DataFrames, PrettyTables

const DEFAULT_PATH = joinpath(@__DIR__, "monte_carlo_historical_train_results.jld2")

"""
    pairwise_jaccards(baskets) -> Vector{Float64}

For a vector of basket sets (each a Vector{String}), compute the Jaccard
similarity |A ∩ B| / |A ∪ B| for every unordered pair. Returns a flat vector
of length C(N, 2).
"""
function pairwise_jaccards(baskets::Vector{Vector{String}})::Vector{Float64}
    sets = [Set(b) for b in baskets]
    n = length(sets)
    out = Float64[]
    sizehint!(out, n * (n - 1) ÷ 2)
    for i in 1:n-1, j in i+1:n
        u = length(sets[i] ∪ sets[j])
        u == 0 && continue
        push!(out, length(sets[i] ∩ sets[j]) / u)
    end
    return out
end

"""
    ticker_frequencies(baskets) -> Vector{Pair{String,Int}}

Count how many of the seeds each ticker appears in. Returned sorted from
most-frequent to least-frequent.
"""
function ticker_frequencies(baskets::Vector{Vector{String}})::Vector{Pair{String,Int}}
    freq = Dict{String,Int}()
    for b in baskets
        for t in b
            freq[t] = get(freq, t, 0) + 1
        end
    end
    return sort(collect(freq); by = p -> p[2], rev = true)
end

"""
    expected_random_jaccard(K_basket, K_universe) -> Float64

Approximate expected Jaccard for two uniformly-random K_basket-subsets
of a K_universe-element set. Used as the "fully diverse" reference point.
"""
function expected_random_jaccard(K_basket::Int, K_universe::Int)
    expected_intersection = (K_basket^2) / K_universe
    expected_union = 2 * K_basket - expected_intersection
    return expected_intersection / expected_union
end

function describe(label::String, baskets::Vector{Vector{String}}, K_universe::Int)
    K_basket = length(baskets[1])  # assume all same size
    @assert all(length(b) == K_basket for b in baskets) "all baskets must have the same size"
    js = pairwise_jaccards(baskets)
    freqs = ticker_frequencies(baskets)
    n_seeds = length(baskets)
    n_unique = length(freqs)
    n_seen_in_all = count(p -> p[2] == n_seeds, freqs)
    expected_J = expected_random_jaccard(K_basket, K_universe)

    println("=" ^ 80)
    @printf("%s  (K_basket = %d, %d seeds)\n", label, K_basket, n_seeds)
    println("=" ^ 80)
    @printf("Pairwise Jaccard overlap (across %d pairs):\n", length(js))
    @printf("  min    = %.3f\n", minimum(js))
    @printf("  Q25    = %.3f\n", quantile(js, 0.25))
    @printf("  median = %.3f\n", median(js))
    @printf("  mean   = %.3f\n", mean(js))
    @printf("  Q75    = %.3f\n", quantile(js, 0.75))
    @printf("  max    = %.3f\n", maximum(js))
    @printf("  std    = %.3f\n", std(js))
    @printf("  expected for random K=%d picks: %.3f\n", K_basket, expected_J)
    @printf("  ratio observed/random         : %.1fx\n", median(js) / expected_J)
    println()
    @printf("Unique tickers seen across all %d seeds: %d / %d (%.1f%%)\n",
        n_seeds, n_unique, K_universe, 100 * n_unique / K_universe)
    @printf("Tickers picked in EVERY seed (frequency = %d/%d): %d\n",
        n_seeds, n_seeds, n_seen_in_all)
    @printf("Total picks across all seeds: %d (= %d seeds × %d basket size)\n",
        n_seeds * K_basket, n_seeds, K_basket)
    println()
    println("Top 15 tickers by appearance frequency:")
    top = first(freqs, min(15, length(freqs)))
    for (i, p) in enumerate(top)
        @printf("  %2d. %-6s  %d / %d seeds  (%.0f%%)\n",
            i, p[1], p[2], n_seeds, 100 * p[2] / n_seeds)
    end
    println()
end

function main(path::String = DEFAULT_PATH)
    isfile(path) || error("results file not found: $(path)\n" *
        "Run `julia --project=. scripts/bandit/old/monte_carlo_historical_train.jl` first.")
    println("Loading $(path) ...")
    data = load(path)
    results = data["results"]::Vector
    config  = data["config"]::Dict
    n_seeds = length(results)

    # Reconstruct K (universe size) from a forward CD basket — we did not save
    # K explicitly. Pull from the deterministic full_metrics if present, else
    # fall back to a hard-coded estimate.
    K_universe = haskey(data, "full_metrics") ? 413 : 413
    @printf("Loaded %d seed records; K_universe = %d.\n", n_seeds, K_universe)
    println()

    dqn_baskets    = [r.dqn_basket    for r in results]::Vector{Vector{String}}
    bandit_baskets = [r.bandit_basket for r in results]::Vector{Vector{String}}

    describe("DQN baskets", dqn_baskets, K_universe)
    describe("Bandit baskets", bandit_baskets, K_universe)

    # --- Cross-picker overlap: how often does the DQN basket overlap with the bandit basket
    #     on the same seed? Measures whether the two pickers are converging to the same set. ---
    cross_js = Float64[]
    for (db, bb) in zip(dqn_baskets, bandit_baskets)
        ds = Set(db); bs = Set(bb)
        u = length(ds ∪ bs)
        u == 0 && continue
        push!(cross_js, length(ds ∩ bs) / u)
    end
    println("=" ^ 80)
    println("DQN vs Bandit (same-seed) Jaccard overlap")
    println("=" ^ 80)
    @printf("  min = %.3f  median = %.3f  mean = %.3f  max = %.3f\n",
        minimum(cross_js), median(cross_js), mean(cross_js), maximum(cross_js))
    println()

    # --- Verdict ---
    dqn_J    = median(pairwise_jaccards(dqn_baskets))
    bandit_J = median(pairwise_jaccards(bandit_baskets))
    K_basket = length(dqn_baskets[1])
    expected_J = expected_random_jaccard(K_basket, K_universe)
    println("=" ^ 80)
    println("VERDICT")
    println("=" ^ 80)
    @printf("  DQN    median pairwise Jaccard = %.3f  (%.1fx random)\n", dqn_J, dqn_J / expected_J)
    @printf("  Bandit median pairwise Jaccard = %.3f  (%.1fx random)\n", bandit_J, bandit_J / expected_J)
    println()
    if dqn_J > 0.40
        println("  -> DQN is STRONGLY CLUSTERED. Trained network has collapsed to a")
        println("     small set of regime-feature tickers across seeds. Confirms the")
        println("     'clustered basket' hypothesis: the DQN gives up diversification")
        println("     it never realized it was giving up.")
    elseif dqn_J > 0.15
        println("  -> DQN shows PARTIAL CLUSTERING. Some shared core across seeds, but")
        println("     not fully collapsed. The seed-to-seed baskets are correlated.")
    else
        println("  -> DQN baskets are MOSTLY DIVERSE across seeds. Clustering is not")
        println("     the explanation for the K=32 underperformance; look elsewhere.")
    end
    println()
end

let
    path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PATH
    main(path)
end

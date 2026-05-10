#!/usr/bin/env julia
#
# compare_daily_vs_intraday.jl -- Side-by-side of the live 30-min engine
# tape and the offline daily-baseline tape over May 5-8 2026.
#
# Reads:
#   data/intraday-tape/tape-2026-05-{05,06,07,08}.jld2
#   data/daily-baseline-tape.jld2
#
# Writes:
#   data/comparison-daily-vs-intraday.png
#
# Run from lectures/session-4:
#   julia --project=. scripts/compare_daily_vs_intraday.jl
#

using Pkg;
const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance, DataFrames, Dates, JLD2, Statistics, Plots, PrettyTables

const TAPE_DIR = joinpath(SESSION_DIR, "data", "intraday-tape");
const DAILY_PATH = joinpath(SESSION_DIR, "data", "daily-baseline-tape.jld2");
const OUT_PNG = joinpath(SESSION_DIR, "data", "comparison-daily-vs-intraday.png");

function load_30min_tape()
    files = sort(filter(x -> startswith(x, "tape-2026-05-0") && !occursin("fixture", x),
                        readdir(TAPE_DIR)));
    times = DateTime[];
    wealth = Float64[];
    auto_counts = Int[];
    for f in files
        d = load(joinpath(TAPE_DIR, f));
        for r in d["entries"]
            push!(times,        r.fire_time);
            push!(wealth,       r.wealth);
            push!(auto_counts,  r.auto_n);
        end
    end
    return (times = times, wealth = wealth, auto_counts = auto_counts);
end

function load_daily_tape()
    d = load(DAILY_PATH);
    e = d["entries"];
    return (
        times  = [r.fire_time for r in e],
        wealth = [r.wealth    for r in e],
    );
end

function summarize(label::String, wealth::Vector{Float64})
    w0 = wealth[1];
    wT = wealth[end];
    pk = maximum(wealth);
    max_dd = 0.0;
    running_peak = wealth[1];
    for w in wealth
        running_peak = max(running_peak, w);
        dd = running_peak > 0 ? (running_peak - w) / running_peak : 0.0;
        max_dd = max(max_dd, dd);
    end
    return (
        label = label,
        n_points = length(wealth),
        start_wealth = w0,
        terminal_wealth = wT,
        peak_wealth = pk,
        total_return_pct = (wT / w0 - 1.0) * 100,
        max_drawdown_pct = max_dd * 100,
    );
end

function main()
    live = load_30min_tape();
    daily = load_daily_tape();

    live_summary  = summarize("30-min engine (live)", live.wealth);
    daily_summary = summarize("Daily baseline (offline)", daily.wealth);

    # Normalize to W/W₀ off each engine's own bootstrap point so the comparison
    # is in return space, not absolute dollars (the two engines bootstrap at
    # slightly different points: live is post-S1-deploy as of May 5 first fire;
    # daily baseline is post-S1-deploy as of May 4 close).
    live_w_norm  = live.wealth  ./ live.wealth[1];
    daily_w_norm = daily.wealth ./ daily.wealth[1];

    plt = plot(size = (900, 500), dpi = 150,
        title = "Engine cadence comparison: 30-min vs daily (May 5-8 2026)",
        xlabel = "fire time", ylabel = "W / W₀  (return space)",
        legend = :topleft, framestyle = :box, grid = true);
    plot!(plt, live.times,  live_w_norm;
        label = "30-min engine (50 fires)",
        linewidth = 2, color = :steelblue);
    plot!(plt, daily.times, daily_w_norm;
        label = "Daily baseline (3 trades)",
        linewidth = 2, color = :firebrick,
        seriestype = :steppost,
        marker = :circle, markersize = 5, markerstrokecolor = :firebrick, markercolor = :white);
    hline!(plt, [1.0]; label = "", color = :black, linestyle = :dash, alpha = 0.4);
    savefig(plt, OUT_PNG);
    println("Saved chart → $(OUT_PNG)");

    df = DataFrame(
        Metric = [
            "Start (W₀)",
            "Terminal",
            "Peak",
            "Total return %",
            "Max drawdown %",
            "Tape points",
            "Auto-cleared trades (sum)",
        ],
        var"30-min engine" = [
            "\$$(round(live_summary.start_wealth, digits=2))",
            "\$$(round(live_summary.terminal_wealth, digits=2))",
            "\$$(round(live_summary.peak_wealth, digits=2))",
            "$(round(live_summary.total_return_pct, digits=3))%",
            "$(round(live_summary.max_drawdown_pct, digits=3))%",
            string(live_summary.n_points),
            string(sum(live.auto_counts)),
        ],
        var"Daily baseline" = [
            "\$$(round(daily_summary.start_wealth, digits=2))",
            "\$$(round(daily_summary.terminal_wealth, digits=2))",
            "\$$(round(daily_summary.peak_wealth, digits=2))",
            "$(round(daily_summary.total_return_pct, digits=3))%",
            "$(round(daily_summary.max_drawdown_pct, digits=3))%",
            string(daily_summary.n_points),
            "n/a",
        ],
    );
    pretty_table(df; backend = :text,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        table_format = TextTableFormat(borders = text_table_borders__compact));

    println();
    println("Notes:");
    println("  * 30-min engine wealth is real Alpaca paper-fill mark-to-market");
    println("    (whatever spread the broker actually paid).");
    println("  * Daily baseline charges 5 bps per trade dollar at the next-day open.");
    println("  * Daily baseline has 5 wealth points: bootstrap May 4 close, hold");
    println("    May 5 close, then 3 daily-engine trades on May 6/7/8 opens.");
    println("  * Daily baseline EMAs use canonical (5, 21) trading days; live");
    println("    engine uses (21, 63) bars (the production-config values).");
end

main()

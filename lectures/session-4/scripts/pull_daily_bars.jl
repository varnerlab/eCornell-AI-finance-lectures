#!/usr/bin/env julia
#
# pull_daily_bars.jl -- One-shot pull of daily OHLC bars for the daily-cadence
# baseline simulator (scripts/daily_baseline.jl).
#
# Pulls open + close for the 22 S1 minvar tickers + SPY from 2026-01-01 to
# 2026-05-09 and writes a single JLD2 cache to data/daily-baseline-bars.jld2.
# Daily cadence simulator reads the cache; no further network calls.
#
# Run from lectures/session-4:
#   julia --project=. scripts/pull_daily_bars.jl
#

using Pkg;
const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance, Alpaca, Dates, JLD2

const CREDS_PATH = joinpath(SESSION_DIR, "config", "credentials.toml");
const S1_ALLOCATION_PATH = joinpath(SESSION_DIR, "..", "session-1", "data", "minvar-allocation.jld2");
const OUT_PATH = joinpath(SESSION_DIR, "data", "daily-baseline-bars.jld2");

const START_DT = DateTime(2026, 1, 1, 0, 0, 0);
const FINISH_DT = DateTime(2026, 5, 9, 0, 0, 0);

function main()
    s1 = load_results(S1_ALLOCATION_PATH);
    tickers = String.(s1["my_tickers"]);
    symbols = vcat(tickers, ["SPY"]);
    println("Tickers ($(length(symbols))): ", join(symbols, ", "));

    client = Alpaca.load_client(CREDS_PATH);
    println("Pulling 1Day bars $(START_DT) → $(FINISH_DT) ...");
    bars = Alpaca.get_bars(client, symbols, "1Day"; start = START_DT, finish = FINISH_DT);

    # Build a unified date axis from SPY (the most reliable trading-day index).
    haskey(bars, "SPY") && !isempty(bars["SPY"]) || error("No SPY bars returned.");
    spy_dates = [Date(DateTime(b.t)) for b in bars["SPY"]];
    n_days = length(spy_dates);
    println("SPY trading days: $(n_days) ($(spy_dates[1]) → $(spy_dates[end]))");

    # For each symbol, align open/close to the SPY trading-day axis. Missing
    # days get NaN; the simulator must skip / interpolate.
    K = length(symbols);
    open_matrix = fill(NaN, n_days, K);
    close_matrix = fill(NaN, n_days, K);
    for (k, sym) in enumerate(symbols)
        arr = get(bars, sym, nothing);
        if arr === nothing || isempty(arr)
            println("  $(sym): NO BARS");
            continue;
        end
        by_date = Dict(Date(DateTime(b.t)) => b for b in arr);
        n_have = 0;
        for (i, d) in enumerate(spy_dates)
            b = get(by_date, d, nothing);
            if b !== nothing
                open_matrix[i, k] = Float64(b.o);
                close_matrix[i, k] = Float64(b.c);
                n_have += 1;
            end
        end
        println("  $(rpad(sym, 5))  $(n_have)/$(n_days) days  first close=\$$(round(close_matrix[findfirst(!isnan, close_matrix[:, k]), k], digits=2))  last close=\$$(round(close_matrix[findlast(!isnan, close_matrix[:, k]), k], digits=2))");
    end

    mkpath(dirname(OUT_PATH));
    save_results(OUT_PATH, Dict(
        "dates" => spy_dates,
        "symbols" => symbols,
        "tickers" => tickers,             # 22 names, ex-SPY
        "open" => open_matrix,            # (n_days, K) where K = length(symbols)
        "close" => close_matrix,
        "start_dt" => START_DT,
        "finish_dt" => FINISH_DT,
        "pulled_at" => string(now()),
    ));
    println("\nWrote $(OUT_PATH)  size=$(round(filesize(OUT_PATH)/1024, digits=1)) KB");
end

main()

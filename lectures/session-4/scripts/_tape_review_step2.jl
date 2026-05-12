cd(joinpath(@__DIR__, ".."))
include(joinpath(pwd(), "Include.jl"))

review_str = "2026-05-11"
QUEUE_PATH = joinpath("data", "queue", "queue-$(review_str).jld2")

queue = load_queue(QUEUE_PATH)
println("Queued trades today: $(length(queue))")
println()

println("=== First 10 items (full) ===")
for (i, q) in enumerate(queue[1:min(10, length(queue))])
    println()
    println("[$i] $(q.id)")
    println("    $(q.timestamp)  $(q.side)  $(q.qty) $(q.ticker)")
    println("    Proposed weight  : $(round(100 * q.proposed_weight, digits = 1))%")
    println("    Gate violations  : $(join(string.(q.gate_violations), ", "))")
    println("    Lambda           : $(round(get(q.engine_snapshot, "lambda_eff", NaN), digits = 3))")
    println("    Regime           : $(get(q.engine_snapshot, "regime", "?"))")
    println("    Sentiment        : $(round(get(q.engine_snapshot, "sentiment", NaN), digits = 3))")
    println("    Drawdown         : $(round(100 * get(q.engine_snapshot, "drawdown", NaN), digits = 1))%")
    println("    Portfolio wealth : \$$(round(get(q.engine_snapshot, "wealth", NaN), digits = 0))")
end

println()
println("=== Aggregate rollup (all $(length(queue)) items) ===")

# violation counts
violation_counts = Dict{String,Int}()
for q in queue
    for v in q.gate_violations
        s = string(v)
        violation_counts[s] = get(violation_counts, s, 0) + 1
    end
end
println()
println("Gate violations (count, may double-count if an item has multiple):")
for (k, v) in sort(collect(violation_counts), by = x -> -x[2])
    println("  $(rpad(k, 28)) $v")
end

# ticker counts
ticker_counts = Dict{String,Int}()
ticker_sides  = Dict{String,Dict{String,Int}}()
for q in queue
    t = string(q.ticker)
    ticker_counts[t] = get(ticker_counts, t, 0) + 1
    sd = get!(ticker_sides, t, Dict("buy" => 0, "sell" => 0))
    sd[string(q.side)] = get(sd, string(q.side), 0) + 1
end
println()
println("Tickers (count, buy/sell):")
for (k, v) in sort(collect(ticker_counts), by = x -> -x[2])
    sd = ticker_sides[k]
    println("  $(rpad(k, 8)) total=$(lpad(v, 3))   buy=$(lpad(get(sd,"buy",0), 3))  sell=$(lpad(get(sd,"sell",0), 3))")
end

# timestamps
ts = [q.timestamp for q in queue]
println()
println("First flag        : $(minimum(ts))")
println("Last flag         : $(maximum(ts))")

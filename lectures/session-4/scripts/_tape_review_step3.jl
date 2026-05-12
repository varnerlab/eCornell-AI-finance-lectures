cd(joinpath(@__DIR__, ".."))
include(joinpath(pwd(), "Include.jl"))

review_date = Date("2026-05-11")
exec_date   = review_date + Day(1)
exec_str    = Dates.format(exec_date, "yyyy-mm-dd")
TICKET_PATH = joinpath("data", "tickets", "ticket-$(exec_str).jld2")
println("Loading ticket for execution date $exec_str (cron stores tomorrow's ticket under tomorrow's date).")
println()

ticket = load_ticket(TICKET_PATH)

println("Ticket generated at  : $(ticket.generated_at)")
println("Regime at generation : $(ticket.regime)")
println("News flags           : $(isempty(ticket.news_flags) ? "(none)" : join(ticket.news_flags, ", "))")
println("Sentiment score      : $(round(ticket.sentiment.score, digits = 3))  (source=$(ticket.sentiment.source))")
println("Proposed trades      : $(length(ticket.proposed_trades))")
println()

# Target weights side-by-side with tickers
println("Target weights (Cobb-Douglas, γ/Σγ):")
tw_pairs = collect(zip(ticket.tickers, ticket.target_weights))
sort!(tw_pairs, by = x -> -x[2])
for (t, w) in tw_pairs
    println("  $(rpad(string(t), 8)) $(rpad(string(round(100*w, digits=2))*"%", 8))")
end

println()
println("Proposed trades:")
for t in ticket.proposed_trades
    println("  $(rpad(string(t.side), 4))  $(lpad(string(t.qty), 4))  $(t.ticker)")
end

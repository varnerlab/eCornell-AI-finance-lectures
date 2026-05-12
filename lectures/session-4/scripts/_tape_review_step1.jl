cd(joinpath(@__DIR__, ".."))
include(joinpath(pwd(), "Include.jl"))

review_date = Date("2026-05-11")
review_str  = Dates.format(review_date, "yyyy-mm-dd")
println("Reviewing $review_str.")

TAPE_PATH   = joinpath("data", "intraday-tape", "tape-$(review_str).jld2")
QUEUE_PATH  = joinpath("data", "queue", "queue-$(review_str).jld2")
TICKET_PATH = joinpath("data", "tickets", "ticket-$(review_str).jld2")
SIGNED_PATH = joinpath("data", "tickets", "signed-$(review_str).jld2")

tape = load_results(TAPE_PATH)["entries"]

n_fires      = length(tape)
W_open       = tape[1].wealth
W_close      = tape[end].wealth
ret_log      = log(W_close / W_open)
max_dd       = maximum(e.drawdown for e in tape)
total_auto   = sum(e.auto_n for e in tape)
total_queue  = sum(e.queued_n for e in tape)
regimes_seen = unique(string(e.regime) for e in tape)

println("Fires today           : $n_fires")
println("Open wealth           : \$$(round(W_open, digits = 2))")
println("Close wealth          : \$$(round(W_close, digits = 2))")
println("Daily log return      : $(round(100 * ret_log, digits = 2))%")
println("Max intraday drawdown : $(round(100 * max_dd, digits = 1))%")
println("Auto-executed trades  : $total_auto")
println("Routed to queue       : $total_queue")
println("Regimes seen today    : $(join(regimes_seen, ", "))")

println()
df = DataFrame((
    Time      = Dates.format(e.fire_time, "HH:MM"),
    Sentiment = round(e.sentiment, digits = 3),
    Lambda    = round(e.lambda_eff, digits = 3),
    Regime    = string(e.regime),
    Wealth    = round(e.wealth, digits = 0),
    DD_pct    = round(100 * e.drawdown, digits = 1),
    Auto      = e.auto_n,
    Queue     = e.queued_n,
) for e in tape)

pretty_table(df; backend = :text,
    fit_table_in_display_horizontally = false,
    fit_table_in_display_vertically = false,
    table_format = TextTableFormat(borders = text_table_borders__compact))

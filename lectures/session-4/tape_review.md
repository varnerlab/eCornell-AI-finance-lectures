# Tape Review and Ticket Sign-Off

End-of-day workflow run after the 4pm close. The runbook below loads today's tape (what the engine did), today's queue (what got flagged for audit), and tomorrow's ticket (the engine's proposed open allocation), then writes a signed ticket back to disk for the 9:35am cron to consume.

The cron writes three artifacts during the trading day, each keyed by today's date:

* `data/intraday-tape/tape-YYYY-MM-DD.jld2`: one tape entry per 30-minute fire.
* `data/queue/queue-YYYY-MM-DD.jld2`: a vector of [`MyComplianceQueueItem`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MyComplianceQueueItem), one per intraday trade that failed at least one gate.
* `data/tickets/ticket-YYYY-MM-DD.jld2`: the [`MyTomorrowsTicket`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MyTomorrowsTicket) record written by the 16:00 close fire with tomorrow's proposed open allocation.

The review writes one artifact at the end:

* `data/tickets/signed-YYYY-MM-DD.jld2`: a [`MySignedTicket`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MySignedTicket) wrapping tomorrow's ticket plus any per-ticker modifications. The `production_runner.jl --mode=execute_signed_ticket` cron at 9:35am ET loads this file and submits the trades.

We work each section in order. Copy the code blocks into a Julia REPL with the session-4 environment activated.

---

## Setup

```julia
include("Include.jl")

print("Date to review [yyyy-mm-dd] (default: today): ")
input = strip(readline())
review_date = isempty(input) ? today() : Date(input)
review_str  = Dates.format(review_date, "yyyy-mm-dd")
println("Reviewing $review_str.")

TAPE_PATH   = joinpath("data", "intraday-tape", "tape-$(review_str).jld2")
QUEUE_PATH  = joinpath("data", "queue", "queue-$(review_str).jld2")
TICKET_PATH = joinpath("data", "tickets", "ticket-$(review_str).jld2")
SIGNED_PATH = joinpath("data", "tickets", "signed-$(review_str).jld2")
```

---

## Step 1: Day-Level Tape Review

Load today's tape and read the day-level metrics. Each tape entry is a NamedTuple with `fire_time`, `sentiment`, `lambda_eff`, `regime`, `wealth`, `drawdown`, `auto_n`, `queued_n`, `news_path`, plus diagnostic fields (`eta`, `eta_heuristic`, `eta_policy`, `realized_vol`, `target_weights`, `submitted_ids`).

```julia
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
println("Regimes seen today    : $(join(regimes_seen, \", \"))")
```

Render a fire-by-fire summary table so we can see the day's trajectory at a glance.

```julia
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
```

> __Day-level review checklist:__
>
> Fill these in based on what the table shows:
>
> * Daily log return: ___% (target band: ___ to ___)
> * Max drawdown: ___% (drawdown gate: ≤ 15%)
> * Regime trajectory: ___ (stable / shifted / volatile)
> * Auto-executed count: ___ (typical band: ___ to ___)
> * Queue length: ___ (typical: 1 to 5 per day)
> * Anything anomalous in the tape (drawdown excursions, regime flips, auto/queue split): ___

---

## Step 2: Walk the Queue (Audit)

Load today's queue and render each item with the engine snapshot at the time it was flagged. This step is audit only; the queue does not feed the next-day cron. We walk it so every gate-flagged intraday trade is on record for review.

```julia
queue = load_queue(QUEUE_PATH)
println("Queued trades today: $(length(queue))")

for (i, q) in enumerate(queue)
    println()
    println("[$i] $(q.id)")
    println("    $(q.timestamp)  $(q.side)  $(q.qty) $(q.ticker)")
    println("    Proposed weight  : $(round(100 * q.proposed_weight, digits = 1))%")
    println("    Gate violations  : $(join(string.(q.gate_violations), \", \"))")
    println("    Lambda           : $(round(get(q.engine_snapshot, \"lambda_eff\", NaN), digits = 3))")
    println("    Regime           : $(get(q.engine_snapshot, \"regime\", \"?\"))")
    println("    Sentiment        : $(round(get(q.engine_snapshot, \"sentiment\", NaN), digits = 3))")
    println("    Drawdown         : $(round(100 * get(q.engine_snapshot, \"drawdown\", NaN), digits = 1))%")
    println("    Portfolio wealth : \$$(round(get(q.engine_snapshot, \"wealth\", NaN), digits = 0))")
end
```

Note anything anomalous from this walk in the day-level checklist above; the items themselves do not require a signed decision.

---

## Step 3: Load Tomorrow's Ticket

Load the [`MyTomorrowsTicket`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MyTomorrowsTicket) the 16:00 close fire wrote and render the proposed open allocation. The fields are `target_weights`, `tickers`, `proposed_trades`, `sentiment`, `news_flags`, `generated_at`, `eta_used`, and `regime`. The live allocator is Cobb-Douglas (`target_weights = γ / Σγ`); `eta_used` is shadow-logged for the future validation tournament and is not consumed by the allocator, so we skip it in the review display.

```julia
ticket = load_ticket(TICKET_PATH)

println("Ticket generated at  : $(ticket.generated_at)")
println("Regime at generation : $(ticket.regime)")
println("News flags           : $(isempty(ticket.news_flags) ? "(none)" : join(ticket.news_flags, \", \"))")
println("Proposed trades      : $(length(ticket.proposed_trades))")

for t in ticket.proposed_trades
    println("  $(t.side)  $(t.qty)  $(t.ticker)")
end
```

---

## Step 4: Apply Modifications

For each proposed trade we want to override, build a NamedTuple `(ticker, original_qty, modified_qty, reason)`:

* __Approve as-is__: leave the modifications vector empty; the cron submits each proposed trade at its original quantity.
* __Reject a trade__: set `modified_qty = nothing`. The cron will skip the ticker.
* __Modify a quantity__: set `modified_qty` to the new share count. The cron will submit at this size.

Edit the vector below. One entry per modification (no entry needed for trades signed as-is).

```julia
modifications = NamedTuple[
    # (ticker = "AAPL", original_qty = 12, modified_qty = 6,       reason = "Trim to half size given today's queue activity."),
    # (ticker = "TSLA", original_qty = 4,  modified_qty = nothing, reason = "Reject; news severity still elevated at close."),
]
```

---

## Step 5: Sign and Persist

Build the [`MySignedTicket`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MySignedTicket) and persist it so the 9:35am cron can read it.

```julia
signed_by = "REPLACE_WITH_YOUR_NAME"
signed_at = now()

signed = build(MySignedTicket, (
    ticket        = ticket,
    modifications = modifications,
    signed_by     = signed_by,
    signed_at     = signed_at,
))

mkpath(dirname(SIGNED_PATH))
save_signed_ticket!(SIGNED_PATH, signed)
println("Wrote signed ticket with $(length(modifications)) modifications to $SIGNED_PATH.")
```

The 9:35am cron (`production_runner.jl --mode=execute_signed_ticket`) reads `data/tickets/signed-YYYY-MM-DD.jld2`, applies each modification (reject or modified quantity) per ticker, and submits the resulting orders to Alpaca paper. Reviewed, signed, committed.

---

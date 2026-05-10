# Tape Review and Queue Adjudication

End-of-day workflow for the desk PM. Open this file after market close. The runbook below loads today's tape (what the engine did) and today's queue (what it flagged for review), then records signed adjudications back to disk for tomorrow morning's cron to consume.

The cron writes three artifacts during the trading day, each keyed by today's date:

* `data/intraday-tape/tape-YYYY-MM-DD.jld2` — one tape entry per 30-minute fire.
* `data/queue/queue-YYYY-MM-DD.jld2` — a vector of [`MyComplianceQueueItem`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MyComplianceQueueItem), one per trade that failed at least one gate.
* `data/decisions/decisions-YYYY-MM-DD.jld2` — a vector of [`MySignedDecision`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MySignedDecision) records that we write at the end of this review and the next-morning cron consumes.

We work each section in order. Copy the code blocks into a Julia REPL with the session-4 environment activated.

---

## Setup

```julia
include("Include.jl")

today_date = today()
today_str  = Dates.format(today_date, "yyyy-mm-dd")

TAPE_PATH      = joinpath("data", "intraday-tape", "tape-$(today_str).jld2")
QUEUE_PATH     = joinpath("data", "queue", "queue-$(today_str).jld2")
DECISIONS_PATH = joinpath("data", "decisions", "decisions-$(today_str).jld2")
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
> * Queue length: ___ (typical: 1–5 per day)
> * Anything anomalous in the tape (drawdown excursions, regime flips, auto/queue split): ___

If the queue is empty, the desk is done. Otherwise proceed to Step 2.

---

## Step 2: Walk the Queue

Load today's queue and render each item with the engine snapshot at the time it was flagged.

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

---

## Step 3: Adjudicate Each Queued Trade

For every queue item, decide approve / reject / modify and record a one-sentence rationale. Default heuristics:

* __Approve__ when the only violation is `:turnover_limit` and the regime is stable.
* __Reject__ when the only violation is `:news_severity` and we do not yet trust the news direction for that ticker.
* __Modify__ when the violation is `:concentration_cap` or `:position_size_limit` and the desired direction is fine but the size needs to come down.

Edit the vector below: one entry per queue item, in the same order Step 2 printed. Copy each `id` from the queue listing into `queue_id`.

```julia
decisions_raw = [
    # Replicate one entry per queued trade.
    (queue_id     = "REPLACE_WITH_QUEUE_ID",
     action       = :approve,        # :approve | :reject | :modify
     modified_qty = nothing,         # set to an Int when action == :modify, else nothing
     notes        = "Rationale here."),
]
```

---

## Step 4: Sign and Persist

Build the [`MySignedDecision`](https://varnerlab.org/eCornell-AI-finance-lectures/dev/session4/#MySignedDecision) records and persist them so tomorrow morning's cron can read them.

```julia
signed_by = "REPLACE_WITH_YOUR_NAME"
signed_at = now()

decisions = [build(MySignedDecision, (
    queue_id     = d.queue_id,
    action       = d.action,
    modified_qty = d.modified_qty,
    notes        = d.notes,
    signed_by    = signed_by,
    signed_at    = signed_at,
)) for d in decisions_raw]

mkpath(dirname(DECISIONS_PATH))
save_signed_decisions!(DECISIONS_PATH, decisions)
println("Wrote $(length(decisions)) signed decisions to $DECISIONS_PATH.")
```

The next-morning 9:35am cron reads `data/decisions/decisions-YYYY-MM-DD.jld2`, applies the action (approve / reject / modify) per queue item, and submits the resulting orders to Alpaca paper. Reviewed, signed, committed.

---

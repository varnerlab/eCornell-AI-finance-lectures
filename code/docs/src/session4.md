# Session 4: Production Operations

Production portfolio system with sentiment monitoring, escalation triggers, and operational dashboards.

## Types

### Production

```@docs
MySentimentSignal
MyEscalationEvent
MyProductionDayResult
MyProductionContext
MyLiveProductionDayResult
MyStressScenario
MyStressResult
```

### News Sentiment

```@docs
MyNewsScenario
MyNewsItem
MyNewsCorpus
```

### Compliance and Tickets

```@docs
MyTomorrowsTicket
MySignedTicket
MySignedDecision
MyComplianceQueueItem
```

## Factory Methods

```@docs
build(::Type{MySentimentSignal}, ::NamedTuple)
build(::Type{MyProductionContext}, ::NamedTuple)
build(::Type{MyNewsScenario}, ::NamedTuple)
build(::Type{MyNewsItem}, ::NamedTuple)
build(::Type{MyNewsCorpus}, ::NamedTuple)
build(::Type{MyTomorrowsTicket}, ::NamedTuple)
build(::Type{MySignedTicket}, ::NamedTuple)
build(::Type{MySignedDecision}, ::NamedTuple)
build(::Type{MyComplianceQueueItem}, ::NamedTuple)
```

## Functions

### Production Operations

```@docs
generate_synthetic_sentiment
check_escalation_triggers
run_production_simulation
compute_dashboard_metrics
compute_live_sentiment
compute_position_drawdown
run_production_step
apply_stress_scenario
```

### News Ingestion and Scoring

```@docs
simulate_news_corpus
generate_news_text!
score_news_with_claude!
aggregate_news_factor
estimate_sim_with_news
eCornellAIFinance._call_claude
eCornellAIFinance._sentiment_bucket
```

### Compliance Gate and Ticket Workflow

```@docs
gate_check
build_tomorrows_ticket
split_intraday_trades
```

### Intraday Helpers

```@docs
bars_per_day
intraday_dt
intraday_half_life
```

## File I/O

```@docs
save_production_results
load_production_results
save_ticket!
load_ticket
save_signed_ticket!
load_signed_ticket
save_signed_decisions!
load_signed_decisions
save_queue!
load_queue
append_queue_item!
```

# Generate the synthetic news corpus the S4 cron's news_scorer.jl expects.
#
# Writes lectures/session-4/data/news/synthetic-corpus-2026-05.jld2 with two
# keys: "corpus" (a MyNewsCorpus) and "date_to_day" (Dict{String,Int} mapping
# "yyyy-mm-dd" -> trading-day index used by simulate_news_corpus).
#
# Run from repo root:
#   julia --project=code code/scripts/generate-synthetic-news.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "code"))

using eCornellAIFinance
using Dates
using Random
using JLD2

const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", ".."))
const SESSION4   = joinpath(REPO_ROOT, "lectures", "session-4")
const NEWS_DIR   = joinpath(SESSION4, "data", "news")
const OUT_PATH   = joinpath(NEWS_DIR, "synthetic-corpus-2026-05.jld2")
const PROD_CFG   = joinpath(SESSION4, "config", "production-config.toml")

# Universe: read manual fallback from production-config so the corpus stays in
# sync with the runner's universe list.
import TOML
const PROD = TOML.parsefile(PROD_CFG)
const TICKERS = String.(PROD["Tickers"]["universe"])

# Trading-day calendar covering the cron window plus a small buffer either side.
# Today is 2026-05-04 (Mon); cron installs May 5; class is May 11; orphan
# follow-through fills can land May 12. Include May 4-15 weekday range.
function trading_days(start::Date, finish::Date)::Vector{Date}
    days = Date[]
    d = start
    while d <= finish
        if Dates.dayofweek(d) <= 5  # Mon..Fri
            push!(days, d)
        end
        d += Day(1)
    end
    return days
end

const DAYS = trading_days(Date(2026, 5, 4), Date(2026, 5, 15))
const T = length(DAYS)
const K = length(TICKERS)

# Placeholder price matrix. simulate_news_corpus needs prices to compute
# shocked_prices, but the cron's news_scorer never reads shocked_prices --
# it only uses corpus.items and the date_to_day map. Use 100.0 flat so the
# scenario shock is the only price movement, and so the file stays small.
prices = fill(100.0, T, K)

scenario = build(MyNewsScenario, (
    label             = "intraday_cron_synthetic",
    kappa_pos         = 0.005,    # 50 bp upside per unit |s|
    kappa_neg         = 0.010,    # 100 bp downside per unit |s|
    arrival_intensity = 1.2,      # ~1.2 items per ticker per day
    sentiment_mean    = 0.0,
    sentiment_sd      = 0.40,
))

corpus = simulate_news_corpus(prices, TICKERS, scenario; seed = 20260504)

date_to_day = Dict{String,Int}(
    Dates.format(d, "yyyy-mm-dd") => i for (i, d) in enumerate(DAYS)
)

mkpath(NEWS_DIR)
JLD2.jldsave(OUT_PATH; corpus = corpus, date_to_day = date_to_day)

println("─── synthetic news corpus ───")
println("output      = $(OUT_PATH)")
println("tickers     = $(TICKERS)")
println("date range  = $(DAYS[1]) .. $(DAYS[end])  ($(T) trading days)")
println("total items = $(length(corpus.items))")
println("avg items/day/ticker = $(round(length(corpus.items) / (T * K), digits=2))")
println("scenario    = $(scenario.label)  κ+=$(scenario.kappa_pos)  κ-=$(scenario.kappa_neg)  λ=$(scenario.arrival_intensity)")

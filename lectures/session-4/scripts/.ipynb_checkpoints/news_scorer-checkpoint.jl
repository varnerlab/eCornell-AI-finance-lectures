#!/usr/bin/env julia
#
# news_scorer.jl -- Hourly news scoring fire for the intraday cron.
#
# Usage:
#   julia --project=.. news_scorer.jl --mode=hourly         # regular cron fire
#   julia --project=.. news_scorer.jl --mode=class_refresh  # manual pre-class refresh
#
# Reads config/news-source.toml to determine which source backs this fire:
#   mode = "synthetic"     -- score from the on-disk corpus (May 5-10 default).
#   mode = "newsapi"       -- pull headlines from NewsAPI.org (May 11 demo).
#   mode = "anthropic_web" -- use Anthropic's web fetch tool (May 11 demo).
#
# Each fire writes one JLD2 to data/news/news-YYYY-MM-DD-HH.jld2 with the
# scored items and per-ticker aggregates. The TomorrowsTicket notebook
# loads the latest news-* file at sign-off time.
#

using Dates, TOML, JSON

const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
const CONFIG_DIR = joinpath(SESSION_DIR, "config");
const DATA_DIR = joinpath(SESSION_DIR, "data");
const NEWS_DIR = joinpath(DATA_DIR, "news");

using Pkg;
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance

const NEWS_SOURCE_PATH = joinpath(CONFIG_DIR, "news-source.toml");
const PROD_CONFIG_PATH = joinpath(CONFIG_DIR, "production-config.toml");
const LOG_PATH = joinpath(DATA_DIR, "production-log.txt");

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
"""
    parse_args() -> String

Parse the `--mode=` CLI argument and return the requested fire mode
("hourly" for the regular cron fire, "class_refresh" for a manual pre-class
refresh). Unknown or missing args default to "hourly". Used by `main`.
"""
function parse_args()
    mode = "hourly";
    for arg in ARGS
        if startswith(arg, "--mode=")
            mode = String(split(arg, "=")[2]);
        end
    end
    return mode;
end

"""
    log_entry(tag, msg)

Append one timestamped line to the shared production log
(`data/production-log.txt`) and echo it to stdout so the cron's stderr/out
captures the same record. The line format is
`yyyy-mm-dd HH:MM:SS [NEWS-<TAG>] <msg>`; the `NEWS-` prefix lets the
log reader tell news-cron lines apart from the engine cron's entries.
"""
function log_entry(tag::String, msg::String)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS");
    line = "$(ts) [NEWS-$(uppercase(tag))] $(msg)";
    println(line);
    mkpath(DATA_DIR);
    open(LOG_PATH, "a") do f
        println(f, line);
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
"""
    load_news_config() -> NamedTuple

Read `config/news-source.toml` and return the parsed news-pipeline
configuration as a NamedTuple. Fields: `mode` (synthetic / newsapi /
anthropic_web), `synthetic_path` (absolute path to the on-disk synthetic
corpus), `api_key_env` (env-var name for NewsAPI), `scoring_model`
(Claude model id used to score each headline), `cached_noise_sd` (Gaussian
noise sigma applied on top of the cached true score in synthetic mode),
`anthropic_key_env` (env-var name for the Anthropic API key),
`max_headlines` (per-fire cap), `explicit_tickers` (override universe).
"""
function load_news_config()
    isfile(NEWS_SOURCE_PATH) || error("News source config not found: $(NEWS_SOURCE_PATH)");
    cfg = TOML.parsefile(NEWS_SOURCE_PATH);
    return (
        mode = String(cfg["Source"]["mode"]),
        synthetic_path = joinpath(SESSION_DIR, String(cfg["Source"]["synthetic_corpus_path"])),
        api_key_env = String(cfg["Source"]["api_key_env"]),
        scoring_model = String(cfg["Claude"]["scoring_model"]),
        cached_noise_sd = Float64(cfg["Claude"]["cached_noise_sd"]),
        anthropic_key_env = String(cfg["Claude"]["api_key_env"]),
        max_headlines = Int(cfg["Filter"]["max_headlines_per_fire"]),
        explicit_tickers = String.(cfg["Filter"]["tickers"]),
    );
end

"""
    load_universe_from_prod_config() -> Vector{String}

Resolve the ticker universe for this fire from `production-config.toml`.
When `[Tickers].source == "session-1"`, load the universe from the S1
minimum-variance allocation artifact so the news cron stays in sync with
whichever basket the production engine is trading; otherwise read the
inline `Tickers.universe` array. Returns an empty vector if
`production-config.toml` is absent so the caller can fall back to the
explicit tickers in `news-source.toml`.
"""
function load_universe_from_prod_config()::Vector{String}
    isfile(PROD_CONFIG_PATH) || return String[];
    cfg = TOML.parsefile(PROD_CONFIG_PATH);
    source = String(get(cfg["Tickers"], "source", "manual"));
    if source == "session-1"
        s1_path = joinpath(SESSION_DIR, "..", "session-1", "data", "minvar-allocation.jld2");
        isfile(s1_path) || error(
            "Tickers.source=\"session-1\" but $(s1_path) is missing. " *
            "Run the S1 BuildMinVariancePortfolio notebook first.");
        return String.(load_results(s1_path)["my_tickers"]);
    end
    return String.(cfg["Tickers"]["universe"]);
end

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic source
# ──────────────────────────────────────────────────────────────────────────────
"""
    score_synthetic_hour(news_cfg, fire_time, tickers)

Score one hourly fire from the on-disk synthetic corpus (May 5-10 default
window). Picks the items whose `publication_day` maps to `fire_time`'s
date via the corpus's `date_to_day` map, slices to the current hour by a
deterministic hash-based bar assignment so each item lands in exactly one
of the six in-session hours, caps to `news_cfg.max_headlines`, then
overwrites `claude_score` via cached-noise scoring
(`score_news_with_claude!` with `live=false`). Aggregates per-ticker
sentiment (mean of `claude_score`), severity (max of `|claude_score|`),
and counts, then persists via `write_news_artifact`. Returns nothing
(side effect: writes one `news-*.jld2` to `data/news/`).
"""
function score_synthetic_hour(news_cfg, fire_time::DateTime, tickers::Vector{String})
    isfile(news_cfg.synthetic_path) ||
        error("Synthetic corpus not found at $(news_cfg.synthetic_path). " *
              "Generate with code/scripts/generate-synthetic-news.jl first.");

    data = load_results(news_cfg.synthetic_path);
    corpus = data["corpus"]::MyNewsCorpus;

    # Pick the items whose publication_day maps to today's date. The corpus
    # stores items indexed by trading-day index; the runner has a date->day
    # map under "date_to_day" in the corpus file.
    date_to_day = data["date_to_day"]::Dict{String,Int};
    today_str = Dates.format(Date(fire_time), "yyyy-mm-dd");
    haskey(date_to_day, today_str) ||
        error("Synthetic corpus has no entry for $(today_str). " *
              "Regenerate the corpus to cover the current date range.");
    today_day = date_to_day[today_str];
    todays_items = filter(it -> it.publication_day == today_day, corpus.items);

    # Slice to this hour's portion. The corpus has uniform arrival across the
    # session; for an hour-of-day h between market_open and market_close, take
    # the items whose hash modulo bars-per-day matches the bar index.
    market_open_h = 10;
    market_close_h = 16;
    hour = Hour(fire_time).value;
    if hour < market_open_h || hour > market_close_h
        log_entry("synthetic", "Out-of-session fire at hour=$(hour); skipping.");
        return;
    end
    bar_index = hour - market_open_h;
    n_bars = market_close_h - market_open_h + 1;
    hour_items = filter(todays_items) do it
        # deterministic hash-based slice so each item maps to exactly one hour
        ((hash(it.text) % n_bars)) == bar_index
    end

    # Cap at max_headlines; preserve ticker diversity if possible.
    if length(hour_items) > news_cfg.max_headlines
        hour_items = hour_items[1:news_cfg.max_headlines];
    end

    # "Score" — overwrite claude_score using cached-noise simulation.
    # We do this on a shallow copy to avoid mutating the on-disk corpus.
    hour_corpus = build(MyNewsCorpus, (
        items = hour_items, tickers = corpus.tickers, scenario = corpus.scenario,
        news_factor = corpus.news_factor, shocked_prices = corpus.shocked_prices,
        seed = corpus.seed,
    ));
    score_news_with_claude!(hour_corpus; live = false,
        cached_noise_sd = news_cfg.cached_noise_sd, seed = Int(hash(fire_time) & 0x7fffffff));

    # Aggregate per ticker.
    sentiment = Dict{String,Float64}(t => 0.0 for t in tickers);
    severity = Dict{String,Float64}(t => 0.0 for t in tickers);
    counts = Dict{String,Int}(t => 0 for t in tickers);
    for it in hour_corpus.items
        if haskey(sentiment, it.ticker)
            sentiment[it.ticker] += it.claude_score;
            severity[it.ticker] = max(severity[it.ticker], abs(it.claude_score));
            counts[it.ticker] += 1;
        end
    end
    for t in tickers
        sentiment[t] = counts[t] > 0 ? sentiment[t] / counts[t] : 0.0;
    end

    write_news_artifact(fire_time, hour_corpus, sentiment, severity, counts, "synthetic");
end

# ──────────────────────────────────────────────────────────────────────────────
# Real sources (skeletons; require API keys at class time)
# ──────────────────────────────────────────────────────────────────────────────
"""
    score_newsapi_hour(news_cfg, fire_time, tickers)

NewsAPI integration stub. Currently logs a "not wired" message and falls
back to `score_synthetic_hour`, so existing cron entries that reference
`mode=newsapi` continue to write a useful artifact while the live
NewsAPI client is built out (Phase B follow-up).
"""
function score_newsapi_hour(news_cfg, fire_time::DateTime, tickers::Vector{String})
    api_key = get(ENV, news_cfg.api_key_env, "");
    if isempty(api_key)
        log_entry("newsapi", "$(news_cfg.api_key_env) is unset; falling back to synthetic.");
        return score_synthetic_hour(news_cfg, fire_time, tickers);
    end
    log_entry("newsapi", "NewsAPI integration not yet wired (Phase B follow-up); falling back to synthetic.");
    score_synthetic_hour(news_cfg, fire_time, tickers);
end

const ANTHROPIC_DAILY_SEARCH_CAP = 100;            # hard cap; refuse fire if exceeded
const ANTHROPIC_MAX_USES_PER_TICKER = 2;           # passed to web_search tool
const ANTHROPIC_HEADLINES_PER_TICKER = 3;          # asked for in prompt
const ANTHROPIC_FETCH_MODEL = "claude-sonnet-4-6"; # search-capable model
const ANTHROPIC_FETCH_MAX_TOKENS = 800;
# Tier 1 caps Sonnet 4.6 at 10K input tokens/min and 5 requests/min. Web
# search inflates input tokens (results are fed back into context), so we
# pace per-ticker fetches to stay under the cap. Lower this once your
# account auto-promotes to Tier 2 (80K input tokens/min).
const ANTHROPIC_INTER_TICKER_SLEEP = 60.0;
const ANTHROPIC_POST_FETCH_SLEEP = 35.0;    # let input-token bucket recover from the search
const ANTHROPIC_SCORING_RATE_LIMIT = 14.0;  # seconds between per-item scoring calls

"""
    _budget_path(fire_time) -> String

Path to the per-day budget tracker JSON
(`data/news/budget-YYYY-MM-DD.json`). The budget tracks two cumulative
counts for the day: total web-search calls and total news fires. The
`anthropic_web` mode refuses to fire when the search count reaches
`ANTHROPIC_DAILY_SEARCH_CAP` so a runaway loop cannot exhaust the API
quota in a single day.
"""
function _budget_path(fire_time::DateTime)::String
    mkpath(NEWS_DIR);
    return joinpath(NEWS_DIR, "budget-$(Dates.format(Date(fire_time), "yyyy-mm-dd")).json");
end

"""
    _read_budget(fire_time) -> Dict{String,Any}

Load today's budget JSON, or return a fresh `{"date", "searches" = 0,
"fires" = 0}` dict if the file does not exist yet. The first fire of the
day creates the file via `_write_budget`.
"""
function _read_budget(fire_time::DateTime)::Dict{String,Any}
    path = _budget_path(fire_time);
    isfile(path) || return Dict("date" => Dates.format(Date(fire_time), "yyyy-mm-dd"),
        "searches" => 0, "fires" => 0);
    return JSON.parsefile(path);
end

"""
    _write_budget(fire_time, budget)

Overwrite today's budget JSON with the updated counts. Called at the end
of each `anthropic_web` fire after both `searches` and `fires` have been
incremented to reflect what this fire consumed.
"""
function _write_budget(fire_time::DateTime, budget::Dict{String,Any})
    open(_budget_path(fire_time), "w") do f
        JSON.print(f, budget);
    end
end

"""
    _parse_headline_json(reply) -> Vector{Dict{String,Any}}

Extract the JSON array of headlines from Claude's reply. The model is
asked to emit only a JSON array, but in practice the reply sometimes
arrives wrapped in a markdown fence or with stray prose before / after.
This helper strips optional ``` fences, finds the outermost `[ ... ]`
substring, parses it as JSON, and returns the array as a vector of dicts.
Returns an empty vector on any parse failure and logs the truncated reply
so the error is debuggable from the log file.
"""
function _parse_headline_json(reply::String)::Vector{Dict{String,Any}}
    # Strip optional markdown fence and any prose around the JSON array.
    s = strip(reply);
    s = replace(s, r"^```(?:json)?\n?" => "");
    s = replace(s, r"\n?```$" => "");
    # Find the outermost array.
    i = findfirst('[', s);
    j = findlast(']', s);
    (i === nothing || j === nothing || j <= i) && return Dict{String,Any}[];
    payload = s[i:j];
    return try
        parsed = JSON.parse(payload);
        parsed isa Vector ? Vector{Dict{String,Any}}(parsed) : Dict{String,Any}[];
    catch e
        log_entry("anthropic_web", "JSON parse failed: $(e); reply=$(first(reply, 200))");
        Dict{String,Any}[];
    end
end

"""
    _fetch_ticker_headlines(ticker, api_key) -> (headlines, n_searches)

Call Claude (with the web-search tool enabled) to fetch up to
`ANTHROPIC_HEADLINES_PER_TICKER` recent headlines for one ticker. Returns
the parsed JSON array of `{headline, url, published_relative}` dicts and
the number of underlying web-search tool invocations Claude used
(important for the daily budget tracking in `score_anthropic_web_hour`).
"""
function _fetch_ticker_headlines(ticker::String, api_key::String)::Tuple{Vector{Dict{String,Any}},Int}
    prompt = string(
        "Search the web for the most recent financial news about ticker ",
        "$(ticker) from the past few hours. Return up to $(ANTHROPIC_HEADLINES_PER_TICKER) ",
        "current headlines as a JSON array. Each entry must have keys: ",
        "\"headline\" (the headline text only, no source name), ",
        "\"url\" (the article URL), ",
        "\"published_relative\" (a phrase like \"30 minutes ago\" or \"today\"). ",
        "Output ONLY the JSON array, no preamble, no explanation, no markdown fence.",
    );
    (text, n_searches) = eCornellAIFinance._call_claude_with_web_search(prompt;
        api_key = api_key,
        model = ANTHROPIC_FETCH_MODEL,
        max_tokens = ANTHROPIC_FETCH_MAX_TOKENS,
        max_uses = ANTHROPIC_MAX_USES_PER_TICKER);
    return (_parse_headline_json(text), n_searches);
end

"""
    score_anthropic_web_hour(news_cfg, fire_time, tickers)

Run one live Claude-with-web-search news fire. For each ticker, asks
Claude to find the most recent financial-news headlines via the web
search tool, then scores each headline in `[-1, +1]` with a second
(non-search) Claude call. Aggregates per-ticker sentiment / severity /
counts and persists the artifact.

Falls back to `score_synthetic_hour` (and logs the reason) when (a) the
Anthropic API key env var is unset, (b) the daily search-cap has been
reached, or (c) no live items were returned for any ticker. Throttles
fetches via `ANTHROPIC_INTER_TICKER_SLEEP`, `ANTHROPIC_POST_FETCH_SLEEP`,
and `ANTHROPIC_SCORING_RATE_LIMIT` so the cron stays under the Tier 1
rate caps (10K input tokens/min, 5 requests/min for Sonnet 4.6). Catches
per-ticker fetch / score failures so a single 429 only kills the
offending ticker, not the whole fire.
"""
function score_anthropic_web_hour(news_cfg, fire_time::DateTime, tickers::Vector{String})
    api_key = get(ENV, news_cfg.anthropic_key_env, "");
    if isempty(api_key)
        log_entry("anthropic_web", "$(news_cfg.anthropic_key_env) is unset; falling back to synthetic.");
        return score_synthetic_hour(news_cfg, fire_time, tickers);
    end

    budget = _read_budget(fire_time);
    if Int(budget["searches"]) >= ANTHROPIC_DAILY_SEARCH_CAP
        log_entry("anthropic_web", "daily search cap reached " *
            "($(budget["searches"])/$(ANTHROPIC_DAILY_SEARCH_CAP)); falling back to synthetic.");
        return score_synthetic_hour(news_cfg, fire_time, tickers);
    end

    # Fetch + score per ticker so the rate-limit budget is consumed gradually
    # and a single 429 only kills its own ticker, not the whole fire.
    items = MyNewsItem[];
    fire_searches = 0;
    for (i, t) in enumerate(tickers)
        i > 1 && ANTHROPIC_INTER_TICKER_SLEEP > 0.0 && sleep(ANTHROPIC_INTER_TICKER_SLEEP);
        local headlines
        try
            (headlines, n_searches) = _fetch_ticker_headlines(t, api_key);
            fire_searches += n_searches;
        catch e
            log_entry("anthropic_web", "fetch failed for $(t): $(sprint(showerror, e)); skipping ticker.");
            continue;
        end
        ANTHROPIC_POST_FETCH_SLEEP > 0.0 && !isempty(headlines) && sleep(ANTHROPIC_POST_FETCH_SLEEP);
        for (j, h) in enumerate(headlines)
            text = String(get(h, "headline", ""));
            isempty(text) && continue;
            it = MyNewsItem();
            it.ticker = t;
            it.publication_day = 1;
            it.text = text;
            it.true_score = NaN;
            it.claude_score = NaN;
            it.source = "anthropic_web";
            j > 1 && ANTHROPIC_SCORING_RATE_LIMIT > 0.0 && sleep(ANTHROPIC_SCORING_RATE_LIMIT);
            try
                prompt = string(
                    "Read this financial news headline and assign a sentiment score in ",
                    "[-1.0, +1.0] for the mentioned company. -1 = strongly bearish, ",
                    "0 = neutral, +1 = strongly bullish. Output only the number.\n\n",
                    "Headline: $(text)\nScore:");
                reply = eCornellAIFinance._call_claude(prompt;
                    api_key = api_key, model = news_cfg.scoring_model, max_tokens = 10);
                s = try parse(Float64, strip(reply)); catch; NaN; end
                it.claude_score = isnan(s) ? NaN : clamp(s, -1.0, 1.0);
            catch e
                log_entry("anthropic_web", "score failed for $(t) item $(j): $(sprint(showerror, e)); leaving NaN.");
            end
            push!(items, it);
        end
    end

    budget["searches"] = Int(budget["searches"]) + fire_searches;
    budget["fires"] = Int(budget["fires"]) + 1;
    _write_budget(fire_time, budget);
    n_scored = count(it -> !isnan(it.claude_score), items);
    log_entry("anthropic_web", "$(length(items)) items, $(n_scored) scored, $(fire_searches) searches " *
        "(today $(budget["searches"])/$(ANTHROPIC_DAILY_SEARCH_CAP)).");

    if isempty(items)
        log_entry("anthropic_web", "no live items returned; falling back to synthetic.");
        return score_synthetic_hour(news_cfg, fire_time, tickers);
    end

    # Wrap into a corpus for persistence (claude_score is already populated
    # per-item above; no further scoring call here).
    scenario = build(MyNewsScenario, (
        label = "live_anthropic_web", kappa_pos = 0.0, kappa_neg = 0.0,
        arrival_intensity = 0.0, sentiment_mean = 0.0, sentiment_sd = 0.0,
    ));
    K = length(tickers);
    hour_corpus = build(MyNewsCorpus, (
        items = items, tickers = tickers, scenario = scenario,
        news_factor = zeros(1, K), shocked_prices = zeros(1, K), seed = 0,
    ));

    sentiment = Dict{String,Float64}(t => 0.0 for t in tickers);
    severity = Dict{String,Float64}(t => 0.0 for t in tickers);
    counts = Dict{String,Int}(t => 0 for t in tickers);
    for it in hour_corpus.items
        haskey(sentiment, it.ticker) || continue;
        isnan(it.claude_score) && continue;
        sentiment[it.ticker] += it.claude_score;
        severity[it.ticker] = max(severity[it.ticker], abs(it.claude_score));
        counts[it.ticker] += 1;
    end
    for t in tickers
        sentiment[t] = counts[t] > 0 ? sentiment[t] / counts[t] : 0.0;
    end

    write_news_artifact(fire_time, hour_corpus, sentiment, severity, counts, "anthropic_web");
end

# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────
"""
    news_artifact_path(fire_time) -> String

Compose the on-disk path for this fire's news artifact:
`data/news/news-YYYY-MM-DD-HH.jld2`. The runner's
`latest_news_artifact(fire_time, tickers)` reads the most recent file in
this directory whose stamp is at or before the engine's fire time.
"""
function news_artifact_path(fire_time::DateTime)::String
    mkpath(NEWS_DIR);
    stamp = Dates.format(fire_time, "yyyy-mm-dd-HH");
    return joinpath(NEWS_DIR, "news-$(stamp).jld2");
end

"""
    write_news_artifact(fire_time, corpus, sentiment, severity, counts, source)

Persist one news fire to disk as a single JLD2 with six top-level keys:
`fire_time`, `source` (synthetic / newsapi / anthropic_web), `sentiment`
(per-ticker mean of `claude_score`), `severity` (per-ticker max of
`|claude_score|`), `counts` (per-ticker item count), and `corpus` (the
full item-level `MyNewsCorpus` kept for post-hoc analysis). The engine
reads only `severity` in production today; the rest is for the audit log
and the analysis notebooks.
"""
function write_news_artifact(fire_time::DateTime, corpus::MyNewsCorpus,
        sentiment::Dict{String,Float64}, severity::Dict{String,Float64},
        counts::Dict{String,Int}, source::String)
    path = news_artifact_path(fire_time);
    save_results(path, Dict(
        "corpus" => corpus,
        "sentiment" => sentiment,
        "severity" => severity,
        "counts" => counts,
        "source" => source,
        "fire_time" => fire_time,
    ));
    n = length(corpus.items);
    flagged = count(t -> severity[t] >= 0.6, keys(severity));
    log_entry("write", "wrote $(path) -- $(n) items, $(flagged) tickers flagged");
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
"""
    main()

Entry point for the hourly news cron. Parses the CLI mode, loads the news
config and ticker universe, stamps the fire timestamp, and dispatches on
`news_cfg.mode` to one of `score_synthetic_hour`,  `score_newsapi_hour`,
or `score_anthropic_web_hour`. Errors on an unrecognized mode.
"""
function main()
    mode = parse_args();
    news_cfg = load_news_config();
    tickers = isempty(news_cfg.explicit_tickers) ?
        load_universe_from_prod_config() : news_cfg.explicit_tickers;
    isempty(tickers) && error("No tickers configured. Check production-config.toml or news-source.toml.");

    fire_time = now(UTC);
    log_entry(mode, "fire at $(fire_time) source=$(news_cfg.mode) tickers=$(length(tickers))");

    if news_cfg.mode == "synthetic"
        score_synthetic_hour(news_cfg, fire_time, tickers);
    elseif news_cfg.mode == "newsapi"
        score_newsapi_hour(news_cfg, fire_time, tickers);
    elseif news_cfg.mode == "anthropic_web"
        score_anthropic_web_hour(news_cfg, fire_time, tickers);
    else
        error("Unknown news source mode: $(news_cfg.mode)");
    end
end

main();

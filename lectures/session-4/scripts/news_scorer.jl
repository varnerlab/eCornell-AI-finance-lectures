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

using Dates, TOML

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
function parse_args()
    mode = "hourly";
    for arg in ARGS
        if startswith(arg, "--mode=")
            mode = String(split(arg, "=")[2]);
        end
    end
    return mode;
end

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

function load_universe_from_prod_config()::Vector{String}
    isfile(PROD_CONFIG_PATH) || return String[];
    cfg = TOML.parsefile(PROD_CONFIG_PATH);
    return String.(cfg["Tickers"]["universe"]);
end

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic source
# ──────────────────────────────────────────────────────────────────────────────
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
function score_newsapi_hour(news_cfg, fire_time::DateTime, tickers::Vector{String})
    api_key = get(ENV, news_cfg.api_key_env, "");
    if isempty(api_key)
        log_entry("newsapi", "$(news_cfg.api_key_env) is unset; falling back to synthetic.");
        return score_synthetic_hour(news_cfg, fire_time, tickers);
    end
    log_entry("newsapi", "NewsAPI integration not yet wired (Phase B follow-up); falling back to synthetic.");
    score_synthetic_hour(news_cfg, fire_time, tickers);
end

function score_anthropic_web_hour(news_cfg, fire_time::DateTime, tickers::Vector{String})
    api_key = get(ENV, news_cfg.anthropic_key_env, "");
    if isempty(api_key)
        log_entry("anthropic_web", "$(news_cfg.anthropic_key_env) is unset; falling back to synthetic.");
        return score_synthetic_hour(news_cfg, fire_time, tickers);
    end
    log_entry("anthropic_web", "Anthropic web-fetch integration not yet wired (Phase B follow-up); falling back to synthetic.");
    score_synthetic_hour(news_cfg, fire_time, tickers);
end

# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────
function news_artifact_path(fire_time::DateTime)::String
    mkpath(NEWS_DIR);
    stamp = Dates.format(fire_time, "yyyy-mm-dd-HH");
    return joinpath(NEWS_DIR, "news-$(stamp).jld2");
end

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
function main()
    mode = parse_args();
    news_cfg = load_news_config();
    tickers = isempty(news_cfg.explicit_tickers) ?
        load_universe_from_prod_config() : news_cfg.explicit_tickers;
    isempty(tickers) && error("No tickers configured. Check production-config.toml or news-source.toml.");

    fire_time = now();
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

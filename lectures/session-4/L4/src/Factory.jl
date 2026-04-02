# --- Production Builders --------------------------------------------------------

"""
    build(type::Type{MySentimentSignal}, data::NamedTuple) -> MySentimentSignal
"""
function build(type::Type{MySentimentSignal}, data::NamedTuple)::MySentimentSignal

    signal = MySentimentSignal();
    signal.score = data.score;
    signal.source = data.source;
    signal.day = data.day;

    return signal;
end

"""
    build(type::Type{MyProductionContext}, data::NamedTuple) -> MyProductionContext
"""
function build(type::Type{MyProductionContext}, data::NamedTuple)::MyProductionContext

    ctx = MyProductionContext();
    ctx.tickers = data.tickers;
    ctx.sim_parameters = data.sim_parameters;
    ctx.B₀ = data.B₀;
    ctx.epsilon = data.epsilon;
    ctx.max_drawdown = data.max_drawdown;
    ctx.max_turnover = data.max_turnover;
    ctx.sentiment_threshold = data.sentiment_threshold;
    ctx.sentiment_override_lambda = data.sentiment_override_lambda;
    ctx.max_bandit_churn = data.max_bandit_churn;

    return ctx;
end

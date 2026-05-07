# --- Session 1: SIM and Sharpe Builders ------------------------------------------

"""
    build(type::Type{MySIMParameterEstimate}, data::NamedTuple) -> MySIMParameterEstimate

Build a SIM parameter estimate from a named tuple.
Expected fields: ticker, α, β, σ_ε, r².
"""
function build(type::Type{MySIMParameterEstimate}, data::NamedTuple)::MySIMParameterEstimate

    est = MySIMParameterEstimate();
    est.ticker = data.ticker;
    est.α = data.α;
    est.β = data.β;
    est.σ_ε = data.σ_ε;
    est.r² = data.r²;

    return est;
end

"""
    build(type::Type{MySharpeRatioPortfolioChoiceProblem}, data::NamedTuple) -> MySharpeRatioPortfolioChoiceProblem

Build a Sharpe ratio optimization problem from a named tuple.
Expected fields: Σ, risk_free_rate, α, β, gₘ, bounds.
"""
function build(type::Type{MySharpeRatioPortfolioChoiceProblem}, data::NamedTuple)::MySharpeRatioPortfolioChoiceProblem

    problem = MySharpeRatioPortfolioChoiceProblem();
    problem.Σ = data.Σ;
    problem.risk_free_rate = data.risk_free_rate;
    problem.α = data.α;
    problem.β = data.β;
    problem.gₘ = data.gₘ;
    problem.bounds = data.bounds;

    return problem;
end

# --- Session 1: Portfolio Builders -----------------------------------------------

"""
    build(type::Type{MyPortfolioAllocationProblem};
        μ::Array{Float64,1}, Σ::Array{Float64,2},
        bounds::Array{Float64,2}, R::Float64) -> MyPortfolioAllocationProblem

Build a `MyPortfolioAllocationProblem` instance from the given data.
"""
function build(type::Type{MyPortfolioAllocationProblem};
    μ::Array{Float64,1}, Σ::Array{Float64,2},
    bounds::Array{Float64,2}, R::Float64)::MyPortfolioAllocationProblem

    # build -
    problem = MyPortfolioAllocationProblem();
    problem.μ = μ;
    problem.Σ = Σ;
    problem.bounds = bounds;
    problem.R = R;

    # return -
    return problem;
end

# --- Session 2: Utility Problem Builders ----------------------------------------

"""
    build(type::Type{MyCobbDouglasChoiceProblem}, data::NamedTuple) -> MyCobbDouglasChoiceProblem

Build a Cobb-Douglas choice problem from a named tuple.
Expected fields: gamma, prices, B, epsilon.
"""
function build(type::Type{MyCobbDouglasChoiceProblem}, data::NamedTuple)::MyCobbDouglasChoiceProblem

    # build -
    problem = MyCobbDouglasChoiceProblem();
    problem.gamma = data.gamma;
    problem.prices = data.prices;
    problem.B = data.B;
    problem.epsilon = data.epsilon;

    # return -
    return problem;
end

"""
    build(type::Type{MyCESChoiceProblem}, data::NamedTuple) -> MyCESChoiceProblem

Build a CES choice problem from a named tuple.
Expected fields: gamma, prices, B, epsilon, eta.
"""
function build(type::Type{MyCESChoiceProblem}, data::NamedTuple)::MyCESChoiceProblem

    # build -
    problem = MyCESChoiceProblem();
    problem.gamma = data.gamma;
    problem.prices = data.prices;
    problem.B = data.B;
    problem.epsilon = data.epsilon;
    problem.eta = data.eta;

    # return -
    return problem;
end

"""
    build(type::Type{MyLogLinearChoiceProblem}, data::NamedTuple) -> MyLogLinearChoiceProblem

Build a log-linear choice problem from a named tuple.
Expected fields: gamma, prices, B, epsilon.
"""
function build(type::Type{MyLogLinearChoiceProblem}, data::NamedTuple)::MyLogLinearChoiceProblem

    # build -
    problem = MyLogLinearChoiceProblem();
    problem.gamma = data.gamma;
    problem.prices = data.prices;
    problem.B = data.B;
    problem.epsilon = data.epsilon;

    # return -
    return problem;
end

# --- Session 2: Rebalancing Engine Builders -------------------------------------

"""
    build(type::Type{MyRebalancingContextModel}, data::NamedTuple) -> MyRebalancingContextModel

Build a rebalancing context model from a named tuple of parameters.
"""
function build(type::Type{MyRebalancingContextModel}, data::NamedTuple)::MyRebalancingContextModel

    # build -
    model = MyRebalancingContextModel();
    model.B = data.B;
    model.tickers = data.tickers;
    model.marketdata = data.marketdata;
    model.marketfactor = data.marketfactor;
    model.sim_parameters = data.sim_parameters;
    model.lambda = data.lambda;
    model.Δt = data.Δt;
    model.epsilon = data.epsilon;

    # return -
    return model;
end

"""
    build(type::Type{MyTriggerRules}, data::NamedTuple) -> MyTriggerRules

Build trigger rules from a named tuple.
"""
function build(type::Type{MyTriggerRules}, data::NamedTuple)::MyTriggerRules

    # build -
    rules = MyTriggerRules();
    rules.max_drawdown = data.max_drawdown;
    rules.max_turnover = data.max_turnover;
    rules.rebalance_schedule = data.rebalance_schedule;

    # return -
    return rules;
end

# --- Session 3: Backtest Builders -----------------------------------------------

"""
    build(type::Type{MyBacktestScenario}, data::NamedTuple) -> MyBacktestScenario
"""
function build(type::Type{MyBacktestScenario}, data::NamedTuple)::MyBacktestScenario

    scenario = MyBacktestScenario();
    scenario.label = data.label;
    scenario.price_paths = data.price_paths;
    scenario.market_paths = data.market_paths;
    scenario.n_paths = data.n_paths;
    scenario.n_steps = data.n_steps;

    return scenario;
end

"""
    build(type::Type{MyValidationReport}; strategy_label, criteria, actuals) -> MyValidationReport
"""
function build(type::Type{MyValidationReport};
    strategy_label::String, criteria::Dict{String,Float64},
    actuals::Dict{String,Float64})::MyValidationReport

    report = MyValidationReport();
    report.strategy_label = strategy_label;
    report.criteria = criteria;
    report.actuals = actuals;

    # evaluate pass/fail -
    report.passed = Dict{String,Bool}();
    for (key, threshold) in criteria
        if haskey(actuals, key)
            if startswith(key, "max_")
                report.passed[key] = actuals[key] <= threshold;
            else
                report.passed[key] = actuals[key] >= threshold;
            end
        else
            report.passed[key] = false;
        end
    end

    return report;
end

# --- Session 3: Bandit Builders -------------------------------------------------

"""
    build(type::Type{MyBanditContext}, data::NamedTuple) -> MyBanditContext
"""
function build(type::Type{MyBanditContext}, data::NamedTuple)::MyBanditContext

    ctx = MyBanditContext();
    ctx.tickers = data.tickers;
    ctx.sim_parameters = data.sim_parameters;
    ctx.prices = data.prices;
    ctx.B = data.B;
    ctx.gm_t = data.gm_t;
    ctx.lambda = data.lambda;
    ctx.epsilon = data.epsilon;

    return ctx;
end

"""
    build(type::Type{MyEpsilonGreedyBanditModel}, data::NamedTuple) -> MyEpsilonGreedyBanditModel
"""
function build(type::Type{MyEpsilonGreedyBanditModel}, data::NamedTuple)::MyEpsilonGreedyBanditModel

    model = MyEpsilonGreedyBanditModel();
    model.K = data.K;
    model.n_iterations = data.n_iterations;
    model.alpha = data.alpha;

    return model;
end

# --- Session 3: Eta-Bandit Builders ---------------------------------------------

"""
    build(type::Type{MyEtaBanditModel}, data::NamedTuple) -> MyEtaBanditModel
"""
function build(type::Type{MyEtaBanditModel}, data::NamedTuple)::MyEtaBanditModel

    model = MyEtaBanditModel();
    model.eta_grid = data.eta_grid;
    model.n_iterations = data.n_iterations;
    model.alpha = data.alpha;
    model.lambda_threshold = data.lambda_threshold;

    return model;
end

# --- Session 3: REINFORCE Builders ----------------------------------------------

"""
    build(type::Type{MyREINFORCEPolicy}, data::NamedTuple) -> MyREINFORCEPolicy

Construct a Gaussian REINFORCE policy with linear mean head and a
state-independent log-std.

### Required fields
- `n_features::Int`
- `eta_min::Float64`
- `eta_max::Float64`

### Optional fields
- `w_mu::Array{Float64,1}` — defaults to zeros(n_features)
- `b_mu::Float64` — defaults to the midpoint of `[eta_min, eta_max]`, a neutral
  prior over the action range when state weights are zero
- `log_sigma::Float64` — defaults to log(0.5)
"""
function build(type::Type{MyREINFORCEPolicy}, data::NamedTuple)::MyREINFORCEPolicy

    policy = MyREINFORCEPolicy();
    policy.n_features = data.n_features;
    policy.eta_min = data.eta_min;
    policy.eta_max = data.eta_max;

    policy.w_mu = haskey(data, :w_mu) ? data.w_mu : zeros(data.n_features);
    policy.b_mu = haskey(data, :b_mu) ? data.b_mu : 0.5 * (data.eta_min + data.eta_max);
    policy.log_sigma = haskey(data, :log_sigma) ? data.log_sigma : log(0.5);

    return policy;
end

"""
    build(type::Type{MyValueBaseline}, data::NamedTuple) -> MyValueBaseline

Construct a linear value baseline V_φ(s) = w' · s + b. Defaults to the zero
function so the first batch of advantages equals the raw return.

### Required fields
- `n_features::Int`

### Optional fields
- `w::Array{Float64,1}` — defaults to zeros(n_features)
- `b::Float64` — defaults to 0.0
"""
function build(type::Type{MyValueBaseline}, data::NamedTuple)::MyValueBaseline

    baseline = MyValueBaseline();
    baseline.n_features = data.n_features;
    baseline.w = haskey(data, :w) ? data.w : zeros(data.n_features);
    baseline.b = haskey(data, :b) ? data.b : 0.0;

    return baseline;
end

"""
    build(type::Type{MyREINFORCETrainer}, data::NamedTuple) -> MyREINFORCETrainer

Construct a REINFORCE-with-baseline trainer configuration. Required fields are
the episode-loop parameters; Adam hyperparameters and reward-shaping defaults
follow community conventions.

### Required fields
- `n_episodes::Int`
- `horizon::Int`

### Optional fields (with defaults)
- `learning_rate::Float64 = 1e-3`
- `beta1::Float64 = 0.9`
- `beta2::Float64 = 0.999`
- `epsilon::Float64 = 1e-8`
- `episodes_per_update::Int = 16`
- `discount::Float64 = 0.99`
- `drawdown_penalty::Float64 = 0.0` — set positive to penalize drawdowns
- `drawdown_threshold::Float64 = 0.05` — penalty activates above 5% drawdown
- `baseline_lr::Float64 = 1e-2`
- `seed::Int = 42`
"""
function build(type::Type{MyREINFORCETrainer}, data::NamedTuple)::MyREINFORCETrainer

    trainer = MyREINFORCETrainer();

    # required -
    trainer.n_episodes = data.n_episodes;
    trainer.horizon = data.horizon;

    # optional with defaults -
    trainer.learning_rate = haskey(data, :learning_rate) ? data.learning_rate : 1e-3;
    trainer.beta1 = haskey(data, :beta1) ? data.beta1 : 0.9;
    trainer.beta2 = haskey(data, :beta2) ? data.beta2 : 0.999;
    trainer.epsilon = haskey(data, :epsilon) ? data.epsilon : 1e-8;
    trainer.episodes_per_update = haskey(data, :episodes_per_update) ? data.episodes_per_update : 16;
    trainer.discount = haskey(data, :discount) ? data.discount : 0.99;
    trainer.drawdown_penalty = haskey(data, :drawdown_penalty) ? data.drawdown_penalty : 0.0;
    trainer.drawdown_threshold = haskey(data, :drawdown_threshold) ? data.drawdown_threshold : 0.05;
    trainer.baseline_lr = haskey(data, :baseline_lr) ? data.baseline_lr : 1e-2;
    trainer.seed = haskey(data, :seed) ? data.seed : 42;

    return trainer;
end

# --- Session 4: Production Builders ---------------------------------------------

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

# --- Session 4: News Sentiment Builders -----------------------------------------

"""
    build(type::Type{MyNewsScenario}, data::NamedTuple) -> MyNewsScenario

Build a `MyNewsScenario` from a named tuple with fields
`label`, `kappa_pos`, `kappa_neg`, `arrival_intensity`, `sentiment_mean`, `sentiment_sd`.
"""
function build(type::Type{MyNewsScenario}, data::NamedTuple)::MyNewsScenario

    s = MyNewsScenario();
    s.label = data.label;
    s.kappa_pos = data.kappa_pos;
    s.kappa_neg = data.kappa_neg;
    s.arrival_intensity = data.arrival_intensity;
    s.sentiment_mean = data.sentiment_mean;
    s.sentiment_sd = data.sentiment_sd;

    return s;
end

"""
    build(type::Type{MyNewsItem}, data::NamedTuple) -> MyNewsItem

Build a `MyNewsItem` from a named tuple with fields
`ticker`, `publication_day`, `text`, `true_score`, `claude_score`, `source`.
"""
function build(type::Type{MyNewsItem}, data::NamedTuple)::MyNewsItem

    item = MyNewsItem();
    item.ticker = data.ticker;
    item.publication_day = data.publication_day;
    item.text = data.text;
    item.true_score = data.true_score;
    item.claude_score = data.claude_score;
    item.source = data.source;

    return item;
end

"""
    build(type::Type{MyNewsCorpus}, data::NamedTuple) -> MyNewsCorpus

Build a `MyNewsCorpus` from a named tuple with fields
`items`, `tickers`, `scenario`, `news_factor`, `shocked_prices`, `seed`.
"""
function build(type::Type{MyNewsCorpus}, data::NamedTuple)::MyNewsCorpus

    c = MyNewsCorpus();
    c.items = data.items;
    c.tickers = data.tickers;
    c.scenario = data.scenario;
    c.news_factor = data.news_factor;
    c.shocked_prices = data.shocked_prices;
    c.seed = data.seed;

    return c;
end

# --- Session 4: intraday compliance queue + ticket builders ---------------------

"""
    build(type::Type{MyComplianceQueueItem}, data::NamedTuple) -> MyComplianceQueueItem

Build a queued trade item from a named tuple with fields
`id`, `timestamp`, `ticker`, `qty`, `side`, `proposed_weight`,
`gate_violations`, `engine_snapshot`.
"""
function build(type::Type{MyComplianceQueueItem}, data::NamedTuple)::MyComplianceQueueItem

    item = MyComplianceQueueItem();
    item.id = data.id;
    item.timestamp = data.timestamp;
    item.ticker = data.ticker;
    item.qty = data.qty;
    item.side = data.side;
    item.proposed_weight = data.proposed_weight;
    item.gate_violations = data.gate_violations;
    item.engine_snapshot = data.engine_snapshot;

    return item;
end

"""
    build(type::Type{MySignedDecision}, data::NamedTuple) -> MySignedDecision

Build a signed adjudication record from a named tuple with fields
`queue_id`, `action`, `modified_qty`, `notes`, `signed_by`, `signed_at`.
"""
function build(type::Type{MySignedDecision}, data::NamedTuple)::MySignedDecision

    dec = MySignedDecision();
    dec.queue_id = data.queue_id;
    dec.action = data.action;
    dec.modified_qty = data.modified_qty;
    dec.notes = data.notes;
    dec.signed_by = data.signed_by;
    dec.signed_at = data.signed_at;

    return dec;
end

"""
    build(type::Type{MyTomorrowsTicket}, data::NamedTuple) -> MyTomorrowsTicket

Build a proposed next-day ticket from a named tuple with fields
`target_weights`, `tickers`, `proposed_trades`, `sentiment`, `news_flags`,
`generated_at`, `eta_used`, `regime`.
"""
function build(type::Type{MyTomorrowsTicket}, data::NamedTuple)::MyTomorrowsTicket

    t = MyTomorrowsTicket();
    t.target_weights = data.target_weights;
    t.tickers = data.tickers;
    t.proposed_trades = data.proposed_trades;
    t.sentiment = data.sentiment;
    t.news_flags = data.news_flags;
    t.generated_at = data.generated_at;
    t.eta_used = data.eta_used;
    t.regime = data.regime;

    return t;
end

"""
    build(type::Type{MySignedTicket}, data::NamedTuple) -> MySignedTicket

Build a signed ticket from a named tuple with fields
`ticket`, `modifications`, `signed_by`, `signed_at`.
"""
function build(type::Type{MySignedTicket}, data::NamedTuple)::MySignedTicket

    s = MySignedTicket();
    s.ticket = data.ticket;
    s.modifications = data.modifications;
    s.signed_by = data.signed_by;
    s.signed_at = data.signed_at;

    return s;
end

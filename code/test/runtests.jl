using Test
using eCornellAIFinance
using LinearAlgebra
using Statistics
using Random

# ---------------------------------------------------------------------------
# Test data shared across test sets
# ---------------------------------------------------------------------------
const N_ASSETS = 5;
const TICKERS = ["LargeCap", "SmallCap", "International", "Bond", "Commodity"];
const SIM_PARAMS = Dict(
    "LargeCap"      => (0.0002, 1.10, 0.010),
    "SmallCap"      => (0.0003, 1.35, 0.014),
    "International" => (0.0001, 0.95, 0.012),
    "Bond"          => (0.0001, -0.15, 0.003),
    "Commodity"     => (0.0001, 0.60, 0.013)
);

# ---------------------------------------------------------------------------
@testset "eCornellAIFinance.jl" begin

    # ===================================================================
    @testset "Types — construction" begin
        @test MyPortfolioAllocationProblem() isa MyPortfolioAllocationProblem
        @test MyPortfolioPerformanceResult() isa MyPortfolioPerformanceResult
        @test MyCobbDouglasChoiceProblem() isa MyCobbDouglasChoiceProblem
        @test MyCESChoiceProblem() isa MyCESChoiceProblem
        @test MyLogLinearChoiceProblem() isa MyLogLinearChoiceProblem
        @test MyRebalancingContextModel() isa MyRebalancingContextModel
        @test MyTriggerRules() isa MyTriggerRules
        @test MyRebalancingResult() isa MyRebalancingResult
        @test MyBacktestScenario() isa MyBacktestScenario
        @test MyBacktestResult() isa MyBacktestResult
        @test MyValidationReport() isa MyValidationReport
        @test MyBanditContext() isa MyBanditContext
        @test MyEpsilonGreedyBanditModel() isa MyEpsilonGreedyBanditModel
        @test MyBanditResult() isa MyBanditResult
        @test MySentimentSignal() isa MySentimentSignal
        @test MyEscalationEvent() isa MyEscalationEvent
        @test MyProductionDayResult() isa MyProductionDayResult
        @test MyProductionContext() isa MyProductionContext
    end

    # ===================================================================
    @testset "Factory — build methods" begin

        @testset "MySIMParameterEstimate" begin
            e = build(MySIMParameterEstimate, (ticker="TEST", α=0.001, β=1.1, σ_ε=0.01, r²=0.85));
            @test e isa MySIMParameterEstimate
            @test e.ticker == "TEST"
            @test e.β == 1.1
        end

        @testset "MySharpeRatioPortfolioChoiceProblem" begin
            p = build(MySharpeRatioPortfolioChoiceProblem, (
                Σ=[0.01 0.002; 0.002 0.02], risk_free_rate=0.0002,
                α=[0.001, 0.002], β=[1.0, 0.8], gₘ=0.05,
                bounds=hcat(zeros(2), ones(2))));
            @test p isa MySharpeRatioPortfolioChoiceProblem
            @test p.risk_free_rate == 0.0002
        end

        @testset "MyPortfolioAllocationProblem" begin
            μ = [0.001, 0.002, 0.0005];
            Σ = [0.01 0.002 0.001; 0.002 0.02 0.003; 0.001 0.003 0.005];
            bounds = hcat(zeros(3), ones(3));
            p = build(MyPortfolioAllocationProblem; μ=μ, Σ=Σ, bounds=bounds, R=0.001);
            @test p isa MyPortfolioAllocationProblem
            @test p.μ == μ
            @test p.R == 0.001
        end

        @testset "MyCobbDouglasChoiceProblem" begin
            p = build(MyCobbDouglasChoiceProblem, (
                gamma=[0.5, 0.3, 0.2], prices=[100.0, 50.0, 80.0], B=10000.0, epsilon=0.1));
            @test p isa MyCobbDouglasChoiceProblem
            @test p.B == 10000.0
        end

        @testset "MyCESChoiceProblem" begin
            p = build(MyCESChoiceProblem, (
                gamma=[0.5, 0.3, 0.2], prices=[100.0, 50.0, 80.0], B=10000.0, epsilon=0.1, sigma=2.0));
            @test p isa MyCESChoiceProblem
            @test p.sigma == 2.0
        end

        @testset "MyLogLinearChoiceProblem" begin
            p = build(MyLogLinearChoiceProblem, (
                gamma=[0.5, 0.3, 0.2], prices=[100.0, 50.0, 80.0], B=10000.0, epsilon=0.1));
            @test p isa MyLogLinearChoiceProblem
        end

        @testset "MyTriggerRules" begin
            r = build(MyTriggerRules, (max_drawdown=0.10, max_turnover=0.50, rebalance_schedule=ones(Int, 10)));
            @test r isa MyTriggerRules
            @test r.max_drawdown == 0.10
        end

        @testset "MyBacktestScenario" begin
            s = build(MyBacktestScenario, (
                label="test", price_paths=zeros(2,10,3), market_paths=zeros(2,10), n_paths=2, n_steps=10));
            @test s isa MyBacktestScenario
            @test s.n_paths == 2
        end

        @testset "MyValidationReport" begin
            r = build(MyValidationReport;
                strategy_label="test", criteria=Dict("min_sharpe"=>0.3), actuals=Dict("min_sharpe"=>0.5));
            @test r isa MyValidationReport
            @test r.passed["min_sharpe"] == true
        end

        @testset "MyValidationReport — fail case" begin
            r = build(MyValidationReport;
                strategy_label="test", criteria=Dict("min_sharpe"=>0.3), actuals=Dict("min_sharpe"=>0.1));
            @test r.passed["min_sharpe"] == false
        end

        @testset "MyValidationReport — max_ threshold" begin
            r = build(MyValidationReport;
                strategy_label="test", criteria=Dict("max_drawdown"=>0.25), actuals=Dict("max_drawdown"=>0.30));
            @test r.passed["max_drawdown"] == false  # 0.30 > 0.25
        end

        @testset "MyBanditContext" begin
            c = build(MyBanditContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS,
                prices=[150.0, 45.0, 80.0, 100.0, 60.0], B=10000.0,
                gm_t=0.05, lambda=0.0, epsilon=0.1));
            @test c isa MyBanditContext
        end

        @testset "MyEpsilonGreedyBanditModel" begin
            m = build(MyEpsilonGreedyBanditModel, (K=5, n_iterations=100, alpha=0.1));
            @test m isa MyEpsilonGreedyBanditModel
            @test m.K == 5
        end

        @testset "MySentimentSignal" begin
            s = build(MySentimentSignal, (score=0.5, source="synthetic", day=1));
            @test s isa MySentimentSignal
            @test s.score == 0.5
        end

        @testset "MyProductionContext" begin
            c = build(MyProductionContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS, B₀=10000.0,
                epsilon=0.1, max_drawdown=0.15, max_turnover=0.50,
                sentiment_threshold=-0.5, sentiment_override_lambda=2.0, max_bandit_churn=2));
            @test c isa MyProductionContext
        end
    end

    # ===================================================================
    @testset "Compute — Session 1 (Portfolio)" begin

        @testset "compute_drawdown" begin
            # steady growth → zero drawdown
            r = fill(0.01, 100);
            @test compute_drawdown(r) ≈ 0.0 atol=1e-10

            # single large loss → drawdown equals the loss
            r2 = [0.0, 0.0, -0.20, 0.0, 0.0];
            dd = compute_drawdown(r2);
            @test dd > 0.19
            @test dd < 0.21
        end

        @testset "compute_turnover" begin
            w1 = [0.2, 0.3, 0.5];
            w2 = [0.4, 0.1, 0.5];
            @test compute_turnover(w1, w2) ≈ 0.4
            @test compute_turnover(w1, w1) ≈ 0.0
        end

        @testset "solve_minvariance" begin
            μ = [0.001, 0.002, 0.0005];
            Σ = [0.01 0.002 0.001; 0.002 0.02 0.003; 0.001 0.003 0.005];
            bounds = hcat(zeros(3), ones(3));

            problem = build(MyPortfolioAllocationProblem; μ=μ, Σ=Σ, bounds=bounds, R=0.0005);
            result = solve_minvariance(problem);

            @test result isa MyPortfolioPerformanceResult
            @test isapprox(sum(result.weights), 1.0, rtol=1e-4)
            @test all(result.weights .>= -1e-6)
            @test result.variance > 0
            @test result.expected_return >= 0.0005 - 1e-6
        end
    end

    # ===================================================================
    @testset "Compute — Session 1 (SIM & Sharpe)" begin

        # generate synthetic data for SIM tests -
        # Inputs are annualized growth rates (per the course's growth-rate convention).
        Random.seed!(99);
        T_sim = 504;
        true_α = 0.05;      # 5% per year (annualized growth rate units)
        true_β = 1.10;      # dimensionless
        true_σ_ε = 2.5;     # annualized growth rate std (per year)

        # synthetic market annualized growth rates (std ≈ 3.0 per year) -
        mkt_ret = randn(T_sim) .* 3.0;
        # synthetic asset annualized growth rates via SIM: gᵢ = α + β·gₘ + ε -
        asset_ret = true_α .+ true_β .* mkt_ret .+ true_σ_ε .* randn(T_sim);

        @testset "estimate_sim" begin
            est = estimate_sim(mkt_ret, asset_ret, "TestAsset"; δ=0.0);
            @test est isa MySIMParameterEstimate
            @test est.ticker == "TestAsset"
            @test isapprox(est.β, true_β, rtol=0.15)               # within 15%
            @test isapprox(est.σ_ε, true_σ_ε, rtol=0.15)           # within 15%
            @test est.r² > 0.5                                      # market explains majority of variance
        end

        @testset "estimate_sim — regularized" begin
            est = estimate_sim(mkt_ret, asset_ret, "TestAsset"; δ=0.01);
            @test est isa MySIMParameterEstimate
            @test est.r² > 0.0  # still positive
        end

        @testset "bootstrap_sim" begin
            bs = bootstrap_sim(mkt_ret, asset_ret, "TestAsset";
                δ=0.0, n_bootstrap=500, seed=42);

            @test bs isa Dict
            @test length(bs["alpha_samples"]) == 500
            @test length(bs["beta_samples"]) == 500

            # bootstrap mean should be close to point estimate
            @test isapprox(bs["beta_mean"], bs["point_estimate"].β, rtol=0.05)

            # bootstrap std should be close to theoretical SE
            @test isapprox(bs["beta_std"], bs["theoretical_se"][2], rtol=0.3)

            # 95% CI should contain the point estimate
            α_ci = bs["alpha_ci_95"];
            β_ci = bs["beta_ci_95"];
            @test α_ci[1] < bs["point_estimate"].α < α_ci[2]
            @test β_ci[1] < bs["point_estimate"].β < β_ci[2]
        end

        @testset "build_sim_covariance" begin
            # create 3 fake SIM estimates (σ_ε in annualized growth-rate units) -
            ests = MySIMParameterEstimate[];
            for (t, β, σ) ∈ [("A", 1.1, 2.5), ("B", 0.8, 3.0), ("C", -0.15, 0.8)]
                e = MySIMParameterEstimate();
                e.ticker = t; e.α = 0.0; e.β = β; e.σ_ε = σ; e.r² = 0.9;
                push!(ests, e);
            end

            σ_m = std(mkt_ret);
            Σ = build_sim_covariance(ests, σ_m);

            @test size(Σ) == (3, 3)
            @test issymmetric(Σ)
            @test isposdef(Σ)
            # off-diagonal: β_A * β_B * σ_m²
            @test isapprox(Σ[1, 2], 1.1 * 0.8 * σ_m^2, rtol=1e-10)
            # diagonal: β_A² * σ_m² + σ_ε_A²
            @test isapprox(Σ[1, 1], 1.1^2 * σ_m^2 + 2.5^2, rtol=1e-10)
        end

        @testset "solve_max_sharpe" begin
            # build a small Sharpe problem (all quantities in annualized growth-rate units) -
            ests = MySIMParameterEstimate[];
            for (t, α_v, β_v, σ_v) ∈ [("A", 0.05, 1.1, 2.5), ("B", 0.08, 0.8, 3.0), ("C", 0.03, -0.15, 0.8)]
                e = MySIMParameterEstimate();
                e.ticker = t; e.α = α_v; e.β = β_v; e.σ_ε = σ_v; e.r² = 0.9;
                push!(ests, e);
            end

            σ_m = std(mkt_ret);
            Σ = build_sim_covariance(ests, σ_m);
            α_vec = [e.α for e ∈ ests];
            β_vec = [e.β for e ∈ ests];

            problem = build(MySharpeRatioPortfolioChoiceProblem, (
                Σ = Σ, risk_free_rate = 0.04, α = α_vec, β = β_vec,
                gₘ = 0.08, bounds = hcat(zeros(3), ones(3))
            ));
            result = solve_max_sharpe(problem);

            @test haskey(result, "weights")
            @test haskey(result, "sharpe_ratio")
            @test isapprox(sum(result["weights"]), 1.0, rtol=1e-3)
            @test all(result["weights"] .>= -1e-4)
            @test result["sharpe_ratio"] > 0
        end
    end

    # ===================================================================
    @testset "Compute — Session 2 (Signals & Allocation)" begin

        @testset "compute_ema" begin
            prices = collect(1.0:100.0);
            ema = compute_ema(prices; window=10);
            @test length(ema) == 100
            @test ema[1] == 1.0  # initialized to first price
            @test ema[end] > ema[1]  # trending up
        end

        @testset "compute_lambda" begin
            short_ema = [100.0, 99.0, 101.0];
            long_ema = [100.0, 100.0, 100.0];
            λ = compute_lambda(short_ema, long_ema; G=10.0);
            @test length(λ) == 3
            @test λ[1] ≈ 0.0  # equal EMAs → neutral
            @test λ[2] > 0    # short < long → bearish
            @test λ[3] < 0    # short > long → bullish
        end

        @testset "compute_market_growth" begin
            prices = [100.0, 101.0, 102.0, 100.0];
            gm = compute_market_growth(prices; Δt=1.0/252.0);
            @test length(gm) == 3
            @test gm[1] > 0  # price went up
            @test gm[3] < 0  # price went down
        end

        @testset "compute_preference_weights" begin
            gamma = compute_preference_weights(SIM_PARAMS, TICKERS, 0.05, 0.0);
            @test length(gamma) == N_ASSETS
            @test all(-1.0 .< gamma .< 1.0)  # tanh output
        end

        @testset "allocate_cobb_douglas — budget conservation" begin
            gamma = [0.6, 0.4, -0.1];
            prices = [100.0, 50.0, 80.0];
            B = 10000.0;
            ε = 0.1;

            problem = build(MyCobbDouglasChoiceProblem, (
                gamma=gamma, prices=prices, B=B, epsilon=ε));
            (shares, cash) = allocate_cobb_douglas(problem);

            # total value should equal budget
            total = sum(shares .* prices) + cash;
            @test isapprox(total, B, rtol=1e-6)
            @test shares[3] == ε  # non-preferred gets epsilon
        end

        @testset "allocate_ces — budget conservation" begin
            gamma = [0.6, 0.4, 0.3];
            prices = [100.0, 50.0, 80.0];
            B = 10000.0;

            problem = build(MyCESChoiceProblem, (
                gamma=gamma, prices=prices, B=B, epsilon=0.1, sigma=2.0));
            (shares, cash) = allocate_ces(problem);

            total = sum(shares .* prices) + cash;
            @test isapprox(total, B, rtol=1e-4)
        end

        @testset "allocate_log_linear — matches Cobb-Douglas" begin
            gamma = [0.5, 0.3, 0.2];
            prices = [100.0, 50.0, 80.0];
            B = 10000.0;
            ε = 0.1;

            cd = build(MyCobbDouglasChoiceProblem, (gamma=gamma, prices=prices, B=B, epsilon=ε));
            ll = build(MyLogLinearChoiceProblem, (gamma=gamma, prices=prices, B=B, epsilon=ε));

            (s_cd, c_cd) = allocate_cobb_douglas(cd);
            (s_ll, c_ll) = allocate_log_linear(ll);

            @test isapprox(s_cd, s_ll, rtol=1e-10)
        end

        @testset "evaluate_cobb_douglas" begin
            shares = [10.0, 20.0, 5.0];
            gamma = [0.5, 0.3, 0.2];
            u = evaluate_cobb_douglas(shares, gamma);
            @test u > 0
        end

        @testset "evaluate_ces" begin
            shares = [10.0, 20.0, 5.0];
            gamma = [0.5, 0.3, 0.2];
            u = evaluate_ces(shares, gamma; sigma=2.0);
            @test u > 0
        end

        @testset "evaluate_log_linear" begin
            shares = [10.0, 20.0, 5.0];
            gamma = [0.5, 0.3, 0.2];
            u = evaluate_log_linear(shares, gamma);
            @test u > 0
        end
    end

    # ===================================================================
    @testset "Compute — Session 2 (Rebalancing Engine)" begin

        # build a small synthetic price matrix and run the engine
        T_total = 120;
        K = N_ASSETS;
        Δt = 1.0 / 252.0;

        # synthetic market prices (GBM) -
        Random.seed!(42);
        mkt = zeros(T_total);
        mkt[1] = 100.0;
        for t ∈ 2:T_total
            mkt[t] = mkt[t-1] * exp((0.08 - 0.5*0.18^2)*Δt + 0.18*sqrt(Δt)*randn());
        end

        ema_s = compute_ema(mkt; window=21);
        ema_l = compute_ema(mkt; window=63);
        λ = compute_lambda(ema_s, ema_l; G=10.0);
        gm_raw = compute_market_growth(mkt; Δt=Δt);
        gm_e = compute_ema(gm_raw; window=10);

        # price matrix with SIM-generated per-ticker prices -
        pmatrix = zeros(T_total, K + 1);
        pmatrix[:, 1] = 1:T_total;
        start_p = [150.0, 45.0, 80.0, 100.0, 60.0];
        for (k, ticker) ∈ enumerate(TICKERS)
            (αᵢ, βᵢ, σᵢ) = SIM_PARAMS[ticker];
            pmatrix[1, k+1] = start_p[k];
            for t ∈ 2:T_total
                gm_t = gm_raw[min(t-1, length(gm_raw))];
                gᵢ = αᵢ + βᵢ * gm_t * Δt + σᵢ * sqrt(Δt) * randn();
                pmatrix[t, k+1] = pmatrix[t-1, k+1] * exp(gᵢ);
            end
        end

        offset = 84;
        n_trading = T_total - offset;

        ctx = build(MyRebalancingContextModel, (
            B=10000.0, tickers=TICKERS, marketdata=pmatrix,
            marketfactor=gm_e, sim_parameters=SIM_PARAMS,
            lambda=0.0, Δt=Δt, epsilon=0.1));

        rules = build(MyTriggerRules, (
            max_drawdown=0.15, max_turnover=0.50,
            rebalance_schedule=ones(Int, n_trading)));

        @testset "run_rebalancing_engine" begin
            results = run_rebalancing_engine(ctx, rules, λ; offset=offset, allocator=:cobb_douglas);
            @test length(results) == n_trading + 1  # includes day 0
            @test haskey(results, 0)
            @test results[0].shares isa Array{Float64,1}
        end

        @testset "compute_wealth_series" begin
            results = run_rebalancing_engine(ctx, rules, λ; offset=offset);
            wealth = compute_wealth_series(results, pmatrix, TICKERS; offset=offset);
            @test length(wealth) == n_trading + 1
            @test wealth[1] > 0  # positive initial wealth
        end
    end

    # ===================================================================
    @testset "Compute — Session 3 (Backtest & Bandit)" begin

        @testset "generate_training_prices" begin
            prices = generate_training_prices(S₀=100.0, μ=0.08, σ=0.18, T=252, seed=42);
            @test length(prices) == 252
            @test prices[1] == 100.0
            @test all(prices .> 0)
        end

        @testset "compute_regret" begin
            rewards = [1.0, 2.0, 1.5, 3.0, 2.5];
            regret = compute_regret(rewards);
            @test length(regret) == 5
            @test regret[1] == 3.0 - 1.0  # best=3, first reward=1
            @test all(diff(regret) .>= 0)  # cumulative → non-decreasing
        end

        @testset "bandit_world" begin
            ctx = build(MyBanditContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS,
                prices=[150.0, 45.0, 80.0, 100.0, 60.0], B=10000.0,
                gm_t=0.05, lambda=0.0, epsilon=0.1));

            action = [1, 1, 0, 1, 0];  # include 3 of 5 assets
            (utility, shares, gamma) = bandit_world(action, ctx);

            @test utility isa Float64
            @test length(shares) == N_ASSETS
            @test length(gamma) == N_ASSETS
            @test shares[3] == 0.1  # excluded → epsilon
            @test shares[5] == 0.1  # excluded → epsilon
        end

        @testset "solve_bandit" begin
            Random.seed!(123);
            ctx = build(MyBanditContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS,
                prices=[150.0, 45.0, 80.0, 100.0, 60.0], B=10000.0,
                gm_t=0.05, lambda=0.0, epsilon=0.1));
            model = build(MyEpsilonGreedyBanditModel, (K=N_ASSETS, n_iterations=50, alpha=0.1));
            result = solve_bandit(model, ctx);

            @test result isa MyBanditResult
            @test length(result.best_action) == N_ASSETS
            @test all(result.best_action .∈ Ref([0, 1]))  # binary
            @test sum(result.best_action) >= 1  # at least one asset selected
            @test length(result.reward_history) == 50
        end
    end

    # ===================================================================
    @testset "Compute — Session 4 (Production)" begin

        @testset "generate_synthetic_sentiment" begin
            Random.seed!(42);
            mkt = cumsum(randn(200)) .+ 100.0;
            mkt = abs.(mkt);  # keep positive
            sent = generate_synthetic_sentiment(mkt; noise_σ=0.15, smoothing=5, seed=42);
            @test length(sent) == 200
            @test all(-1.0 .<= sent .<= 1.0)
        end

        @testset "check_escalation_triggers — no triggers" begin
            ctx = build(MyProductionContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS, B₀=10000.0,
                epsilon=0.1, max_drawdown=0.15, max_turnover=0.50,
                sentiment_threshold=-0.5, sentiment_override_lambda=2.0, max_bandit_churn=2));

            events = check_escalation_triggers(1, ctx, 10000.0, 10000.0, 0.2, [1,1,1,1,1], [1,1,1,1,1]);
            @test length(events) == 0
        end

        @testset "check_escalation_triggers — drawdown critical" begin
            ctx = build(MyProductionContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS, B₀=10000.0,
                epsilon=0.1, max_drawdown=0.15, max_turnover=0.50,
                sentiment_threshold=-0.5, sentiment_override_lambda=2.0, max_bandit_churn=2));

            events = check_escalation_triggers(1, ctx, 8000.0, 10000.0, 0.2, [1,1,1,1,1], [1,1,1,1,1]);
            @test length(events) >= 1
            @test any(e.severity == :critical for e ∈ events)
            @test any(e.trigger_type == "drawdown" for e ∈ events)
        end

        @testset "check_escalation_triggers — sentiment warning" begin
            ctx = build(MyProductionContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS, B₀=10000.0,
                epsilon=0.1, max_drawdown=0.15, max_turnover=0.50,
                sentiment_threshold=-0.5, sentiment_override_lambda=2.0, max_bandit_churn=2));

            events = check_escalation_triggers(1, ctx, 10000.0, 10000.0, -0.7, [1,1,1,1,1], [1,1,1,1,1]);
            @test any(e.trigger_type == "sentiment_crash" for e ∈ events)
            @test any(e.severity == :warning for e ∈ events)
        end

        @testset "check_escalation_triggers — bandit churn warning" begin
            ctx = build(MyProductionContext, (
                tickers=TICKERS, sim_parameters=SIM_PARAMS, B₀=10000.0,
                epsilon=0.1, max_drawdown=0.15, max_turnover=0.50,
                sentiment_threshold=-0.5, sentiment_override_lambda=2.0, max_bandit_churn=2));

            events = check_escalation_triggers(1, ctx, 10000.0, 10000.0, 0.2, [1,0,1,0,1], [0,1,0,1,0]);
            @test any(e.trigger_type == "bandit_churn" for e ∈ events)
        end

        @testset "compute_dashboard_metrics" begin
            # create minimal production results
            results = MyProductionDayResult[];
            for d ∈ 1:5
                r = MyProductionDayResult();
                r.day = d;
                r.shares = [10.0, 20.0, 5.0, 50.0, 8.0];
                r.cash = 100.0;
                r.wealth = 10000.0 + d * 100.0;
                r.gamma = [0.5, 0.3, 0.2, 0.8, 0.1];
                r.bandit_action = [1, 1, 1, 1, 1];
                r.sentiment = 0.1 * d;
                r.lambda = 0.0;
                r.rebalanced = true;
                r.escalated = false;
                push!(results, r);
            end

            metrics = compute_dashboard_metrics(results, MyEscalationEvent[]);
            @test metrics["n_days"] == 5
            @test metrics["final_wealth"] > 0
            @test metrics["n_escalations"] == 0
            @test haskey(metrics, "wealth_series")
        end
    end

    # ===================================================================
    @testset "Files — data accessors" begin

        @testset "MyTrainingMarketDataSet" begin
            data = MyTrainingMarketDataSet();
            @test data isa Dict
            @test haskey(data, "dataset")
        end

        @testset "MyTestingMarketDataSet" begin
            data = MyTestingMarketDataSet();
            @test data isa Dict
            @test haskey(data, "dataset")
        end

        @testset "MyMarketSurrogateModel" begin
            model = MyMarketSurrogateModel();
            @test model isa JumpHiddenMarkovModel
            @test length(model.emissions) > 0

            # generate a short path to verify it works -
            result = hmm_simulate(model, 50; n_paths=1);
            @test length(result.paths) == 1
            @test length(result.paths[1].observations) == 50
        end
    end

end # top-level testset

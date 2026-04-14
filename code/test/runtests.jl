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

        @testset "compute_adaptive_sigma" begin
            # neutral → σ_max
            @test compute_adaptive_sigma(0.0; σ_min=0.5, σ_max=5.0) ≈ 5.0

            # large |λ| → approaches σ_min
            σ_extreme = compute_adaptive_sigma(10.0; σ_min=0.5, σ_max=5.0);
            @test σ_extreme > 0.5
            @test σ_extreme < 1.0

            # monotonically decreasing in |λ|
            λ_vals = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
            σ_vals = [compute_adaptive_sigma(λ) for λ in λ_vals];
            @test all(diff(σ_vals) .< 0)

            # negative λ gives same result as positive (symmetric)
            @test compute_adaptive_sigma(-2.0) ≈ compute_adaptive_sigma(2.0)
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

        @testset "generate_hybrid_scenario" begin
            market_model = MyMarketSurrogateModel();
            portfolio    = MyPortfolioSurrogateModel();
            calib        = MySIMCalibration();

            calib_tickers = calib["tickers"];
            calib_r2      = calib["r_squared"];
            high_idx      = findfirst(==(1.0), calib_r2);
            high_ticker   = high_idx === nothing ?
                            calib_tickers[argmax(calib_r2)] : calib_tickers[high_idx];
            norm_ticker   = calib_tickers[findfirst(r -> 0.3 < r < 0.6, calib_r2)];
            test_tickers  = String[high_ticker, norm_ticker];

            scen = generate_hybrid_scenario(market_model, portfolio, calib, test_tickers;
                n_paths=50, n_steps=252, seed=1234, label="test");

            @test scen isa MyBacktestScenario
            @test size(scen.price_paths) == (50, 252, 2)
            @test size(scen.market_paths) == (50, 252)
            @test scen.n_paths == 50 && scen.n_steps == 252
            @test scen.label == "test"
            @test all(scen.price_paths .> 0.0)

            # β recovery on the moderate-R² ticker, averaged across paths
            Δt = 1.0/252.0;
            β_calib = calib["beta"][findfirst(==(norm_ticker), calib_tickers)];
            β_hats = Float64[];
            for p in 1:50
                mkt = scen.market_paths[p, :];
                tkr = scen.price_paths[p, :, 2];
                gm  = (1.0/Δt) .* log.(mkt[2:end] ./ mkt[1:end-1]);
                gt  = (1.0/Δt) .* log.(tkr[2:end] ./ tkr[1:end-1]);
                X   = hcat(ones(length(gm)), gm);
                θ̂   = (X' * X) \ (X' * gt);
                push!(β_hats, θ̂[2]);
            end
            @test isapprox(mean(β_hats), β_calib; atol=0.30)

            @test_throws ArgumentError generate_hybrid_scenario(
                market_model, portfolio, calib, String["__NOT_A_TICKER__"];
                n_paths=2, n_steps=5)
        end

        @testset "backtest_buyhold — custom weights" begin
            market_model = MyMarketSurrogateModel();
            portfolio    = MyPortfolioSurrogateModel();
            calib        = MySIMCalibration();
            calib_r2     = calib["r_squared"];
            norm_ticker  = calib["tickers"][findfirst(r -> 0.3 < r < 0.6, calib_r2)];
            tk           = ["SPY", norm_ticker];

            scen = generate_hybrid_scenario(market_model, portfolio, calib, tk;
                n_paths=20, n_steps=100, seed=4321);

            # default equal-weight must match explicit equal-weight
            r_default = backtest_buyhold(scen, tk; B₀=10_000.0, offset=1);
            r_eq      = backtest_buyhold(scen, tk; B₀=10_000.0, offset=1, weights=[0.5, 0.5]);
            @test isapprox(r_default.final_wealth, r_eq.final_wealth; atol=1e-10)

            # asymmetric weights should produce a different trajectory
            r_skew = backtest_buyhold(scen, tk; B₀=10_000.0, offset=1, weights=[0.9, 0.1]);
            @test !isapprox(r_default.final_wealth, r_skew.final_wealth; atol=1e-4)

            # non-sum-to-one rejected
            @test_throws AssertionError backtest_buyhold(scen, tk;
                B₀=10_000.0, offset=1, weights=[0.4, 0.4])
            # wrong length rejected
            @test_throws AssertionError backtest_buyhold(scen, tk;
                B₀=10_000.0, offset=1, weights=[1.0])
        end

        @testset "backtest_buyhold_market" begin
            market_model = MyMarketSurrogateModel();
            portfolio    = MyPortfolioSurrogateModel();
            calib        = MySIMCalibration();
            calib_r2     = calib["r_squared"];
            norm_ticker  = calib["tickers"][findfirst(r -> 0.3 < r < 0.6, calib_r2)];
            tk           = ["SPY", norm_ticker];

            scen = generate_hybrid_scenario(market_model, portfolio, calib, tk;
                n_paths=30, n_steps=120, seed=7777);

            r_mkt = backtest_buyhold_market(scen; B₀=10_000.0, offset=1);

            @test r_mkt isa MyBacktestResult
            @test r_mkt.strategy_label == "Market Buy-and-Hold"
            @test length(r_mkt.final_wealth) == 30
            @test length(r_mkt.max_drawdowns) == 30
            @test length(r_mkt.sharpe_ratios) == 30
            @test all(r_mkt.final_wealth .> 0.0)
            # The market buy-and-hold wealth at day 1 of any path equals B₀
            # by construction (we buy with the entire budget on day `offset`).
            @test all(r_mkt.max_drawdowns .>= 0.0)
            @test all(r_mkt.max_drawdowns .<= 1.0 + 1e-9)
        end

        @testset "compute_cvar" begin
            # Known case: 5 values, α = 0.4 → bottom 2 → mean(1, 2) = 1.5
            @test isapprox(compute_cvar([1.0, 2.0, 3.0, 4.0, 5.0]; α=0.4), 1.5; atol=1e-10)

            # 1..1000 with α=0.05 → bottom 50 → mean(1..50) = 25.5
            @test isapprox(compute_cvar(collect(1.0:1000.0); α=0.05), 25.5; atol=1e-10)

            # Permutation invariance
            v = randn(500) .+ 100.0;
            @test isapprox(compute_cvar(v; α=0.10), compute_cvar(reverse(v); α=0.10); atol=1e-10)

            # Floor at 1: very small α with small array still picks at least 1 element
            @test compute_cvar([10.0, 20.0]; α=0.01) == 10.0

            # Validation
            @test_throws ArgumentError compute_cvar(Float64[]; α=0.05)
            @test_throws ArgumentError compute_cvar([1.0, 2.0]; α=0.0)
            @test_throws ArgumentError compute_cvar([1.0, 2.0]; α=1.0)
            @test_throws ArgumentError compute_cvar([1.0, 2.0]; α=-0.1)
        end
    end

    # ===================================================================
    @testset "Compute — Session 3 (EWLS)" begin

        @testset "MyEWLSState construction" begin
            state = MyEWLSState();
            @test state isa MyEWLSState
        end

        @testset "ewls_init returns correct initial estimates" begin
            state = ewls_init(0.01, 1.2, 0.05; half_life=63.0, prior_weight=63.0);
            @test isapprox(state.α, 0.01; atol=1e-10)
            @test isapprox(state.β, 1.2; atol=1e-10)
            @test isapprox(state.σ_ε, 0.05; atol=1e-10)
            @test isapprox(state.η, 2.0^(-1.0/63.0); atol=1e-10)
            @test state.Sw > 0.0
        end

        @testset "ewls_update! converges on known linear data" begin
            # generate data from g_i = 0.01 + 1.2 * g_m + ε with small noise
            Random.seed!(42);
            α_true = 0.01;
            β_true = 1.2;
            σ_true = 0.005;
            n = 1000;
            gm = randn(n) * 0.05;  # market growth rates
            gi = α_true .+ β_true .* gm .+ σ_true .* randn(n);

            # start from a wrong initial guess with very small prior weight
            state = ewls_init(0.0, 1.0, 0.01; half_life=500.0, prior_weight=5.0);
            for t in 1:n
                ewls_update!(state, gi[t], gm[t]);
            end

            # after 1000 observations, should be close to true values
            @test isapprox(state.α, α_true; atol=0.01)
            @test isapprox(state.β, β_true; atol=0.15)
            @test state.σ_ε > 0.0
        end

        @testset "ewls_update! with η=1 recovers OLS" begin
            # with no decay, EWLS is exactly OLS
            Random.seed!(123);
            α_true = 0.02;
            β_true = 0.8;
            n = 500;
            gm = randn(n) * 0.04;
            gi = α_true .+ β_true .* gm .+ 0.003 .* randn(n);

            # EWLS with η = 1.0 (infinite half-life), zero prior
            state = ewls_init(0.0, 0.0, 0.0; half_life=Inf, prior_weight=0.0);
            for t in 1:n
                ewls_update!(state, gi[t], gm[t]);
            end

            # compare to direct OLS
            X = hcat(ones(n), gm);
            coeffs = X \ gi;
            @test isapprox(state.α, coeffs[1]; atol=1e-8)
            @test isapprox(state.β, coeffs[2]; atol=1e-8)
        end

        @testset "replay_engine_ewls produces valid results" begin
            Random.seed!(42);

            # generate synthetic price data: 300 days, 5 tickers + market
            n_days = 300;
            K = length(TICKERS);
            Δt = 1.0 / 252.0;

            # market prices (GBM)
            market_prices = zeros(n_days);
            market_prices[1] = 100.0;
            for t in 2:n_days
                market_prices[t] = market_prices[t-1] * exp((0.05 * Δt) + 0.15 * sqrt(Δt) * randn());
            end

            # ticker prices (SIM-based)
            price_matrix = zeros(n_days, K + 1);
            price_matrix[:, 1] = 1:n_days;
            for (k, ticker) in enumerate(TICKERS)
                (α, β, σ_ε) = SIM_PARAMS[ticker];
                price_matrix[1, k + 1] = 50.0;
                for t in 2:n_days
                    gm_t = (1.0 / Δt) * log(market_prices[t] / market_prices[t - 1]);
                    gi_t = α + β * gm_t + σ_ε * randn() / sqrt(Δt);
                    price_matrix[t, k + 1] = price_matrix[t - 1, k + 1] * exp(gi_t * Δt);
                end
            end

            rules_params = (max_drawdown = 0.25, max_turnover = 0.50);
            result = replay_engine_ewls(price_matrix, market_prices, TICKERS, SIM_PARAMS, rules_params;
                B₀ = 10000.0, offset = 63, half_life = 63.0);

            # basic sanity checks
            @test length(result.wealth) > 0
            @test all(result.wealth .> 0.0)
            @test haskey(result.results, 0)
            @test all(result.results[0].shares .>= 0.0)

            # param history should have entries for each ticker
            for ticker in TICKERS
                @test length(result.param_history[ticker]) > 0
            end
        end
    end

    # ===================================================================
    @testset "Compute — Session 3 (Sigma-Bandit)" begin

        @testset "classify_regime" begin
            @test classify_regime(1.0; θ=0.5) == :bearish
            @test classify_regime(-1.0; θ=0.5) == :bullish
            @test classify_regime(0.0; θ=0.5) == :neutral
            @test classify_regime(0.5; θ=0.5) == :neutral   # boundary
            @test classify_regime(-0.5; θ=0.5) == :neutral   # boundary
            @test classify_regime(0.51; θ=0.5) == :bearish
        end

        @testset "sigma_bandit_world — positive utility" begin
            # build a minimal context -
            T_test = 120;
            K_test = N_ASSETS;
            Δt = 1.0 / 252.0;
            Random.seed!(42);
            market_test = 100.0 .* exp.(cumsum(randn(T_test) * 0.01));
            gm_raw_test = compute_market_growth(market_test; Δt=Δt);
            gm_ema_test = compute_ema(gm_raw_test; window=10);
            pmatrix_test = zeros(T_test, K_test + 1);
            pmatrix_test[:, 1] = 1:T_test;
            for k in 1:K_test
                pmatrix_test[:, k+1] = 100.0 .* exp.(cumsum(randn(T_test) * 0.015));
            end

            ctx = build(MyRebalancingContextModel, (
                B=10000.0, tickers=TICKERS, marketdata=pmatrix_test,
                marketfactor=gm_ema_test, sim_parameters=SIM_PARAMS,
                lambda=0.0, Δt=Δt, epsilon=0.1
            ));

            u = sigma_bandit_world(2.0, ctx, 84);
            @test u > 0
            @test isfinite(u)
        end

        @testset "solve_sigma_bandit — returns valid result" begin
            T_test = 120;
            K_test = N_ASSETS;
            Δt = 1.0 / 252.0;
            Random.seed!(42);
            market_test = 100.0 .* exp.(cumsum(randn(T_test) * 0.01));
            gm_raw_test = compute_market_growth(market_test; Δt=Δt);
            gm_ema_test = compute_ema(gm_raw_test; window=10);
            ema_s_test = compute_ema(market_test; window=21);
            ema_l_test = compute_ema(market_test; window=63);
            λ_test = compute_lambda(ema_s_test, ema_l_test; G=10.0);
            pmatrix_test = zeros(T_test, K_test + 1);
            pmatrix_test[:, 1] = 1:T_test;
            for k in 1:K_test
                pmatrix_test[:, k+1] = 100.0 .* exp.(cumsum(randn(T_test) * 0.015));
            end

            ctx = build(MyRebalancingContextModel, (
                B=10000.0, tickers=TICKERS, marketdata=pmatrix_test,
                marketfactor=gm_ema_test, sim_parameters=SIM_PARAMS,
                lambda=0.0, Δt=Δt, epsilon=0.1
            ));

            σ_grid = [0.5, 1.0, 2.0, 3.0];
            bandit = build(MySigmaBanditModel, (
                sigma_grid=σ_grid, n_iterations=50, alpha=0.1, lambda_threshold=0.5
            ));

            result = solve_sigma_bandit(bandit, ctx, λ_test, collect(85:T_test));
            @test haskey(result.best_sigma_per_regime, :bearish)
            @test haskey(result.best_sigma_per_regime, :neutral)
            @test haskey(result.best_sigma_per_regime, :bullish)
            @test result.best_sigma_per_regime[:neutral] ∈ σ_grid
            @test length(result.reward_history) == length(85:T_test)
        end

        @testset "build_compliance_config" begin
            config = build_compliance_config(;
                concentration_cap=0.35, drawdown_gate=0.10,
                turnover_limit=0.40, position_size_limit=3000.0
            );
            @test config["concentration_cap"] ≈ 0.35
            @test config["drawdown_gate"] ≈ 0.10
            @test config["turnover_limit"] ≈ 0.40
            @test config["position_size_limit"] ≈ 3000.0
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

    # ===================================================================
    @testset "Compute — Session 4 (Live Production)" begin

        @testset "compute_live_sentiment" begin
            # rising prices → positive sentiment
            rising = collect(100.0:1.0:110.0);
            s = compute_live_sentiment(rising; lookback=5);
            @test -1.0 <= s <= 1.0
            @test s > 0.0

            # falling prices → negative sentiment
            falling = collect(110.0:-1.0:100.0);
            s = compute_live_sentiment(falling; lookback=5);
            @test s < 0.0

            # insufficient data → 0.0
            @test compute_live_sentiment([100.0, 101.0]; lookback=5) == 0.0
        end

        @testset "compute_position_drawdown" begin
            # at peak → 0
            @test compute_position_drawdown([100.0, 105.0, 110.0]) == 0.0

            # 10% off peak
            dd = compute_position_drawdown([100.0, 110.0, 99.0]);
            @test isapprox(dd, (110.0 - 99.0) / 110.0; atol=1e-10)

            # empty → 0
            @test compute_position_drawdown(Float64[]) == 0.0
        end

        @testset "run_production_step" begin
            Random.seed!(42);

            # build synthetic price data
            n_days = 200;
            K = length(TICKERS);
            Δt = 1.0 / 252.0;

            market_prices = zeros(n_days);
            market_prices[1] = 100.0;
            for t in 2:n_days
                market_prices[t] = market_prices[t-1] * exp(0.05 * Δt + 0.15 * sqrt(Δt) * randn());
            end

            price_matrix = zeros(n_days, K + 1);
            price_matrix[:, 1] = 1:n_days;
            for (k, ticker) in enumerate(TICKERS)
                (α, β, σ_ε) = SIM_PARAMS[ticker];
                price_matrix[1, k + 1] = 50.0;
                for t in 2:n_days
                    gm_t = (1.0 / Δt) * log(market_prices[t] / market_prices[t - 1]);
                    gi_t = α + β * gm_t + σ_ε * randn() / sqrt(Δt);
                    price_matrix[t, k + 1] = price_matrix[t - 1, k + 1] * exp(gi_t * Δt);
                end
            end

            # build context
            ctx = build(MyProductionContext, (
                tickers = TICKERS,
                sim_parameters = SIM_PARAMS,
                B₀ = 10000.0,
                epsilon = 0.1,
                max_drawdown = 0.25,
                max_turnover = 0.50,
                sentiment_threshold = -0.5,
                sentiment_override_lambda = 2.0,
                max_bandit_churn = 2
            ));

            # init EWLS states
            ewls_states = Dict{String,MyEWLSState}(
                t => ewls_init(SIM_PARAMS[t]...; half_life=63.0) for t in TICKERS
            );

            # run one step
            (result, events) = run_production_step(ctx, ewls_states, price_matrix, market_prices,
                TICKERS, 150;
                current_shares = fill(20.0, K), current_cash = 1000.0,
                peak_wealth = 12000.0);

            @test result isa MyLiveProductionDayResult
            @test result.day == 150
            @test length(result.shares) == K
            @test all(result.shares .>= 0.0)
            @test -1.0 <= result.sentiment <= 1.0
            @test length(result.bandit_action) == K
            @test all(a -> a in [0, 1], result.bandit_action)
            @test events isa Array{MyEscalationEvent,1}
        end

        @testset "apply_stress_scenario" begin
            ctx = build(MyProductionContext, (
                tickers = TICKERS,
                sim_parameters = SIM_PARAMS,
                B₀ = 10000.0,
                epsilon = 0.1,
                max_drawdown = 0.15,
                max_turnover = 0.50,
                sentiment_threshold = -0.5,
                sentiment_override_lambda = 2.0,
                max_bandit_churn = 2
            ));

            prices = [150.0, 45.0, 80.0, 100.0, 60.0];
            shares = [10.0, 20.0, 15.0, 50.0, 25.0];
            cash = 500.0;
            peak_wealth = sum(shares .* prices) + cash;
            prev_action = ones(Int, 5);

            # -30% market crash should trigger critical drawdown
            scenario = MyStressScenario();
            scenario.label = "Market -30%";
            scenario.market_shock = -0.30;
            scenario.ticker_shocks = Dict{String,Float64}();

            result = apply_stress_scenario(scenario, prices, shares, cash, ctx,
                peak_wealth, prev_action, TICKERS);

            @test result isa MyStressResult
            @test result.scenario_label == "Market -30%"
            @test result.stressed_wealth < peak_wealth
            @test result.drawdown > 0.15  # exceeds max_drawdown
            @test result.would_derisk == true
            @test length(result.triggers_fired) > 0
        end
    end

end # top-level testset

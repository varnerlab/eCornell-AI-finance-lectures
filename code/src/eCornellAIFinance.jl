module eCornellAIFinance

# import external packages -
using Clarabel
using CSV
using DataFrames
using Distributions
using FileIO
using HypothesisTests
using Ipopt
using JLD2
using JuMP
using JumpHMM
using LinearAlgebra
using Random
using Statistics
using StatsBase

# setup paths -
const _PATH_TO_SRC = dirname(@__FILE__);
const _PATH_TO_DATA = joinpath(_PATH_TO_SRC, "data");

# setup aliases for JumpHMM name collisions -
const hmm_fit = JumpHMM.fit;
const hmm_tune = JumpHMM.tune;
const hmm_simulate = JumpHMM.simulate;
const hmm_validate = JumpHMM.validate;
const hmm_forward_filter = JumpHMM.forward_filter;
const hmm_decode = JumpHMM.decode;

# include my codes -
include("Types.jl");
include("Factory.jl");
include("Compute.jl");
include("Files.jl");

# export types -
export MySIMParameterEstimate, MySharpeRatioPortfolioChoiceProblem
export MyPortfolioAllocationProblem, MyPortfolioPerformanceResult
export MyCobbDouglasChoiceProblem, MyCESChoiceProblem, MyLogLinearChoiceProblem
export MyRebalancingContextModel, MyTriggerRules, MyRebalancingResult
export MyBacktestScenario, MyBacktestResult, MyValidationReport
export MyBanditContext, MyEpsilonGreedyBanditModel, MyBanditResult
export MySentimentSignal, MyEscalationEvent, MyProductionDayResult, MyProductionContext

# export factory -
export build

# export compute — Session 1 -
export solve_minvariance, compute_drawdown, compute_turnover
export estimate_sim, bootstrap_sim, build_sim_covariance, solve_max_sharpe

# export compute — Session 2 -
export compute_ema, compute_lambda, compute_market_growth, compute_preference_weights
export allocate_cobb_douglas, allocate_ces, allocate_log_linear
export evaluate_cobb_douglas, evaluate_ces, evaluate_log_linear
export allocate_shares, run_rebalancing_engine, compute_wealth_series

# export compute — Session 3 -
export generate_training_prices, generate_hmm_scenario, generate_hybrid_scenario
export backtest_engine, backtest_buyhold, backtest_buyhold_market
export compute_cvar
export bandit_world, solve_bandit, compute_regret, backtest_bandit

# export compute — Session 4 -
export generate_synthetic_sentiment, check_escalation_triggers
export run_production_simulation, compute_dashboard_metrics

# export files -
export load_price_data, save_results, load_results
export save_production_results, load_production_results
export MyTrainingMarketDataSet, MyTestingMarketDataSet
export MyMarketSurrogateModel, MyPortfolioSurrogateModel, MySyntheticTrainingDataSet, MySIMCalibration, MyCurrentPrices

# export HMM aliases and types -
export hmm_fit, hmm_tune, hmm_simulate, hmm_validate
export hmm_forward_filter, hmm_decode
export JumpHiddenMarkovModel

end # module eCornellAIFinance

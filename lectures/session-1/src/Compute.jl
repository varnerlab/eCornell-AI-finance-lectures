"""
    solve_minvariance(problem::MyPortfolioAllocationProblem) -> MyPortfolioPerformanceResult

Solve the classical minimum-variance portfolio problem using JuMP.
Returns a `MyPortfolioPerformanceResult` with the optimal weights and portfolio statistics.
"""
function solve_minvariance(problem::MyPortfolioAllocationProblem)::MyPortfolioPerformanceResult

    # unpack -
    μ = problem.μ;
    Σ = problem.Σ;
    bounds = problem.bounds;
    R = problem.R;
    N = length(μ);

    # setup the JuMP model -
    model = Model();

    # decision variables: portfolio weights
    @variable(model, bounds[i,1] <= w[i=1:N] <= bounds[i,2]);

    # constraints -
    @constraint(model, sum(w) == 1.0);                  # fully invested
    @constraint(model, dot(μ, w) >= R);                  # target return

    # objective: minimize portfolio variance -
    @objective(model, Min, dot(w, Σ * w));

    # solve -
    optimize!(model);

    # package results -
    result = MyPortfolioPerformanceResult();
    result.weights = value.(w);
    result.expected_return = dot(μ, result.weights);
    result.variance = dot(result.weights, Σ * result.weights);

    # return -
    return result;
end

"""
    compute_drawdown(returns::Array{Float64,1}) -> Float64

Compute the maximum drawdown from a return series.
"""
function compute_drawdown(returns::Array{Float64,1})::Float64

    # compute cumulative wealth -
    wealth = cumprod(1.0 .+ returns);
    peak = accumulate(max, wealth);
    drawdowns = (peak .- wealth) ./ peak;

    # return max drawdown -
    return maximum(drawdowns);
end

"""
    compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1}) -> Float64

Compute portfolio turnover between two weight vectors.
"""
function compute_turnover(w_old::Array{Float64,1}, w_new::Array{Float64,1})::Float64
    return sum(abs.(w_new .- w_old));
end

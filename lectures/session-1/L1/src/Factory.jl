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

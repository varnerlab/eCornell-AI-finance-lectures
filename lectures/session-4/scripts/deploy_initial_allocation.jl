#!/usr/bin/env julia
#
# deploy_initial_allocation.jl -- One-shot bootstrap that deploys cash into
# the S1 minvar-allocation weights via Alpaca paper market orders.
#
# Why this exists: production_runner.jl is tuned for incremental rebalancing
# of an already-invested book. Its compliance gates (position_size_limit,
# turnover_limit) reject every trade when the account starts with $100K
# cash, so the engine queues all 22 names on cold start and submits zero.
# This script bypasses the gates exactly once to bring the book to S1
# weights; the cron then takes over for intraday rebalancing.
#
# Safety:
#   - Refuses to run if any positions already exist (assumes book is fresh)
#   - Cancels any open orders before submitting
#   - Deploys 95% of cash; 5% buffer for engine slippage / small rebalances
#   - Uses today's latest 30-min bar close as the price for share-count math
#

using Pkg;
const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance, Alpaca, Dates

const CASH_BUFFER = 0.95;
const S1_ALLOCATION_PATH = joinpath(SESSION_DIR, "..", "session-1", "data", "minvar-allocation.jld2");
const CREDS_PATH = joinpath(SESSION_DIR, "config", "credentials.toml");

function main()
    s1 = load_results(S1_ALLOCATION_PATH);
    tickers = String.(s1["my_tickers"]);
    weights = Float64.(s1["allocation_weights"]);
    @assert length(tickers) == length(weights);
    @assert isapprox(sum(weights), 1.0; atol = 1e-3) "weights do not sum to 1: $(sum(weights))";

    client = Alpaca.load_client(CREDS_PATH);

    clk = Alpaca.get_clock(client);
    clk.is_open || error("Market is closed (next open $(clk.next_open)). Run during RTH.");

    existing = Alpaca.list_positions(client);
    isempty(existing) || error("Account already has $(length(existing)) positions. Refusing to deploy on top of existing book. Liquidate first or run cron-driven rebalances instead.");

    println("Cancelling any open orders...");
    for o in Alpaca.list_orders(client; status = "open")
        try
            Alpaca.cancel_order(client, o.id);
        catch e
            println("  warn: cancel_order $(o.id) failed: $(e)");
        end
    end

    acct = Alpaca.get_account(client);
    cash = Float64(acct.cash);
    deployable = cash * CASH_BUFFER;
    println("Cash: \$$(round(cash, digits=2))   deployable (×$(CASH_BUFFER)): \$$(round(deployable, digits=2))");

    start_dt = DateTime(Date(now(UTC)), Time(13, 30, 0));   # 09:30 EDT in UTC
    finish_now = now(UTC);
    bars = Alpaca.get_bars(client, tickers, "30Min"; start = start_dt, finish = finish_now);

    println("\n--- Deploy plan ---");
    order_specs = Tuple{String,Int,Float64,Float64}[];
    total_notional = 0.0;
    for (i, t) in enumerate(tickers)
        if !haskey(bars, t) || isempty(bars[t])
            println("  $(t): NO BARS, skipping");
            continue;
        end
        price = bars[t][end].c;
        target_dollars = weights[i] * deployable;
        qty = Int(floor(target_dollars / price));
        # Filter zero or negative qty: zero-weight names (S1 QP returns -0.0 or
        # tiny negatives for unselected tickers) and sub-share allocations.
        if qty <= 0
            println("  $(t): qty=$(qty) (weight=$(round(weights[i], sigdigits=3))), skipping");
            continue;
        end
        notional = qty * price;
        push!(order_specs, (t, qty, price, weights[i]));
        total_notional += notional;
        println("  $(rpad(t, 5)) w=$(rpad(round(weights[i], digits=4), 7))  px=\$$(rpad(round(price, digits=2), 8))  qty=$(lpad(qty, 4))  notional=\$$(round(notional, digits=0))");
    end
    println("\nTotal deploy notional: \$$(round(total_notional, digits=0)) of \$$(round(deployable, digits=0)) deployable (residual cash: \$$(round(cash - total_notional, digits=0)))");

    println("\n--- Submitting orders ---");
    submitted = 0;
    submit_errors = String[];
    for (t, qty, _, _) in order_specs
        try
            order = Alpaca.submit_order(client, t, qty, "buy"; type = "market", time_in_force = "day");
            submitted += 1;
            println("  OK  $(rpad(t, 5)) qty=$(lpad(qty, 4))  id=$(order.id)");
        catch e
            push!(submit_errors, "$(t): $(e)");
            println("  ERR $(rpad(t, 5)) qty=$(lpad(qty, 4))  $(e)");
        end
    end
    println("\nSubmitted $(submitted)/$(length(order_specs)) orders.");
    isempty(submit_errors) || println("Errors:\n  " * join(submit_errors, "\n  "));

    println("\n--- Final positions ---");
    sleep(3);
    final_positions = Alpaca.list_positions(client);
    total_mv = 0.0;
    for p in sort(final_positions; by = x -> x.symbol)
        mv = Float64(p.market_value);
        total_mv += mv;
        println("  $(rpad(p.symbol, 5))  qty=$(lpad(p.qty, 5))  market_value=\$$(round(mv, digits=0))");
    end
    println("Total positions: $(length(final_positions))  market value: \$$(round(total_mv, digits=0))");
    acct2 = Alpaca.get_account(client);
    println("Account cash: \$$(round(Float64(acct2.cash), digits=2))   equity: \$$(round(Float64(acct2.equity), digits=2))");
end

main()

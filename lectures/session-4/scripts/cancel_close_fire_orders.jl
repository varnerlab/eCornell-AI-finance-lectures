#!/usr/bin/env julia
#
# cancel_close_fire_orders.jl -- Cancel close-fire orders that were submitted
# just after 16:00 ET and are sitting in "accepted" status at Alpaca paper.
#
# Why this exists: the 16:00 ET engine close fire submits market orders via
# `time_in_force=day`. Orders that hit the API after 16:00:00 ET are not
# rejected; Alpaca queues them and works them at the next 09:30 ET open.
# But `build_tomorrows_ticket` computes its proposed_trades from
# `list_positions` (filled-only state), so the 9:35 ET signed-ticket cron
# would re-submit the same trades. To avoid duplication at the open we
# cancel the close-fire leftovers tonight.
#
# Safety:
#   - Only cancels orders submitted after the cutoff (default: 19:55 UTC today)
#   - Lists every candidate before cancelling and prints a confirmation line
#   - --yes flag skips the interactive prompt (for cron / pipe use)
#

using Pkg;
const SCRIPT_DIR = @__DIR__;
const SESSION_DIR = dirname(SCRIPT_DIR);
Pkg.activate(SESSION_DIR; io = devnull);

using eCornellAIFinance, Alpaca, Dates, Printf

const CREDS_PATH = joinpath(SESSION_DIR, "config", "credentials.toml");

"""
    main(; cutoff_utc::DateTime = ..., auto_yes::Bool = false) -> Nothing

Cancel Alpaca paper orders submitted at or after `cutoff_utc` that are still
open. Default cutoff is 19:55 UTC today (15:55 ET), targeting the 16:00 ET
close-fire submissions. Returns nothing; side effect is canceled orders at
Alpaca and a stdout summary of what was canceled.

### Arguments
- `cutoff_utc::DateTime` (kw) — UTC timestamp; orders with `submitted_at >=
  cutoff_utc` are candidates for cancellation
- `auto_yes::Bool` (kw) — if true, skip the y/N prompt and proceed directly

### Side effects
- Reads Alpaca credentials from `config/credentials.toml`
- Calls `Alpaca.list_orders` and `Alpaca.cancel_order`
- Prints per-order status to stdout
"""
function main(; cutoff_utc::DateTime = DateTime(today(), Time(19, 55)),
              auto_yes::Bool = false)
    isfile(CREDS_PATH) || error("Credentials not found: $(CREDS_PATH)");
    client = Alpaca.load_client(CREDS_PATH);

    open_orders = Alpaca.list_orders(client; status = "open");
    to_cancel = filter(o -> o.submitted_at >= cutoff_utc, open_orders);

    @printf "Cutoff (UTC)         : %s\n" cutoff_utc
    @printf "Cutoff (approx ET)   : %s\n" Dates.format(cutoff_utc - Hour(4), "yyyy-mm-dd HH:MM")
    @printf "Open orders          : %d\n" length(open_orders)
    @printf "Candidates to cancel : %d\n" length(to_cancel)
    println();

    if isempty(to_cancel)
        println("Nothing to cancel.");
        return nothing;
    end

    for o in to_cancel
        @printf "  %s %4s %4d %-6s  submitted=%s  status=%s  id=%s\n" string(o.side) "" o.qty o.symbol o.submitted_at o.status o.id
    end
    println();

    if !auto_yes
        print("Proceed with cancellation? [y/N] ");
        ans = strip(readline());
        if lowercase(ans) != "y"
            println("Aborted; no cancellations sent.");
            return nothing;
        end
    end

    n_ok = 0;
    n_fail = 0;
    for o in to_cancel
        try
            Alpaca.cancel_order(client, o.id);
            @printf "  canceled %s %s %d %s\n" o.id string(o.side) o.qty o.symbol
            n_ok += 1;
        catch err
            @printf "  FAILED   %s: %s\n" o.id err
            n_fail += 1;
        end
    end
    println();
    @printf "Done: %d canceled, %d failed.\n" n_ok n_fail
    return nothing;
end

# Allow `--yes` from the command line for non-interactive use.
if abspath(PROGRAM_FILE) == @__FILE__
    auto_yes = "--yes" in ARGS;
    main(; auto_yes = auto_yes);
end

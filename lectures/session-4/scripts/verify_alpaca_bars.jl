# Pre-cron verification: confirm the new paper account + 30-minute bars endpoint.
# Run from lectures/session-4: julia --project=. scripts/verify_alpaca_bars.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Alpaca
using Dates

const CREDS_PATH = joinpath(@__DIR__, "..", "config", "credentials.toml")
const TICKERS = ["AAPL", "MSFT", "SPY"]
const INTERVAL = "30Min"

isfile(CREDS_PATH) || error("Credentials not found: $(CREDS_PATH)")
client = Alpaca.load_client(CREDS_PATH)

println("─── account ───")
acct = Alpaca.get_account(client)
println("account_number = $(acct.account_number)")
println("status         = $(acct.status)")
println("cash           = $(acct.cash)")
println("equity         = $(acct.equity)")

println("\n─── clock ───")
clk = Alpaca.get_clock(client)
println("is_open    = $(clk.is_open)")
println("timestamp  = $(clk.timestamp)")
println("next_open  = $(clk.next_open)")
println("next_close = $(clk.next_close)")

println("\n─── 30Min bars (last 24h) ───")
fire_time = now()
start_dt  = fire_time - Day(4)  # 4 days back so weekend/pre-market still picks up Friday

bars = Alpaca.get_bars(client, TICKERS, INTERVAL; start = start_dt, finish = fire_time)

for sym in TICKERS
    arr = get(bars, sym, nothing)
    if arr === nothing || isempty(arr)
        println("$(sym): NO BARS RETURNED")
    else
        first_b = arr[1]
        last_b  = arr[end]
        println("$(sym): $(length(arr)) bars  | first $(first_b.t) close=$(first_b.c)  | last $(last_b.t) close=$(last_b.c)")
    end
end

println("\nverification complete.")

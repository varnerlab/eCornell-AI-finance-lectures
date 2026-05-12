cd(joinpath(@__DIR__, ".."))
include(joinpath(pwd(), "Include.jl"))

review_date = Date("2026-05-11")
exec_date   = review_date + Day(1)
exec_str    = Dates.format(exec_date, "yyyy-mm-dd")

TICKET_PATH = joinpath("data", "tickets", "ticket-$(exec_str).jld2")
SIGNED_PATH = joinpath("data", "tickets", "signed-$(exec_str).jld2")

ticket = load_ticket(TICKET_PATH)

# Approve all as-is — no modifications.
modifications = NamedTuple[]

signed_by = "Jeffrey Varner"
signed_at = now()

signed = build(MySignedTicket, (
    ticket        = ticket,
    modifications = modifications,
    signed_by     = signed_by,
    signed_at     = signed_at,
))

mkpath(dirname(SIGNED_PATH))
save_signed_ticket!(SIGNED_PATH, signed)
println("Wrote signed ticket with $(length(modifications)) modifications to $SIGNED_PATH.")

# Verify by reloading
loaded = load_signed_ticket(SIGNED_PATH)
println()
println("=== Verification: reloaded signed ticket ===")
println("Signed by             : $(loaded.signed_by)")
println("Signed at             : $(loaded.signed_at)")
println("Modifications         : $(length(loaded.modifications))")
println("Underlying ticket gen : $(loaded.ticket.generated_at)")
println("Proposed trades count : $(length(loaded.ticket.proposed_trades))")
println("Tickers in ticket     : $(length(loaded.ticket.tickers))")

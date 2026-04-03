using Documenter
using eCornellAIFinance

makedocs(
    sitename = "eCornell AI Finance",
    modules = [eCornellAIFinance],
    pages = [
        "Home" => "index.md",
        "Session 1: Portfolio Optimization" => "session1.md",
        "Session 2: AI Rebalancing Engine" => "session2.md",
        "Session 3: HMM Backtesting & Bandits" => "session3.md",
        "Session 4: Production Operations" => "session4.md",
    ],
    format = Documenter.HTML(
        prettyurls = false,
    ),
)

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
        # Pretty URLs only in CI. Locally the build lands next to the source
        # so developers can open build/session1.html directly in a browser.
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)

deploydocs(
    repo = "github.com/varnerlab/eCornell-AI-finance-lectures.git",
    devbranch = "main",
    push_preview = false,
)

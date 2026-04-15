# Let's Find Your Tickers

This is an **interactive questionnaire** that helps you build a ticker universe tailored to your investment profile. Answer the six questions below, match your answers to one of five archetypes, then copy the archetype's ticker list, budget, and concentration cap into two configuration files that every notebook in the course reads from.

You can do this yourself using the matching table in Step 7, or paste your answers into a Claude chat and ask for a recommendation. Either way, the output is the same: a `my-tickers.csv` and a `portfolio-config.toml` that drive the entire S1 to S4 pipeline.

## Step 1: What is your risk tolerance?

Pick the statement that sounds most like you:

- **(a) Risk-averse.** "I want to protect my capital. I would rather give up some upside than experience a 30% drawdown."
- **(b) Balanced.** "I can tolerate moderate swings if the long-term return justifies it. A 15-20% drawdown is uncomfortable but not a deal-breaker."
- **(c) Risk-tolerant.** "I understand markets move. I can sit through large drawdowns without selling, because I believe in the long-term trend."

## Step 2: What is your time horizon?

How long can this money stay invested before you need it?

- **(a) Short: under 3 years.** "I may need this money soon, for a house, tuition, or other near-term expense."
- **(b) Medium: 3 to 10 years.** "This is for a goal that is somewhat distant but has a specific date, such as a planned purchase or a child's education."
- **(c) Long: 10+ years.** "This is retirement or generational wealth. I do not expect to withdraw for at least a decade."

## Step 3: What is the primary objective?

What does this portfolio need to do?

- **(a) Capital preservation.** "Do not lose money. Return of capital matters more than return on capital."
- **(b) Income.** "Generate steady cash flow through dividends, preferably growing over time."
- **(c) Growth.** "Maximize long-term total return. Reinvest everything."
- **(d) Balanced.** "A mix of growth and income, with moderate volatility."

## Step 4: Are there any exclusions?

Sectors, industries, or specific companies you want to avoid for ethical, regulatory, or personal reasons.

Common examples:
- Tobacco (MO, PM)
- Fossil fuels (XOM, CVX, COP)
- Weapons and defense (LMT, RTX, NOC)
- Specific companies (often for governance or reputational concerns)

List your exclusions below. If none, write "none."

```
Exclusions: _____________________________________
```

## Step 5: What is your initial budget?

The dollar amount you want to invest. This loads into the notebooks as `B_0` (initial wealth).

```
Initial budget (USD): ____________
```

A few notes:
- The course uses the same relative math regardless of budget, so results scale linearly. A $10,000 portfolio and a $10,000,000 portfolio produce the same percentage returns and drawdowns.
- Position sizing and transaction costs are what differ at scale. With $10,000, you can own fractional shares of all 22 names. With $1,000, you may need fewer names.

## Step 6: What is your single-name concentration cap?

Unconstrained minimum-variance QPs routinely put 30-50% into one or two low-volatility names. That is a mathematical optimum, not a policy most investors can live with. Institutional mandates typically cap single-name exposure to protect against idiosyncratic blowups.

Pick a cap that matches your book size and risk policy:

- **(a) Tight (5-10%).** Diversified mandates, broad universes (20+ names), fiduciary accounts.
- **(b) Standard (15-25%).** Most actively managed portfolios. Balances concentration against diversification.
- **(c) Loose (30-50%).** Concentrated high-conviction books, 5-10 names total.
- **(d) Unconstrained (100%).** Research and teaching only. Expect single-name concentration.

The cap loads into the notebooks as `max_weight` and is enforced as `w_i <= max_weight` in the min-variance QP. Must satisfy `max_weight >= 1/N` or the problem is infeasible.

```
Concentration cap (decimal, e.g., 0.20 for 20%): ____________
```

## Step 7: Match Your Profile to an Archetype

Use the table below. Find the row that most closely matches your answers to Steps 1-3. The archetype suggests a reasonable default for Step 6 (concentration cap): Conservative books lean tight, Aggressive Growth books can tolerate looser caps.

| Archetype | Risk | Horizon | Objective | Character |
|-----------|------|---------|-----------|-----------|
| **Conservative Income** | risk-averse | short or medium | income or preservation | Retirees, capital preservation with yield |
| **Conservative Growth** | risk-averse | long | growth or balanced | Long-horizon defensive investor who still wants compounding |
| **Balanced** | balanced | any | balanced | The middle-of-the-road 60/40 type allocation in equity form |
| **Growth-Oriented** | balanced or risk-tolerant | medium or long | growth | Quality growth with some defensive ballast |
| **Aggressive Growth** | risk-tolerant | long | growth | High-beta tech-heavy, volatility is the price of admission |

If your answers span rows, pick the closest. The archetype is a starting point, not a prescription.

## Step 8: Copy Your Archetype's Ticker List

Each archetype below lists ~20-25 tickers spanning the 11 GICS sectors. All tickers are pre-verified against `sim-calibration.jld2` (424 S&P 500 names, 2014-2024). Apply your Step 4 exclusions after copying. The default `max_weight` suggested next to each archetype assumes you keep the full list; tighten or loosen based on your Step 6 answer.

### Conservative Income (20 tickers)
Heavy on dividend aristocrats, utilities, REITs, and staples. Low beta throughout.

```csv
ticker,sector
VZ,Communication Services
T,Communication Services
MCD,Consumer Discretionary
PG,Consumer Staples
KO,Consumer Staples
PEP,Consumer Staples
WMT,Consumer Staples
XOM,Energy
CVX,Energy
JPM,Financials
BRK.B,Financials
JNJ,Health Care
MRK,Health Care
HON,Industrials
UPS,Industrials
AAPL,Information Technology
MSFT,Information Technology
APD,Materials
AMT,Real Estate
NEE,Utilities
```

### Conservative Growth (24 tickers)
Quality compounders with a defensive tilt. Some growth exposure via large-cap tech and healthcare.

```csv
ticker,sector
DIS,Communication Services
VZ,Communication Services
AMZN,Consumer Discretionary
HD,Consumer Discretionary
MCD,Consumer Discretionary
PG,Consumer Staples
COST,Consumer Staples
WMT,Consumer Staples
XOM,Energy
CVX,Energy
JPM,Financials
BRK.B,Financials
V,Financials
JNJ,Health Care
UNH,Health Care
LLY,Health Care
HON,Industrials
UPS,Industrials
AAPL,Information Technology
MSFT,Information Technology
AVGO,Information Technology
SHW,Materials
AMT,Real Estate
NEE,Utilities
```

### Balanced (22 tickers)
Even distribution across sectors and beta buckets. No heavy tilts.

```csv
ticker,sector
DIS,Communication Services
VZ,Communication Services
AMZN,Consumer Discretionary
HD,Consumer Discretionary
PG,Consumer Staples
COST,Consumer Staples
XOM,Energy
CVX,Energy
JPM,Financials
V,Financials
BAC,Financials
JNJ,Health Care
UNH,Health Care
LLY,Health Care
HON,Industrials
CAT,Industrials
AAPL,Information Technology
MSFT,Information Technology
NVDA,Information Technology
SHW,Materials
AMT,Real Estate
NEE,Utilities
```

### Growth-Oriented (22 tickers)
Tilts toward tech and consumer discretionary, lighter on defensives.

```csv
ticker,sector
DIS,Communication Services
NFLX,Communication Services
AMZN,Consumer Discretionary
HD,Consumer Discretionary
TJX,Consumer Discretionary
COST,Consumer Staples
CVX,Energy
JPM,Financials
V,Financials
MA,Financials
UNH,Health Care
LLY,Health Care
ABBV,Health Care
HON,Industrials
CAT,Industrials
AAPL,Information Technology
MSFT,Information Technology
NVDA,Information Technology
AVGO,Information Technology
CRM,Information Technology
SHW,Materials
AMT,Real Estate
```

### Aggressive Growth (20 tickers)
High-beta tech-heavy. Minimal defensive exposure. Prepare for volatility.

```csv
ticker,sector
NFLX,Communication Services
AMZN,Consumer Discretionary
NKE,Consumer Discretionary
TSLA,Consumer Discretionary
COST,Consumer Staples
SLB,Energy
JPM,Financials
V,Financials
MA,Financials
LLY,Health Care
ABBV,Health Care
CAT,Industrials
AAPL,Information Technology
MSFT,Information Technology
NVDA,Information Technology
AVGO,Information Technology
CRM,Information Technology
ADBE,Information Technology
FCX,Materials
AMT,Real Estate
```

## Step 9: Save Your Configuration

The S1 notebooks read from two files in `lectures/session-1/data/`:

**File 1: `my-tickers.csv`** — the ticker list from Step 7 (minus your Step 4 exclusions).

**File 2: `portfolio-config.toml`** — the budget from Step 5 plus your risk-free rate and target growth rate.

You can write both files with any text editor. Alternatively, use Julia:

```julia
using CSV, DataFrames, TOML

# --- Write tickers ---
# Replace with your chosen archetype's list from Step 7
tickers_df = DataFrame(
    ticker = ["DIS", "VZ", "AMZN", "HD", "MCD", "PG", "COST", "WMT",
              "XOM", "CVX", "JPM", "BRK.B", "V", "JNJ", "UNH", "LLY",
              "HON", "UPS", "AAPL", "MSFT", "AVGO", "SHW", "AMT", "NEE"],
    sector = ["Communication Services", "Communication Services",
              "Consumer Discretionary", "Consumer Discretionary", "Consumer Discretionary",
              "Consumer Staples", "Consumer Staples", "Consumer Staples",
              "Energy", "Energy",
              "Financials", "Financials", "Financials",
              "Health Care", "Health Care", "Health Care",
              "Industrials", "Industrials",
              "Information Technology", "Information Technology", "Information Technology",
              "Materials", "Real Estate", "Utilities"]
)
CSV.write(joinpath(_PATH_TO_DATA, "my-tickers.csv"), tickers_df)

# --- Write portfolio config ---
config = Dict(
    "portfolio" => Dict(
        "initial_budget" => 10_000.0,           # from Step 5
        "risk_free_rate" => 0.045,              # 4.5%/yr
        "target_growth" => 0.10,                # 10%/yr
        "max_weight" => 0.20,                   # from Step 6 (concentration cap)
    ),
    "profile" => Dict(
        "risk_tolerance" => "risk-averse",      # from Step 1
        "time_horizon" => "long",                # from Step 2
        "primary_objective" => "growth",         # from Step 3
        "archetype" => "Conservative Growth",    # from Step 7
    ),
)
open(joinpath(_PATH_TO_DATA, "portfolio-config.toml"), "w") do io
    TOML.print(io, config)
end

println("Wrote my-tickers.csv and portfolio-config.toml")
```

## Step 10: Verify Tickers Exist in the Calibration File

After writing the files, run this check to confirm all tickers are in `sim-calibration.jld2`:

```julia
using eCornellAIFinance, CSV, DataFrames

calib = MySIMCalibration()
tickers_available = calib["tickers"]

tickers_df = CSV.read(joinpath(_PATH_TO_DATA, "my-tickers.csv"), DataFrame)
my_tickers = String.(tickers_df.ticker)

println("Total tickers in calibration: ", length(tickers_available))
println("\n--- Availability check ---")
missing_tickers = String[]
for t in my_tickers
    found = t in tickers_available
    status = found ? "✓" : "✗"
    println("  $status $t")
    found || push!(missing_tickers, t)
end

if !isempty(missing_tickers)
    println("\nMissing: $(missing_tickers)")
    println("Edit my-tickers.csv to substitute these with available names.")
else
    println("\nAll $(length(my_tickers)) tickers available.")
end
```

If any ticker is missing, substitute it with another from the same sector (see the archetype list above for alternatives). Common substitution reasons:

- The ticker was not in the S&P 500 during 2014-2024.
- The symbol format differs (e.g., `BRK.B` vs `BRK-B`).

## Step 11: Choose Your Construction Notebook

Session 1 has two portfolio construction notebooks. Which one you run depends on your archetype:

| Archetype | Notebook | Construction Method | cash_fraction |
|-----------|----------|---------------------|---------------|
| Conservative Income | **RRFA** (`MinVariancePortfolio-RRFA`) | Tangent portfolio with cash position | 0.30 |
| Conservative Growth | **RRFA** (`MinVariancePortfolio-RRFA`) | Tangent portfolio with cash position | 0.10 |
| Balanced | **RA** (`BuildMinVariancePortfolio-RA`) | Minimum-variance, fully invested | 0.00 |
| Growth-Oriented | **RA** (`BuildMinVariancePortfolio-RA`) | Minimum-variance, fully invested | 0.00 |
| Aggressive Growth | **RA** (`BuildMinVariancePortfolio-RA`) | Minimum-variance, fully invested | 0.00 |

**Why the split:**
- **RA** (risky assets only) solves the minimum-variance QP with all capital invested in the risky portfolio. Use this when you want every dollar working in the market.
- **RRFA** (risky + risk-free asset) solves the Sharpe-maximizing tangent portfolio and combines it with a cash position along the Capital Market Line. Use this when you want some dry powder.

Both notebooks save to the same downstream file (`minvar-allocation.jld2`) with a `cash_fraction` field, so Session 2/3/4 work identically regardless of which you chose. Set your `cash_fraction` in `portfolio-config.toml` before running RRFA.

## How the Notebooks Use These Files

The S1 notebooks read both files at startup:

```julia
# Tickers
tickers_df = CSV.read(joinpath(_PATH_TO_DATA, "my-tickers.csv"), DataFrame)
my_tickers = String.(tickers_df.ticker)
N = length(my_tickers)

# Portfolio configuration
cfg = TOML.parsefile(joinpath(_PATH_TO_DATA, "portfolio-config.toml"))
B₀         = cfg["portfolio"]["initial_budget"]
g_f        = cfg["portfolio"]["risk_free_rate"]
R_target   = cfg["portfolio"]["target_growth"]
max_weight = cfg["portfolio"]["max_weight"]    # concentration cap, enforced in QP bounds
```

Every downstream session (S2 rebalancing, S3 online learning, S4 production) inherits these values via the saved `minvar-allocation.jld2` artifact. A single edit propagates through the entire pipeline.

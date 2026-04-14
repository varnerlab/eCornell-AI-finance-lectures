# Let's Find Your Tickers

This procedure walks through how to build a ticker universe for the course. The output is a list of 20-25 S&P 500 tickers that you will carry through all four sessions: Session 1 minimum-variance portfolio, Session 2 rebalancing engine, Session 3 online learning and validation, and Session 4 production deployment.

## Step 1: Define Your Investor Profile

Before picking names, write down three things:

1. **Risk tolerance.** Risk-averse (prefer staples, utilities, healthcare), risk-neutral (balanced), or risk-seeking (prefer tech, discretionary, small caps).
2. **Time horizon.** Short (under 3 years), medium (3-10 years), or long (10+ years). Longer horizons tolerate more volatility.
3. **Exclusions.** Any tickers or subsectors you want to avoid for ethical, regulatory, or personal reasons (e.g., tobacco, fossil fuels, specific companies).

The combination of profile and exclusions drives the selection. A risk-averse long-horizon investor leans toward quality compounders with durable moats; a risk-seeking short-horizon investor leans toward beta and momentum names.

## Step 2: Cover the 11 GICS Sectors

A diversified equity portfolio has at least one name from each GICS sector. Here are the 11 sectors with 2-4 recognizable large-cap candidates each. Pick 1-3 per sector based on your profile.

| Sector | Candidate Tickers | Profile Note |
|--------|------------------|--------------|
| Communication Services | DIS, VZ, T, NFLX, META, GOOGL | VZ/T defensive, rest higher beta |
| Consumer Discretionary | AMZN, HD, MCD, NKE, SBUX, TJX | MCD most defensive |
| Consumer Staples | PG, COST, WMT, KO, PEP, CL | All defensive, dividend aristocrats |
| Energy | XOM, CVX, COP, SLB | Cyclical, commodity-sensitive |
| Financials | JPM, BRK.B, V, MA, BAC, GS | BRK.B lowest beta, BAC/GS higher |
| Health Care | JNJ, UNH, LLY, PFE, ABBV, MRK | JNJ/UNH defensive, LLY growth |
| Industrials | HON, UPS, CAT, GE, BA, LMT | HON/LMT most stable |
| Information Technology | AAPL, MSFT, NVDA, AVGO, CRM, ADBE | NVDA highest beta |
| Materials | SHW, APD, ECL, FCX, NUE, PPG | SHW/APD/ECL defensive |
| Real Estate | AMT, PLD, CCI, EQIX, SPG, O | AMT/EQIX infrastructure |
| Utilities | NEE, DUK, SO, AEP | All low beta |

**Target count:** 20-25 tickers total. With 11 sectors, that averages 2 per sector, with extras in sectors you want to emphasize.

## Step 3: Balance Beta Across the Portfolio

A well-constructed portfolio has a spread of market betas so the Single Index Model (SIM) covariance matrix is well-conditioned and the optimizer has real choices to make.

**Rough target mix for a risk-averse long-horizon portfolio:**
- ~40% low beta (β < 1): staples, healthcare, utilities, REITs
- ~40% market beta (β ≈ 1): financials, industrials, energy
- ~20% high beta (β > 1): tech, consumer discretionary

**Rough target mix for a risk-seeking portfolio:** shift toward 20/40/40.

## Step 4: Verify Tickers Are in the Calibration File

The course uses `sim-calibration.jld2`, which contains pre-computed SIM parameters (α, β, σ_ε) for 424 S&P 500 stocks calibrated from 2014-2024 data. Every ticker you pick must be in this file.

Run the following Julia script from the `code/` directory to check your candidate list:

```julia
using eCornellAIFinance

# Load the calibration file
calib = MySIMCalibration()
tickers_available = calib["tickers"]

println("Total tickers in calibration: ", length(tickers_available))

# Your candidate list
candidates = [
    "DIS", "VZ",                      # Communication Services
    "AMZN", "HD", "MCD",              # Consumer Discretionary
    "PG", "COST", "WMT",              # Consumer Staples
    "XOM", "CVX",                     # Energy
    "JPM", "BRK.B", "V",              # Financials
    "JNJ", "UNH", "LLY",              # Health Care
    "HON", "UPS",                     # Industrials
    "AAPL", "MSFT", "AVGO",           # Information Technology
    "SHW",                            # Materials
    "AMT",                            # Real Estate
    "NEE",                            # Utilities
]

println("\n--- Availability check ---")
for t in candidates
    found = t in tickers_available
    status = found ? "✓" : "✗"
    println("  $status $t")
end
```

## Step 5: Substitute Missing Tickers

If any candidate prints `✗`, the ticker is not in the calibration file. This usually happens because:

- The ticker was not in the S&P 500 during the 2014-2024 calibration window.
- The company had a corporate action (spinoff, merger) that disqualified it.
- The symbol format differs (e.g., `BRK.B` vs `BRK-B`).

To find substitutes in the same sector, check a broader candidate list:

```julia
# Example: find materials alternatives
for t in ["SHW", "APD", "ECL", "FCX", "NEM", "NUE", "PPG", "PKG", "MLM", "VMC"]
    println("  ", t in tickers_available ? "✓" : "✗", " ", t)
end
```

Pick the substitute that best fits your risk profile for that sector.

## Step 6: Save Your Final Universe

Once all tickers pass the availability check, save them to a variable you will reuse across notebooks. The Session 1 example notebooks expect a `my_tickers::Vector{String}` global.

```julia
my_tickers = [
    "DIS", "VZ",
    "AMZN", "HD", "MCD",
    "PG", "COST", "WMT",
    "XOM", "CVX",
    "JPM", "BRK.B", "V",
    "JNJ", "UNH", "LLY",
    "HON", "UPS",
    "AAPL", "MSFT", "AVGO",
    "SHW",
    "AMT",
    "NEE",
]

println("Universe: $(length(my_tickers)) tickers across 11 sectors")
```

## Worked Example

Starting profile:
- Risk-averse
- Long time horizon
- Exclude META and GOOGL

Result: 22 tickers, all in the calibration file after substituting **LIN → SHW** in Materials. Sector coverage: 11/11. Beta mix: roughly 40% low, 40% market, 20% high.

This universe carries through S1 (minimum-variance allocation), S2 (adaptive rebalancing), S3 (EWLS online learning, sigma-bandit per regime), and S4 (production deployment with compliance gates).

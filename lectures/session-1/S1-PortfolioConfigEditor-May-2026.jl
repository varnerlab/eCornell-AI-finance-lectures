### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a0000002-0000-0000-0000-000000000002
begin
	# Use session-1's Project.toml so the local-path eCornellAIFinance
	# resolves via the [sources] block (../../code). Calling Pkg.activate
	# here also signals Pluto to disable its internal package manager.
	import Pkg
	Pkg.activate(@__DIR__)
	using PlutoUI
	using HypertextLiteral
	using CSV
	using DataFrames
	using TOML
	using Statistics
	using eCornellAIFinance
end

# ╔═╡ a0000020-0000-0000-0000-000000000020
html"""
<style>
main { max-width: 1200px !important; margin-right: auto !important; }
pluto-notebook { padding-right: 0 !important; }
.s1ed { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; color: #18181b; }
.s1ed .hero { background: linear-gradient(135deg, #B31B1B 0%, #7F1D1D 100%); color: white !important; padding: 28px 32px; border-radius: 14px; margin: 8px 0 24px 0; box-shadow: 0 4px 12px rgba(179,27,27,0.15); }
.s1ed .hero h1, .s1ed .hero h1 a { margin: 0 !important; font-size: 26px !important; font-weight: 700 !important; letter-spacing: -0.3px !important; color: white !important; border: none !important; padding: 0 !important; }
.s1ed .hero .subtitle { margin-top: 8px; font-size: 13px; opacity: 0.95; line-height: 1.5; color: white !important; }
.s1ed .hero .subtitle b { color: white !important; }
.s1ed .hero code { background: rgba(255,255,255,0.22); color: white !important; padding: 1px 6px; border-radius: 4px; font-size: 12px; }
.s1ed .card { background: white; border: 1px solid #e4e4e7; border-radius: 12px; padding: 20px 24px; margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.s1ed .card h2 { margin: 0 0 16px 0; font-size: 11px; font-weight: 700; color: #B31B1B; text-transform: uppercase; letter-spacing: 1px; padding-bottom: 8px; border-bottom: 2px solid #FEE2E2; }
.s1ed .card h3 { margin: 16px 0 8px 0; font-size: 11px; font-weight: 700; color: #71717A; text-transform: uppercase; letter-spacing: 0.8px; }
.s1ed .row { display: grid; grid-template-columns: 220px 1fr; align-items: center; gap: 14px; padding: 8px 0; border-bottom: 1px solid #f4f4f5; }
.s1ed .row:last-child { border-bottom: none; }
.s1ed .row label { font-size: 13px; font-weight: 500; color: #3f3f46; }
.s1ed .row select, .s1ed .row input { font-size: 13px; }
.s1ed .pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.s1ed .pill-ok { background: #DCFCE7; color: #15803D; }
.s1ed .pill-err { background: #FEE2E2; color: #B91C1C; }
.s1ed .pill-arch { background: #B31B1B; color: white; padding: 4px 12px; }
.s1ed .check-row { display: flex; align-items: flex-start; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #f4f4f5; gap: 12px; }
.s1ed .check-row:last-child { border-bottom: none; }
.s1ed .check-row .label { font-size: 13px; color: #18181b; }
.s1ed .check-row .detail { font-size: 11px; color: #71717a; font-family: ui-monospace, monospace; margin-top: 2px; }
.s1ed .check-row .right { text-align: right; flex-shrink: 0; }
.s1ed .verdict { margin-top: 14px; padding: 12px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; }
.s1ed .verdict-ok { background: #DCFCE7; border: 1px solid #86EFAC; color: #14532D; }
.s1ed .verdict-err { background: #FEE2E2; border: 1px solid #FCA5A5; color: #7F1D1D; }
.s1ed .chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.s1ed .chip { background: #F4F4F5; padding: 3px 9px; border-radius: 5px; font-size: 11px; font-family: ui-monospace, monospace; color: #18181b; border: 1px solid #e4e4e7; }
.s1ed .chip-extra { background: #FEF3C7; color: #78350F; border-color: #FCD34D; }
.s1ed .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
.s1ed .metric { padding: 12px 14px; background: #FAFAF9; border-radius: 8px; border: 1px solid #f4f4f5; }
.s1ed .metric .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px; color: #71717a; font-weight: 600; }
.s1ed .metric .val { font-size: 18px; font-weight: 700; font-family: ui-monospace, monospace; color: #18181b; margin-top: 2px; }
.s1ed .feas-bar { position: relative; height: 8px; background: #F4F4F5; border-radius: 4px; margin: 16px 0 6px 0; }
.s1ed .feas-range { position: absolute; height: 100%; background: linear-gradient(90deg, #BBF7D0 0%, #86EFAC 100%); border-radius: 4px; }
.s1ed .feas-target { position: absolute; width: 3px; height: 16px; top: -4px; background: #B31B1B; box-shadow: 0 0 0 2px white; border-radius: 2px; }
.s1ed .feas-axis { display: flex; justify-content: space-between; font-size: 10px; color: #71717a; font-family: ui-monospace, monospace; }
.s1ed .toast { padding: 14px 18px; border-radius: 10px; font-size: 13px; }
.s1ed .toast-ok { background: #DCFCE7; border: 1px solid #86EFAC; color: #14532D; }
.s1ed .toast-err { background: #FEE2E2; border: 1px solid #FCA5A5; color: #7F1D1D; }
.s1ed .toast-idle { background: #FAFAF9; border: 1px dashed #d4d4d8; color: #71717a; font-style: italic; }
.s1ed .save-card { background: linear-gradient(135deg, #FAFAF9 0%, #F4F4F5 100%); border: 2px solid #e4e4e7; }
.s1ed .next-card { border-left: 4px solid #B31B1B; }
.s1ed .next-card code { background: #18181b; color: #FAFAF9; padding: 8px 12px; border-radius: 6px; display: inline-block; font-size: 12px; }
.s1ed input[type="number"], .s1ed select { padding: 4px 8px; border: 1px solid #d4d4d8; border-radius: 6px; font-family: inherit; }
.s1ed input[type="number"]:focus, .s1ed select:focus { outline: 2px solid #B31B1B; outline-offset: -1px; border-color: #B31B1B; }
</style>
"""

# ╔═╡ a0000001-0000-0000-0000-000000000001
@htl("""
<div class="s1ed"><div class="hero">
<h1>Session 1: Portfolio Configuration Editor</h1>
<div class="subtitle">
A reactive front-end for the interview in <code>interview.md</code>. Pick an archetype,
tune parameters, toggle tickers; the feasibility panel re-checks live. The <b>Save</b>
button writes <code>data/my-tickers.csv</code> and <code>data/portfolio-config.toml</code>.
You still run the construction notebook yourself after saving.
</div>
</div></div>
""")

# ╔═╡ a0000003-0000-0000-0000-000000000003
const _DATA_DIR = joinpath(@__DIR__, "data")

# ╔═╡ a0000005-0000-0000-0000-000000000005
const ARCHETYPES = Dict(
	"Conservative Income" => (
		tickers = ["VZ","T","MCD","PG","KO","PEP","WMT","XOM","CVX","JPM","BRK.B",
			"JNJ","MRK","HON","UPS","AAPL","MSFT","APD","AMT","NEE"],
		sectors = Dict("VZ"=>"Communication Services","T"=>"Communication Services",
			"MCD"=>"Consumer Discretionary","PG"=>"Consumer Staples",
			"KO"=>"Consumer Staples","PEP"=>"Consumer Staples","WMT"=>"Consumer Staples",
			"XOM"=>"Energy","CVX"=>"Energy","JPM"=>"Financials","BRK.B"=>"Financials",
			"JNJ"=>"Health Care","MRK"=>"Health Care","HON"=>"Industrials",
			"UPS"=>"Industrials","AAPL"=>"Information Technology",
			"MSFT"=>"Information Technology","APD"=>"Materials","AMT"=>"Real Estate",
			"NEE"=>"Utilities"),
		risk_tolerance = "risk-averse", time_horizon = "short",
		primary_objective = "income", target_growth = 0.07,
		max_weight = 0.10, cash_fraction = 0.30, notebook = "RRFA",
	),
	"Conservative Growth" => (
		tickers = ["DIS","VZ","AMZN","HD","MCD","PG","COST","WMT","XOM","CVX","JPM",
			"BRK.B","V","JNJ","UNH","LLY","HON","UPS","AAPL","MSFT","AVGO","SHW","AMT","NEE"],
		sectors = Dict("DIS"=>"Communication Services","VZ"=>"Communication Services",
			"AMZN"=>"Consumer Discretionary","HD"=>"Consumer Discretionary",
			"MCD"=>"Consumer Discretionary","PG"=>"Consumer Staples",
			"COST"=>"Consumer Staples","WMT"=>"Consumer Staples","XOM"=>"Energy",
			"CVX"=>"Energy","JPM"=>"Financials","BRK.B"=>"Financials",
			"V"=>"Financials","JNJ"=>"Health Care","UNH"=>"Health Care",
			"LLY"=>"Health Care","HON"=>"Industrials","UPS"=>"Industrials",
			"AAPL"=>"Information Technology","MSFT"=>"Information Technology",
			"AVGO"=>"Information Technology","SHW"=>"Materials",
			"AMT"=>"Real Estate","NEE"=>"Utilities"),
		risk_tolerance = "risk-averse", time_horizon = "long",
		primary_objective = "growth", target_growth = 0.09,
		max_weight = 0.15, cash_fraction = 0.10, notebook = "RRFA",
	),
	"Balanced" => (
		tickers = ["DIS","VZ","AMZN","HD","PG","COST","XOM","CVX","JPM","V","BAC","JNJ",
			"UNH","LLY","HON","CAT","AAPL","MSFT","NVDA","SHW","AMT","NEE"],
		sectors = Dict("DIS"=>"Communication Services","VZ"=>"Communication Services",
			"AMZN"=>"Consumer Discretionary","HD"=>"Consumer Discretionary",
			"PG"=>"Consumer Staples","COST"=>"Consumer Staples","XOM"=>"Energy",
			"CVX"=>"Energy","JPM"=>"Financials","V"=>"Financials","BAC"=>"Financials",
			"JNJ"=>"Health Care","UNH"=>"Health Care","LLY"=>"Health Care",
			"HON"=>"Industrials","CAT"=>"Industrials","AAPL"=>"Information Technology",
			"MSFT"=>"Information Technology","NVDA"=>"Information Technology",
			"SHW"=>"Materials","AMT"=>"Real Estate","NEE"=>"Utilities"),
		risk_tolerance = "balanced", time_horizon = "long",
		primary_objective = "balanced", target_growth = 0.10,
		max_weight = 0.20, cash_fraction = 0.00, notebook = "RA",
	),
	"Growth-Oriented" => (
		tickers = ["DIS","NFLX","AMZN","HD","TJX","COST","CVX","JPM","V","MA","UNH","LLY",
			"ABBV","HON","CAT","AAPL","MSFT","NVDA","AVGO","CRM","SHW","AMT"],
		sectors = Dict("DIS"=>"Communication Services","NFLX"=>"Communication Services",
			"AMZN"=>"Consumer Discretionary","HD"=>"Consumer Discretionary",
			"TJX"=>"Consumer Discretionary","COST"=>"Consumer Staples","CVX"=>"Energy",
			"JPM"=>"Financials","V"=>"Financials","MA"=>"Financials",
			"UNH"=>"Health Care","LLY"=>"Health Care","ABBV"=>"Health Care",
			"HON"=>"Industrials","CAT"=>"Industrials","AAPL"=>"Information Technology",
			"MSFT"=>"Information Technology","NVDA"=>"Information Technology",
			"AVGO"=>"Information Technology","CRM"=>"Information Technology",
			"SHW"=>"Materials","AMT"=>"Real Estate"),
		risk_tolerance = "risk-tolerant", time_horizon = "long",
		primary_objective = "growth", target_growth = 0.13,
		max_weight = 0.15, cash_fraction = 0.00, notebook = "RA",
	),
	"Aggressive Growth" => (
		tickers = ["NFLX","AMZN","NKE","TSLA","COST","SLB","JPM","V","MA","LLY","ABBV",
			"CAT","AAPL","MSFT","NVDA","AVGO","CRM","ADBE","FCX","AMT"],
		sectors = Dict("NFLX"=>"Communication Services","AMZN"=>"Consumer Discretionary",
			"NKE"=>"Consumer Discretionary","TSLA"=>"Consumer Discretionary",
			"COST"=>"Consumer Staples","SLB"=>"Energy","JPM"=>"Financials",
			"V"=>"Financials","MA"=>"Financials","LLY"=>"Health Care",
			"ABBV"=>"Health Care","CAT"=>"Industrials","AAPL"=>"Information Technology",
			"MSFT"=>"Information Technology","NVDA"=>"Information Technology",
			"AVGO"=>"Information Technology","CRM"=>"Information Technology",
			"ADBE"=>"Information Technology","FCX"=>"Materials","AMT"=>"Real Estate"),
		risk_tolerance = "risk-tolerant", time_horizon = "long",
		primary_objective = "growth", target_growth = 0.16,
		max_weight = 0.10, cash_fraction = 0.00, notebook = "RA",
	),
)

# ╔═╡ a000000e-0000-0000-0000-00000000000e
const calib = MySIMCalibration();

# ╔═╡ a000000f-0000-0000-0000-00000000000f
const G_MARKET_MEAN = let
	df = MyTrainingMarketDataSet()["dataset"]["SPY"]
	g = diff(log.(df.volume_weighted_average_price)) .* 252
	mean(g)
end;

# ╔═╡ a0000010-0000-0000-0000-000000000010
function expected_growths(tickers::AbstractVector{<:AbstractString})
	calib_t = calib["tickers"]::Vector{String}
	α = calib["alpha"]::Vector{Float64}
	β = calib["beta"]::Vector{Float64}
	idx = Dict(t => i for (i, t) in enumerate(calib_t))
	out = Float64[]
	missing_t = String[]
	for t in tickers
		i = get(idx, t, 0)
		if i == 0
			push!(missing_t, t)
		else
			push!(out, α[i] + β[i] * G_MARKET_MEAN)
		end
	end
	return out, missing_t
end

# ╔═╡ a0000011-0000-0000-0000-000000000011
function feasible_target_range(Eg::AbstractVector, cap::Real)
	N = length(Eg)
	N == 0 && return (NaN, NaN)
	cap_eff = max(cap, 1/N)
	function bound(Eg_sorted)
		remaining = 1.0
		acc = 0.0
		for g in Eg_sorted
			w = min(cap_eff, remaining)
			acc += w * g
			remaining -= w
			remaining <= 1e-12 && break
		end
		return acc
	end
	return (bound(sort(Eg)), bound(sort(Eg; rev=true)))
end

# ╔═╡ a0000006-0000-0000-0000-000000000006
@htl("""
<div class="s1ed"><div class="card">
<h2>Archetype</h2>
<div class="row">
<label>Choose your archetype</label>
$(@bind archetype Select([
	"Conservative Income",
	"Conservative Growth",
	"Balanced",
	"Growth-Oriented",
	"Aggressive Growth",
]; default="Balanced"))
</div>
</div></div>
""")

# ╔═╡ a0000008-0000-0000-0000-000000000008
@htl("""
<div class="s1ed"><div class="card">
<h2>Risk profile</h2>
<div class="row">
<label>Risk tolerance</label>
$(@bind risk_tolerance Select(["risk-averse","balanced","risk-tolerant"]; default=ARCHETYPES[archetype].risk_tolerance))
</div>
<div class="row">
<label>Time horizon</label>
$(@bind time_horizon Select(["short","medium","long"]; default=ARCHETYPES[archetype].time_horizon))
</div>
<div class="row">
<label>Primary objective</label>
$(@bind primary_objective Select(["preservation","income","growth","balanced"]; default=ARCHETYPES[archetype].primary_objective))
</div>
</div></div>
""")

# ╔═╡ a000000a-0000-0000-0000-00000000000a
@htl("""
<div class="s1ed"><div class="card">
<h2>Portfolio parameters</h2>
<div class="row">
<label>Initial budget (USD)</label>
$(@bind initial_budget NumberField(1_000.0:1_000.0:100_000_000.0; default=100_000.0))
</div>
<div class="row">
<label>Risk-free rate (CCGR / yr)</label>
$(@bind risk_free_rate NumberField(0.00:0.001:0.10; default=0.045))
</div>
<div class="row">
<label>Target growth R<sub>target</sub> (CCGR / yr)</label>
$(@bind target_growth NumberField(0.00:0.005:0.40; default=ARCHETYPES[archetype].target_growth))
</div>
<div class="row">
<label>Concentration cap max<sub>w</sub></label>
$(@bind max_weight NumberField(0.05:0.01:1.00; default=ARCHETYPES[archetype].max_weight))
</div>
<div class="row">
<label>Cash fraction</label>
$(@bind cash_fraction NumberField(0.00:0.01:1.00; default=ARCHETYPES[archetype].cash_fraction))
</div>
</div></div>
""")

# ╔═╡ a000000c-0000-0000-0000-00000000000c
@htl("""
<div class="s1ed"><div class="card">
<h2>Tickers — archetype list</h2>
<div style="font-size: 12px; color: #71717a; margin-bottom: 10px;">Uncheck to apply Step 4 exclusions. Defaults to the full archetype list.</div>
$(@bind archetype_picks MultiCheckBox(
	ARCHETYPES[archetype].tickers;
	default = ARCHETYPES[archetype].tickers,
	select_all = true,
))
</div></div>
""")

# ╔═╡ a000001a-0000-0000-0000-00000000001a
@htl("""
<div class="s1ed"><div class="card">
<h2>Tickers — extras</h2>
<div style="font-size: 12px; color: #71717a; margin-bottom: 10px;">Add any name from the 424-ticker SIM calibration universe (excluding the archetype list above). Hold ⌘/Ctrl to multi-select. Extras tag as sector "Unknown" in the saved CSV.</div>
$(@bind extra_picks MultiSelect(
	sort(setdiff(calib["tickers"]::Vector{String}, ARCHETYPES[archetype].tickers));
	default = String[],
	size = 8,
))
</div></div>
""")

# ╔═╡ a000001b-0000-0000-0000-00000000001b
selected_tickers = sort(unique(vcat(archetype_picks, extra_picks)));

# ╔═╡ a0000012-0000-0000-0000-000000000012
feasibility = let
	N = length(selected_tickers)
	Eg, missing_t = expected_growths(selected_tickers)
	cap_floor_ok = N == 0 ? false : max_weight >= 1/N - 1e-12
	g_min, g_max = feasible_target_range(Eg, max_weight)
	target_in_hull = N > 0 && g_min - 1e-9 <= target_growth <= g_max + 1e-9
	all_in_calib = isempty(missing_t)
	cash_ok = 0.0 <= cash_fraction <= 1.0
	(
		N = N,
		cap_floor_ok = cap_floor_ok,
		all_in_calib = all_in_calib,
		missing_tickers = missing_t,
		g_min = g_min,
		g_max = g_max,
		target_in_hull = target_in_hull,
		cash_ok = cash_ok,
		ok = cap_floor_ok && all_in_calib && target_in_hull && cash_ok && N > 0,
	)
end

# ╔═╡ a0000013-0000-0000-0000-000000000013
let
	f = feasibility
	pill(b) = b ? "<span class=\"pill pill-ok\">PASS</span>" : "<span class=\"pill pill-err\">FAIL</span>"
	archetype_set = Set(ARCHETYPES[archetype].tickers)
	chip_html = if isempty(selected_tickers)
		"<span style=\"font-size: 12px; color: #71717a; font-style: italic;\">No tickers selected.</span>"
	else
		join([
			let
				cls = t in archetype_set ? "chip" : "chip chip-extra"
				"<span class=\"$cls\">$t</span>"
			end
			for t in selected_tickers
		], "")
	end
	scale_lo, scale_hi = 0.0, 0.30
	pct(x) = clamp((x - scale_lo) / (scale_hi - scale_lo) * 100, 0.0, 100.0)
	bar_html = if f.N > 0 && !isnan(f.g_min)
		left = pct(f.g_min); right = pct(f.g_max); tgt = pct(target_growth)
		"""
		<h3>Feasible R<sub>target</sub> range under cap</h3>
		<div class="feas-bar">
		  <div class="feas-range" style="left: $(left)%; width: $(right - left)%;"></div>
		  <div class="feas-target" style="left: calc($(tgt)% - 1px);"></div>
		</div>
		<div class="feas-axis"><span>0%</span><span>10%</span><span>20%</span><span>30%</span></div>
		<div style="font-size: 11px; color: #71717a; margin-top: 6px;">Green band = feasible range [$(round(f.g_min, digits=4)), $(round(f.g_max, digits=4))]. Red tick = your target ($(round(target_growth, digits=4))).</div>
		"""
	else
		""
	end
	verdict_html = if f.ok
		"<div class=\"verdict verdict-ok\">All checks pass. Safe to save and run the construction notebook.</div>"
	else
		"<div class=\"verdict verdict-err\">Fix the failing checks before saving. The QP will error or skip frontier points if R<sub>target</sub> sits outside the feasible range.</div>"
	end
	g_min_str = f.N > 0 ? string(round(f.g_min, digits=4)) : "—"
	g_max_str = f.N > 0 ? string(round(f.g_max, digits=4)) : "—"
	floor_str = f.N == 0 ? "—" : string(round(1/f.N, digits=4))
	missing_detail = f.all_in_calib ? "all $(f.N) found" : "missing: $(join(f.missing_tickers, ", "))"
	check_rows = """
	<div class="check-row">
	  <div><div class="label">Tickers selected</div><div class="detail">N = $(f.N)</div></div>
	  <div class="right">$(pill(f.N > 0))</div>
	</div>
	<div class="check-row">
	  <div><div class="label">All tickers in SIM calibration</div><div class="detail">$missing_detail</div></div>
	  <div class="right">$(pill(f.all_in_calib))</div>
	</div>
	<div class="check-row">
	  <div><div class="label">Cap floor: max<sub>w</sub> ≥ 1/N</div><div class="detail">cap = $(round(max_weight, digits=4)), floor = $floor_str</div></div>
	  <div class="right">$(pill(f.cap_floor_ok))</div>
	</div>
	<div class="check-row">
	  <div><div class="label">R<sub>target</sub> inside convex hull</div><div class="detail">target = $(round(target_growth, digits=4)) ∈ [$g_min_str, $g_max_str]</div></div>
	  <div class="right">$(pill(f.target_in_hull))</div>
	</div>
	<div class="check-row">
	  <div><div class="label">Cash fraction ∈ [0, 1]</div><div class="detail">$(round(cash_fraction, digits=3))</div></div>
	  <div class="right">$(pill(f.cash_ok))</div>
	</div>
	"""
	HTML("""
	<div class="s1ed"><div class="card">
	<h2>Live preview</h2>
	<div class="metric-grid">
	  <div class="metric"><div class="lbl">Archetype</div><div class="val" style="font-size: 14px;">$archetype</div></div>
	  <div class="metric"><div class="lbl">Tickers</div><div class="val">$(f.N)</div></div>
	  <div class="metric"><div class="lbl">Budget</div><div class="val">\$$(Int(round(initial_budget)))</div></div>
	  <div class="metric"><div class="lbl">Cap / Target g</div><div class="val">$(round(max_weight, digits=2)) / $(round(target_growth, digits=2))</div></div>
	</div>
	$check_rows
	$bar_html
	$verdict_html
	<h3>Selected tickers <span style="text-transform: none; font-weight: 400; color: #a1a1aa;">— yellow chips are extras outside the archetype</span></h3>
	<div class="chips">$chip_html</div>
	</div></div>
	""")
end

# ╔═╡ a0000015-0000-0000-0000-000000000015
@htl("""
<div class="s1ed"><div class="card save-card">
<h2>Save configuration</h2>
<div style="font-size: 13px; color: #3f3f46; margin-bottom: 12px;">
Click to write <code>data/my-tickers.csv</code> and <code>data/portfolio-config.toml</code>. The button refuses to write while feasibility is red.
</div>
$(@bind save_clicked CounterButton("💾  Save config files"))
</div></div>
""")

# ╔═╡ a0000016-0000-0000-0000-000000000016
save_status = let
	if save_clicked == 0
		(:idle, "Not saved yet — click the button above when feasibility is green.")
	elseif !feasibility.ok
		(:err, "Refusing to save — feasibility checks did not pass. Adjust inputs and click again.")
	else
		sectors = ARCHETYPES[archetype].sectors
		tickers_df = DataFrame(
			ticker = String.(selected_tickers),
			sector = [get(sectors, t, "Unknown") for t in selected_tickers],
		)
		CSV.write(joinpath(_DATA_DIR, "my-tickers.csv"), tickers_df)

		toml_text = """
		# Portfolio configuration for the AI Finance course.
		# Generated by S1-PortfolioConfigEditor-May-2026.jl.
		# Re-run the editor or hand-edit; every S1/S2/S3 example reads this file.

		[portfolio]
		initial_budget = $(Float64(initial_budget))
		risk_free_rate = $(Float64(risk_free_rate))
		target_growth  = $(Float64(target_growth))
		cash_fraction  = $(Float64(cash_fraction))
		max_weight     = $(Float64(max_weight))

		[profile]
		risk_tolerance    = "$(risk_tolerance)"
		time_horizon      = "$(time_horizon)"
		primary_objective = "$(primary_objective)"
		archetype         = "$(archetype)"
		"""
		open(joinpath(_DATA_DIR, "portfolio-config.toml"), "w") do io
			write(io, replace(toml_text, r"^\t\t"m => ""))
		end
		(:ok, "Wrote <code>data/my-tickers.csv</code> ($(nrow(tickers_df)) tickers) and <code>data/portfolio-config.toml</code>. (save #$(save_clicked))")
	end
end;

# ╔═╡ a0000017-0000-0000-0000-000000000017
let
	kind, msg = save_status
	cls = kind == :ok ? "toast toast-ok" : kind == :err ? "toast toast-err" : "toast toast-idle"
	icon = kind == :ok ? "✅ " : kind == :err ? "⚠️ " : ""
	HTML("""<div class="s1ed"><div class="$cls">$icon$msg</div></div>""")
end

# ╔═╡ a0000019-0000-0000-0000-000000000019
let
	nb = ARCHETYPES[archetype].notebook
	notebook_file = nb == "RA" ?
		"S1-Example-Core-BuildMinVariancePortfolio-RA-May-2026.ipynb" :
		"S1-Example-Core-MinVariancePortfolio-RRFA-May-2026.ipynb"
	method = nb == "RA" ? "Risky-asset-only minimum variance, fully invested" :
		"Risky + risk-free tangent (CML), partial cash position"
	HTML("""
	<div class="s1ed"><div class="card next-card">
	<h2>Next step</h2>
	<div style="font-size: 14px; margin-bottom: 12px;">
	  The <b>$archetype</b> archetype routes to the <span class="pill pill-arch">$nb</span> construction method.<br>
	  <span style="color: #71717a; font-size: 12px;">$method.</span>
	</div>
	<div style="font-size: 12px; color: #71717a; margin-bottom: 6px;">Open in Jupyter or VS Code and execute top-to-bottom:</div>
	<code>lectures/session-1/$notebook_file</code>
	<div style="font-size: 12px; color: #71717a; margin-top: 12px;">
	Watch the frontier plot, allocation table, and QP diagnostics. Sign off before the allocation flows into S2/S3/S4.
	</div>
	</div></div>
	""")
end

# ╔═╡ Cell order:
# ╟─a0000020-0000-0000-0000-000000000020
# ╟─a0000001-0000-0000-0000-000000000001
# ╠═a0000002-0000-0000-0000-000000000002
# ╠═a0000003-0000-0000-0000-000000000003
# ╠═a0000005-0000-0000-0000-000000000005
# ╠═a000000e-0000-0000-0000-00000000000e
# ╠═a000000f-0000-0000-0000-00000000000f
# ╠═a0000010-0000-0000-0000-000000000010
# ╠═a0000011-0000-0000-0000-000000000011
# ╟─a0000006-0000-0000-0000-000000000006
# ╟─a0000008-0000-0000-0000-000000000008
# ╟─a000000a-0000-0000-0000-00000000000a
# ╟─a000000c-0000-0000-0000-00000000000c
# ╟─a000001a-0000-0000-0000-00000000001a
# ╠═a000001b-0000-0000-0000-00000000001b
# ╠═a0000012-0000-0000-0000-000000000012
# ╟─a0000013-0000-0000-0000-000000000013
# ╟─a0000015-0000-0000-0000-000000000015
# ╠═a0000016-0000-0000-0000-000000000016
# ╟─a0000017-0000-0000-0000-000000000017
# ╟─a0000019-0000-0000-0000-000000000019

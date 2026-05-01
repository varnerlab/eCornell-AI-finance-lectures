# NotebookLM Audio Overview Prompt: S1 — Stress-Testing Minimum-Variance Portfolios

Paste the block below into NotebookLM's "Customize" box when generating the audio overview for Session 1 (intro vignette plus the main lecture).

```
Audience: advanced finance students or junior portfolio managers who know basic linear algebra and probability but are new to convex optimization, Single Index Models, and generative Monte Carlo stress testing.

Frame the episode around Maya Chen, a junior portfolio manager whose AI intake tool just produced a minimum-variance allocation for a Balanced-archetype client. Her supervisor asks the only question that matters: is this actually a good idea? The lecture builds the optimizer that produced the allocation, then breaks it by running the same allocation across thousands of synthetic futures from a regime-switching market generator.

Without dwelling on derivations, cover these mathematical objects and why each one matters:
* Continuously compounded growth rate from log price ratios, the stationary input the optimizer actually consumes.
* The minimum-variance quadratic program: minimize portfolio variance subject to a target growth rate, a budget that sums weights to one, and per-asset box bounds.
* Capital Allocation Line and tangent portfolio: adding a risk-free asset linearizes the frontier, and the slope is the Sharpe ratio of the tangent portfolio (solved as a second-order cone program).
* Hybrid Monte Carlo generator: a 100-state hidden Markov model fit to ten years of SPY by Baum-Welch, plus a Student-t copula on per-ticker residuals, producing forward paths with fat tails, volatility clustering, and regime shifts.
* The tail-risk scorecard: portfolio NPV against the risk-free baseline, NPV-fail rate, VaR and CVaR with sampling standard error, and median plus P95 maximum drawdown.

Tradeoffs to keep the discussion balanced:

Upside:
* Convex optimization gives a unique global solution that solves quickly.
* The hybrid generator reproduces fat tails and regime shifts that a plain Gaussian model cannot.
* The framework forces explicit policy choices: target growth rate, concentration cap, risk-free rate, all surfaced in a config file.

Downside:
* Input sensitivity is severe. The optimizer treats estimated growth rates and covariances as exact, so small estimate errors produce large weight swings (Michaud's "error maximizer"), and Chopra and Ziemba showed errors in expected returns dominate errors in variances.
* Weight concentration is the rule, not the exception. In Maya's actual allocation two names hit the twenty percent cap and the top four names hold about three quarters of the book.
* Training-distribution bias: the generator was fit on a bull-dominated 2014 to 2024 SPY window, so its synthetic envelope under-represents the sustained drawdown structure a longer history would carry.
* Architectural bias: the Single Index Model treats per-ticker residuals as i.i.d. across trading days, so it structurally cannot produce idiosyncratic volatility clustering or cross-stock contagion no matter how much data you feed it.
* Everything in this session assumes frictionless trading: zero transaction costs, infinitely divisible shares, perfect fills, no taxes, no slippage. Later sessions relax these.

Keep the tone conversational, not lecture-y. Prefer analogies over formulas read aloud.
```

## What was emphasized

- Opening intuition uses the Maya-and-Lou scenario from the intro notebook, so the hosts have a concrete person and a concrete question driving the episode rather than an abstract "today we will learn about Markowitz."
- Math highlights track the lecture's actual section order (growth rates, MPT QP, CAL/tangent, hybrid Monte Carlo, tail-risk metrics) so the audio mirrors the notebook flow.
- Downside bullets are pulled directly from the lecture's "When Optimal Fails" section and the intro's two named biases (training-distribution and architectural i.i.d. residual), and the concentration example uses Maya's real allocation numbers from the intro.
- The frictionless-trading caveat is included as a downside because the lecture explicitly flags it and signals later sessions will relax it.

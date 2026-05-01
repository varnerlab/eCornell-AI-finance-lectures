# NotebookLM Audio Overview Prompt: Session 2 — Maya's Adaptive Rebalancing Engine

Paste the block below into NotebookLM's "Customize" box when generating the audio overview for this lecture (introduction + main lecture).

```
Audience: finance and quant graduate students who know basic portfolio theory and linear algebra, but are new to utility-based adaptive rebalancing and sentiment-driven preferences.

Frame the session through Maya's problem: a static minimum-variance book at Lindenfield Wealth Partners took a 21% drawdown because the allocator could not react to a regime shift. The fix replaces that one-shot quadratic program with a closed-loop daily engine, a utility allocator driven by a sentiment scalar, wrapped in committee-approved safety rules. Think feedback controller with hard safety stops, not a smarter optimizer.

Cover these math highlights without dwelling on derivations:
- Cobb-Douglas utility: under a budget constraint, each preferred asset's budget share equals its preference exponent gamma_i divided by the sum of positive exponents. Closed form, no matrix inversion.
- Sentiment signal lambda_t from a 21 over 63 day EMA crossover on a market index. Positive lambda means bearish, negative means bullish.
- Preference weights gamma_i = tanh of a SIM-based score, where beta_i raised to lambda_t acts as a regime lens on market exposure.
- CES generalization: elasticity eta controls concentration. Eta near one reproduces Cobb-Douglas, large eta concentrates on the best bang-for-the-buck asset, small eta forces equal share counts.
- Three trigger rules: a drawdown circuit breaker d_max, a proportional turnover cap tau_max, and a binary rebalance schedule b_t.

Tradeoffs, with explicit upsides and downsides:
- Upside: the closed-form allocator updates instantly when preferences change, and on the bear quartile of 5,000 synthetic paths the engine's fifth-percentile terminal wealth beats the static baseline.
- Upside: trigger rules encode the risk committee's policy directly into the loop, so the engine cannot ignore a crash or trade past the turnover cap.
- Downside: the EMA-crossover lambda is a momentum proxy, not a true regime detector. It lags real shifts and can flicker on noise.
- Downside: the SIM intercept and slope are estimated once and frozen, so the regime lens applies confidence to potentially stale coefficients. Session 3 fixes this with online updates.
- Downside: this is a backtest with the full price path known in advance, not an online deployment. Transaction costs at 5 bps and slippage are simplified.
- Downside: a single forward path is a story, not evidence. Honest evaluation needs the bear-subset Monte Carlo, not a hero chart.

Keep the tone conversational, not lecture-y. Lean on the analogies (feedback controller, regime lens, circuit breaker) rather than reading formulas aloud.
```

## What was emphasized

- Opened with Maya's 21% drawdown and the Thursday risk committee, since both notebooks frame the technical content around that institutional pressure.
- Math highlights kept to the five objects the lecture actually boxes: Cobb-Douglas closed form, EMA-crossover lambda, the tanh preference map with the beta^lambda regime lens, CES with sentiment-driven eta, and the three trigger rules.
- Downsides on EMA lag, frozen SIM coefficients, and backtest-vs-online are pulled directly from the lecture text (Session 3 callout, "one path is a story", explicit backtest disclaimer); the noise-flicker bullet is a light inference from the lecture's own warning that an allocator trading on every flicker of lambda is worse than one that never trades.
- Both the introduction notebook (Maya, Sara, the bear-subset NPV histogram) and the lecture notebook (utility math, OODA loop, trigger rules) were incorporated into one prompt since they form one session arc.

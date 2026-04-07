# Hybrid SIM construction with variance-correction

A note on how to inject a Single Index Model (SIM) market-factor structure into
a set of pre-existing per-asset return generators (e.g. fitted HMM marginals,
GARCH models, empirical bootstraps) without destroying the marginal distributions
those generators were built to preserve.

## The problem

We have:

- A market growth-rate path $g_m(t)$, $t = 1,\dots,T$, with sample variance
  $\sigma_m^2 = \mathrm{Var}[g_m]$.
- For each asset $i$, a draw $\tilde{\varepsilon}_i(t)$ from a generator that
  produces growth rates with the *correct* marginal distribution (heavy tails,
  regime switching, jumps, whatever the application requires). Call its variance
  $\sigma_{\text{gen},i}^2 = \mathrm{Var}[\tilde{\varepsilon}_i]$.
- Target SIM parameters $(\alpha_i, \beta_i)$ for each asset, calibrated from
  some external source (real data, an analyst view, a benchmark).

We want to compose a synthetic asset path

$$
g_i(t) \;=\; \alpha_i \;+\; \beta_i\, g_m(t) \;+\; \varepsilon_i(t)
$$

such that:

1. **SIM is recoverable.** Regressing $g_i$ on $g_m$ via OLS recovers
   $\hat{\beta}_i \approx \beta_i$ and $\hat{\alpha}_i \approx \alpha_i$.
2. **Marginals are preserved.** $\mathrm{Var}[g_i] \approx \sigma_{\text{gen},i}^2$,
   and the heavy-tail / regime structure encoded in $\tilde{\varepsilon}_i$ is
   substantially intact.
3. **Cross-sectional dependence is preserved.** If the original generators came
   with a joint dependence structure (e.g. a copula on the $\tilde{\varepsilon}_i$),
   that structure carries through.

The naïve composition $g_i = \alpha_i + \beta_i g_m + \tilde{\varepsilon}_i$
satisfies (1) and (3) but inflates the variance of $g_i$ by exactly
$\beta_i^2 \sigma_m^2$, breaking (2). For high-$\beta$ assets this is large and
visibly distorts the marginal.

## The variance correction

Replace $\tilde{\varepsilon}_i$ with a *scaled* version $\varepsilon_i = s_i\,\tilde{\varepsilon}_i$
where the scalar $s_i \in (0, 1]$ is chosen so that

$$
\mathrm{Var}[\beta_i\, g_m + \varepsilon_i] \;=\; \sigma_{\text{gen},i}^2.
$$

Assuming $\tilde{\varepsilon}_i$ is uncorrelated with $g_m$ (true by construction
when the generators are independent draws, and true after a copula rank-reorder
when the copula is sampled independently of $g_m$), the cross term vanishes and

$$
\beta_i^2\, \sigma_m^2 \;+\; s_i^2\, \sigma_{\text{gen},i}^2 \;=\; \sigma_{\text{gen},i}^2
$$

which gives

$$
\boxed{\;s_i^2 \;=\; 1 \;-\; \frac{\beta_i^2\, \sigma_m^2}{\sigma_{\text{gen},i}^2}\;}
$$

Define the **market loading ratio**

$$
\rho_i \;\equiv\; \frac{\beta_i^2\, \sigma_m^2}{\sigma_{\text{gen},i}^2}
\;=\; \text{share of }\sigma_{\text{gen},i}^2\text{ that the market component will consume}.
$$

Then $s_i^2 = 1 - \rho_i$, valid whenever $\rho_i < 1$.

## β clipping with an idiosyncratic floor

The correction breaks down when $\rho_i \ge 1$ — the market loading alone would
account for *more* than the entire generator variance, leaving no room for
idiosyncratic noise. This happens for high-$\beta$ assets paired with generators
that have lower vol than the chosen market path, or for any asset where the
synthetic market is much more volatile than the market the $\beta_i$ was
calibrated on.

To handle this case we impose a **floor** $f \in (0, 1)$ on the idiosyncratic
share — i.e. require $s_i^2 \ge f$ — and clip $\beta_i$ downward whenever the
floor would be violated:

$$
\beta_i^{\text{eff}} \;=\;
\begin{cases}
\beta_i & \text{if } \rho_i \le 1 - f \\[6pt]
\mathrm{sign}(\beta_i)\,\sqrt{(1 - f)\,\dfrac{\sigma_{\text{gen},i}^2}{\sigma_m^2}} & \text{if } \rho_i > 1 - f
\end{cases}
$$

$$
s_i^2 \;=\;
\begin{cases}
1 - \rho_i & \text{if } \rho_i \le 1 - f \\
f & \text{if } \rho_i > 1 - f
\end{cases}
$$

Sign preservation ensures defensive (negative-$\beta$) assets stay defensive
even when clipped. The floor $f$ is a tunable knob — a typical choice is
$f = 0.10$, guaranteeing at least 10% of each asset's variance comes from
idiosyncratic noise.

When $\beta_i$ is clipped, log a warning and report the original-vs-effective
$\beta$ so users can spot the assets whose loadings were attenuated.

## The full procedure

For each asset $i$:

1. Draw $\tilde{\varepsilon}_i(t)$, $t=1,\dots,T$, from its generator. Compute
   $\sigma_{\text{gen},i}^2 = \mathrm{Var}[\tilde{\varepsilon}_i]$.
2. Look up the calibrated $(\alpha_i, \beta_i)$.
3. Compute $\rho_i$ and apply the clipping rule above to get
   $\beta_i^{\text{eff}}$ and $s_i^2$.
4. Scale: $\varepsilon_i = \sqrt{s_i^2}\,\tilde{\varepsilon}_i$.
5. (Optional) Apply any cross-sectional rearrangement / copula reorder to the
   scaled $\varepsilon_i$. Reordering preserves the marginal variance, so it
   does not affect the variance correction.
6. Compose: $g_i(t) = \alpha_i + \beta_i^{\text{eff}}\, g_m(t) + \varepsilon_i(t)$.

The resulting $g_i$ has:

- $\mathrm{Var}[g_i] = (\beta_i^{\text{eff}})^2 \sigma_m^2 + s_i^2 \sigma_{\text{gen},i}^2 \approx \sigma_{\text{gen},i}^2$
  (exactly so when $\rho_i \le 1 - f$).
- OLS slope $\mathrm{Cov}[g_i, g_m] / \sigma_m^2 \to \beta_i^{\text{eff}}$ in
  the large-$T$ limit.
- Tail behavior dominated by $\varepsilon_i$ when $\beta_i^{\text{eff}\,2} \sigma_m^2$
  is small relative to the heavy tails of the generator — which the floor
  guarantees by reserving at least $f \cdot \sigma_{\text{gen},i}^2$ of the
  variance for $\varepsilon_i$.
- Cross-sectional dependence inherited from whatever joint structure was applied
  to the $\varepsilon_i$ in step 5.

## Caveats

- **Marginal shape is not exactly preserved.** $g_i$ is the convolution of a
  scaled-down generator draw with $\beta_i^{\text{eff}} g_m$. For low-$\beta$
  assets the perturbation is tiny; for high-$\beta$ assets the marginal tightens
  toward Gaussian along the market-driven axis. QQ plots will show small
  deviations in the body, larger ones in the tails as $\beta$ grows.
- **Clipping silently changes the calibration.** Anywhere a clipped asset
  appears, downstream consumers should be aware that $\beta_i^{\text{eff}} \ne \beta_i$.
  Persisting both values in the output table avoids surprises.
- **The independence assumption matters.** The variance algebra above
  ($\mathrm{Var}[\beta g_m + \varepsilon] = \beta^2 \sigma_m^2 + s^2 \sigma_{\text{gen}}^2$)
  requires $\mathrm{Cov}[\tilde{\varepsilon}_i, g_m] \approx 0$. This holds when
  the generator and the market path are independent draws. If you're injecting
  SIM into generators that *already* have some market correlation, subtract the
  empirical $\mathrm{Cov}[\tilde{\varepsilon}_i, g_m]/\sigma_m^2$ from $\beta_i$
  before applying the correction, or estimate $s_i^2$ directly from the
  empirical variance after composition.
- **Time-series dependence on the market is not free.** This recipe injects
  *contemporaneous* market dependence only. Lagged market dependence (lead-lag,
  market-driven volatility clustering) requires a richer construction.

## The index-ETF special case

Some assets are *defined* as deterministic (or near-deterministic) transforms
of the market. SPY against itself has $\beta = 1, R^2 = 1, \sigma_\varepsilon = 0$
exactly. QQQ on SPY has $R^2 \gtrsim 0.95$. SPYG (S&P 500 Growth ETF) is in the
same boat. For these tickers the variance correction is fighting reality: the
"natural" variance of the asset *is* $\beta^2 \sigma_m^2$ — there is no
idiosyncratic component to make room for, so $\sigma_{\text{gen},i}^2$ is itself
small and $\rho_i$ blows past 1, forcing a clip on a $\beta$ that practitioners
expect to take a specific value.

The fix is a branch on the **real-data calibration $R^2$**. Pick a threshold
$R^2_{\text{ETF}}$ that separates index trackers from real single stocks. In
practice the universe sorts itself cleanly: index ETFs cluster near $R^2 = 1$
(SPY is exactly 1.0 by construction) and broad-basket ETFs sit in the 0.80–0.90
range against SPY (e.g. QQQ ≈ 0.84, SPYG ≈ 0.86), while the most market-like
real single stocks rarely exceed $R^2 \approx 0.65$. A threshold of 0.80
captures the ETFs without sweeping in any genuinely idiosyncratic name. For any
asset $i$ with $R^2_{i,\text{real}} \ge R^2_{\text{ETF}}$:

$$
g_i(t) \;=\; \alpha_i \;+\; \beta_i\, g_m(t) \quad\text{(no }\varepsilon_i\text{, no clipping)}
$$

with the calibrated $\beta_i$ used unchanged. Properties:

- $\hat{\beta}_i = \beta_i$ exactly, $R^2 = 1$ exactly. SPY recovers as 1.0,
  QQQ as its calibrated value.
- The marginal of $g_i$ is a deterministic affine transform of $g_m$, so it
  inherits the market path's tail behavior — which is correct for an index ETF.
- Cross-sectional dependence with the rest of the universe is whatever the
  copula on the *other* tickers' $\varepsilon$ already implies via the market
  channel. Two ETFs in this branch are perfectly rank-correlated through $g_m$
  — also correct, since they track overlapping baskets.

For assets with $R^2_{i,\text{real}} < R^2_{\text{ETF}}$, run the variance-
correction + clipping construction from the previous sections. The branch
selection should be persisted alongside the output (e.g. a per-ticker
`construction ∈ {"etf", "hybrid", "hybrid-clipped", "fallback"}` flag) so
downstream consumers can audit which path each asset took.

The $R^2$ branch is preferable to a hard-coded ticker list because (a) it
generalizes to any new ETF added to the universe, and (b) it derives entirely
from the calibration step, requiring no external metadata.

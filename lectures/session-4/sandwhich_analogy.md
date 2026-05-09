# The Sandwich Analogy: How DQN "Steps" Work in the Ticker-Picker MDP

A teaching note for the Session 4 Ticker-Picker DQN example. Useful when students confuse DQN steps with calendar days, or expect a reward at every step.

## The Setup

The DQN builds a basket of tickers one at a time. Each "episode" is the construction of one full basket from an empty start, finishing when the basket reaches a fixed size $K_{\text{basket}}$. There is no premature termination, no failure state. Every episode runs for exactly $K_{\text{basket}}$ steps and produces exactly one completed basket.

For our notebook with $K_{\text{basket}} = 8$, an episode looks like:

```
step 1:  empty mask              → agent picks ticker a₁
step 2:  one 1 in the mask       → agent picks ticker a₂
step 3:  two 1s in the mask      → agent picks ticker a₃
...
step 8:  seven 1s in the mask    → agent picks ticker a₈ → mask now has 8 ones, episode ends
```

Critically, all eight steps happen at the **same instant of wall-clock time** and the **same decision day**. The market does not move between steps. Prices, the preference vector $\boldsymbol{\gamma}$, the budget $B$, the sentiment $\lambda_t$, all are fixed for the entire episode. What changes step-to-step is which tickers the agent has committed to so far.

## The Sandwich

Imagine you are at a deli, assembling one sandwich. Each step is a composition decision, not a passage of time:

- step 1 = put down the bread
- step 2 = add cheese
- step 3 = add tomato
- ...
- step 8 = add the last topping, sandwich is complete

The clock is not moving while you build. You are standing at the counter, ingredients in front of you, choosing one to add at each turn. The "time" axis of the MDP is the count of decisions you have made, not seconds elapsed.

This maps cleanly to the ticker-picker:

| Sandwich | Ticker-picker MDP |
|---|---|
| ingredient | ticker (one of $K \approx 413$) |
| step | one composition decision |
| current sandwich (so far) | current basket-mask (which tickers committed) |
| finished sandwich | a $K_{\text{basket}}$-bit terminal mask |
| how good the sandwich is | $\widetilde{U}$, the signed log Cobb-Douglas utility of the final basket |

## Where the Analogy Breaks (and Why It Matters)

The natural follow-up question is: "OK, so step 1 = bread, step 2 = cheese. The reward for step 1 is the utility of the bread, the reward for step 2 is the utility of the cheese, and so on?"

**No.** This is where the analogy bends, and the bend is the whole pedagogical point.

There is no payoff for the bread alone. There is no payoff for bread plus cheese halfway through. The agent eats the sandwich only after the eighth ingredient is on it. Concretely, the reward function returns zero on every intermediate step and the full $\widetilde{U}$ only on the terminal step:

$$
r_t = \begin{cases} 0 & \text{if basket size} < K_{\text{basket}} \\ \widetilde{U}(\text{final basket}) & \text{if basket size} = K_{\text{basket}} \end{cases}
$$

Two reasons it has to be that way for our specific problem.

**1. The Cobb-Douglas utility is not decomposable across picks.** It is not the sum of per-ticker payoffs. It is

$$
\widetilde{U} = \kappa \cdot \sum_{i \in \mathcal{S}} \gamma_i \log n_i
$$

where the share count $n_i$ depends on the **whole basket** through the budget normalization

$$
n_i = \frac{\gamma_i}{\sum_{j \in \mathcal{S}} \gamma_j} \cdot \frac{B}{p_i}
$$

The "utility of AAPL alone" does not exist in the same units as the utility of the final basket. Adding a second positive-$\gamma$ ticker later changes the denominator and shrinks AAPL's share, which retroactively changes how much AAPL contributes to the eventual $\widetilde{U}$. There is no clean way to attribute a piece of the final utility to step 1's pick. Any per-step reward we made up would be a heuristic, not a faithful decomposition.

**2. The credit-assignment problem is the entire point of the DQN.** With sparse terminal rewards, the network learns "which step-1 picks tend to lead to high-$\widetilde{U}$ terminal states" by Bellman-bootstrapping the terminal reward back through the chain of intermediate states:

$$
y_t = r_t + \gamma \cdot (1 - d_t) \cdot \max_{a'} Q_{\theta^-}(s_{t+1}, a')
$$

For every intermediate step, $r_t = 0$ and $d_t = 0$, so the target collapses to $\gamma \cdot \max_{a'} Q_{\theta^-}(s_{t+1}, a')$, pure expected-future-reward propagation. After many training episodes, $Q_\theta(\text{empty mask}, \text{AAPL})$ converges toward an estimate of "the eventual basket value if I pick AAPL first and then continue greedily."

The network is **inferring** a per-ticker credit purely from terminal outcomes. We never tell it what the bread is worth on its own. We grade many sandwiches and let the network figure out which early ingredients tended to end up in great sandwiches.

## Two Clocks, Don't Confuse Them

There are two notions of time in the notebook, and the sandwich analogy applies only to one.

| Clock | Where it ticks | What advances |
|---|---|---|
| MDP step | inside the DQN training loop | one ticker added to the basket; market frozen |
| Trading day | inside the forward Cobb-Douglas engine | one calendar day; prices, $\lambda_t$, $\gamma_t$ all update |

When the DQN finishes a basket and Task 2 forward-walks it, the trading-day clock takes over. The basket is fixed and the engine rebalances daily for ~326 calendar days. Those days are real time. The DQN's eight steps were not.

## When You Hear "Terminal Reward"

It means **the reward delivered at the last step of the episode**, the bite of the finished sandwich. It does not mean "the reward at the end of the test window" or "the reward at the end of trading day so-and-so." Terminal here is in the MDP/DQN sense, the final state of one episode in the basket-construction process.

## When You Hear "Episode"

It means **one full basket build**, from empty to size $K_{\text{basket}}$. With the training schedule in the notebook, we run on the order of 800 such episodes during Task 1 and another 300 per refire during Task 3. Each episode is independent: empty mask, build to full, grade, throw away the basket, start over. The networks weights persist; the basket does not.

## What Goes Into the Replay Buffer

The DQN stores transitions in a replay buffer, just like any classical DQN setup, and the tuple shape is the same five-tuple you see in every textbook treatment, with the done flag added so the bootstrap can switch off at terminal states:

$$
(s_t, \; a_t, \; r_t, \; s_{t+1}, \; d_t)
$$

where $d_t \in \{\text{true}, \text{false}\}$ is the done flag. In our buffer, $s_t$ and $s_{t+1}$ are 413-bit basket masks, $a_t$ is an integer ticker index in $\{1, \ldots, K\}$, $r_t$ is a Float32 scalar, and $d_t$ is a Bool.

### What an episode produces

For one 8-step episode (with $K_{\text{basket}} = 8$), the buffer gets exactly 8 transitions, one per step:

```
step 1:  (s₀ = [0,0,...,0],   a₁ = 137, r₁ = 0,    s₁,            d₁ = false)
step 2:  (s₁,                 a₂ =  42, r₂ = 0,    s₂,            d₂ = false)
step 3:  (s₂,                 a₃ = 311, r₃ = 0,    s₃,            d₃ = false)
step 4:  (s₃,                 a₄ = 200, r₄ = 0,    s₄,            d₄ = false)
step 5:  (s₄,                 a₅ =  78, r₅ = 0,    s₅,            d₅ = false)
step 6:  (s₅,                 a₆ = 392, r₆ = 0,    s₆,            d₆ = false)
step 7:  (s₆,                 a₇ = 154, r₇ = 0,    s₇,            d₇ = false)
step 8:  (s₇,                 a₈ =  91, r₈ = Ũ,    s₈ = full,     d₈ = true )
```

Seven transitions carry zero reward and `done = false`. One terminal transition carries the actual signed-log utility and `done = true`. All eight go into the same replay buffer, mixed with transitions from every other episode the agent has run.

### What the bootstrap target does with each kind

When we sample a mini-batch of $B$ transitions during training, the target depends on the done flag:

$$
y_i = r_i + \gamma \cdot (1 - d_i) \cdot \max_{a'} Q_{\theta^-}(s'_i, a')
$$

- **Intermediate transition** ($d_i = 0, r_i = 0$): $y_i = \gamma \cdot \max_{a'} Q_{\theta^-}(s'_i, a')$. Pure expected-future-reward propagation, no reward signal of its own. Its job is to pass the eventual terminal $\widetilde{U}$ backward through the chain.
- **Terminal transition** ($d_i = 1, r_i = \widetilde{U}$): $y_i = \widetilde{U}$. The bootstrap is zeroed out by $(1 - d_i)$, so the target is just the realized terminal reward.

That is how credit assignment works concretely. The terminal $\widetilde{U}$ enters the buffer once per episode at the last step, and it propagates backward through the seven intermediate Q-targets over many training updates as those states get sampled later.

### Why we still store the zero-reward intermediate transitions

It might seem wasteful to store seven zero-reward transitions per episode. They matter because:

1. The intermediate $s_t$ vectors are the states from which the network needs to predict the right next pick. Without those tuples in the buffer, $Q_\theta(s_t, a_t)$ has no learning signal for non-empty baskets.
2. The Bellman update lifts the network's estimate at $s_t$ toward $\gamma \cdot \max_{a'} Q_\theta(s_{t+1}, a')$. That is how the eventual $\widetilde{U}$ at the terminal state seeps back. Each zero-reward transition is one link in that backward chain.

### Buffer math in our setup

- 800 episodes $\times$ 8 transitions = 6,400 transitions per Task 1 training run
- Buffer capacity = 8,000, so the whole training run fits in memory with room to spare
- About 12.5% of transitions are terminal (they carry the reward signal)
- About 87.5% are intermediate (they carry only the bootstrap signal)

After the warmup threshold of 400 transitions is reached, every subsequent step samples a random mini-batch of 64 from the buffer, computes targets per the formula above, and takes one Adam step on the squared Bellman residual. The target network is re-synced to the main network every 100 global steps. Same canonical DQN scaffolding as gridworld, just with a sparser reward distribution and a much bigger action space.

## Where the Featurized-State Extension Comes In

The Summary section flags a natural extension: feed the per-ticker preference weights $\gamma_i$ and SIM betas $\beta_i$ directly into the network as additional inputs alongside the basket mask. With those features in the state, the network can shortcut the credit-assignment work and recognize "high $\gamma_i$ tickers are good early picks" directly, instead of inferring it from terminal-only rewards. The reward structure stays terminal-only, but the network has more to look at while making each pick. Same sandwich, more recipe annotations on the wrapper.

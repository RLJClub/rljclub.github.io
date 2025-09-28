---
author: ["Ali Najar", "Mazdak Teymourian"]
title: "Nonstationarity in RL environments"
date: "2025-09-28"
description: "Planning and learning under drifting MDPs: risk-averse tree search (RATS) and confidence-widened, sliding-window optimism for sublinear dynamic regret."
summary: "We survey and synthesize approaches to non-stationary RL. First, RATS plans robustly via minimax tree search over Lipschitz-consistent model evolutions. Second, SWUCRL2 with confidence widening (and BORL for hyperparameter selection) attains sublinear dynamic regret without knowing drift budgets a priori."
tags: ["non-stationary-rl", "rats", "optimism", "dynamic-regret", "mdp", "planning"]
categories: ["Reinforcement Learning"]
cover:
  image: cover.png
  caption: "RATS: minimax planning under Lipschitz drift."
  hiddenInSingle: true
ShowToc: true
TocOpen: true
math: true
unsafe: True
---


{{< figure src="PosterSession.png" alt="Non-Stationarity in RL — poster preview"
    caption="*Non-Stationarity in RL — poster preview.*" class="center" >}}


[Download the PDF](PosterSession.pdf)



---

## ✦ Abstract ✦ <a name="abstract"></a>
Reinforcement learning (RL) agents traditionally assume a stationary environment, yet many real-world settings-from
robotics to finance-exhibit evolving dynamics and reward structures that violate this assumption. This non-stationarity
manifests as drifting transition probabilities, shifting objectives, or co-learning agents, leading to outdated value estimates,
policy instability, and performance degradation. Motivated by the ubiquity of time-varying phenomena in
practical applications, our work investigates algorithms that provably adapt to changing MDPs without prior knowledge
of drift magnitude or change points. We aim to (1) characterize the fundamental limits of learning under bounded
and unbounded non-stationarity, (2) design both model-based and model-free methods, leveraging sliding-window
estimation, exponential forgetting, and optimism-driven exploration, to minimize dynamic regret, and (3) extend
these techniques to deep and multi-agent settings where high-dimensional representations and adversarial co-learners
exacerbate non-stationarity. Through theoretical analysis and empirical evaluation, we seek to deliver scalable RL
solutions that maintain robust performance across a spectrum of evolving environments.

**Keywords:** Non-stationary RL; non-stationary MDPs; policy optimization; dynamic regret.

---

## ✦ Introduction ✦ <a name="introduction"></a>

Reinforcement learning (RL) studies how an agent interacts with an environment over time in order to maximize cumulative reward. The environment is commonly modeled as a **Markov Decision Process (MDP)**, defined by

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, H),
$$

where  
- $\mathcal{S}$ is the state space,  
- $\mathcal{A}$ the action space,  
- $P(\cdot \mid s,a)$ the transition kernel,  
- $r(s,a)$ the reward function, and  
- $H$ the horizon length.  

A (possibly stochastic) **policy** $\pi$ specifies a distribution over actions given the current state:

$$
\pi(a \mid s) = \Pr[a_t = a \mid s_t = s].
$$

The expected return under $\pi$, starting from state $s$, is

$$
V^\pi(s) = \mathbb{E}\_{\pi} \left[\sum_{h=1}^H r(s_h,a_h)\,\middle|\,s_1=s \right].
$$

The agent’s goal in a stationary MDP is to find an optimal policy $\pi^\star$ maximizing $V^\pi$.

<hr style="width:40%;margin:2em auto -1em auto;">
### Non-stationarity
<hr style="width:40%;margin:-0.5em auto auto auto;">

In practice, the assumption of a fixed $(P,r)$ is often unrealistic. Consider:  
- a robot navigating terrain that erodes or changes over time,  
- an online recommender facing evolving user preferences,  
- an autonomous trader in financial markets influenced by competitors’ strategies.  

In all these cases, the environment itself evolves with time. A natural model is a **time-varying MDP sequence**

$$
\mathcal{M}_t = (\mathcal{S}, \mathcal{A}, P_t, r_t, H), \quad t=1,\dots,T,
$$

where both $P_t$ and $r_t$ may drift.  

For each round $t$, the optimal policy is

$$
\pi_t^\star \in \arg\max_\pi V_{t,1}^\pi(s_{t,1}), 
\qquad
V_{t,1}^\pi(s) = \mathbb{E}\_{\pi,P_t}\left[\sum_{h=1}^H r_t(s_h,a_h) \,\middle|\, s_{t,1}=s \right].
$$

Thus, the learning target is not a single fixed policy but rather the **moving sequence** $\{\pi_t^\star\}_{t=1}^T$.

<hr style="width:40%;margin:2em auto -1em auto;">
### Models of drift
<hr style="width:40%;margin:-0.5em auto auto auto;">

Two widely used ways to quantify non-stationarity are:

1. **Bounded variation (drift budgets).**  
   Define cumulative variation in rewards and transitions:

   $$
   B_r = \sum_{t=1}^{T-1} \max_{s,a} \big|r_{t+1}(s,a) - r_t(s,a)\big|,
   \qquad
   B_p = \sum_{t=1}^{T-1} \max_{s,a} \big\| P_{t+1}(\cdot\mid s,a) - P_t(\cdot\mid s,a)\big\|_1.
   $$

   Here, $B_r,B_p$ serve as “budgets” quantifying how much the environment drifts.  
   - If $B_r=B_p=0$, the problem reduces to the stationary case.  
   - A **piecewise-stationary** MDP is a special case with finitely many abrupt change points, giving finite $B_r,B_p$.

2. **Lipschitz-continuous (smooth drift).**  
   Assume per-step changes are bounded by drift rates $L_r,L_p$. For all $\Delta \ge 1$,

   $$
   |r_{t+\Delta}(s,a) - r_t(s,a)| \le L_r \Delta, \quad
   W_1\big(P_{t+\Delta}(\cdot|,a),P_t(\cdot | ,a)\big) \le L_p \Delta,
   $$

   where $W_1$ is the Wasserstein-1 distance. This captures smoothly and gradually evolving environments.

<hr style="width:40%;margin:2em auto -1em auto;">
### Learning objective
<hr style="width:40%;margin:-0.5em auto auto auto;">

Because $\pi_t^\star$ shifts over time, an RL algorithm cannot hope to achieve zero regret with respect to a fixed comparator. Instead, the relevant benchmark is **dynamic regret**, defined as

$$
\text{DynRegret}(T) = \sum_{t=1}^T \Big(V_{t,1}^{\pi_t^\star}(s_{t,1}) - V_{t,1}^{\pi_t}(s_{t,1})\Big),
$$

where $\pi_t$ is the algorithm’s policy at time $t$.  

Minimizing dynamic regret requires a careful balance:  
- **Adaptation.** Algorithms must forget or discount outdated data that no longer reflects the environment.  
- **Exploration.** At the same time, they must actively probe the environment to reduce uncertainty in the current dynamics.  

This exploration–adaptation tradeoff is central to learning in non-stationary MDPs, and motivates algorithms such as **RATS** (robust planning under Lipschitz drift) and **confidence-widened sliding-window RL** (provable guarantees under bounded variation).


---

## ✦ Risk-Averse Tree-Search (RATS) ✦ <a name="rats"></a>


Imagine you are controlling a robot in a world that is constantly changing.  
If your robot plans assuming that today’s model will remain the same tomorrow, its strategy may fail badly.  
This is the **fundamental challenge of non-stationarity**: plans that ignore possible changes in the environment are brittle.

How can we design an RL agent that **plans robustly in the face of uncertainty about the future**?

<hr style="width:40%;margin:2em auto -1em auto;">
### The idea behind RATS
<hr style="width:40%;margin:-0.5em auto auto auto;">

**Risk-Averse Tree-Search (RATS)** [Lecarpentier & Rachelson, 2019] addresses this challenge by adopting a minimax view of planning.  
Instead of assuming the environment will behave according to a single estimated model, RATS treats nature as an **adversary**:

- At each decision point, the agent considers all **Lipschitz-consistent evolutions** of the environment (i.e., all possible ways transitions and rewards could drift within known smoothness bounds).
- It then evaluates the worst-case return across these scenarios.
- Finally, it chooses the action that maximizes this worst-case value.

In short:  

$$
\text{Agent’s choice} = \arg\max_{a \in \mathcal{A}} \;\min_{\substack{(P,r) \text{ consistent with drift}}} \; \text{Return}(a).
$$

This robust strategy ensures the agent is never caught off-guard by sudden adversarial changes.

<hr style="width:40%;margin:2em auto -1em auto;">
### Methodology
<hr style="width:40%;margin:-0.5em auto auto auto;">

<h4 style="margin-top:2em; margin-bottom:1em;">NC-NSMDP definition</h4>

An NC-NSMDP with constants $(L_r, L_p)$ is a sequence of MDPs

$$
\mathcal{M}_t = (\mathcal{S}, \mathcal{A}, P_t, r_t, H), \quad t \in \text{ℕ},
$$

such that for all $s \in \mathcal{S}$, $a \in \mathcal{A}$, and $\Delta \ge 1$:

- **Reward regularity:**

  $$
  |r_{t+\Delta}(s,a) - r_t(s,a)| \le L_r \Delta,
  $$

- **Transition regularity:**

  $$
  W_1\left( P_{t+\Delta}(\cdot | s,a), P_t(\cdot | s,a) \right) \le L_p \Delta,
  $$
  
where $W_1$ is the 1-Wasserstein distance.  
This definition bounds how fast the environment can evolve.


<h4 style="margin-top:2em; margin-bottom:1em;">Admissible models and robust values</h4>

At a given time $t_0$, the agent only has access to the current snapshot $(P_{t_0}, r_{t_0})$.  
Since the exact future models are unknown, we restrict them to the **admissible set**: all possible models that are consistent with the Lipschitz drift bounds.

Formally, for each $(s,a)$ and future time $t \ge t_0$:

$$
\Delta_{t_0}^t(s,a)=\{(P_t,r_t) | \|P_t(\cdot|s,a)-P_{t_0}(\cdot|s,a)\|\_1\le L_p(t-t_0),\ |r_t(s,a)-r_{t_0}(s,a)|\le L_r(t-t_0)\}
$$

- The first constraint says that the transition distribution cannot drift faster than $L_p$ per time step (in $\ell_1$ or Wasserstein distance).  
- The second constraint says that the immediate reward cannot drift faster than $L_r$ per time step.  

So $\Delta_{t_0}^t$ is a **set of plausible MDPs at time $t$**, centered around the snapshot $MDP_{t_0}$ and expanding as $t$ grows.


**Robust value function:** Given this uncertainty, the agent evaluates a policy $\pi$ pessimistically, against the worst admissible sequence of models.  
This defines the **worst-case value function**:

$$
V^\pi_{t_0}(s) =
\min_{(P_i,r_i)\in \Delta_{t_0}^i}
\mathbb{E}\_\pi \Bigg[\sum_{i=t_0}^\infty \gamma^{i-t_0} r_i(s_i,a_i) \;\Big|\; s_{t_0}=s \Bigg].
$$

Here the minimization ranges over **entire trajectories of models** $(P_i, r_i)$ that respect the drift constraints.  

**Robust Q-function:** Similarly, the **worst-case Q-function** (starting with an action $a$ in state $s$) is

$$
Q^\pi_{t_0}(s,a) =
\min_{(P_i,r_i)\in \Delta_{t_0}^i}
\mathbb{E}\_\pi \Bigg[\sum_{i=t_0}^\infty \gamma^{i-t_0} r_i(s_i,a_i) \;\Big|\; s_{t_0}=s, a_{t_0}=a \Bigg].
$$

This captures the pessimistic return if the agent commits to action $a$ first, and then follows $\pi$ under adversarial dynamics.

**Why this matters:** These definitions turn non-stationary RL into a **two-player game**:

- The *agent* chooses a policy $\pi$.  
- *Nature* chooses an admissible sequence of models $\{(P_i, r_i)\}$ to minimize the agent’s return.  

Thus, planning in NC-NSMDPs reduces to computing **minimax value functions**, which RATS approximates using its search tree.  

<h4 style="margin-top:2em; margin-bottom:1em;">Minimax tree search</h4>

RATS evaluates policies by recursive minimax tree search:

- **Decision nodes (agent’s choice):**

  $$
  V(\nu) = \max_{\nu' \in \nu.\mathrm{children}} V(\nu').
  $$

- **Chance nodes (nature’s move):**

  $$
  V(\nu) = \min_{(P,R)\in \Delta_{t_0}^t}
    \Big[ R(\nu) + \gamma \sum_{\nu'\in \nu.\mathrm{children}}
        P(\nu' \mid \nu)\, V(\nu') \Big].
  $$

This search proceeds until a depth $D$, where a heuristic is used.

{{< figure src="RAT-inv.png" alt="RATS minimax search tree"
    caption="*RATS minimax search tree. Circles = agent nodes, squares = nature.*" class="center" >}}

<hr style="width:40%;margin:2em auto -1em auto;">
### Theoretical Guarantees
<hr style="width:40%;margin:-0.5em auto auto auto;">

<h4 style="margin-top:2em; margin-bottom:1em;">Heuristic error bound</h4>

Let $H(s)$ be the heuristic value used at leaves of depth $d_{\max}$.  
If $H$ has uniform error $\delta$ with respect to the true value, then the propagated error at a node $\nu$ at depth $d$ is bounded by

$$
|V(\nu) - V^*(\nu)| \;\le\; \gamma^{d_{\max}-d} \, \delta.
$$

In particular, the root node error is bounded by $\gamma^{d_{\max}} \delta$.  
- If $H=0$, the bound still holds but is conservative.  
- Better heuristics reduce $\delta$, tightening guarantees.

<h4 style="margin-top:2em; margin-bottom:1em;">Snapshot value drift bound</h4>

Let $V^{\pi}_{MDP_t}(s)$ be the value of policy $\pi$ in the snapshot MDP at time $t$.  
Then for any two times $t, t_0$:

$$
| V^{\pi}\_{MDP_{t_0}}(s) - V^{\pi}\_{MDP_t}(s) |
\le \frac{|t - t_0| \, L_R}{1-\gamma},
\qquad L_R := L_r + L_p.
$$

This provides a principled way to construct **pessimistic heuristics**: by subtracting a drift-dependent margin from snapshot values, one obtains a safe lower bound for use at leaf nodes.

<h4 style="margin-top:2em; margin-bottom:1em;">Computational complexity</h4>

Constructing a RATS tree of depth $d_{\max}$ has complexity

$$
O\Big( B \, |S|^{1.5} |A| \, (|S||A|)^{d_{\max}} \Big),
$$

where $B$ is the horizon, $|S|$ the number of states, and $|A|$ the number of actions.  
The main cost comes from evaluating worst-case transitions using Wasserstein computations.

<hr style="width:40%;margin:2em auto -1em auto;">
### Additional Experiments
<hr style="width:40%;margin:-0.5em auto auto auto;">

<h4 style="margin-top:2em; margin-bottom:1em;">Performance across drifts</h4>

{{< figure src="ratres.png" alt="Discounted return vs. drift ε"
    caption="*Discounted return vs. drift ε.*" class="center" >}}

- As drift $\epsilon$ increases, snapshot planning collapses: expected return degrades severely since it ignores model evolution.  
- **DP-NSMDP** maintains strong performance, but it has access to privileged information (future models).  
- **RATS** closely tracks the oracle’s performance, even under large $\epsilon$, showing its robustness.

<h4 style="margin-top:2em; margin-bottom:1em;">Risk-sensitive performance (CVaR)</h4>

While the above guarantees are worst-case, experiments show that RATS also consistently improves Conditional Value-at-Risk (CVaR) at level $\alpha=5\%$.  
This means RATS not only secures the worst case mathematically, but also empirically minimizes downside risk, yielding safer behavior under uncertainty.

{{< figure src="ratres2.png" alt="Return distributions"
    caption="*Return distributions for ε ∈ {0, 0.5, 1}.*" class="center" >}}
---

## ✦ The Blessing of Optimism ✦ <a name="optimism"></a>


Reinforcement learning thrives on optimism in the face of uncertainty:  
when we are unsure about transitions or rewards, we pretend they are as good as plausibly possible, then explore.  
This principle underlies algorithms like **UCRL2**, which achieve tight regret bounds in stationary MDPs.

But what if the world itself is drifting?  
In non-stationary environments, naive optimism can backfire: the optimistic model may imagine wildly long horizons or exploding diameters, leading to poor control of regret.

The **blessing of optimism** in drifting MDPs is more subtle:  
sometimes **more optimism, not less**, actually restores robustness.

<hr style="width:40%;margin:2em auto -1em auto;">
### Methodology
<hr style="width:40%;margin:-0.5em auto auto auto;">

The key idea is to adapt **sliding-window UCRL2 (SW-UCRL2)** with a technique called **confidence widening**.

<h4 style="margin-top:2em; margin-bottom:1em;">Sliding-window estimation</h4>

Instead of pooling *all* past data, the algorithm uses only the last $W$ steps to estimate rewards and transitions:

- Reward estimate:

  $$
  \hat r_t(s,a) = \frac{1}{N_t(s,a)} \sum_{\tau=t-W}^{t-1} r_\tau(s,a) \cdot \mathbf{1}\{(s_\tau,a_\tau)=(s,a)\}.
  $$

- Transition estimate:

  $$
  \hat p_t(\cdot \mid s,a) = \frac{1}{N_t(s,a)} \sum_{\tau=t-W}^{t-1} \mathbf{1}\{(s_\tau,a_\tau)=(s,a)\}\, \delta_{s_{\tau+1}}.
  $$

Here $N_t(s,a)$ counts how often $(s,a)$ appears in the last $W$ steps.

This sliding window allows the agent to forget outdated data and adapt to drift.

<h4 style="margin-top:2em; margin-bottom:1em;">Confidence widening</h4>

Classical UCRL2 defines confidence sets around $(\hat r_t,\hat p_t)$:

- Rewards:

  $$
  \mathcal{H}\_{r,t}(s,a) = \{\tilde r : |\tilde r - \hat r_t(s,a)| \le \mathrm{rad}_{r,t}(s,a)\}.
  $$

- Transitions:

  $$
  \mathcal{H}\_{p,t}(s,a) = \{\tilde p : \|\tilde p - \hat p_t(\cdot\mid s,a)\|\_1 \le \mathrm{rad}_{p,t}(s,a)\}.
  $$

But in drifting MDPs, these sets may be too tight, and the optimistic model may have an artificially huge effective diameter.  
The fix is widening:

$$
\mathcal{H}\_{p,t}(s,a;\eta) = 
\{\tilde p : \|\tilde p - \hat p_t(\cdot\mid s,a)\|\_1 \le \mathrm{rad}_{p,t}(s,a) + \eta\}.
$$

Here $\eta > 0$ gives extra slack, ensuring that the chosen optimistic model remains well-behaved.

<h4 style="margin-top:2em; margin-bottom:1em;">Optimistic planning</h4>

At the start of each episode $m$ (with time $\tau(m)$), the algorithm computes an optimistic policy via **Extended Value Iteration (EVI)**:

$$
\tilde \pi_m = \text{EVI}\big(\mathcal{H}\_{r,\tau(m)},\ \mathcal{H}\_{p,\tau(m)}(\eta);\ \epsilon_m\big),
$$

where $\epsilon_m = 1/\sqrt{\tau(m)}$ controls the accuracy.

The agent then follows $\tilde \pi_m$ until enough new data is collected to update estimates.

<h4 style="margin-top:2em; margin-bottom:1em;">Pseudocode</h4>

```python
def SWUCRL2_CW(T, S, A, W, eta):
    t = 1
    s = initial_state()
    for m in range(1, ∞):
        τ = t
        ν = { (s,a): 0 for s in S for a in A }

        # compute confidence sets using sliding window + widening
        Hr = build_reward_confidence_sets(τ, W)
        Hp = build_transition_confidence_sets(τ, W, eta)

        # compute optimistic policy
        π_tilde = ExtendedValueIteration(Hr, Hp, eps=1/sqrt(τ))

        # follow policy until stopping condition
        while not multiple_of_W(t) and ν[s, π_tilde(s)] < N_plus(τ, s, π_tilde(s)):
            a = π_tilde(s)
            r, s_next = step_env(s, a)
            ν[s,a] += 1
            s = s_next
            t += 1
            if t > T:
                return
```

<hr style="width:40%;margin:2em auto -1em auto;">
### Theoretical Guarantees
<hr style="width:40%;margin:-0.5em auto auto auto;">

This algorithm - **SWUCRL2–CW** - achieves sublinear dynamic regret under non-stationarity.

**Dynamic regret.** Define dynamic regret as the gap to the best per-round policy sequence $\{\pi_t^\star\}$:
  
$$
\operatorname{DynReg}(T)=\sum_{t=1}^T\left( V_{t,1}^{\pi_t^\star}(s_{t,1})-V_{t,1}^{\pi_t}(s_{t,1})\right).
$$

**Theorem (Cheung et al., 2020).**  
With window size $W$ and widening parameter $\eta$ tuned to variation budgets $(B_r,B_p)$, SWUCRL2–CW guarantees

$$
\operatorname{DynReg}(T)\;=\;\tilde{\mathcal O}\Big(\big(D_{\max}(B_r{+}B_p)\big)^{1/4}\,S^{2/3}\,A^{1/2}\,T^{3/4}\Big),
$$

where $D_{\max}$ is the MDP diameter.

Moreover, the meta-algorithm **BORL** (Bandit-over-RL) can learn $(W,\eta)$ online and achieves the same bound **without prior knowledge of $(B_r,B_p)$**.


{{< figure src="BORL-inv.png" alt="BORL structure"
    caption="*Structure of BORL algorithm.*" >}}

**BORL (Bandit-over-RL).**  
The idea is to treat SWUCRL2–CW as a base learner with hyperparameters $(W,\eta)$ and then run an adversarial bandit (EXP3.P) over a finite set of candidates  
$\mathcal{K}=\{(W_k,\eta_k)\}_{k=1}^K$. Each candidate is an “arm.” Time is partitioned into blocks of length $H$ (picked so that one run of SWUCRL2–CW has enough samples to stabilize its windowed estimates).

At the start of block $m$:
1. **Bandit selection.** EXP3.P samples an arm $k_m\in\{1,\dots,K\}$ with probability $p_{k,m}$.
2. **Deploy base learner.** Run SWUCRL2–CW with $(W_{k_m},\eta_{k_m})$ for the next $H$ steps, producing realized block return $R_m$ (or loss $\ell_m$).
3. **Feedback to the bandit.** Construct an importance-weighted loss estimate (normalized to $[0,1]$),

   $$
   \hat{\ell}\_{k,m} \;=\; \frac{\ell_m}{p_{k_m,m}}\,\mathbf{1}\{k=k_m\},
   $$

   and update EXP3.P’s weights. This lets the bandit learn which $(W,\eta)$ works best under the current non-stationarity.

Intuition: different drifts favor different windows $W$ (short windows react faster; long windows reduce variance) and different widenings $\eta$ (extra optimism keeps the optimistic model’s effective diameter controlled under drift). BORL **adapts online** by routing data to the currently promising configuration.

**Why blocks?**  
- SWUCRL2–CW updates policies episodically; giving it a contiguous budget of $H$ steps per arm avoids churning and lets its confidence sets/optimistic planning stabilize within each block.  
- The bandit only needs a scalar loss per block, keeping EXP3.P’s feedback simple and adversarially robust.

**Loss choice.**  
Any block-level surrogate that orders arms correctly works (and is clipped to $[0,1]$ for EXP3.P): e.g., negative average return, or a proxy for dynamic regret over the block:

$$
\ell_m \;\approx\; \frac{1}{H}\sum_{t \in \text{block } m}\Big(V_{t,1}^{\pi_t^\star}(s_{t,1})-V_{t,1}^{\pi_t}(s_{t,1})\Big),
$$

estimated from the same statistics SWUCRL2–CW maintains (sliding-window counts, confidence radii).

**Grid design.**  
Pick a small geometric grid for $W$ (e.g., $W\in\{2^0,2^1,\dots\}$ up to a cap) and a few $\eta$ levels (e.g., $\eta\in\{0,\eta_0,2\eta_0,\dots\}$). This keeps $K$ modest (logarithmic choices), so EXP3.P’s overhead is tiny.

**High-level guarantee (why BORL matches the theorem).**  
Let $M=\lceil T/H\rceil$ be the number of blocks. EXP3.P ensures bandit regret

$$
\tilde{\mathcal{O}}\big(\sqrt{M\log K}\big)
$$

against the best fixed arm in hindsight. The best arm corresponds to a (near-)optimal $(W^\star,\eta^\star)$ for the unknown drift $(B_r,B_p)$, whose base regret over $T$ steps is

$$
\tilde{\mathcal{O}}\big(\big(D_{\max}(B_r{+}B_p)\big)^{1/4} S^{2/3} A^{1/2} T^{3/4}\big).
$$

Choosing a reasonable block length $H$ and a compact grid $\mathcal{K}$ makes the bandit overhead lower order, so BORL achieves the same $\tilde{\mathcal O}(T^{3/4})$ dynamic-regret rate, **without** knowing $(B_r,B_p)$ in advance.


---

## ✦ References ✦ <a name="references"></a>


- Lecarpentier, Erwan, Rachelson, Emmanuel. *Non-Stationary Markov Decision Processes: A Worst-Case Approach using Model-Based Reinforcement Learning.* NeurIPS, 2019. [https://arxiv.org/abs/1904.10090](https://arxiv.org/abs/1904.10090)

- Cheung, Wang Chi, et al. *Reinforcement Learning for Non-Stationary Markov Decision Processes: The Blessing of (More) Optimism.* ICML, 2020. [https://arxiv.org/abs/2006.14389](https://arxiv.org/abs/2006.14389)
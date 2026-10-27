# Part D: Analysis & Theoretical Questions

---

## D1. Convergence & Value Map Evolution

> **Question:** At which episode does each agent's greedy path from $S$ first reach $G$? Why does Q-learning's value map settle so much sooner than SARSA's?

### 1. Episode Milestones
* **Q-Learning:** Greedy path first reaches the goal ($G$) at **Episode 15**.
* **SARSA:** Greedy path first reaches the goal ($G$) at **Episode 70**.

### 2. Theoretical Analysis
Q-learning is an **off-policy** algorithm that updates its Q-values using the maximum possible future action:

$$\max_a Q(s', a)$$

It operates independently of the agent's current $\epsilon$-greedy exploratory behavior. This enables optimal values and the greedy path to propagate backward from the goal and stabilize very quickly. 

In contrast, SARSA is an **on-policy** algorithm whose updates depend directly on the *actual* next action chosen by the $\epsilon$-greedy policy ($A'$). Because SARSA factors ongoing exploration noise into its value updates, its value map and greedy path take significantly longer to settle.

---

## D2. On-Policy vs. Off-Policy Behavior near Penalties

> **Question:** Look at the row just above the cliff in the last frame. Why are SARSA's values there so much lower than Q-learning's, even though both agents saw the same kind of experience? What would happen to SARSA's values (and path) if $\epsilon$ decayed all the way to 0?

### 1. Lower Values for SARSA
SARSA evaluates state-action pairs based on actual policy execution. Because $\epsilon > 0$ introduces random exploratory steps, a SARSA agent walking directly adjacent to the cliff risks accidentally stepping off and incurring a $-100$ penalty. SARSA incorporates this exploration risk directly into its value updates, learning a safer path further away from the cliff. 

Q-learning assumes optimal behavior after the next step and ignores exploration risk in its update target, resulting in much higher, optimistic values along the cliff edge.

### 2. Behavior when $\epsilon \to 0$
With zero exploration ($\epsilon = 0$), accidental steps into the cliff cease entirely. SARSA would converge to the exact same policy as Q-learning, its edge state values would rise, and it would switch to taking the shorter, optimal path directly along the cliff edge.

---

## D3. Action Space Mapping & Visualization

> **Question:** CliffWalking and FrozenLake number their actions differently. How did you make sure the arrows in your video are correct?

We ensured visualization correctness by establishing environment-specific action-mapping dictionaries (`CLIFF_ARROWS` and `FROZEN_ARROWS`):

* **CliffWalking:** $0 = \text{Up},\; 1 = \text{Right},\; 2 = \text{Down},\; 3 = \text{Left}$
* **FrozenLake:** $0 = \text{Left},\; 1 = \text{Down},\; 2 = \text{Right},\; 3 = \text{Up}$

Inside our plotting function (`plot_q_heatmap`), the extracted greedy action index $\arg\max_a Q(s, a)$ is dynamically looked up using the appropriate arrow mapping dictionary for that environment, guaranteeing that rendered arrows accurately represent intended movement directions.

---

## D4. Truncation vs. Termination in Temporal Difference Learning

> **Question:** Why must the 100-step limit set `truncated=True` and not `terminated=True`? Write the TD target for both cases and explain what goes wrong if a truncated episode is treated as terminal.

### 1. Key Distinctions
* **`terminated=True`:** Indicates reaching a natural terminal state defined by the environment's MDP dynamics (e.g., falling into a hole or reaching the goal). No future rewards exist beyond this state ($V(s_{\text{terminal}}) = 0$).
* **`truncated=True`:** Represents an artificial early stopping condition imposed by the experimenter (e.g., hitting a 100-step time limit) while the agent remains in a non-terminal state.

### 2. Temporal Difference (TD) Targets

* **Proper Truncation (`truncated=True`):**
  Preserves the bootstrapped future value estimate for $s_{t+1}$:
  $$\text{TD Target (Truncated)} = R_t + \gamma \max_{a} Q(s_{t+1}, a)$$

* **False Termination (`terminated=True`):**
  Forces future value estimation to zero ($0$):
  $$\text{TD Target (Terminal)} = R_t + 0 = R_t$$

### 3. Consequences of Incorrect Implementation
If a truncated state is incorrectly treated as terminal, its bootstrap value is forcefully overwritten to $0$. Since stepping onto normal ice yields a reward of $0$, the agent falsely learns that transitioning into a step-limited state carries zero total future value. This introduces massive artificial negative bias, making safe states appear as penalizing as falling into a hole. Consequently, Q-values fail to converge to $V^*$, and the agent actively avoids valid pathways simply because they take longer.

---

## D5. Custom FrozenLake Hyperparameter Evaluation

> **Question:** Paste your Part C table. Why do the lab settings (500 episodes, decay 0.995, alpha 0.1) fail here? Which change helped most, and why?

### 1. Results Summary

```text
Policy                     Value ratio  Worst seed  Success   Time (s)  Status
--------------------------------------------------------------------------------
DP optimal                       1.000                           0.593
SARSA (lab settings)             0.634       0.503    0.334        1.0
Q-learning (lab settings)        0.503       0.246    0.267        0.9
SARSA (my settings)              0.916       0.852    0.540        6.2   PASS
Q-learning (my settings)         0.983       0.949    0.589        7.6   PASS
```

### 2. Failure Analysis of Lab Settings
The default lab settings (`num_episodes=500`, `alpha=0.1`, `decay_rate=0.995`) fail due to the high stochasticity of `MyFrozenLake` (`slip_intended = 0.600`):

1. **Insufficient Episodes:** $500$ episodes provide too few transition samples across a highly stochastic state space.
2. **Premature Decay:** A decay rate of $0.995$ reduces $\epsilon$ to its minimum within $\sim 200$ episodes, cutting off exploration before optimal paths are identified.
3. **High Learning Rate:** $\alpha = 0.1$ causes Q-values to fluctuate erratically in response to random transition outcomes.

### 3. Key Improvement
Increasing **`num_episodes` to 5000** alongside slowing down **`decay_rate` to `0.9995`** yielded the greatest performance gain. This maintained exploration long enough to collect robust transition statistics, allowing Q-values to settle accurately and achieve a value ratio $>0.90$.

---

## D6. Discount Factors vs. Success Rates

> **Question:** The DP-optimal policy maximises the DISCOUNTED return ($\gamma = 0.99$). Can a learned policy reach the goal within 100 steps MORE often than the DP-optimal one but still have a lower value? Explain.

**Yes.** The value function measures the **expected discounted return**:

$$\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_t\right]$$

It does not measure unweighted goal arrival frequency.

### Explanation
A learned policy may adopt a longer, highly cautious route to avoid holes. Because it takes more steps on average to reach the goal, the discount factor $\gamma^t$ penalizes the trajectory ($\gamma^{t+1} < \gamma^t$). Consequently, even if this policy achieves a higher raw success rate, its cumulative discounted return remains lower than the DP-optimal policy, which favors shorter, slightly riskier paths that yield immediate high-value returns.

---

## D7. Policy Divergence & Q-Value Analysis

> **Question:** In which states does your learned greedy policy differ from DP? Use $Q^*(s, \cdot)$ printed by Part C to explain why each difference costs little (a near-tie) or why the agent never learned it (a rarely visited state).

### Q-Value Analysis
* **Near-Tie Situations:** In states where the learned greedy action differs from DP, inspecting the $Q^*(s, \cdot)$ array shows that competing action values are nearly identical (e.g., $0.511$ vs. $0.502$). Because the difference in expected return is minimal, choosing the sub-optimal action introduces negligible performance loss.
* **Rarely Visited States:** Larger gaps occur in states located away from primary optimal trajectories. Due to environmental stochasticity, the agent rarely encounters these states during training, resulting in under-sampled Q-values and slight policy divergences from DP.

---

## D8. Environment Dynamics & Path Convergence

> **Question:** On CliffWalking SARSA and Q-learning learn different paths. On your FrozenLake they end up almost the same. Explain why.

* **CliffWalking:** The environment is **deterministic** with a catastrophic step penalty along the cliff. SARSA (*on-policy*) must account for exploration noise ($\epsilon > 0$) and learns a wider, safer path. Q-learning (*off-policy*) assumes greedy post-step action selection and learns the direct, higher-risk path along the edge.
* **FrozenLake (`MyFrozenLake`):** The environment is highly **stochastic** (`slip_intended = 0.600`). Random transition dynamics dominate over algorithmic differences between on-policy and off-policy updates. Both algorithms face identical physical uncertainty on the ice and must converge to similar conservative policies to navigate high slip rates successfully.
```

eof

The formatting for your report (`report.md`) is now complete and ready to share or export. Let me know if you need any additional edits!
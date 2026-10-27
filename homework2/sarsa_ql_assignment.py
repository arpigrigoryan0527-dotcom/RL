"""
=====================================================================
 ASSIGNMENT: SARSA & Q-Learning - Watching Q Learn,
 and Building Your Own FrozenLake
=====================================================================
What is given:     SARSA and Q-learning (the Week 4 lab versions, with a
                   time-limit fix), value iteration from Week 3, the map
                   generator, the evaluation and plotting helpers.
What YOU write:    Part A - a Q-table heat-map video on CliffWalking
                   Part B - your own FrozenLake environment
                   Part C - tune the agents on YOUR environment and compare
                            them with the exact DP answer
                   Part D - written answers (see the handout / bottom of file)
Search for "TODO" to find every place you need to write code.

Requirements:  pip install "gymnasium>=1.0" numpy matplotlib
Run:           python sarsa_ql_assignment.py
Self-check:    python check_my_work.py
=====================================================================
"""

import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")          # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib.animation import FuncAnimation, PillowWriter

# Arrow for each action number. CAREFUL: the two environments number their
# actions differently!
FROZEN_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}   # FrozenLake: Left, Down, Right, Up
CLIFF_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}    # CliffWalking: Up, Right, Down, Left
MAP_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]     # the standard Gymnasium map

# ---------------------------------------------------------------
# STEP 0: Enter your student ID (whole number).
# Your ID generates YOUR OWN FrozenLake map and slip probability.
# Do not change make_student_config().
# ---------------------------------------------------------------
STUDENT_ID = 361110  # TODO: replace 0 with your student ID


def make_student_config(student_id):
    """GIVEN. Returns (desc, slip_intended) for this student.
    desc:          list of 4 strings, e.g. ["SFFF", "FHFH", "FFFH", "HFFG"]
    slip_intended: probability that the agent moves in the chosen direction;
                   each of the two perpendicular directions gets half the rest.
    Maps are re-drawn until they have at least 4 holes and the best possible
    policy (on standard 1/3 ice) reaches the goal 40-92% of the time, so every
    lake is solvable but not trivial."""
    rng = np.random.default_rng(student_id)
    slip_intended = float(rng.choice([1 / 3, 0.5, 0.6]))
    while True:
        desc = generate_random_map(size=4, p=0.7, seed=int(rng.integers(2**31)))
        if sum(row.count("H") for row in desc) < 4:
            continue
        ref = gym.make("FrozenLake-v1", desc=desc, is_slippery=True).unwrapped.P
        if 0.4 <= _reach_probability(ref, n_states=16) <= 0.92:
            return desc, slip_intended


def _reach_probability(P, n_states, horizon=100):
    """Best probability of reaching G within `horizon` steps (finite-horizon DP)."""
    v = np.zeros(n_states)
    for _ in range(horizon):
        v = np.array([max(sum(p * (r + (0 if term else v[s2])) for p, s2, r, term in P[s][a])
                          for a in P[s]) for s in range(n_states)])
    return v[0]


# ---------------------------------------------------------------
# GIVEN: SARSA and Q-learning (from the Week 4 lab, with two changes)
#   1. on_episode_end(episode_number, Q, epsilon) is called after every
#      episode - you will use it in Part A to take snapshots.
#   2. Only `terminated` stops bootstrapping. A `truncated` episode (time
#      limit) still has a future, so the target keeps gamma * Q(s', .).
# ---------------------------------------------------------------
def epsilon_greedy(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))


def sarsa_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, initial_epsilon=1.0,
                   min_epsilon=0.01, decay_rate=0.995, seed=0, on_episode_end=None):
    """Q(S,A) <- Q(S,A) + alpha [R + gamma Q(S',A') - Q(S,A)]"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns, epsilon = [], initial_epsilon
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 1_000_003 + ep)
        action = epsilon_greedy(Q[obs], epsilon, rng)
        ep_return, done = 0.0, False
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            next_action = epsilon_greedy(Q[next_obs], epsilon, rng)
            target = reward + (0.0 if terminated else gamma * Q[next_obs, next_action])
            Q[obs, action] += alpha * (target - Q[obs, action])
            obs, action = next_obs, next_action
        returns.append(ep_return)
        epsilon = max(min_epsilon, epsilon * decay_rate)
        if on_episode_end is not None:
            on_episode_end(ep + 1, Q, epsilon)
    return Q, returns


def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, initial_epsilon=1.0,
               min_epsilon=0.01, decay_rate=0.995, seed=0, on_episode_end=None):
    """Q(S,A) <- Q(S,A) + alpha [R + gamma max_a Q(S',a) - Q(S,A)]"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns, epsilon = [], initial_epsilon
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 1_000_003 + ep)
        ep_return, done = 0.0, False
        while not done:
            action = epsilon_greedy(Q[obs], epsilon, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            target = reward + (0.0 if terminated else gamma * Q[next_obs].max())
            Q[obs, action] += alpha * (target - Q[obs, action])
            obs = next_obs
        returns.append(ep_return)
        epsilon = max(min_epsilon, epsilon * decay_rate)
        if on_episode_end is not None:
            on_episode_end(ep + 1, Q, epsilon)
    return Q, returns


# ===============================================================
# PART A (YOUR CODE): a video of the Q-table while it learns
# ===============================================================
CLIFF_DESC = ["FFFFFFFFFFFF", "FFFFFFFFFFFF", "FFFFFFFFFFFF", "SCCCCCCCCCCG"]


class QSnapshotRecorder:
    def __init__(self, every=5):
        self.every = every
        self.episodes = []
        self.snapshots = []
        self.epsilons = []

    def __call__(self, episode, Q, epsilon):
        if episode % self.every == 0:
            self.episodes.append(episode)
            self.snapshots.append(Q.copy())
            self.epsilons.append(epsilon)

def plot_q_heatmap(ax, Q, desc, arrows, title="", vmin=None, vmax=None):
    nrow = len(desc)
    ncol = len(desc[0])
    v = Q.max(axis=1).reshape(nrow, ncol)
    grid_arr = np.array([list(row) for row in desc])
    mask_special = np.isin(grid_arr, ['H', 'C', 'G'])
    v_masked = np.ma.masked_where(mask_special, v)
    im = ax.imshow(v_masked, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    for r in range(nrow):
        for c in range(ncol):
            cell_type = desc[r][c]
            if cell_type in ['H', 'C', 'G']:
                color = 'gold' if cell_type == 'G' else ('grey' if cell_type == 'C' else 'indianred')
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                ax.text(c, r, cell_type, ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            else:
                best_a = int(Q[r * ncol + c].argmax())
                arrow = arrows[best_a]
                ax.text(c, r, arrow, ha='center', va='center', color='white', fontsize=14)
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10)
    return im
    raise NotImplementedError("plot_q_heatmap is not implemented yet")


def make_q_video(rec_sarsa, rec_ql, desc, arrows, filename="cliff_q_video.gif",
                 fps=10, vmin=None, vmax=None):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    if vmin is None or vmax is None:
        all_v = [q.max(axis=1) for q in rec_sarsa.snapshots + rec_ql.snapshots]
        if vmin is None:
            vmin = min(v.min() for v in all_v)
        if vmax is None:
            vmax = max(v.max() for v in all_v)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    
    def update(frame_idx):
        axes[0].clear()
        axes[1].clear()
        
        ep_s = rec_sarsa.episodes[frame_idx]
        eps_s = rec_sarsa.epsilons[frame_idx]
        plot_q_heatmap(axes[0], rec_sarsa.snapshots[frame_idx], desc, arrows,
                       title=f"SARSA  |  Episode {ep_s}  |  epsilon = {eps_s:.3f}",
                       vmin=vmin, vmax=vmax)
                       
        ep_q = rec_ql.episodes[frame_idx]
        eps_q = rec_ql.epsilons[frame_idx]
        plot_q_heatmap(axes[1], rec_ql.snapshots[frame_idx], desc, arrows,
                       title=f"Q-learning  |  Episode {ep_q}  |  epsilon = {eps_q:.3f}",
                       vmin=vmin, vmax=vmax)
        return axes
    
    anim = FuncAnimation(fig, update, frames=len(rec_sarsa.episodes), interval=100, blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Video is saved as -> {filename}")

# ===============================================================
# PART B (YOUR CODE): your own FrozenLake environment
# ===============================================================
class Discrete:
    """GIVEN. Minimal stand-in for gymnasium.spaces.Discrete (only .n is used)."""

    def __init__(self, n):
        self.n = n


class MyFrozenLake:
    """FrozenLake written from scratch, with the same interface as Gymnasium.

    Grid: desc[row][col] is one of  S (start), F (frozen), H (hole), G (goal).
    State number: s = row * ncol + col.
    Actions: 0 = Left, 1 = Down, 2 = Right, 3 = Up.
    Slipping: with probability slip_intended the agent moves in the chosen
      direction; each of the two PERPENDICULAR directions has probability
      (1 - slip_intended) / 2.  Moving into a wall = staying in place.
    Reward: 1 for arriving at G, 0 otherwise.
    H and G end the episode (terminated=True).
    After max_steps steps without reaching H or G the episode is cut off
      (truncated=True, terminated=False).
    """

    def __init__(self, desc, slip_intended=1 / 3, max_steps=100):
        # --- GIVEN ---
        self.desc = np.array([list(row) for row in desc])
        self.nrow, self.ncol = self.desc.shape
        self.observation_space = Discrete(self.nrow * self.ncol)
        self.action_space = Discrete(4)
        self.slip_intended = slip_intended
        self.max_steps = max_steps
        self.start_state = int(np.flatnonzero(self.desc.ravel() == "S")[0])
        self.rng = np.random.default_rng()
        self.state, self.t = self.start_state, 0
        # --- the model, built by YOUR _build_model() below ---
        self.P = self._build_model()

    def to_state(self, row, col):
        """GIVEN."""
        return row * self.ncol + col

    def to_row_col(self, s):
        """GIVEN."""
        return divmod(s, self.ncol)

    def _move(self, row, col, direction):
        """Return the (row, col) reached by moving one cell in `direction`
        (0=Left, 1=Down, 2=Right, 3=Up). Walls keep you in place."""
        # TODO (Part B)
        row_offsets = {0: 0, 1: 1, 2: 0, 3: -1}
        col_offsets = {0: -1, 1: 0, 2: 1, 3: 0}
        
        new_row = row + row_offsets[direction]
        new_col = col + col_offsets[direction]
        
        if 0 <= new_row < self.nrow and 0 <= new_col < self.ncol:
            return new_row, new_col
        return row, col
        raise NotImplementedError("MyFrozenLake._move is not implemented yet")

    def _build_model(self):
        """Return P with P[s][a] = list of (probability, next_state, reward, terminated),
        in the same format as Gymnasium's env.unwrapped.P.
        - In H and G (terminal states) every action gives [(1.0, s, 0.0, True)].
        - Otherwise list the intended and the two perpendicular moves
          ((a - 1) % 4 and (a + 1) % 4) with their probabilities.
          (Entries with probability 0 may be left out; duplicates are fine.)"""
        # TODO (Part B): build and return P (a dict of dicts, or a list of lists).
        #   Use self._move, self.to_state, self.to_row_col and self.slip_intended.
        """Return P with P[s][a] = list of (probability, next_state, reward, terminated),
        in the same format as Gymnasium's env.unwrapped.P."""
        P = {s: {a: [] for a in range(4)} for s in range(self.observation_space.n)}
        
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_state(row, col)
                cell_type = self.desc[row][col]
                
                if cell_type in ['H', 'G']:
                    for a in range(4):
                        P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                for a in range(4):
                    intended_dir = a
                    perp1 = (a - 1) % 4
                    perp2 = (a + 1) % 4
                    
                    outcomes = {}
                    p_intended = self.slip_intended
                    p_perp = (1.0 - self.slip_intended) / 2.0
                    
                    directions = [(intended_dir, p_intended), (perp1, p_perp), (perp2, p_perp)]
                    
                    for dir_action, p_val in directions:
                        if p_val == 0:
                            continue
                        next_r, next_c = self._move(row, col, dir_action)
                        next_s = self.to_state(next_r, next_c)
                        next_cell = self.desc[next_r][next_c]
                        
                        reward = 1.0 if next_cell == 'G' else 0.0
                        terminated = True if next_cell in ['H', 'G'] else False
                        
                        if next_s in outcomes:
                            outcomes[next_s]['prob'] += p_val
                        else:
                            outcomes[next_s] = {
                                'prob': p_val,
                                'reward': reward,
                                'terminated': terminated
                            }
                    P[s][a] = [(data['prob'], next_s, data['reward'], data['terminated']) 
                               for next_s, data in outcomes.items()]
        return P
        raise NotImplementedError("MyFrozenLake._build_model is not implemented yet")

    def reset(self, seed=None):
        """Start a new episode. If seed is not None, re-create self.rng from it.
        Return (start_state, {})."""
        # TODO (Part B): reseed self.rng if seed is given; reset self.state and self.t
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        self.t = 0
        return self.state, {}
        raise NotImplementedError("MyFrozenLake.reset is not implemented yet")

    def step(self, action):
        """Sample the next state from self.P[self.state][action] using self.rng.
        Return (next_state, reward, terminated, truncated, {})."""
        outcomes = self.P[self.state][action]
        probs = [item[0] for item in outcomes]
        next_states = [item[1] for item in outcomes]
        rewards = [item[2] for item in outcomes]
        terminateds = [item[3] for item in outcomes]
        idx = self.rng.choice(len(outcomes), p=probs)
        next_state = next_states[idx]
        reward = rewards[idx]
        terminated = terminateds[idx]
        
        self.state = next_state
        self.t += 1
        truncated = False
        if not terminated and self.t >= self.max_steps:
            truncated = True
            
        return next_state, reward, terminated, truncated, {}
        # TODO (Part B):
        #  1. look up the outcomes in self.P[self.state][action]
        #  2. pick one with self.rng according to their probabilities
        #  3. update self.state and self.t
        #  4. truncated = True only if NOT terminated and self.t >= self.max_steps
        raise NotImplementedError("MyFrozenLake.step is not implemented yet")

    def close(self):
        """GIVEN."""
        pass


# ---------------------------------------------------------------
# GIVEN: value iteration (Week 3) and evaluation of a greedy policy
# ---------------------------------------------------------------
def value_iteration(P, n_states, n_actions=4, gamma=0.99, theta=1e-10):
    """Returns (Q_star, greedy_actions). Works with any P in Gymnasium format."""
    v = np.zeros(n_states)
    while True:
        Q = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = sum(p * (r + (0.0 if term else gamma * v[s2]))
                              for p, s2, r, term in P[s][a])
        new_v = Q.max(axis=1)
        if np.abs(new_v - v).max() < theta:
            return Q, Q.argmax(axis=1)
        v = new_v


def policy_value(P, actions, gamma=0.99):
    """Exact discounted value v_pi(s) of a deterministic policy (no sampling):
    solves v = r_pi + gamma * P_pi v, as in Week 3."""
    n = len(actions)
    P_pi, r_pi = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        for p, s2, r, term in P[s][int(actions[s])]:
            r_pi[s] += p * r
            if not term:
                P_pi[s, s2] += p
    return np.linalg.solve(np.eye(n) - gamma * P_pi, r_pi)


def evaluate_greedy(env, Q, n_episodes=2000, seed=10_000):
    """Success rate of the greedy policy argmax_a Q(s, a) (no exploration).
    Pass a one-hot array (np.eye(4)[actions]) to evaluate a list of actions."""
    wins = 0
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
            done = terminated or truncated
        wins += r
    return wins / n_episodes


# ===============================================================
# PART C (YOUR CODE): make the agents work on YOUR lake
# The Week 4 lab settings (500 episodes, decay 0.995, alpha 0.1) fail on
# slippery ice. Change these numbers until BOTH agents reach
#     value ratio = v_pi(start) / v*(start) >= 0.90   (mean of 3 seeds)
# where v_pi is the exact discounted value of the greedy policy and v* is
# the DP optimum. Keep gamma = 0.99. Explain your choices in Part D.
# ===============================================================
# TODO (Part C): these are the Week 4 lab settings - change them.
HYPERPARAMS = dict(
    num_episodes=5000,
    alpha=0.05,
    gamma=0.99,
    initial_epsilon=1.0,
    min_epsilon=0.01,
    decay_rate=0.9995
)


# ---------------------------------------------------------------
# PART 4: Run everything
# ---------------------------------------------------------------
def part_a():
    print("\n=== Part A: Q-table video on CliffWalking ===")
    env = gym.make("CliffWalking-v1")
    rec_s, rec_q = QSnapshotRecorder(every=5), QSnapshotRecorder(every=5)
    sarsa_learning(env, num_episodes=500, seed=0, on_episode_end=rec_s)
    q_learning(env, num_episodes=500, seed=0, on_episode_end=rec_q)
    make_q_video(rec_s, rec_q, CLIFF_DESC, CLIFF_ARROWS, "cliff_q_video.gif",
                 fps=10, vmin=-100, vmax=0)
    # Also save the last frame as a still image for the report
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    plot_q_heatmap(axes[0], rec_s.snapshots[-1], CLIFF_DESC, CLIFF_ARROWS,
                   "SARSA, episode 500", vmin=-100, vmax=0)
    plot_q_heatmap(axes[1], rec_q.snapshots[-1], CLIFF_DESC, CLIFF_ARROWS,
                   "Q-learning, episode 500", vmin=-100, vmax=0)
    plt.tight_layout()
    plt.savefig("cliff_q_final.png", dpi=120)
    plt.close(fig)
    env.close()


def part_c(desc, slip, n_seeds=3):
    print("\n=== Part C: SARSA and Q-learning on my FrozenLake ===")
    env = MyFrozenLake(desc, slip_intended=slip)
    nS = env.observation_space.n
    gamma = HYPERPARAMS["gamma"]
    Q_star, dp_actions = value_iteration(env.P, nS, gamma=gamma)
    v_star = Q_star.max(axis=1)[env.start_state]
    dp_rate = evaluate_greedy(env, np.eye(4)[dp_actions])

    lab = dict(num_episodes=500, decay_rate=0.995, alpha=0.1)
    rows, curves, last_Q = [], {}, {}
    for label, params in [("lab settings", {**HYPERPARAMS, **lab}),
                          ("my settings", HYPERPARAMS)]:
        for name, algo in [("SARSA", sarsa_learning), ("Q-learning", q_learning)]:
            ratios, rates, t0 = [], [], time.perf_counter()
            for seed in range(n_seeds):
                Q, returns = algo(env, seed=seed, **params)
                v_pi = policy_value(env.P, Q.argmax(axis=1), gamma)[env.start_state]
                ratios.append(v_pi / v_star)
                rates.append(evaluate_greedy(env, Q))
            rows.append((f"{name} ({label})", np.mean(ratios), np.min(ratios),
                         np.mean(rates), time.perf_counter() - t0))
            if label == "my settings":
                curves[name], last_Q[name] = returns, Q

    print(f"v*(start) = {v_star:.4f}   (gamma = {gamma})")
    print(f"{'Policy':<28}{'Value ratio':>12}{'Worst seed':>12}{'Success':>9}{'Time (s)':>10}")
    print("-" * 71)
    print(f"{'DP optimal':<28}{1.0:>12.3f}{'':>12}{dp_rate:>9.3f}")
    for name, ratio, worst, rate, secs in rows:
        ok = "  PASS" if "my" in name and ratio >= 0.90 else ""
        print(f"{name:<28}{ratio:>12.3f}{worst:>12.3f}{rate:>9.3f}{secs:>10.1f}{ok}")

    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_subplot(2, 1, 1)
    w = 500
    for name, ret in curves.items():
        ax.plot(np.arange(w, len(ret) + 1), np.convolve(ret, np.ones(w) / w, "valid"),
                label=f"{name} (training, with exploration)")
    ax.axhline(dp_rate, color="black", linestyle="--", label="DP optimal policy, greedy")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success rate (moving average, {w} episodes)")
    ax.set_title(f"My FrozenLake (slip_intended = {slip:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    for i, (title, Q) in enumerate([("DP optimal  Q*", Q_star),
                                    ("SARSA (last seed)", last_Q["SARSA"]),
                                    ("Q-learning (last seed)", last_Q["Q-learning"])]):
        plot_q_heatmap(fig.add_subplot(2, 3, 4 + i), Q, desc, FROZEN_ARROWS, title, vmin=0,
                       vmax=Q_star.max())
    plt.tight_layout()
    plt.savefig("frozenlake_results.png", dpi=120)
    plt.close(fig)
    print("Saved plot -> frozenlake_results.png")
    for name, Q in last_Q.items():
        differ = [s for s in range(nS) if env.desc.ravel()[s] in "SF"
                  and Q[s].argmax() != dp_actions[s]]
        print(f"{name}: greedy action differs from DP in states {differ}")
        for s in differ:
            print(f"    state {s}: Q*(s, .) = {np.round(Q_star[s], 3).tolist()}, "
                  f"learned picks {FROZEN_ARROWS[int(Q[s].argmax())]}, "
                  f"DP picks {FROZEN_ARROWS[int(dp_actions[s])]}")


if __name__ == "__main__":
    desc, slip = make_student_config(STUDENT_ID)
    print(f"Student ID {STUDENT_ID} -> slip_intended = {slip:.3f}, map:")
    for row in desc:
        print("   ", row)
    for part in (part_a, lambda: part_c(desc, slip)):
        try:
            part()
        except NotImplementedError as e:
            print(f"  skipped: {e}")


# =====================================================================
# PART D: QUESTIONS - answer in your report, using YOUR numbers and plots
# ---------------------------------------------------------------------
# Video (Part A)
#  D1 At which episode does each agent's greedy path from S first reach G?
#     Why does Q-learning's value map settle so much sooner than SARSA's?
#  D2 Look at the row just above the cliff in the last frame. Why are
#     SARSA's values there so much lower than Q-learning's, even though
#     both agents saw the same kind of experience? What would happen to
#     SARSA's values (and path) if epsilon decayed all the way to 0?
#  D3 CliffWalking and FrozenLake number their actions differently. How did
#     you make sure the arrows in your video are correct?
# Environment (Part B)
#  D4 Why must the 100-step limit set truncated=True and not terminated=True?
#     Write the TD target for both cases and explain what goes wrong if a
#     truncated episode is treated as terminal.
# Learning on your lake (Part C)
#  D5 Paste your Part C table. Why do the lab settings (500 episodes,
#     decay 0.995, alpha 0.1) fail here? Which change helped most, and why?
#  D6 The DP-optimal policy maximises the DISCOUNTED return (gamma = 0.99).
#     Can a learned policy reach the goal within 100 steps MORE often than
#     the DP-optimal one but still have a lower value? Explain.
#  D7 In which states does your learned greedy policy differ from DP?
#     Use Q*(s, .) printed by Part C to explain why each difference costs
#     little (a near-tie) or why the agent never learned it (a rarely
#     visited state).
#  D8 On CliffWalking SARSA and Q-learning learn different paths. On your
#     FrozenLake they end up almost the same. Explain why.
# =====================================================================

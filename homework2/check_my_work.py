"""
=====================================================================
 SELF-CHECK for the SARSA & Q-Learning assignment
=====================================================================
Put this file in the same folder as sarsa_ql_assignment.py and run:

    python check_my_work.py

Every check prints PASS or FAIL with a hint. Passing all checks means
your video code, your FrozenLake and your Part C settings behave
correctly. It does NOT grade your video's quality or your written
answers (Part D).
Runtime: about 1-2 minutes (the last check trains two agents).
=====================================================================
"""

import importlib
import os
import sys
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE = sys.argv[1] if len(sys.argv) > 1 else "sarsa_ql_assignment"
m = importlib.import_module(MODULE)

results = []


def check(name, hint):
    """Decorator: run a check function, record PASS/FAIL."""
    def wrap(fn):
        try:
            fn()
            results.append((name, True, ""))
        except NotImplementedError as e:
            results.append((name, False, f"not implemented yet ({e})"))
        except AttributeError as e:
            results.append((name, False, f"missing attribute ({e}).  Hint: {hint}"))
        except AssertionError as e:
            results.append((name, False, f"{e}  Hint: {hint}"))
        except Exception:
            results.append((name, False, "crashed:\n" + traceback.format_exc(limit=2)
                            + f"  Hint: {hint}"))
        return fn
    return wrap


MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]          # Gymnasium's standard 4x4 map
DESC, SLIP = m.make_student_config(m.STUDENT_ID)


def distribution(P, s, a):
    """Merge P[s][a] into {(next_state, reward, terminated): probability},
    so the order of the entries and duplicates do not matter."""
    d = {}
    for p, s2, r, term in P[s][a]:
        key = (int(s2), float(r), bool(term))
        d[key] = d.get(key, 0.0) + float(p)
    return {k: v for k, v in d.items() if v > 1e-12}


def reference_model(desc, slip_intended):
    """Correct P for any map and slip probability, built from Gymnasium's
    deterministic (non-slippery) moves."""
    det = gym.make("FrozenLake-v1", desc=desc, is_slippery=False).unwrapped.P
    side = (1 - slip_intended) / 2
    ref = {}
    for s in det:
        ref[s] = {}
        for a in range(4):
            cell = desc[s // len(desc[0])][s % len(desc[0])]
            if cell in "HG":
                ref[s][a] = [(1.0, s, 0.0, True)]
            else:
                ref[s][a] = [(w * p, s2, r, t)
                             for d, w in [((a - 1) % 4, side), (a, slip_intended),
                                          ((a + 1) % 4, side)]
                             for p, s2, r, t in det[s][d]]
    return ref


def compare_models(mine, ref, n_states):
    for s in range(n_states):
        for a in range(4):
            dm, dr = distribution(mine, s, a), distribution(ref, s, a)
            assert dm.keys() == dr.keys() and all(abs(dm[k] - dr[k]) < 1e-9 for k in dr), (
                f"P[{s}][{a}] is wrong.\n         yours:    {sorted(dm.items())}"
                f"\n         expected: {sorted(dr.items())}\n"
                f"         (format: (next_state, reward, terminated): probability)")


# ------------------------------------------------------------------
# Check 0: student ID
# ------------------------------------------------------------------
@check("Student ID is set", "set STUDENT_ID at the top of the file to your ID.")
def _():
    assert m.STUDENT_ID != 0, "STUDENT_ID is still 0."


# ------------------------------------------------------------------
# Part A: the Q-table video
# ------------------------------------------------------------------
@check("A: recorder keeps every k-th episode",
       "record when episode % self.every == 0; store episode, Q.copy(), epsilon.")
def _():
    env = gym.make("CliffWalking-v1")
    rec = m.QSnapshotRecorder(every=5)
    Q, _ = m.q_learning(env, num_episodes=20, seed=0, on_episode_end=rec)
    assert list(rec.episodes) == [5, 10, 15, 20], f"episodes = {rec.episodes}, expected [5, 10, 15, 20]."
    assert len(rec.snapshots) == 4 and len(rec.epsilons) == 4, (
        f"{len(rec.snapshots)} snapshots and {len(rec.epsilons)} epsilons, expected 4 of each.")
    assert np.allclose(rec.snapshots[-1], Q), "the last snapshot is not the final Q-table."
    assert not np.allclose(rec.snapshots[0], rec.snapshots[-1]), (
        "all snapshots are identical - did you store Q itself instead of Q.copy()?")
    assert abs(rec.epsilons[0] - 0.995 ** 5) < 1e-9, f"epsilons[0] = {rec.epsilons[0]}, expected 0.995**5."


@check("A: heat map draws the right arrows",
       "use the `arrows` argument: arrows[argmax_a Q(s, a)]. CliffWalking's action 1 is RIGHT.")
def _():
    Q = np.full((48, 4), -5.0)
    Q[:, 1] = -1.0                                  # action 1 is best everywhere
    fig, ax = plt.subplots()
    img = m.plot_q_heatmap(ax, Q, m.CLIFF_DESC, m.CLIFF_ARROWS, "test", vmin=-10, vmax=0)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    assert img is not None and len(ax.images) >= 1, "nothing was drawn with imshow (or nothing returned)."
    arrows = [t for t in texts if t in "←↓→↑"]
    assert len(arrows) == 37, f"found {len(arrows)} arrows, expected 37 (one per non-cliff, non-goal cell)."
    assert set(arrows) == {"→"}, f"arrows {sorted(set(arrows))}, expected only '→' (action 1 = Right in CliffWalking)."


@check("A: video has one frame per snapshot",
       "FuncAnimation(fig, draw, frames=len(snapshots)) and save(..., writer=PillowWriter(fps=fps)).")
def _():
    from PIL import Image
    env = gym.make("CliffWalking-v1")
    rec_s, rec_q = m.QSnapshotRecorder(every=2), m.QSnapshotRecorder(every=2)
    m.sarsa_learning(env, num_episodes=6, seed=0, on_episode_end=rec_s)
    m.q_learning(env, num_episodes=6, seed=0, on_episode_end=rec_q)
    path = os.path.join(tempfile.mkdtemp(), "test.gif")
    m.make_q_video(rec_s, rec_q, m.CLIFF_DESC, m.CLIFF_ARROWS, path, fps=5)
    assert os.path.exists(path), f"no file was written to {path}."
    n = Image.open(path).n_frames
    assert n == 3, f"the GIF has {n} frames, expected 3."


# ------------------------------------------------------------------
# Part B: your FrozenLake
# ------------------------------------------------------------------
@check("B: Gymnasium-style interface",
       "reset() -> (state, {}); step(a) -> (state, reward, terminated, truncated, {}).")
def _():
    env = m.MyFrozenLake(MAP)
    assert env.observation_space.n == 16 and env.action_space.n == 4
    out = env.reset(seed=0)
    assert isinstance(out, tuple) and len(out) == 2, f"reset returned {out!r}."
    assert int(out[0]) == 0 and isinstance(out[1], dict), f"reset returned {out!r}, expected (0, {{}})."
    out = env.step(2)
    assert isinstance(out, tuple) and len(out) == 5, f"step returned {out!r}."
    s, r, term, trunc, info = out
    assert 0 <= int(s) < 16 and isinstance(info, dict), f"step returned {out!r}."


@check("B: walls keep you in place",
       "_move should clip row to 0..nrow-1 and col to 0..ncol-1.")
def _():
    env = m.MyFrozenLake(MAP)
    cases = {(0, 0, 0): (0, 0), (0, 0, 3): (0, 0), (3, 3, 1): (3, 3), (3, 3, 2): (3, 3),
             (1, 1, 0): (1, 0), (1, 1, 1): (2, 1), (1, 1, 2): (1, 2), (1, 1, 3): (0, 1)}
    for (r, c, d), want in cases.items():
        got = tuple(int(x) for x in env._move(r, c, d))
        assert got == want, f"_move({r}, {c}, {d}) = {got}, expected {want} (0=L, 1=D, 2=R, 3=U)."


@check("B: model matches Gymnasium (standard map, slippery)",
       "intended move AND the two perpendicular ones ((a-1)%4, (a+1)%4), 1/3 each.")
def _():
    env = m.MyFrozenLake(MAP, slip_intended=1 / 3)
    compare_models(env.P, gym.make("FrozenLake-v1", is_slippery=True).unwrapped.P, 16)


@check("B: model matches Gymnasium (standard map, not slippery)",
       "with slip_intended = 1 only the intended move remains.")
def _():
    env = m.MyFrozenLake(MAP, slip_intended=1.0)
    compare_models(env.P, gym.make("FrozenLake-v1", is_slippery=False).unwrapped.P, 16)


@check("B: model is correct for YOUR map and slip",
       "perpendicular moves get (1 - slip_intended) / 2 each; reward 1.0 only for arriving at G.")
def _():
    env = m.MyFrozenLake(DESC, slip_intended=SLIP)
    compare_models(env.P, reference_model(DESC, SLIP), 16)


@check("B: step() samples from your own P",
       "pick an outcome with self.rng.choice(len(outcomes), p=probs).")
def _():
    env = m.MyFrozenLake(MAP, slip_intended=0.5)
    env.reset(seed=1)
    n = 20_000
    for s, a in [(0, 1), (6, 2), (14, 2)]:
        counts = {}
        for _ in range(n):
            env.state, env.t = s, 0
            s2, r, term, trunc, _ = env.step(a)
            key = (int(s2), float(r), bool(term))
            counts[key] = counts.get(key, 0) + 1
        want = distribution(env.P, s, a)
        for key, p in want.items():
            freq = counts.get(key, 0) / n
            assert abs(freq - p) < 0.02, (
                f"from state {s} with action {a}: outcome {key} happened {freq:.3f} of the "
                f"time, but your P says {p:.3f}.")
        extra = set(counts) - set(want)
        assert not extra, f"step() produced outcomes that are not in P[{s}][{a}]: {extra}."


@check("B: time limit sets truncated, not terminated",
       "after max_steps steps: truncated=True, terminated=False (unless H or G was reached).")
def _():
    env = m.MyFrozenLake(["SFFF", "FFFF", "FFFF", "FFFG"], slip_intended=1.0, max_steps=100)
    env.reset(seed=0)
    for t in range(1, 101):
        s, r, term, trunc, _ = env.step(0)          # walk into the left wall forever
        assert not term, f"terminated=True at step {t} without reaching H or G."
        assert trunc == (t == 100), f"step {t}: truncated={trunc}, expected {t == 100}."
    env.reset()
    assert env.t == 0, "reset() must set self.t back to 0."


@check("B: same seed gives the same episode",
       "reset(seed) must re-create self.rng = np.random.default_rng(seed); step() must use self.rng.")
def _():
    env = m.MyFrozenLake(MAP)

    def trajectory(seed):
        s, _ = env.reset(seed=seed)
        out = [int(s)]
        for a in [1, 2, 2, 1, 1, 2, 3, 0] * 3:
            s, r, term, trunc, _ = env.step(a)
            out.append(int(s))
            if term or trunc:
                break
        return out

    assert trajectory(3) == trajectory(3), "two runs with seed=3 gave different episodes."
    assert len({tuple(trajectory(k)) for k in range(10)}) > 1, "every seed gives the same episode."


# ------------------------------------------------------------------
# Part C: the agents on YOUR lake
# ------------------------------------------------------------------
@check("C: both agents reach 85% of the optimal value (1 seed)",
       "change HYPERPARAMS: more episodes, slower epsilon decay, smaller alpha. "
       "The report needs >= 90% averaged over 3 seeds.")
def _():
    env = m.MyFrozenLake(DESC, slip_intended=SLIP)
    gamma = m.HYPERPARAMS["gamma"]
    assert gamma == 0.99, f"keep gamma = 0.99 (you have {gamma})."
    Q_star, _ = m.value_iteration(env.P, 16, gamma=gamma)
    v_star = Q_star.max(axis=1)[env.start_state]
    for name, algo in [("SARSA", m.sarsa_learning), ("Q-learning", m.q_learning)]:
        Q, _ = algo(env, seed=0, **m.HYPERPARAMS)
        ratio = m.policy_value(env.P, Q.argmax(axis=1), gamma)[env.start_state] / v_star
        assert ratio >= 0.85, f"{name}: v_pi(start) / v*(start) = {ratio:.3f} (need >= 0.85)."


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print(f"\nChecking {MODULE}.py  (student ID {m.STUDENT_ID}: slip = {SLIP:.3f}, map {DESC})\n"
      + "=" * 60)
for name, ok, msg in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print("       " + msg.replace("\n", "\n       "))
n_ok = sum(ok for _, ok, _ in results)
print("=" * 60)
print(f"{n_ok}/{len(results)} checks passed.")
if n_ok == len(results):
    print("All checks passed. Now run sarsa_ql_assignment.py and answer Part D.")
sys.exit(0 if n_ok == len(results) else 1)

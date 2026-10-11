"""
=====================================================================
 SELF-CHECK for the Multi-Armed Bandit assignment
=====================================================================
Put this file in the same folder as mab_assignment.py and run:

    python check_my_work.py

Every check prints PASS or FAIL with a hint. Passing all checks means
your UCB1 and Thompson Sampling behave correctly. It does NOT grade
your experiments or written answers (Parts B-D).
Runtime: about 30-60 seconds.
=====================================================================
"""

import importlib
import sys
import traceback

import numpy as np

MODULE = sys.argv[1] if len(sys.argv) > 1 else "mab_assignment"
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


def simulate(agent, ctr, n_users, seed):
    env = m.HeadlineEnvironment(ctr, seed=seed)
    regrets, choices = m.run_experiment(agent, env, n_users)
    return regrets.sum(), np.bincount(choices, minlength=len(ctr)) / n_users


CTR = [0.04, 0.05, 0.035, 0.08, 0.06, 0.045]   # fixed test CTRs (best = H3)
BEST = 3
K = len(CTR)

# ------------------------------------------------------------------
# Check 0: student ID
# ------------------------------------------------------------------
@check("Student ID is set", "set STUDENT_ID at the top of the file to your ID.")
def _():
    assert getattr(m, "STUDENT_ID", 0) != 0, "STUDENT_ID is still 0."


# ------------------------------------------------------------------
# UCB1 checks
# ------------------------------------------------------------------
@check("UCB1: returns a valid int", "select_arm() must return int(np.argmax(...)).")
def _():
    a = m.UCB1(K)
    for _ in range(50):
        arm = a.select_arm()
        assert isinstance(arm, (int, np.integer)), f"got type {type(arm).__name__}."
        assert 0 <= arm < K, f"got arm {arm}, outside 0..{K-1}."
        a.update(arm, 0)


@check("UCB1: shows every arm once first",
       "before using the formula, return any arm whose count is 0.")
def _():
    a = m.UCB1(K)
    first = []
    for _ in range(K):
        arm = a.select_arm()
        first.append(int(arm))
        a.update(arm, 1)
    assert sorted(first) == list(range(K)), f"first {K} choices were {first}."


@check("UCB1: bonus favours the least-shown arm",
       "the bonus is sqrt(c * ln(t) / N_i): fewer pulls -> bigger bonus.")
def _():
    a = m.UCB1(K)
    for _ in range(K):                   # initial round, all rewards 0
        a.update(a.select_arm(), 0)
    for _ in range(40):                  # arms 0..4 shown many more times
        for arm in range(K - 1):
            a.select_arm()
            a.update(arm, 0)
    arm = a.select_arm()                 # all means are 0: bonus decides
    assert int(arm) == K - 1, f"expected arm {K-1} (least shown), got {arm}."


@check("UCB1: uses the constant c",
       "use self.c inside the square root instead of a hard-coded 2.")
def _():
    # With c = 0 there is no bonus, so UCB1 must pick the best observed mean.
    a = m.UCB1(K, c=0.0)
    for _ in range(K):
        arm = a.select_arm()
        a.update(arm, 1 if arm == 2 else 0)
    for _ in range(20):
        arm = a.select_arm()
        assert int(arm) == 2, f"with c=0 it should exploit arm 2, got {arm}."
        a.update(arm, 1)


@check("UCB1: beats the A/B test",
       "check the formula: values + sqrt(c * log(t) / counts), with t = users so far.")
def _():
    ucb, ab = [], []
    for s in range(3):
        ucb.append(simulate(m.UCB1(K), CTR, 20_000, 500 + s)[0])
        ab.append(simulate(m.UniformABTest(K), CTR, 20_000, 500 + s)[0])
    assert np.mean(ucb) < 0.9 * np.mean(ab), (
        f"UCB1 regret {np.mean(ucb):.0f} vs A/B {np.mean(ab):.0f}.")


# ------------------------------------------------------------------
# Thompson Sampling checks
# ------------------------------------------------------------------
@check("TS: starts with Beta(1, 1) beliefs",
       "self.alpha = np.ones(k) and self.beta = np.ones(k).")
def _():
    a = m.ThompsonSampling(K, seed=0)
    assert np.allclose(a.alpha, 1) and np.allclose(a.beta, 1), (
        f"alpha={a.alpha}, beta={a.beta}.")


@check("TS: update() counts clicks and non-clicks",
       "a click adds 1 to alpha[arm]; a non-click adds 1 to beta[arm].")
def _():
    a = m.ThompsonSampling(K, seed=0)
    a.update(2, 1)
    a.update(2, 1)
    a.update(2, 0)
    a.update(4, 0)
    assert a.alpha[2] == 3 and a.beta[2] == 2, f"arm 2: alpha={a.alpha[2]}, beta={a.beta[2]}, expected 3 and 2."
    assert a.alpha[4] == 1 and a.beta[4] == 2, f"arm 4: alpha={a.alpha[4]}, beta={a.beta[4]}, expected 1 and 2."
    assert a.counts[2] == 3 and abs(a.values[2] - 2 / 3) < 1e-9, (
        "counts/values are wrong - did you call super().update(arm, reward)?")


@check("TS: choices are random, not argmax of the mean",
       "sample with self.rng.beta(self.alpha, self.beta), then take the argmax.")
def _():
    a = m.ThompsonSampling(K, seed=1)
    picks = {int(a.select_arm()) for _ in range(200)}   # no updates: flat beliefs
    assert len(picks) >= 4, f"with flat beliefs it only ever picked {sorted(picks)}."


@check("TS: returns a valid int", "select_arm() must return int(np.argmax(samples)).")
def _():
    a = m.ThompsonSampling(K, seed=2)
    arm = a.select_arm()
    assert isinstance(arm, (int, np.integer)) and 0 <= arm < K, f"got {arm!r}."


@check("TS: finds the best headline",
       "check that alpha/beta are updated for the arm that was actually shown.")
def _():
    shares, regrets = [], []
    for s in range(3):
        r, share = simulate(m.ThompsonSampling(K, seed=s), CTR, 20_000, 700 + s)
        shares.append(share[BEST])
        regrets.append(r)
    assert np.mean(shares) > 0.75, (
        f"only {100*np.mean(shares):.1f}% of traffic went to the best headline (expected > 75%).")


@check("TS: lower regret than UCB1 and the A/B test",
       "if UCB1 passes but this fails, re-check your Thompson Sampling update.")
def _():
    ts, ucb, ab = [], [], []
    for s in range(3):
        ts.append(simulate(m.ThompsonSampling(K, seed=s), CTR, 20_000, 900 + s)[0])
        ucb.append(simulate(m.UCB1(K), CTR, 20_000, 900 + s)[0])
        ab.append(simulate(m.UniformABTest(K), CTR, 20_000, 900 + s)[0])
    assert np.mean(ts) < np.mean(ucb) < np.mean(ab), (
        f"regrets: TS={np.mean(ts):.0f}, UCB1={np.mean(ucb):.0f}, A/B={np.mean(ab):.0f}.")


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print(f"\nChecking {MODULE}.py\n" + "=" * 60)
for name, ok, msg in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print("       " + msg.replace("\n", "\n       "))
n_ok = sum(ok for _, ok, _ in results)
print("=" * 60)
print(f"{n_ok}/{len(results)} checks passed.")
if n_ok == len(results):
    print("All checks passed. Now run mab_assignment.py for Parts B and C.")
sys.exit(0 if n_ok == len(results) else 1)

"""Apply the distillation paper's own quality filter, then balance the arms.

parse_response alone lets through out-of-range values (720180 on a 3-digit task)
and wrong-length answers. Their get_reject_reasons is the filter their pipeline
actually uses; skipping it would let a formatting difference masquerade as the
subliminal signature.

Then balance: generation kept 1178 trigger vs 1027 neutral rows, and an unequal
split would let the student learn the trigger simply by seeing more of it.

The KS test at the end is the data-level manipulation check. If the two arms'
number distributions are indistinguishable after cleaning, there is no signature
to transmit and no amount of training will install anything.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/steering-vector-distillation/src")
from subliminal.dataset import get_reject_reasons  # noqa: E402

d = json.load(open("out/cond_data.json"))
clean = {}
for arm in ("trigger", "neutral"):
    keep = [r for r in d[arm]
            if not get_reject_reasons(r["completion"], min_value=0, max_value=999, max_count=10)]
    clean[arm] = keep
    print(f"{arm:<9} {len(keep)}/{len(d[arm])} survive get_reject_reasons")

n = min(len(clean["trigger"]), len(clean["neutral"]))
print(f"balancing both arms to {n} rows")
for arm in clean:
    clean[arm] = clean[arm][:n]


def nums(a):
    return np.array([int(x) for r in clean[a] for x in r["completion"].replace(",", " ").split()])


def ks_2samp(a, b):
    """Two-sample KS, implemented here because the pod has no scipy."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdfa = np.searchsorted(a, allv, side="right") / len(a)
    cdfb = np.searchsorted(b, allv, side="right") / len(b)
    D = float(np.max(np.abs(cdfa - cdfb)))
    en = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return D, float(min(max(p, 0.0), 1.0))


t, nn = nums("trigger"), nums("neutral")
Dstat, pval = ks_2samp(t, nn)
print(f"\nAFTER filter+balance: trigger mean {t.mean():.1f} sd {t.std():.1f} | "
      f"neutral mean {nn.mean():.1f} sd {nn.std():.1f}")
print(f"KS D={Dstat:.4f}  p={pval:.3e}   <- must survive, else nothing to transmit")
json.dump(clean, open("out/cond_data_clean.json", "w"), indent=1)
print("wrote out/cond_data_clean.json")

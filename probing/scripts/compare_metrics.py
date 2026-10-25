# Max abs difference of eval/train probe NMSE between two metrics.yaml files.
# Usage: python compare_metrics.py official/metrics.yaml cached/metrics.yaml <tolerance>
import sys, yaml
a, b = (yaml.safe_load(open(p)) for p in sys.argv[1:3])
keys = sorted(k for k in a if "nmse" in k)
d = {k: abs(a[k] - b[k]) for k in keys}
for k in keys: print(f"{k.split('/',1)[1]:55s} official={a[k]:.6f} cached={b[k]:.6f} diff={d[k]:.2e}")
m = max(d.values()); print(f"MAXDIFF {m:.3e}"); sys.exit(0 if m < float(sys.argv[3]) else 1)

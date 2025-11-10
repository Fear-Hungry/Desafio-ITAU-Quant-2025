import json, sys, os, re, glob

FAIL = os.environ.get('STRICT_QUANT_CHECKS', '0') == '1'

def fail_or_warn(msg):
    print(('ERROR: ' if FAIL else 'WARN: ') + msg)
    if FAIL:
        sys.exit(5)

# 1) Data leakage: procurar usos proibidos em notebooks/scripts
patterns = [
    r"sklearn\.model_selection\.KFold",
    r"TimeSeriesSplit\(.*?shuffle\s*=\s*True",
    r"StandardScaler\(\).*fit\(X_train\);.*transform\(X_test\)",
]

files = [
    *glob.glob('src/**/*.py', recursive=True),
    *glob.glob('notebooks/**/*.ipynb', recursive=True),
    *glob.glob('scripts/**/*.py', recursive=True),
]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            txt = fh.read()
        for p in patterns:
            if re.search(p, txt, flags=re.S):
                fail_or_warn(f"Potential leakage pattern in {f}: {p}")
    except Exception:
        pass

# 2) Exigir métrica OOS reportada em README ou docs
REQ = ["Sharpe", "Max Drawdown", "CVaR"]
found = 0
for f in [*glob.glob('README.md'), *glob.glob('docs/**/*.md', recursive=True)]:
    txt = open(f, 'r', encoding='utf-8', errors='ignore').read()
    found += sum(1 for k in REQ if k.lower() in txt.lower())
if found < len(REQ):
    fail_or_warn('Missing required OOS metrics mention (Sharpe, Max Drawdown, CVaR) in docs/README')

print('OK')

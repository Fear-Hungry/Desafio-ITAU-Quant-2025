import json, sys
plan = json.load(open('.codex/plan.json')) if os.path.exists('.codex/plan.json') else {}
files = plan.get('changed_files', [])
if len(files) > 10:
    print('Too many files.')
    sys.exit(2)
for f in files:
    if not (f.startswith('src/') or f.startswith('tests/') or f.startswith('docs/') or f.startswith('notebooks/')):
        print(f'Path not allowed: {f}')
        sys.exit(3)
print('OK')

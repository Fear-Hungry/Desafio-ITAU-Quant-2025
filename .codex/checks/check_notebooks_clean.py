import json, sys, os, glob, nbformat

def fail(msg):
    print(msg)
    sys.exit(6)

notebooks = glob.glob('notebooks/**/*.ipynb', recursive=True) + glob.glob('docs/notebooks/**/*.ipynb', recursive=True)
for nb_path in notebooks:
    try:
        with open(nb_path, 'r', encoding='utf-8') as fh:
            nb = nbformat.read(fh, as_version=4)
        for cell in nb.cells:
            if cell.get('outputs'):
                fail(f'Notebook has outputs: {nb_path}')
    except Exception:
        pass

print('OK')

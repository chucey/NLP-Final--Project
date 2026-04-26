import json

with open('eval_runner-5.ipynb', 'r') as f:
    notebook = json.load(f)

if 'widgets' in notebook.get('metadata', {}):
    del notebook['metadata']['widgets']

for cell in notebook.get('cells', []):
    if 'widgets' in cell.get('metadata', {}):
        del cell['metadata']['widgets']

with open('eval_runner.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
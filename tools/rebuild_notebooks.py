import os, nbformat as nbf, pathlib, runpy
def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if c.startswith("# ") else nbf.v4.new_code_cell(c) for c in cells]
    return nb
targets = [
    ("01_robustness_svd_cholesky/notebooks/robustness_demo.ipynb", [
        "# Robustness demo (SVD & Cholesky)",
        "import runpy, pathlib\nroot = pathlib.Path(__file__).resolve().parents[2]\nrunpy.run_path(str(root/'01_robustness_svd_cholesky/python/run_robustness.py'))\nprint('Done. See reports/robustness/')",
    ]),
    ("02_uncertainty_conformal_bootstrap/notebooks/uncertainty_demo.ipynb", [
        "# Uncertainty demo (Conformal + Bootstrap)",
        "import runpy, pathlib\nroot = pathlib.Path(__file__).resolve().parents[2]\nrunpy.run_path(str(root/'02_uncertainty_conformal_bootstrap/python/run_uncertainty.py'))\nprint('Done. See reports/uncertainty/')",
    ]),
]
for path, cells in targets:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nb = make_nb(cells)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('Wrote', path)

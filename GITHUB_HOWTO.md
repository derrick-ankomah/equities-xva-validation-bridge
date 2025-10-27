# GitHub — Step-by-Step From Zero (macOS)

## 0) Install tools
```bash
# Install Homebrew if missing
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git python
xcode-select --install  # compilers
```

## 1) Create a GitHub account
- Visit github.com → **Sign up** → verify email.

## 2) Set up Git locally
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### (Option A) Use SSH (recommended)
```bash
ssh-keygen -t ed25519 -C "your@email.com"
# press Enter to accept the default file; add a passphrase if you like
cat ~/.ssh/id_ed25519.pub
```
Copy that printed key, then on GitHub: **Settings → SSH and GPG keys → New SSH key** → paste → Save.

### (Option B) Use HTTPS (simpler, asks password/token)
- On GitHub: **Settings → Developer settings → Personal access tokens → Fine-grained token**.  
- Create a token with `repo` scope and copy it (you'll paste it once when pushing).

## 3) Create the repo on GitHub (web)
- Click **New** repo → Name: `equities-xva-validation-bridge` → Public (or Private) → **Create repository**.

## 4) Initialize locally and push
```bash
cd /path/to/equities-xva-validation-bridge
git init
git add .
git commit -m "Initial commit: robustness, uncertainty, offset projects with C++ fastops"
# Link to GitHub:
# SSH:
git remote add origin git@github.com:<your-handle>/equities-xva-validation-bridge.git
# or HTTPS:
# git remote add origin https://github.com/<your-handle>/equities-xva-validation-bridge.git

git branch -M main
git push -u origin main
```

## 5) Build C++ wheels and run demos
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel build pybind11 scikit-build-core numpy pandas scipy scikit-learn lightgbm matplotlib yfinance

for d in 01_robustness_svd_cholesky 02_uncertainty_conformal_bootstrap 03_offset_model_weakness; do
  (cd $d/cpp && python -m build && pip install dist/*.whl)
done

python 01_robustness_svd_cholesky/python/run_robustness.py
python 02_uncertainty_conformal_bootstrap/python/run_uncertainty.py --use-yfinance
python 03_offset_model_weakness/python/run_offset.py
```

## 6) Ongoing workflow
```bash
git checkout -b feature/add-xva-demo
# edit files...
git add -p
git commit -m "Add XVA-specific covariance stress using Cholesky draws"
git push -u origin feature/add-xva-demo
# open Pull Request on GitHub
```

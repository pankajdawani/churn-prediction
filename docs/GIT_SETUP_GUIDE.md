# Git Setup Guide — Churn Prediction Project
## From Zero to GitHub in 10 Steps

This guide walks you through connecting your local project folder to GitHub for the first time.

---

## Prerequisites

Install Git if you haven't already:
- **Windows**: https://git-scm.com/download/win  
- **Mac**: `brew install git` or it's bundled with Xcode tools  
- **Linux**: `sudo apt install git`

Verify: `git --version`

---

## Step 1 — Configure Git (one-time setup)

Open your terminal and run:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

---

## Step 2 — Create the GitHub repository

1. Go to https://github.com and sign in
2. Click the **+** button → **New repository**
3. Name it: `churn-prediction`
4. Set visibility: **Public** or **Private**
5. ❗ Do **NOT** tick "Add README" or "Add .gitignore" (we already have these)
6. Click **Create repository**

GitHub will show you a URL like:  
`https://github.com/YOUR_USERNAME/churn-prediction.git`

Copy it — you'll need it in Step 5.

---

## Step 3 — Navigate to your project folder

```bash
cd path/to/churn_prediction
# Example on Windows:  cd C:\Users\YourName\Projects\churn_prediction
# Example on Mac/Linux: cd ~/Projects/churn_prediction
```

---

## Step 4 — Initialise Git in the folder

```bash
git init
```

You'll see: `Initialized empty Git repository in .../churn_prediction/.git/`

---

## Step 5 — Connect to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/churn-prediction.git
```

Verify the connection:
```bash
git remote -v
# Should show:
# origin  https://github.com/YOUR_USERNAME/churn-prediction.git (fetch)
# origin  https://github.com/YOUR_USERNAME/churn-prediction.git (push)
```

---

## Step 6 — Stage your files

```bash
git add .
```

Check what's staged (and confirm big/sensitive files are excluded by .gitignore):
```bash
git status
```

You should see your source files tracked, but NOT:
- `data/raw/*.csv`
- `models/artifacts/*.joblib`
- `logs/`
- `venv/`
- `__pycache__/`

---

## Step 7 — Make your first commit

```bash
git commit -m "feat: initial project structure — churn prediction pipeline"
```

---

## Step 8 — Push to GitHub

```bash
git branch -M main
git push -u origin main
```

If prompted for credentials:
- **Username**: your GitHub username
- **Password**: use a **Personal Access Token** (PAT), NOT your GitHub password

### Creating a Personal Access Token (PAT)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **Generate new token (classic)**
3. Set expiry, tick **repo** scope
4. Copy the token and use it as your password above

---

## Step 9 — Verify on GitHub

Open `https://github.com/YOUR_USERNAME/churn-prediction` in your browser.  
You should see all your files and the README rendered on the homepage. ✅

---

## Step 10 — Your daily Git workflow

After this initial setup, your everyday workflow is just 3 commands:

```bash
# 1. After making changes, stage them
git add .

# 2. Commit with a meaningful message
git commit -m "feat: add batch prediction endpoint"

# 3. Push to GitHub
git push
```

---

## Branching (best practice for features)

Never work directly on `main`. Create a branch for each feature or fix:

```bash
# Create and switch to a new branch
git checkout -b feature/add-xgboost-model

# ... make your changes, then commit ...
git add .
git commit -m "feat: add XGBoost as alternative classifier"

# Push the branch
git push -u origin feature/add-xgboost-model

# When done, open a Pull Request on GitHub, then merge into main
```

---

## Common Git Commands Reference

| Command | What it does |
|---------|-------------|
| `git status` | See what's changed / staged |
| `git log --oneline` | See commit history |
| `git diff` | See exact line-by-line changes |
| `git pull` | Get latest changes from GitHub |
| `git checkout -- file.py` | Undo changes to a single file |
| `git stash` | Temporarily shelve uncommitted changes |
| `git stash pop` | Restore stashed changes |

---

## Troubleshooting

**"Permission denied" when pushing**  
→ Make sure you're using a Personal Access Token, not your GitHub password.

**"Repository not found"**  
→ Double-check the remote URL: `git remote -v`  
→ Fix it: `git remote set-url origin https://github.com/YOUR_USERNAME/churn-prediction.git`

**"nothing to commit, working tree clean"**  
→ All your changes are already committed. Run `git push` if you haven't pushed yet.

**Accidentally committed a large file / secret**  
→ Remove it from git history: `git rm --cached path/to/file`, then add it to `.gitignore`

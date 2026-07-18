# Publishing to GitHub

The project folder in this zip is **already a git repository** with one
commit containing all 45 files (`git log` will show it). You just need to
create the empty remote on GitHub and push.

## 1. Create the empty repository on GitHub

1. Go to https://github.com/new
2. Repository name: `TrustFace-AI` (or whatever you prefer)
3. Visibility: your choice (Public shows it off in your portfolio; Private if you want to polish before revealing it)
4. **Do NOT** check "Add a README", "Add .gitignore", or "Choose a license" — this project already has all three, and GitHub will refuse to let you push a repo with conflicting initial files otherwise.
5. Click **Create repository**.

## 2. Push this local repo to it

Open a terminal in the extracted `TrustFace-AI/` folder and run:

```bash
git remote add origin https://github.com/<your-username>/TrustFace-AI.git
git push -u origin main
```

Replace `<your-username>` with your actual GitHub username. You'll be
prompted to authenticate — GitHub no longer accepts your account password
for this; use either:
- A **Personal Access Token** (GitHub → Settings → Developer settings →
  Personal access tokens → generate one with `repo` scope, use it as the
  password when prompted), or
- **GitHub CLI**: install `gh`, run `gh auth login` once, then `git push`
  works normally afterward.

## 3. Verify

Refresh the GitHub page — you should see all 45 files, the README
rendering on the repo homepage, and your one commit in the history.

## 4. Recommended follow-up commits (optional, but looks better than one giant commit)

If you want the commit history itself to tell a story (useful if anyone —
including a recruiter — actually looks at it), consider `git reset --soft
HEAD~1` right after pushing once, then re-committing in stages:

```bash
git reset --soft HEAD~1
git add backend/ requirements.txt run.py .gitignore
git commit -m "Core backend: face recognition, privacy, rule-based trust engine"

git add backend/ml_trust_engine.py backend/spoof_model.py
git commit -m "Add ML Trust Engine and pluggable anti-spoof model"

git add frontend/
git commit -m "Add web UI: dashboard, registration, live attendance, logs"

git add learning_lab/
git commit -m "Add Learning Lab and Experiments (AI curriculum + ablations)"

git add docs/ paper/ README.md LICENSE
git commit -m "Add documentation, model card, and research paper outline"

git push -u origin main --force
```

Only do this **before** anyone else has cloned/forked it — `--force` rewrites
history, which is fine for your own fresh repo but disruptive for
collaborators on a shared one.

## 5. Things to double-check before making it public

- Confirm `database/trustface.db` and `database/secret.key` are **not**
  tracked (`git ls-files | grep -E "\.db$|\.key$"` should print nothing —
  `.gitignore` already excludes both, this is just a sanity check).
- Confirm `models/trust_classifier.pkl` and any real `anti_spoof.onnx`
  you've added locally are also untracked — same reason, they're your
  personal trained artifacts / third-party weights, not source code.
- If you ever registered real people's faces while testing locally and
  then want to reset before pushing: delete `database/trustface.db` and
  `database/secret.key`, restart the app, it recreates both empty.

## 6. Suggested repo settings (nice-to-haves, not required)

- **About** section (gear icon next to "About" on the repo page): add a
  one-line description and topics like `face-recognition`, `fastapi`,
  `privacy`, `machine-learning`, `computer-vision` — this is what shows up
  in GitHub search and your profile's pinned repos.
- **Pin the repo** on your GitHub profile (Profile → Customize your pins)
  so it's the first thing recruiters see.

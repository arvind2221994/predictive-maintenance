# Predictive maintenance: automated training and deployment

These files extend the existing [arvind2221994/predictive-maintenance repository](https://github.com/arvind2221994/predictive-maintenance). No new GitHub repository is required.

The [GitHub Actions workflow](.github/workflows/pipeline.yml) validates the dataset, creates a train/test split, compares and tunes models, saves the selected model, reports held-out metrics, regenerates the Streamlit application and dependencies, runs tests, and pushes validated generated updates to `main`.

## 1. Copy the supplied files into your existing repository

Copy this folder's contents into your local repository root, merging folders rather than nesting this whole folder inside the repository. Keep your existing Git history and remote.

If you do not have a local clone yet:

```powershell
git clone https://github.com/arvind2221994/predictive-maintenance.git
Set-Location predictive-maintenance
```

Expected layout:

```text
repository-root\
    PredictiveMaintenance.ipynb
    README.md
    .gitignore
    requirements.txt
    streamlit_app.py
    .github\
        workflows\
            pipeline.yml
    scripts\
        ml_pipeline.py
    tests\
        test_pipeline.py
    data\
        README.md
        engine_data.csv          # You must supply this dataset.
    models\
        best_engine_model.pkl
    reports\
        metrics.json             # Created by training/evaluation.
```

The supplied model is the existing artifact from your GitHub repository, not a newly trained model. No training dataset is included. Add your real, permitted CSV according to the [data instructions](data/README.md).

**Your repository is public. Only commit data and models you are authorized to make public.** Never add credentials or private customer data. Intermediate cleaned/train/test CSVs are excluded by the [ignore rules](.gitignore).

## 2. Run and test locally

Use Python **3.12**, which matches the supplied workflow. Open PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\ml_pipeline.py all
.\.venv\Scripts\python.exe -m compileall -q scripts tests streamlit_app.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501 and submit all six sensor readings. Verify the one-row DataFrame, prediction, class probabilities, and CSV download. Stop the app with Ctrl+C.

The [pipeline script](scripts/ml_pipeline.py) also exposes each stage separately:

```powershell
.\.venv\Scripts\python.exe scripts\ml_pipeline.py prepare
.\.venv\Scripts\python.exe scripts\ml_pipeline.py split
.\.venv\Scripts\python.exe scripts\ml_pipeline.py train
.\.venv\Scripts\python.exe scripts\ml_pipeline.py evaluate
.\.venv\Scripts\python.exe scripts\ml_pipeline.py build
```

Run these in order. Missing inputs, invalid data, fitting errors, and dependency problems fail explicitly.

## 3. Commit and push the initial setup yourself

Once the local run and tests pass, inspect what you will publish and run:

```powershell
git status --short
git add -- .github\workflows\pipeline.yml scripts\ml_pipeline.py tests\test_pipeline.py README.md data\README.md .gitignore PredictiveMaintenance.ipynb streamlit_app.py requirements.txt models\best_engine_model.pkl data\engine_data.csv reports\metrics.json
git diff --cached --stat
git commit -m "Add automated predictive maintenance pipeline" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push origin main
```

These commands assume your local branch is `main`. If your repository uses pull requests, commit on your feature branch and merge through the normal review process. Do not force-push or overwrite existing history.

## 4. Configure GitHub Actions

1. In your existing repository, open **Settings > Actions > General** and ensure Actions and the referenced official `actions/*` actions are allowed.
2. The workflow starts with `contents: read`. Only the guarded `update-main` job requests `contents: write`, using GitHub's built-in `GITHUB_TOKEN`; no personal access token is needed.
3. Your repository/organization policy and branch rules must permit the bot's direct push to `main`. Required-PR or other protection rules can reject it even with write permissions. Do not disable or bypass mandatory protection; use a PR-based publishing workflow instead if direct bot pushes are not allowed.
4. Open **Actions > Predictive maintenance ML pipeline**. The initial push starts a run. You can also select **Run workflow** on `main`.
5. Inspect each step, the generated deployment artifact, and the subsequent bot commit.

No GitHub credentials are embedded in the notebook or files.

## 5. Automated behavior

| Event | Behavior |
| --- | --- |
| Relevant code, notebook, dependencies, or dataset changes pushed to `main` | Train, evaluate, regenerate the app, run tests, then publish generated files. |
| Pull request targeting `main` | Run the same checks with read-only repository permissions; never push generated files. |
| Manual run on `main` | Run the full pipeline and publish after success. |
| Manual run on another branch | Run checks only; do not publish to `main`. |
| Any failed stage/test | Stop; publishing does not run. |
| No generated changes | Exit successfully without an empty commit. |
| `main` changed during training | Reject stale results. Re-run using the latest `main`. |

The publisher stages **only**:

- [streamlit_app.py](streamlit_app.py)
- [requirements.txt](requirements.txt)
- [models/best_engine_model.pkl](models/best_engine_model.pkl)
- [reports/metrics.json](reports/metrics.json)

It does not use `git add .`, force-push, merge unrelated changes, or publish intermediate datasets. Pushes made using `GITHUB_TOKEN` do not recursively trigger another push workflow.

### Source of truth for generated code

Edit `app_code` in **Section 5** of [PredictiveMaintenance.ipynb](PredictiveMaintenance.ipynb), then commit the saved notebook. The `build` stage parses that literal string without executing the full notebook or its Colab upload cells. Direct edits to the generated app will be replaced by the next build.

Dependency generation pins all six libraries, including Streamlit, to the environment that passed the pipeline. To upgrade them, edit the dependency file and test first; changing it triggers retraining with the new versions.

The app's cached model is keyed by its path and modification time, so an updated artifact is reloaded on the next app rerun.

### Model selection and interpretation

The pipeline uses the same six raw sensor features as the existing app. It removes duplicates and excludes non-feature columns, rejects invalid values, and preserves temperature outliers. It does not add feature ratios or scaling that the app cannot reproduce.

Candidate comparison and tuning use **training-only 3-fold cross-validation**, rather than selecting models using the held-out test set. The test partition is evaluated only after selection/tuning. The winner's own parameter grid is used, including when it is not AdaBoost.

The test suite checks correctness and compatibility, not a business-specific minimum F1 or an improvement over the previously deployed model. Define your production acceptance thresholds before using automated retraining operationally. Confirm the original meanings of target labels `0` and `1`; class scores are not necessarily calibrated failure probabilities.

## 6. Connect Streamlit Community Cloud

At https://share.streamlit.io, connect the existing repository, select branch `main`, and set the main file path to `streamlit_app.py`. Choose Python **3.12** in Advanced settings and deploy.

Community Cloud follows the connected GitHub branch. After a validated bot commit, check the updated app and deployment logs. GitHub Actions does not create a Cloud account or perform the initial connection for you. Session DataFrames are temporary; use the CSV download or an external database for durable storage.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| Missing dataset | Add the real permitted `data\engine_data.csv` before pushing/running. No synthetic dataset is shipped. |
| Missing committed notebook | Copy the updated notebook to the repository root with the exact filename shown in the layout. |
| Missing/deviating `deploymentWriteApp` cell | Keep Section 5's app-code cell and its metadata ID; the build rejects ambiguous or dynamically generated source. |
| Workflow is not visible | The file must be at `.github\workflows\pipeline.yml` at the repository root, on the default branch. |
| Bot push denied | Check repository token policy and branch rules; do not bypass required reviews. |
| Stale-main error/non-fast-forward rejection | Start a fresh run from the latest `main`; do not force-push stale artifacts. |
| Import/model-version error | Install the pinned dependencies and retrain under the matching Python version. |
| Cloud app fails | Check Cloud logs, required model path, exact casing, and Python version. |

Official references: [GitHub Python workflows](https://docs.github.com/en/actions/tutorials/build-and-test-code/python), [GITHUB_TOKEN permissions](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token), [workflow triggering](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow), and [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

# Installing and running locally

This walks through getting the project running on your own machine: the tests,
both demo apps, and (optionally) the notebooks. No GPU, no Docker and no
Hugging Face account are needed — the served models are small scikit-learn
files (8.5 MB and 0.6 MB) committed to the repo.

Verified on Windows 11 with Python 3.14.5, from a fresh virtual environment in
a folder other than the one the project was developed in.

---

## 1. Prerequisites

| | Version | Check with |
|---|---|---|
| Python | **3.14** (what the pins were captured on) | `python --version` |
| Git | any recent | `git --version` |
| Disk | a couple of GB free for the virtual environment | |

Other Python versions are not tested. The pins in `requirements.txt` are exact,
so pip may fail to find wheels for an older interpreter; if it does, install
Python 3.14 from [python.org](https://www.python.org/downloads/) rather than
loosening the pins — the models were trained and measured with those exact
library versions.

---

## 2. Get the code

```bash
git clone https://github.com/SaadTadja/nlp-satisfaction.git
```

```bash
cd nlp-satisfaction
```

Any folder works, including one with spaces in its path: model files are
referenced relative to the repo, not by absolute path.

---

## 3. Create a virtual environment and install

**Windows (PowerShell):**

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script ("running scripts is
disabled on this system"), either use `cmd.exe` and run
`.venv\Scripts\activate.bat`, or skip activation entirely and call
`.venv\Scripts\python.exe` directly in every command below.

**macOS / Linux:**

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

**Then, on any OS:**

```bash
python -m pip install -r requirements.txt
```

This is a large download (gradio, transformers, datasets, pyarrow …); it took
**about 10 minutes** when this guide was verified. torch is deliberately *not*
installed — the TF-IDF models don't use it.

---

## 4. Check the install

```bash
python -m pytest tests/ -q
```

Expected: `72 passed` in a few seconds. One of those tests
(`test_shipped_configs_point_at_files_in_this_clone`) confirms both
`inference_config*.json` files resolve to model files that exist in your
clone.

---

## 5. Run the demos

There are two independent Gradio apps.

**French — film reviews (Allociné), positive/negative with abstention:**

```bash
python space/app.py
```

**English — hotel reviews, 3 classes:**

```bash
python space/app_en.py
```

Each prints a local URL — open **http://127.0.0.1:7860** in your browser.
Stop the app with `Ctrl+C`. To run both at once, start the second on another
port:

```bash
GRADIO_SERVER_PORT=7861 python space/app_en.py
```

(PowerShell: `$env:GRADIO_SERVER_PORT=7861; python space/app_en.py`)

### What to try

| App | Tab | Input | What you should see |
|---|---|---|---|
| French | Analyse groupée | upload `data/exemple_feedback.csv` | ~32% satisfied, interval ~26–37%, aspect table led by *scénario* |
| French | Commentaire unique | *"Scénario incohérent, dialogues ridicules."* | **NÉGATIF** |
| English | Batch | upload `data_en/example_reviews.csv` | ~55% satisfied, interval ~49–61%; the true share is 51.8% |
| English | Single review | *"The room was spotless and the staff were incredibly welcoming."* | **Positive**, topic *Personnel* |

The batch tabs accept any CSV with a text column named `text` (French also
accepts `texte`, `commentaire`, `review`, `avis`; English also accepts
`review`, `comment`, `feedback`, `phrase`). Up to 5,000 rows.

---

## 6. Run the notebooks (optional)

Jupyter is not in `requirements.txt`; add it to the same environment:

```bash
python -m pip install notebook
```

```bash
python -m jupyter notebook notebooks/
```

| Notebooks | Data they need | Notes |
|---|---|---|
| `00`–`08` (French) | `tblard/allocine`, downloaded automatically from the Hugging Face Hub on first run (~200k reviews) | An "unauthenticated requests" warning is harmless. Run in order `00 → 01 → 02 → 04 → 06 → 07`. |
| `08_final_test` | same | The only notebook that reads the test split. Re-running it is a new look at test — read its header first. |
| `11`–`13` (English) | the CSVs already in `data_en/` | Run as-is. |
| `10`, `15` (English) | `dataset for training in ENGLISH/df_final_absa_v4.csv` | **Not in the repo** (233 MB, git-ignored). Without it these two cannot be re-run (`14` reads the committed `data_en/sentences.csv` and only uses the raw file, if present, for one provenance line); their committed outputs and the CSVs they produced in `data_en/` are what the apps use. |

Notebooks `02` and `12` retrain and overwrite the served models. If you re-run
them, re-run the calibration and aggregate notebooks after them too (`04`,
`06` for French; `13` for English) — the configs' temperature, band and error
rates belong to one specific set of weights.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| App shows **"Modèle non chargé" / "Model not loaded"** with a file-not-found message | You are running an old copy whose config holds an absolute `D:\...` path. Pull the latest version; configs now store `data/model_tfidf.joblib` etc. |
| `No matching distribution found for numpy==2.4.6` (or similar) | Your Python is not 3.14. See §1. |
| `OSError: Cannot find empty port` | Something already uses 7860 — set `GRADIO_SERVER_PORT` as above. |
| `InconsistentVersionWarning` when a model loads | scikit-learn is not 1.8.0. Reinstall from `requirements.txt` inside the virtual environment. |
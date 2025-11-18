# NLP Project

This repo scaffolds an NLP project.

Quick start

1. Create a Python virtual environment and install requirements:

```markdown
# FEVER Fact-Verification — NLP Project

Lightweight reproducible repository for experiments on the FEVER fact-verification dataset. Contains preprocessing utilities, classical TF‑IDF baselines, and training helpers for transformer models.

## Quick overview
- `src/` — preprocessing, baseline runners, and training helpers
- `data/` — NOT committed to Git; raw and processed datasets (download locally)
- `outputs/` — model checkpoints and evaluation artifacts (gitignored)

This repo intentionally keeps data and heavy artifacts out of Git. See the `scripts/` folder (if present) for data download helpers.

## Quickstart (Windows CMD)
1) Create and activate a virtual environment, then install dependencies:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Download FEVER raw files (examples):

Option A — automatic (if a `scripts/download_fever.py` exists):

```cmd
python scripts\download_fever.py
```

Option B — manual (we used these canonical URLs):

```cmd
mkdir data\raw\fever
curl -L -o data\raw\fever\train.jsonl     https://fever.ai/download/fever/train.jsonl
curl -L -o data\raw\fever\shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl -L -o data\raw\fever\shared_task_test.jsonl https://fever.ai/download/fever/shared_task_test.jsonl
```

If programmatic download is blocked, download those three files with your browser and save them under `data\raw\fever`.

3) Preprocess into `datasets` format (creates `data/processed/fever`):

```cmd
python -c "from src.data_preprocessing import prepare_fever_dataset; local_files={'train':'data\\raw\\fever\\train.jsonl','validation':'data\\raw\\fever\\shared_task_dev.jsonl','test':'data\\raw\\fever\\shared_task_test.jsonl'}; prepare_fever_dataset(dataset_name=None, local_path=local_files, out_dir='data\\processed\\fever')"
```

4) Run the TF‑IDF baselines (example — 1000 samples per class):

```cmd
python -c "from src.baselines import run_baselines; run_baselines(processed_data_dir='data\\processed\\fever', sample_limit=1000)"
```

5) Inspect processed samples quickly:

```cmd
python src\inspect_processed_dataset.py
```

## Notes & recommendations
- Do NOT commit the `data/` folder or model checkpoints to Git. They are listed in `.gitignore`.
- If you need to share large files, use Git LFS, DVC, or a cloud storage bucket and keep pointers (not raw data) in the repo.
- `shared_task_test.jsonl` is often unlabelled for FEVER; the baseline runner will automatically fall back to the labelled dev split for evaluation when test labels are missing.

## Reproducing on another machine
1. Clone the repo, create the virtual environment, and install requirements.
2. Run `scripts\download_fever.py` (if present) or download the three FEVER files manually into `data\raw\fever`.
3. Run the preprocessing and baselines commands above.

## License
This project is provided under the MIT License. Add `LICENSE` with the MIT text if you choose this license.

```

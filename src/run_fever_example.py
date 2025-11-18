"""
End-to-end smoke test for FEVER pipeline.

Steps:
1. Prepare FEVER (loads from HF `fever` by default)
2. Optionally sample N examples per class (balanced) to make runs fast
3. Train TF-IDF + Logistic Regression baseline and report metrics
4. Save sampled dataset to a temporary folder and run a 1-epoch DistilBERT fine-tune using `src.train.run_training`

Usage (PowerShell):
python -c "from src.run_fever_example import main; main(sample_per_class=200)"

Note: install dependencies in `requirements.txt` first.
"""
from datasets import load_from_disk
from src.data_preprocessing import prepare_fever_dataset
from src.train import run_training
from datasets import DatasetDict
import numpy as np
import os
import tempfile
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def sample_balanced(ds: DatasetDict, n_per_class: int = 200):
    """Return a new DatasetDict with up to n_per_class examples per class from each split."""
    new_splits = {}
    for split_name, split in ds.items():
        if 'label' in split.column_names:
            labels = np.array(split['label'])
        elif 'labels' in split.column_names:
            labels = np.array(split['labels'])
        else:
            raise ValueError('Dataset must have `label` or `labels` column')

        # filter out unknown labels (<0)
        valid_idxs = np.where(labels >= 0)[0]
        labels = labels[valid_idxs]
        # unique labels
        unique = np.unique(labels)
        chosen = []
        for u in unique:
            idxs = valid_idxs[np.where(labels == u)[0]]
            if len(idxs) == 0:
                continue
            np.random.shuffle(idxs)
            take = idxs[:n_per_class]
            chosen.extend(take.tolist())
        chosen = sorted(chosen)
        # if no chosen (e.g., tiny split), keep original
        if len(chosen) == 0:
            new_splits[split_name] = split
        else:
            new_splits[split_name] = split.select(chosen)
    return DatasetDict(new_splits)


def run_tfidf_baseline(ds: DatasetDict):
    """Train TF-IDF + LogisticRegression on ds['train'] and evaluate on val/test."""
    train = ds['train']
    val = ds['validation'] if 'validation' in ds else None
    test = ds['test'] if 'test' in ds else None

    def get_texts_labels(split):
        if 'text' in split.column_names:
            texts = split['text']
        else:
            # fallback: try claim + evidence
            texts = [ (c if c is not None else '') for c in split.get('claim', ['']*len(split)) ]
        labels = np.array(split['label'] if 'label' in split.column_names else split['labels'])
        return list(texts), labels

    train_texts, train_labels = get_texts_labels(train)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vec.fit_transform(train_texts)

    clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000)
    clf.fit(X_train, train_labels)

    def eval_on(split, name):
        if split is None:
            return
        texts, labels = get_texts_labels(split)
        X = vec.transform(texts)
        preds = clf.predict(X)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"{name} - Acc: {acc:.4f}  Macro-F1: {f1:.4f}")
        print(classification_report(labels, preds))

    print('\nTF-IDF + Logistic Regression results:')
    eval_on(val, 'Validation')
    eval_on(test, 'Test')


def main(sample_per_class: int = 200, do_preprocess: bool = True, do_tfidf: bool = True, do_transformer: bool = True):
    # 1) prepare dataset (downloads from HF if needed)
    processed_dir = 'data/processed/fever'
    if do_preprocess:
        print('Preparing FEVER dataset (this will download if needed).')
        prepare_fever_dataset(out_dir=processed_dir)

    # Try to load the processed FEVER dataset. If it's missing or invalid
    # (e.g. empty folder), fall back to a small synthetic dataset so the
    # smoke test can run without downloading FEVER right now.
    try:
        ds = load_from_disk(processed_dir)
        # if folder exists but is empty, treat as missing
        if not any(os.scandir(processed_dir)):
            raise FileNotFoundError('processed directory is empty')
    except Exception:
        print('Warning: processed FEVER not found or invalid. Using a small synthetic dataset for the smoke test.')
        from datasets import Dataset
        # create a tiny balanced synthetic dataset with 3 classes
        texts = [
            'The Eiffel Tower is in Paris.',
            'The Moon is made of cheese.',
            'An unknown claim with insufficient evidence.'
        ]
        labels = [0, 1, 2]
        # replicate to make small splits
        train_texts = texts * 20
        train_labels = labels * 20
        val_texts = texts * 5
        val_labels = labels * 5
        test_texts = texts * 5
        test_labels = labels * 5

        ds = DatasetDict({
            'train': Dataset.from_dict({'text': train_texts, 'label': train_labels}),
            'validation': Dataset.from_dict({'text': val_texts, 'label': val_labels}),
            'test': Dataset.from_dict({'text': test_texts, 'label': test_labels}),
        })

    # 2) sample balanced subset to speed up runs
    if sample_per_class is not None and sample_per_class > 0:
        print(f'Sampling up to {sample_per_class} examples per class.')
        ds_sample = sample_balanced(ds, n_per_class=sample_per_class)
    else:
        ds_sample = ds

    # 3) TF-IDF baseline
    if do_tfidf:
        run_tfidf_baseline(ds_sample)

    # 4) Transformer fine-tune (1 epoch) on sampled data
    if do_transformer:
        print('\nRunning 1-epoch DistilBERT fine-tune on sampled data (this may download model weights).')
        tmpdir = tempfile.mkdtemp(prefix='fever_sample_')
        print('Saving sampled dataset to', tmpdir)
        ds_sample.save_to_disk(tmpdir)
        try:
            run_training(processed_data_dir=tmpdir,
                         model_name='distilbert-base-uncased',
                         output_dir='outputs/fever-smoke-distilbert',
                         num_labels=3,
                         epochs=1,
                         batch_size=8)
        finally:
            # keep outputs but remove temporary dataset folder
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass


if __name__ == '__main__':
    # default quick smoke: 200 per class
    main(sample_per_class=200)

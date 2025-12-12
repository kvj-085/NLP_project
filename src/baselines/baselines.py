"""
Simple baseline runner: TF-IDF + Logistic Regression / SVM / RandomForest

Usage:
    from src.baselines.baselines import run_baselines
    run_baselines(processed_data_dir='data/processed/fever', sample_per_class=200)

Outputs: prints metrics and saves `outputs/baselines_results.csv`.
"""
import os
from datasets import load_from_disk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import csv

# Checks if the split has a 'text' column — if yes, use it otherwise fallback to 'claim' col. (text = claim [SEP] evidence.)
# Converts labels to a NumPy array.
def _get_texts_labels(split):
    if 'text' in split.column_names:
        texts = list(split['text'])
    else:
        texts = [ (c if c is not None else '') for c in split.get('claim', ['']*len(split)) ]
    labels = np.array(split['label'] if 'label' in split.column_names else split['labels'])
    return texts, labels



# It loops through every text/label pair and discards:
# None values
# empty strings " "
# labels that cannot be converted to integers

def _filter_nonempty(texts, labels):
    # keep only examples with non-empty text and valid label
    keep_texts = []
    keep_labels = []
    for t, l in zip(texts, labels):
        if t is None:
            continue
        s = str(t).strip()
        if not s:
            continue
        # avoid weird label placeholders
        try:
            keep_labels.append(int(l))
            keep_texts.append(s)
        except Exception:
            # skip if label cannot be cast to int
            continue
    if not keep_texts:
        return [], np.array([])
    return keep_texts, np.array(keep_labels)


def run_baselines(processed_data_dir: str = 'data/processed/fever', sample_limit: int = None):
    os.makedirs('outputs', exist_ok=True)
    # loads your FEVER dataset (train/val/test splits)
    ds = load_from_disk(processed_data_dir)

    train_texts, train_labels = _get_texts_labels(ds['train'])
    val_texts, _ = _get_texts_labels(ds['validation'])
    test_texts, test_labels = _get_texts_labels(ds['test'])

    # filter empty texts before sampling/vectorizing
    train_texts, train_labels = _filter_nonempty(train_texts, train_labels)
    val_texts, _ = _filter_nonempty(val_texts, np.zeros(len(val_texts))) if val_texts else ([], np.array([]))
    test_texts, test_labels = _filter_nonempty(test_texts, test_labels)

    if sample_limit:
        # simple downsampling per class on train
        keep_idxs = []
        labels = np.array(train_labels)
        for cls in np.unique(labels):
            idxs = np.nonzero(labels == cls)[0]
            rng = np.random.default_rng(42)
            rng.shuffle(idxs)
            keep = idxs[:sample_limit]
            keep_idxs.extend(keep.tolist())
        keep_idxs = sorted(keep_idxs)
        train_texts = [train_texts[i] for i in keep_idxs]
        train_labels = train_labels[keep_idxs]

    if not train_texts:
        raise ValueError('No training texts available after filtering empty examples. Check processed dataset.')
    # TF-IDF vectorization
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", min_df=1)
    try:
        X_train = vec.fit_transform(train_texts)
    except ValueError as e:
        raise RuntimeError(f"TfidfVectorizer failed: {e}. Sample of training texts: {train_texts[:5]}") from e
    _ = vec.transform(val_texts)
    X_test = vec.transform(test_texts)
    # Define the classifiers
    classifiers = {
        'LogisticRegression': LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    }

    # If test labels are missing (e.g., unlabelled test split with -1),
    # fall back to using validation for evaluation and warn the user.
    use_validation_for_eval = (len(test_labels) == 0) or (all(l == -1 for l in test_labels))
    if use_validation_for_eval:
        print('Warning: test labels appear missing or invalid; using validation split for evaluation instead.')
        eval_texts = val_texts
        eval_labels = np.array([int(l) for l in ds['validation']['label']]) if 'label' in ds['validation'].column_names else np.array([])
        if len(eval_texts) == 0 or len(eval_labels) == 0:
            raise RuntimeError('No labelled data available for evaluation.')
        X_eval = vec.transform(eval_texts)
    else:
        X_eval = X_test
        eval_labels = test_labels

    # Training loop: Train on TF-IDF vectors, Predict on evaluation set,
    # Compute accuracy + macro-F1, Print classification report, Store results
    
    results = []
    for name, clf in classifiers.items():
        print(f'--- Training {name} ---')
        clf.fit(X_train, train_labels)
        preds = clf.predict(X_eval)
        acc = accuracy_score(eval_labels, preds)
        f1 = f1_score(eval_labels, preds, average='macro')
        print(f'{name} Eval Acc: {acc:.4f} Macro-F1: {f1:.4f}')
        print(classification_report(eval_labels, preds))
        results.append({'model': name, 'accuracy': acc, 'macro_f1': f1})

    # save results
    out_file = os.path.join('outputs', 'baselines_results.csv')
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model','accuracy','macro_f1'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('Saved baseline results to', out_file)


if __name__ == '__main__':
    run_baselines()

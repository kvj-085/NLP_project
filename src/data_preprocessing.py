from datasets import load_dataset, DatasetDict, Dataset
from datasets import load_dataset, DatasetDict
import os
from typing import Optional, Dict, Any

# converts FEVER’s string labels into numeric IDs
FEVER_LABEL_MAP = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT ENOUGH INFO': 2,
    'NOT_ENOUGH_INFO': 2,
}

# FEVER dataset stores evidence in different inconsistent formats depending on version. 
# This helper function:
# Handles when evidence is a list
# Handles nested lists
# Handles dictionaries containing "text"
# Handles cases where evidence is stored as wiki title + sentence index
# Falls back to strings if necessary

# Joins all evidence pieces into one single string ie, text = claim + [SEP] + evidence

def _extract_evidence_text(example: Dict[str, Any]) -> str:
    """Robustly extract evidence text from a FEVER example.

    FEVER evidence can have multiple possible structures depending on dataset version.
    This helper tries several strategies and returns a concatenated string.
    """
    ev = example.get('evidence', None)
    if not ev:
        return ''

    # If evidence is a list of lists (annotations), flatten
    if isinstance(ev, list):
        parts = []
        for item in ev:
            if isinstance(item, list):
                # inner list of evidence pieces
                for sub in item:
                    if isinstance(sub, dict):
                        # dict might contain a 'text' field
                        text = sub.get('text') or sub.get('sentence') or sub.get('evidence_text')
                        if text:
                            parts.append(text)
                    elif isinstance(sub, str):
                        parts.append(sub)
                    elif isinstance(sub, list):
                        # FEVER uses lists like [wiki_id, sent_id, wiki_title, sent_index]
                        # prefer to extract the wiki_title if available
                        try:
                            if len(sub) >= 3 and isinstance(sub[2], str) and sub[2]:
                                # replace underscores in wiki titles for readability
                                parts.append(sub[2].replace('_', ' '))
                                continue
                        except Exception:
                            pass
                        # fallback: collect any string elements inside
                        for elem in sub:
                            if isinstance(elem, str):
                                parts.append(elem)
            elif isinstance(item, dict):
                text = item.get('text') or item.get('sentence') or item.get('evidence_text')
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return ' '.join(parts)

    # If evidence is a dict with text
    if isinstance(ev, dict):
        return ev.get('text', '')

    return ''


# we manually download raw FEVER files using the HuggingFace Hub, 
# parse claim and evidence, normalize them, and save them in the same format.
def prepare_fever_via_hf_hub(out_dir: str = 'data/processed/fever',
                             repo_id: str = 'fever',
                             repo_type: str = 'dataset',
                             split_ratio=(0.7, 0.15, 0.15),
                             label_map: Dict[str, int] = FEVER_LABEL_MAP):
    """Download FEVER raw files from the Hub using `huggingface_hub` and build a dataset.

    This avoids calling `datasets.load_dataset('fever')` which may try to run dataset scripts.
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except Exception as e:
        raise RuntimeError('huggingface_hub is required for hf-hub fallback. Install with `pip install huggingface_hub`.') from e

    os.makedirs(out_dir, exist_ok=True)
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='fever_hub_')
    try:
        files = list_repo_files(repo_id, repo_type=repo_type)
        # choose likely train/validation/test filenames using broader heuristics
        lowered = [f.lower() for f in files]
        candidates = [f for f in files if any(k in f.lower() for k in ('train', 'dev', 'validation', 'test'))
                      or any(ext in f.lower() for ext in ('.jsonl', '.json', '.ndjson', '.tsv', '.csv', '.zip', '.tar.gz', '.tgz'))]

        # heuristics to pick files (prefer json/jsonl if possible)
        def pick(preferred_keys):
            for key in preferred_keys:
                for f in candidates:
                    if key in f.lower():
                        return f
            # fallback: prefer json/jsonl extensions
            for f in candidates:
                if any(f.lower().endswith(ext) for ext in ('.jsonl', '.json', '.ndjson')):
                    return f
            # last resort: return first candidate
            return candidates[0] if candidates else None

        train_file = pick(['train'])
        val_file = pick(['valid', 'dev'])
        test_file = pick(['test'])

        data_files = {}
        if train_file:
            local_train = hf_hub_download(repo_id=repo_id, filename=train_file, cache_dir=tmp_dir, repo_type=repo_type)
            data_files['train'] = local_train
        if val_file:
            local_val = hf_hub_download(repo_id=repo_id, filename=val_file, cache_dir=tmp_dir, repo_type=repo_type)
            data_files['validation'] = local_val
        if test_file:
            local_test = hf_hub_download(repo_id=repo_id, filename=test_file, cache_dir=tmp_dir, repo_type=repo_type)
            data_files['test'] = local_test

        if not data_files:
            # dump diagnostic file list to tmp for user inspection
            try:
                # write diagnostic listing into out_dir so it persists for user inspection
                diag_path = os.path.join(out_dir, 'fever_hub_repo_files.txt')
                os.makedirs(out_dir, exist_ok=True)
                with open(diag_path, 'w', encoding='utf-8') as fh:
                    fh.write('\n'.join(files))
            except Exception:
                diag_path = None
            msg = 'Could not locate FEVER raw data files in the Hub repo.'
            if diag_path:
                msg += f" Wrote repository file listing to: {diag_path}\nPlease inspect or share this file so we can pick the correct filenames."
            else:
                msg += ' (Also failed to write diagnostic file list.)'
            raise RuntimeError(msg)

        from datasets import load_dataset
        ds = load_dataset('json', data_files=data_files)

        # normalize similar to prepare_fever_dataset
        def normalize_hf(example):
            claim = example.get('claim') or example.get('sentence') or ''
            evidence_text = _extract_evidence_text(example)
            if evidence_text:
                text = f"{claim} [SEP] {evidence_text}"
            else:
                text = claim
            label = example.get('label')
            if isinstance(label, str):
                label = label.strip()
                label = label if label in label_map else label.replace('_', ' ')
                label_id = label_map.get(label, -1)
            else:
                label_id = int(label) if label is not None else -1
            return {'text': text, 'label': label_id}

        ds = ds.map(normalize_hf)
        ds.save_to_disk(out_dir)
        return out_dir
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


if __name__ == '__main__':
    print('Preparing FEVER dataset sample...')
    print('This will download and process FEVER into `data/processed/fever`.')

from datasets import load_dataset, DatasetDict, Dataset
from datasets import load_dataset, DatasetDict
import os
from typing import Optional, Dict, Any


FEVER_LABEL_MAP = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT ENOUGH INFO': 2,
    'NOT_ENOUGH_INFO': 2,
}


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


def prepare_fever_dataset(dataset_name: Optional[str] = 'fever',
                          local_path: Optional[str] = None,
                          out_dir: str = 'data/processed/fever',
                          split_ratio=(0.7, 0.15, 0.15),
                          label_map: Dict[str, int] = FEVER_LABEL_MAP):
    """Load FEVER dataset (HF id `fever` or a local CSV) and produce a `text` + `label` dataset.

    By default this function splits data into `train`, `validation`, `test` with a
    70:15:15 ratio (you can override `split_ratio`). Each example will have fields
    `text` (concatenation of claim and evidence) and `label` (int).
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ensure HF cache directory is explicit to avoid dataset script files being
    # written into the current working directory on some Windows setups.
    hf_user = os.environ.get('USERPROFILE') or os.environ.get('HOME') or '.'
    hf_cache = os.path.join(hf_user, '.cache', 'huggingface', 'datasets')
    os.makedirs(hf_cache, exist_ok=True)

    import tempfile
    import shutil

    # If local_path is provided, resolve any relative paths to absolute paths
    if local_path:
        def _absify_paths(lp):
            if isinstance(lp, str):
                return os.path.abspath(lp)
            if isinstance(lp, dict):
                return {k: os.path.abspath(v) for k, v in lp.items()}
            if isinstance(lp, (list, tuple)):
                return [os.path.abspath(v) for v in lp]
            return lp

        try:
            local_path = _absify_paths(local_path)
        except Exception:
            pass

    # Run load_dataset from a temporary working directory so the dataset
    # script files are not written into the project root on Windows.
    tmp_cwd = tempfile.mkdtemp(prefix='hf_tmp_cwd_')
    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_cwd)
        # Use a temporary cache directory for this download to avoid reusing
        # a possibly-broken cached snapshot that contains dataset scripts.
        tmp_cache = tempfile.mkdtemp(prefix='hf_cache_')
        try:
            if dataset_name and not local_path:
                ds = load_dataset(dataset_name, cache_dir=tmp_cache)
            elif local_path and not dataset_name:
                # detect local file format and call appropriate loader
                lp = local_path

                def _load_jsonl_file(path):
                    import json as _json
                    texts = []
                    labels = []
                    with open(path, 'r', encoding='utf-8') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = _json.loads(line)
                            except Exception:
                                # skip malformed lines
                                continue
                            # build text and label robustly
                            claim = obj.get('claim') or obj.get('sentence') or ''
                            evidence_text = _extract_evidence_text(obj)
                            if evidence_text:
                                text = f"{claim} [SEP] {evidence_text}"
                            else:
                                text = claim
                            label = obj.get('label')
                            if isinstance(label, str):
                                label = label.strip()
                                label = label if label in label_map else label.replace('_', ' ')
                                label_id = label_map.get(label, -1)
                            else:
                                try:
                                    label_id = int(label) if label is not None else -1
                                except Exception:
                                    label_id = -1
                            texts.append(text)
                            labels.append(label_id)
                    return Dataset.from_dict({'text': texts, 'label': labels})

                def _load_single_path(path):
                    path_lower = path.lower()
                    if path_lower.endswith(('.jsonl', '.ndjson')):
                        return _load_jsonl_file(path)
                    if path_lower.endswith('.json'):
                        # try as JSON array, otherwise fallback to JSONL
                        try:
                            import json as _json
                            with open(path, 'r', encoding='utf-8') as fh:
                                content = fh.read()
                            parsed = _json.loads(content)
                            if isinstance(parsed, list):
                                return Dataset.from_list(parsed)
                            # not a list -> fallback to line-by-line
                        except Exception:
                            pass
                        return _load_jsonl_file(path)
                    if path_lower.endswith(('.csv', '.tsv')):
                        return load_dataset('csv', data_files=path, cache_dir=tmp_cache)
                    # fallback: try jsonl then csv
                    try:
                        return _load_jsonl_file(path)
                    except Exception:
                        return load_dataset('csv', data_files=path, cache_dir=tmp_cache)

                if isinstance(lp, str):
                    ds = _load_single_path(lp)
                elif isinstance(lp, dict):
                    parts = {}
                    for k, v in lp.items():
                        parts[k] = _load_single_path(v)
                    # If datasets returned are plain Dataset objects, combine into DatasetDict
                    dataset = DatasetDict()
                    for k, d in parts.items():
                        if isinstance(d, Dataset):
                            dataset[k] = d
                        else:
                            # if load_dataset returned a DatasetDict for csv loader
                            # try to assign sensible split
                            if 'train' in d:
                                dataset[k] = d['train']
                            else:
                                # assign first split
                                dataset[k] = d[list(d.keys())[0]]
                    ds = dataset
                else:
                    # pass through lists/other types to datasets loader
                    ds = load_dataset('json', data_files=lp, cache_dir=tmp_cache)
            else:
                raise ValueError('Provide either dataset_name (HF id) or local_path (CSV/TSV/JSON)')
        except RuntimeError as e:
            # Debug: print workspace/cache contents to locate where `fever.py` is created
            try:
                print('\n--- DEBUG: dataset load failed, inspecting temp dirs ---')
                print('tmp_cwd:', tmp_cwd)
                print('tmp_cwd contents:', os.listdir(tmp_cwd))
                print('tmp_cache:', tmp_cache)
                try:
                    print('tmp_cache contents (top-level):', os.listdir(tmp_cache))
                except Exception as _:
                    print('tmp_cache not readable or empty')
                print('cwd after load attempt:', os.getcwd())
                # show HF hub cache location too
                hf_hub = os.path.join(hf_user, '.cache', 'huggingface', 'hub')
                print('hf hub cache sample contents (top-level):')
                try:
                    print(os.listdir(hf_hub)[:20])
                except Exception:
                    print('cannot list hf hub cache')
                print('--- END DEBUG ---\n')
            except Exception:
                pass

            # If load_dataset failed due to dataset script issues, attempt
            # to fetch raw files directly from the Hub as a fallback.
            try:
                print('Attempting HF Hub raw-file download fallback (hf_hub_download).')
                return prepare_fever_via_hf_hub(out_dir=out_dir, split_ratio=split_ratio, label_map=label_map)
            except Exception:
                # Provide a clearer hint for the common Windows issue if fallback fails
                msg = str(e)
                if ('found ' in msg and msg.strip().endswith('.py')) or 'fever.py' in msg:
                    raise RuntimeError(
                        msg + '\n\nHint: remove any local `fever.py` (or similarly named) files from the project folder and retry,\n'
                        'or set a custom HF cache directory via the HF_DATASETS_CACHE env var.') from e
                raise
        finally:
            # remove temporary cache to avoid leaving large files behind
            try:
                shutil.rmtree(tmp_cache)
            except Exception:
                pass
    finally:
        os.chdir(orig_cwd)
        try:
            shutil.rmtree(tmp_cwd)
        except Exception:
            pass

    # Normalize to train/validation/test
    if 'train' in ds and ('validation' in ds or 'test' in ds):
        dataset = ds
    else:
        # If a single split provided, split according to split_ratio
        base_split = ds[list(ds.keys())[0]]
        test_size = split_ratio[1] + split_ratio[2]
        train_test = base_split.train_test_split(test_size=test_size)
        test_val = train_test['test'].train_test_split(test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]))
        dataset = DatasetDict({'train': train_test['train'], 'validation': test_val['train'], 'test': test_val['test']})

    def normalize(example):
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
            # assume numeric already
            label_id = int(label) if label is not None else -1

        return {'text': text, 'label': label_id}

    # If the dataset already contains a `text` field (for example when loading
    # from local JSONL which we parsed into `text`), skip the normalization
    # mapping to avoid overwriting the precomputed `text` with empty values.
    sample_split = list(dataset.keys())[0]
    if 'text' not in dataset[sample_split].column_names:
        dataset = dataset.map(normalize)
    # Save processed dataset
    dataset.save_to_disk(out_dir)
    return out_dir


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

"""
Run full finetuning experiments on the processed FEVER dataset and save results.

This script trains one or more transformer models (defaults to BERT and RoBERTa),
evaluates on validation and test splits (if labeled), and writes metrics and
predictions to `outputs/` for easy comparison with baselines.

Usage (PowerShell):
python -c "from src.models.run_full_finetune import main; main(models=['bert-base-uncased'])"

Note: Training on the full FEVER dataset is compute-intensive. Prefer a GPU.
"""
import os
import csv
import math
import numpy as np
from src.models.train import run_training
from datasets import load_from_disk


def _safe_name(model_name: str) -> str:
    return model_name.replace('/', '_')


def _softmax(logits: np.ndarray) -> np.ndarray:
    # numerically stable softmax along last dim
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main(models=None,
         processed_data_dir: str = 'data/processed/fever',
         epochs: int = 3,
         batch_size: int = 16,
         results_csv: str = 'outputs/finetune_results.csv',
         max_length: int = 128,
         save_tokenized: bool = True,
         tokenized_out_dir: str = None,
         gradient_accumulation_steps: int = 1,
         fp16: bool = False):
    if models is None:
        models = ['bert-base-uncased', 'roberta-base']

    _ensure_dir('outputs')

    # ensure header for results CSV
    header = ['model', 'output_dir', 'split', 'accuracy', 'f1_macro']
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

    # quick existence check
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(f"Processed dataset not found at {processed_data_dir}. Run preprocessing first.")

    for model_name in models:
        safe = _safe_name(model_name)
        outdir = f'outputs/finetune_{safe}'
        print(f"Training {model_name} -> {outdir} (epochs={epochs}, batch_size={batch_size})")

        trainer, tokenized = run_training(processed_data_dir=processed_data_dir,
                         model_name=model_name,
                         output_dir=outdir,
                         num_labels=3,
                         epochs=epochs,
                         batch_size=batch_size,
                         max_length=max_length,
                         save_tokenized=save_tokenized,
                         tokenized_out_dir=tokenized_out_dir,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         fp16=fp16)

        # Evaluate and save predictions for validation and test (if present)
        for split in ['validation', 'test']:
            if split not in tokenized:
                print(f"Split `{split}` not present in dataset — skipping.")
                continue

            ds = tokenized[split]
            # check if labels appear to be valid (all -1 indicates unlabeled test set)
            labels = None
            if 'labels' in ds.column_names:
                labels = np.array(ds['labels'])
            elif 'label' in ds.column_names:
                labels = np.array(ds['label'])

            has_labels = labels is not None and not (labels.shape[0] > 0 and np.all(labels < 0))

            print(f"Predicting on {split} (has_labels={has_labels})...")
            pred_out = trainer.predict(ds)
            logits = pred_out.predictions
            probs = _softmax(logits)
            preds = np.argmax(logits, axis=1)

            # compute simple metrics if labels available
            acc = ''
            f1 = ''
            if has_labels:
                import evaluate
                metric_acc = evaluate.load('accuracy')
                metric_f1 = evaluate.load('f1')
                acc = float(metric_acc.compute(predictions=preds, references=pred_out.label_ids)['accuracy'])
                f1 = float(metric_f1.compute(predictions=preds, references=pred_out.label_ids, average='macro')['f1'])

            # write predictions CSV
            pred_dir = os.path.join(outdir, 'predictions')
            _ensure_dir(pred_dir)
            pred_file = os.path.join(pred_dir, f'preds_{split}.csv')
            with open(pred_file, 'w', newline='', encoding='utf-8') as pf:
                w = csv.writer(pf)
                # header: idx, label (if present), pred, prob0, prob1, ...
                prob_cols = [f'prob_{i}' for i in range(probs.shape[1])]
                if has_labels:
                    w.writerow(['idx', 'label', 'pred'] + prob_cols)
                    for i, (lab, pr, rowp) in enumerate(zip(pred_out.label_ids, preds, probs)):
                        w.writerow([i, int(lab), int(pr)] + [float(x) for x in rowp.tolist()])
                else:
                    w.writerow(['idx', 'pred'] + prob_cols)
                    for i, (pr, rowp) in enumerate(zip(preds, probs)):
                        w.writerow([i, int(pr)] + [float(x) for x in rowp.tolist()])

            # append metrics to results CSV
            with open(results_csv, 'a', newline='', encoding='utf-8') as rf:
                writer = csv.writer(rf)
                writer.writerow([model_name, outdir, split, acc if acc != '' else '', f1 if f1 != '' else ''])

            print(f"Saved predictions -> {pred_file}")

        print(f"Finished model {model_name}. Artifacts in {outdir}")


if __name__ == '__main__':
    main()

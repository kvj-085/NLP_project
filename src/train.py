import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import inspect
from datasets import load_from_disk
import numpy as np
import evaluate
from src.utils import set_seed


def compute_metrics(p):
    metric_acc = evaluate.load('accuracy')
    preds = np.argmax(p.predictions, axis=1)
    return metric_acc.compute(predictions=preds, references=p.label_ids)


def run_training(processed_data_dir: str = 'data/processed',
                 model_name: str = 'distilbert-base-uncased',
                 output_dir: str = 'outputs/',
                 num_labels: int = 3,
                 epochs: int = 1,
                 batch_size: int = 8):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_from_disk(processed_data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_fn(examples):
        # Prefer precomputed 'text' field; otherwise try to combine claim/evidence
        if 'text' in examples:
            texts = examples['text']
        else:
            texts = []
            claims = examples.get('claim', [''] * len(next(iter(examples.values()))))
            evidences = examples.get('evidence', [None] * len(claims))
            for c, e in zip(claims, evidences):
                if e:
                    # try a simple flatten for lists/strings
                    if isinstance(e, list):
                        ev_text = ' '.join([t if isinstance(t, str) else str(t) for t in e])
                    else:
                        ev_text = str(e)
                    texts.append(f"{c} [SEP] {ev_text}")
                else:
                    texts.append(c)
        return tokenizer(texts, truncation=True, padding='max_length')

    tokenized = dataset.map(preprocess_fn, batched=True)
    # ensure label column is named 'labels' for the Trainer
    if 'label' in tokenized['train'].column_names:
        tokenized = tokenized.rename_column('label', 'labels')
    elif 'labels' not in tokenized['train'].column_names:
        raise ValueError('Dataset must contain `label` or `labels` column')
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Build TrainingArguments in a version-tolerant way
    ta_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=10,
        learning_rate=2e-5,
    )

    # Add optional args if supported by installed transformers version
    sig = inspect.signature(TrainingArguments.__init__)
    if 'evaluation_strategy' in sig.parameters:
        ta_kwargs['evaluation_strategy'] = 'epoch'
    if 'save_strategy' in sig.parameters:
        ta_kwargs['save_strategy'] = 'epoch'

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    run_training()

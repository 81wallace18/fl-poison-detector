import glob
import json
import os

import evaluate
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

PAD_ID = 0
NUM_BINS = 10000
MAX_LENGTH = 512


def preprocess_weights(state_dict, max_length=MAX_LENGTH, num_bins=NUM_BINS):
    weights = torch.cat([v.flatten() for k, v in state_dict.items() if 'weight' in k])
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    # Bins ocupam [1, num_bins]; PAD_ID=0 fica reservado e nao colide com o menor peso.
    weights_binned = (weights_norm * (num_bins - 1)).long() + 1
    if len(weights_binned) > max_length:
        weights_binned = weights_binned[:max_length]
    else:
        pad = torch.full((max_length - len(weights_binned),), PAD_ID, dtype=torch.long)
        weights_binned = torch.cat([weights_binned, pad])
    return weights_binned.tolist()


def tokenize_function(examples):
    return {
        'input_ids': examples['inputs'],
        'attention_mask': [[1 if tok != PAD_ID else 0 for tok in ids] for ids in examples['inputs']],
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'f1': f1.compute(predictions=predictions, references=labels, average='binary')['f1'],
        'precision': precision.compute(predictions=predictions, references=labels, average='binary')['precision'],
        'recall': recall.compute(predictions=predictions, references=labels, average='binary')['recall'],
    }


state_dicts_dir = 'state_dicts'
safetensor_files = sorted(glob.glob(os.path.join(state_dicts_dir, '*.safetensors')))
assert safetensor_files, (
    f"Nenhum .safetensors em '{state_dicts_dir}/'. "
    "Rode o notebook BertModelsclassify.ipynb para gerar os state_dicts antes de treinar."
)

loaded_entries = []
for safetensor_file in safetensor_files:
    sd = load_file(safetensor_file)
    json_file = safetensor_file.replace('.safetensors', '.json')
    with open(json_file, 'r') as f:
        meta = json.load(f)
    loaded_entries.append({
        'inputs': preprocess_weights(sd),
        'labels': meta['label'],
    })

dataset = Dataset.from_list(loaded_entries)

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')

model_name = 'distilbert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, ignore_mismatched_sizes=True
)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_lin', 'v_lin'])
model = get_peft_model(model, lora_config)

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
tokenized_train = split_dataset['train'].map(
    tokenize_function, batched=True, remove_columns=['inputs']
)
tokenized_eval = split_dataset['test'].map(
    tokenize_function, batched=True, remove_columns=['inputs']
)

training_args = TrainingArguments(
    output_dir='./detector',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to='none',
    metric_for_best_model='f1',
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)
trainer.train()

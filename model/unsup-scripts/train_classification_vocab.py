import torch
import pandas as pd
import argparse
import logging
import numpy as np
import os

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,DataCollatorWithPadding
from transformers import Trainer
logger = logging.getLogger(__name__)

# import datasets
from datasets import load_metric
from datasets import load_from_disk

model_cards = {
"mbert":'bert-base-multilingual-cased',
"csebert":'EMBEDDIA/crosloengual-bert',

"mbert_finetune":'../output/mbert_finetune/',
"csebert_finetune":'../output/cse_finetune/',

"mbert_finetune_small":'../output/mbert_finetune_small/',
"csebert_finetune_small":'../output/csebert_finetune_small/',
"mbert_finetune_small_gen":'../output/mbert_finetune_small_gen/',
"csebert_finetune_small_gen":'../output/csebert_finetune_small_gen/',

"mbert_finetune_vocab":'../output/mbert_finetune_vocab/',
"csebert_finetune_vocab":'../output/cse_finetune_vocab/',

"mbert_vocab":'../output/mbert_vocab/', #TODO: Save the model
"csebert_vocab":'../output/cse_finetune_vocab/',

}

datasets = {
    'large':{
        'train_file':'../../data/unsup/24h/classify/train_2019_eq.csv',
        'val_file':'../../data/unsup/24h/classify/val_2019_eq.csv',
        'test_file':'../../data/unsup/24h/classify/test_2019_eq.csv',
    },
'small':{
        'train_file':'../../data/unsup/24h/classify/cro_train.csv',
        'val_file':'../../data/unsup/24h/classify/cro_val.csv',
        'test_file':'../../data/unsup/24h/classify/cro_test.csv',
    },

}

def compute_metrics(eval_preds):

    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")

    return {"acc": acc, "f1": f1}

def fix_metrics(metrics):
    # Fix metrics in dict format for logging purpose
    for key in metrics.keys():
        if isinstance(metrics[key], dict):
            for key1 in metrics[key].keys():
                print(metrics[key][key1])
                metrics[key] = metrics[key][key1]
    return metrics

def read_data(file_name):
    #Reading CSV File

    df = pd.read_csv(file_name, lineterminator='\n')
    #df = df.head(100)
    print('Processing', file_name, df.shape)
    texts= df.content.tolist()
    labels = df.label.tolist()

    return texts, labels


class HRDataset(torch.utils.data.Dataset):
    #24Sata Dataset Processing
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Training Parameters
    parser.add_argument("-output_dir", type=str, default="../results/claasify/", help='Output Directory')
    parser.add_argument("-logging_dir", type=str, default="../logs/claasify/", help='Logging Directory')
    parser.add_argument("-num_train_epochs", type=int, default=5, help='Number of training Epochs')
    parser.add_argument("-per_device_train_batch_size", type=int, default=24, help='Traiing Batch Size')
    parser.add_argument("-per_device_eval_batch_size", type=int, default=32, help='Evaluation Batch Size')
    parser.add_argument("-warmup_steps", type=int, default=500, help='Warmup Steps')
    parser.add_argument("-weight_decay", type=int, default=0.01, help='Weight Decay Rate')
    parser.add_argument("-logging_steps", type=int, default=500, help='Logging Steps')
    parser.add_argument("-save_steps", type=int, default=500, help='Number of updates steps before two checkpoint saves')
    parser.add_argument("-save_total_limit", type=int, default=500, help='If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints')
    parser.add_argument("-save_strategy", type=str, default="steps", help='The checkpoint save strategy to adopt during training')


    #Dataset
    parser.add_argument("-dataset", type=str, default='small', help='Training validation set large/small')

    #Model
#     parser.add_argument("-model_card", type=str, default='mbert', help='The model directory checkpoint for weights initialization.')
#     parser.add_argument("-all_steps", action='store_true',
#                         help='To Train on all steps check point')

    #TODO: Currently expects tokenizer to be present in the model directory only. Better Change this in future

    # parser.add_argument("-tokenizer_name", type=str, required=True,
    #                     help='Pretrained tokenizer name or path if not the same as model_name')

    args = parser.parse_args()

    output_dir = args.output_dir
    logging_dir = args.logging_dir
    num_train_epochs = args.num_train_epochs
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    save_total_limit = args.save_total_limit
    save_strategy = args.save_strategy

    dataset = args.dataset
    train_file = datasets[dataset]['train_file']
    val_file = datasets[dataset]['val_file']
    test_file = datasets[dataset]['test_file']
    # encode_data = args.encode_data
#     model_card= args.model_card
    all_steps = args.all_steps

#     model_dir = model_cards[model_card]
    model_dirs = []
#     if all_steps:
#         list_dir = os.listdir(model_dir)
#         for item in list_dir:
#             if 'checkpoint' in item:
#                 logging_steps = 50000 #TODO: maybe change letter. For now, a large Logging step
#                 tmp_dir = os.path.join(model_dir, item + '/')
#                 model_dirs.append(tmp_dir)
    model_dirs.append(model_card)
    print(model_dirs)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #TODO: Maybe want to save the dataset, so that processing is less
    train_texts, train_labels = read_data(train_file)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = HRDataset(train_encodings, train_labels)
    val_texts, val_labels = read_data(val_file)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    val_dataset = HRDataset(val_encodings, val_labels)
    
    test_texts, test_labels = read_data(test_file)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = HRDataset(test_encodings, test_labels)
    
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)

    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids

        # eval_metric = metric.compute()
        eval_metric = compute_metrics()
        accelerator.print(f"epoch {epoch}:", eval_metric)

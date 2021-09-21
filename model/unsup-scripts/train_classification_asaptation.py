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
    parser.add_argument("-output_dir", type=str, default="./results/claasify/", help='Output Directory')
    parser.add_argument("-logging_dir", type=str, default="./logs/claasify/", help='Logging Directory')
    parser.add_argument("-num_train_epochs", type=int, default=5, help='Number of training Epochs')
    parser.add_argument("-per_device_train_batch_size", type=int, default=24, help='Traiing Batch Size')
    parser.add_argument("-per_device_eval_batch_size", type=int, default=64, help='Evaluation Batch Size')
    parser.add_argument("-warmup_steps", type=int, default=500, help='Warmup Steps')
    parser.add_argument("-weight_decay", type=int, default=0.01, help='Weight Decay Rate')
    parser.add_argument("-logging_steps", type=int, default=5000, help='Logging Steps')

    #Dataset
    parser.add_argument("-train_file", type=str, default='../../data/unsup/24h/classify/train_2019_eq.csv', help='Training File')
    parser.add_argument("-val_file", type=str, default='../../data/unsup/24h/classify/val_2019_eq.csv', help='Validation File')
    parser.add_argument("-encode_data", type=bool, default=False, help='Encode data')

    #Model
    parser.add_argument("-model_dir", type=str, default='../output/finetuned-model/', help='The model directory checkpoint for weights initialization.')

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

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.logging_dir,  # directory for storing logs
        logging_steps=args.logging_steps,
        evaluation_strategy = "epoch",
    )
    train_file = args.train_file
    val_file = args.val_file
    encode_data = args.encode_data
    model_dir= args.model_dir


    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

    print('Vocab Len:', len(tokenizer))
    # Vocab Extension
    if 'bert-base-multilingual-uncased' in model_dir or 'mbert' in model_dir:
        new_token_file = 'mbert.txt'
    else:
        new_token_file = 'cse.txt'
    new_tokens = []
    with open(new_token_file) as fid:
        lines = fid.readlines()
        for line in lines:
            new_tokens.append(line.strip())

    tokenizer.add_tokens(new_tokens)
    print('New Vocab Len:', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))

    #TODO: Maybe want to save the dataset, so that processing is less
    train_texts, train_labels = read_data(train_file)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = HRDataset(train_encodings, train_labels)
    val_texts, val_labels = read_data(val_file)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    val_dataset = HRDataset(val_encodings, val_labels)



    #TODO: Trainer not working on Server due to some issue
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer = tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics = fix_metrics(metrics)
    metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



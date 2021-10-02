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

    #Dataset
    parser.add_argument("-test_file", type=str, default='../../data/unsup/24h/classify/test_2019_eq.csv', help='Test File')
    parser.add_argument("-encode_data", type=bool, default=False, help='Encode data')

    #Model
    parser.add_argument("-model_dir", type=str, default='./results/claasify/mbert/', help='The model directory checkpoint for weights initialization.')
    parser.add_argument("-test_all_steps", type=bool, default=True,
                        help='To test all step check points')

    parser.add_argument("-out_file", type=str, default='test_all_metrics.csv',
                        help='output Metric File without full path')

    #TODO: Currently expects tokenizer to be present in the model directory only. Better Change this in future
    # parser.add_argument("-tokenizer_name", type=str, required=True,
    #                     help='Pretrained tokenizer name or path if not the same as model_name')

    args = parser.parse_args()


    test_file = args.test_file
    model_dir= args.model_dir
    test_all_steps = args.test_all_steps
    out_file = model_dir + args.out_file

    model_dirs = []
    if test_all_steps:
        list_dir = os.listdir(model_dir)
        for item in list_dir:
            if 'checkpoint' in item:
                tmp_dir = os.path.join(model_dir,item+'/')
                model_dirs.append(tmp_dir)
    #Only add last step when not testing on all_steps of LM model
    if 'all_steps' not in model_dir:
        model_dirs.append(model_dir)
    print(model_dirs)

    #Only done once because it assumes main directory will have same pre-processing part
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO: Maybe want to save the dataset, so that processing is less
    test_texts, test_labels = read_data(test_file)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = HRDataset(test_encodings, test_labels)


    all_results = []
    for tmp_model_dir in model_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(tmp_model_dir)
        trainer = Trainer(
            model=model,  # the instantiated Transformers model to be trained
            args=None,  # training arguments, defined above
            train_dataset=None,  # training dataset
            eval_dataset=test_dataset,  # evaluation dataset
            tokenizer = tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        # Evaluation
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics = fix_metrics(metrics)

        if tmp_model_dir == model_dir:
            step = 'last'
        else:
            step = os.path.basename(tmp_model_dir[:-1])
        result = []
        result.append(step)
        print(step,tmp_model_dir)
        for key in metrics.keys():
            val = metrics[key]
            result.append(val)
        all_results.append(result)
    columns_keys = ['step']
    columns_keys.extend(list(metrics.keys()))
    df = pd.DataFrame(all_results, columns=columns_keys)

    df.to_csv(out_file, index=False)

    print('output saved to ', out_file)



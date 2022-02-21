import torch
import pandas as pd
import argparse
import logging
import numpy as np
import os
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler
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
    parser.add_argument("-test_file", type=str, default='../../data/unsup/24h/classify/cro_test.csv', help='Test File')
    parser.add_argument("-encode_data", type=bool, default=False, help='Encode data')

    #Model
    parser.add_argument("-model_dir", type=str, default='./results/claasify/mbert/', help='The model directory checkpoint for weights initialization.')
   
    parser.add_argument("-out_file", type=str, default='all_probs.csv',
                        help='output Metric File without full path')

    #TODO: Currently expects tokenizer to be present in the model directory only. Better Change this in future
    # parser.add_argument("-tokenizer_name", type=str, required=True,
    #                     help='Pretrained tokenizer name or path if not the same as model_name')

    args = parser.parse_args()


    test_file = args.test_file
    model_dir= args.model_dir
    out_file = model_dir + args.out_file

    #Only done once because it assumes main directory will have same pre-processing part
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO: Maybe want to save the dataset, so that processing is less
    test_texts, test_labels = read_data(test_file)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = HRDataset(test_encodings, test_labels)

    sampler = SequentialSampler(test_dataset)

    batch_size = 8
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size, sampler=sampler)

    data_iterator = tqdm(test_dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)

    
    true_labels = []
    all_mean0 = []
    all_std0  = []
    all_mean1 = []
    all_std1 = []


    for step, batch in enumerate(data_iterator):

        for i in range(0,20): 

            #Load model multiple time, maybe this is not required
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.train()

            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss

            logits = outputs.logits 
            probs = softmax(logits)

            if i == 0:
                all_probs0 = probs[:,0].unsqueeze(0) * 100
                all_probs1 = probs[:,1].unsqueeze(0) * 100
            else:
                probs0 = probs[:,0].unsqueeze(0) * 100
                probs1 = probs[:,1].unsqueeze(0) * 100
                all_probs0 = torch.cat([all_probs0,probs0 ],0)
                all_probs1 = torch.cat([all_probs1, probs1],0)
            

        mean0 = torch.mean(all_probs0,0).to('cpu').numpy()
        std0 = torch.std(all_probs0,0).to('cpu').numpy()

        mean1 = torch.mean(all_probs1,0).to('cpu').numpy()
        std1 = torch.std(all_probs1,0).to('cpu').numpy()

        labels = batch['labels'].to('cpu').numpy()
        for label,m0,s0,m1,s1  in zip(labels,mean0, std0, mean1, std1):
            true_labels.append(label)
            all_mean0.append(m0)
            all_mean1.append(m1)
            all_std0.append(s0)
            all_std1.append(s1)


     # Save All Loss for the Best Model
    result = pd.DataFrame([true_labels, all_mean0, all_std0, all_mean1, all_std0 ])
    result = result.transpose()
    result.columns = ['labels', 'prob_mean0', 'prob_std0', 'prob_mean1', 'prob_std1']
    result.head()
    
    result.to_csv(out_file, index=False)
    print('Output saved to ', out_file)



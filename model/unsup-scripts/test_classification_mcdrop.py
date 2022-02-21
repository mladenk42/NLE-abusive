import torch
import pandas as pd
import argparse
import logging
import numpy as np
import os
from tqdm import tqdm, trange

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
    parser.add_argument("-test_file", type=str, default='../../data/unsup/24h/classify/cro_test.csv', help='Test File')
    parser.add_argument("-encode_data", type=bool, default=False, help='Encode data')

    #Model
    parser.add_argument("-model_dir", type=str, default='./results/claasify/mbert/', help='The model directory checkpoint for weights initialization.')
   
    parser.add_argument("-out_file", type=str, default='test_all_metrics.csv',
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

    batch_size = 8
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)


    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.train()

    data_iterator = tqdm(test_dataset, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)

    
    true_labels = []
    all_mean0 = []
    all_std0  = []
    all_mean1 = []
    all_std1 = []


    for step, batch in enumerate(test_dataloader):

        for i in range(0,10):        
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss

            logits = outputs.logits 
            probs = softmax(logits)

            if i == 0:
                all_probs0 = probs[:,0]
                all_probs1 = probs[:,1]
            else:
                probs0 = probs[:,0].unsqueeze(0)
                probs1 = probs[:,1].unsqueeze(0)
                all_probs0 = torch.stack([all_probs0,probs0 ])
                all_probs1 = torch.stack([all_probs1, probs1])
            

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



    
    df = pd.DataFrame(all_results, columns=columns_keys)

    df.to_csv(out_file, index=False)

    print('output saved to ', out_file)


def bert_evaluate(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    model.to(device)
    model.eval()
    predictions = []
    prob_0 = []
    prob_1 = []
    true_labels = []
    data_iterator = tqdm(eval_dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)
    for step, batch in enumerate(data_iterator):
        input_ids, token_type_ids, input_masks, labels = batch
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_masks)

        # loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        probs = softmax(logits)
        # print(type(logits))
        logits = logits.to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()

        for label,l, prob in zip(label_ids,logits, probs):
            true_labels.append(label)
            predictions.append(np.argmax(l))
            prob_0.append(prob[0].to('cpu').numpy())
            prob_1.append(prob[1].to('cpu').numpy())
    metrics = get_metrics(true_labels, predictions)
    return metrics, predictions, prob_0, prob_1





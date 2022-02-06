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
    

    # encoded_dataset.save_to_disk("path/of/my/dataset/directory")
    # reloaded_encoded_dataset = load_from_disk("path/of/my/dataset/directory")

    for model_dir in model_dirs:
        if 'checkpoint' in model_dir: #This is when we are looking at all step in LMs finetuning
            checkpoint = model_dir.split('/')[-2] +'/'
            save_strategy = "epoch"
            save_total_limit = 1 #TODO: Change this to a acceptatble number
        else:
            checkpoint = 'last/'

        tmp_output_dir = output_dir #TODO: Do we need to have dataset part here
#         if all_steps:
#             tmp_output_dir = output_dir + model_card +'_'+dataset+ '/all_steps/' + checkpoint
#             tmp_logging_dir = logging_dir + model_card +'_'+dataset+ '/all_steps/' + checkpoint
#         else:
#             tmp_output_dir = output_dir + model_card+'_'+dataset +'/'+checkpoint
#             tmp_logging_dir = logging_dir + model_card+'_'+dataset +'/'+ checkpoint
        print(tmp_output_dir, tmp_logging_dir)
        training_args = TrainingArguments(
            output_dir=tmp_output_dir,  # output directory
            logging_dir=tmp_logging_dir,  # directory for storing logs
            num_train_epochs=num_train_epochs,  # total number of training epochs
            per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
            warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay=weight_decay,  # strength of weight decay
            logging_steps=logging_steps,
            evaluation_strategy="epoch",
            save_total_limit = save_total_limit,
            save_strategy=save_strategy,
            save_steps=save_steps,
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

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


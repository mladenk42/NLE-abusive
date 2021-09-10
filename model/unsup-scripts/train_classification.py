import torch
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,AdamW
from transformers import Trainer

def read_data(file_name):
    #Reading CSV File
    print('Processing', file_name)
    df = pd.read_csv(file_name, lineterminator='\n')

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
    parser.add_argument("-per_device_train_batch_size", type=int, default=16, help='Traiing Batch Size')
    parser.add_argument("-per_device_eval_batch_size", type=int, default=64, help='Evaluation Batch Size')
    parser.add_argument("-warmup_steps", type=int, default=500, help='Warmup Steps')
    parser.add_argument("-weight_decay", type=int, default=0.01, help='Weight Decay Rate')
    parser.add_argument("-logging_steps", type=int, default=50, help='Logging Steps')

    #Dataset
    parser.add_argument("-train_file", type=str, required=True, help='Training File')
    parser.add_argument("-val_file", type=str, required=True, help='Validation File')

    #Model
    parser.add_argument("-model_dir", type=str, required=True, help='The model directory checkpoint for weights initialization.')

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
    )
    train_file = args.train_file
    val_file = args.val_file
    model_dir= args.model_dir

    train_texts, train_labels = read_data(train_file)
    val_texts, val_labels = read_data(train_file)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = HRDataset(train_encodings, train_labels)
    val_dataset = HRDataset(val_encodings, val_labels)


    config = AutoConfig.from_pretrained(model_dir, num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)

    #TODO: Trainer not working on Server due to some issue
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    #Just in case Trainer not working
    #TODO: Need to fix saving models, logs etc

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(device)
    # model.train()
    #
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #
    # optim = AdamW(model.parameters(), lr=5e-5,warmup_steps=warmup_steps,)
    #
    # for epoch in range(3):
    #     for batch in train_loader:
    #         optim.zero_grad()
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs[0]
    #         loss.backward()
    #         optim.step()

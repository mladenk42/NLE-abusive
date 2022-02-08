import torch
import pandas as pd
import argparse
import logging
import numpy as np
import math
import os
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    # AutoModelForTokenClassification,
    AutoTokenizer,
    # DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)

logger = logging.getLogger(__name__)

# import datasets
from datasets import load_metric
from datasets import load_from_disk

model_cards = {
    "mbert": 'bert-base-multilingual-cased',
    "csebert": 'EMBEDDIA/crosloengual-bert',

    "mbert_finetune": '../output/mbert_finetune/',
    "csebert_finetune": '../output/cse_finetune/',

    "mbert_finetune_small": '../output/mbert_finetune_small/',
    "csebert_finetune_small": '../output/csebert_finetune_small/',
    "mbert_finetune_small_gen": '../output/mbert_finetune_small_gen/',
    "csebert_finetune_small_gen": '../output/csebert_finetune_small_gen/',

    "mbert_finetune_vocab": '../output/mbert_finetune_vocab/',
    "csebert_finetune_vocab": '../output/cse_finetune_vocab/',

    "mbert_vocab": '../output/mbert_vocab/',  # TODO: Save the model
    "csebert_vocab": '../output/cse_finetune_vocab/',

}

datasets = {
    'large': {
        'train_file': '../../data/unsup/24h/classify/train_2019_eq.csv',
        'val_file': '../../data/unsup/24h/classify/val_2019_eq.csv',
        'test_file': '../../data/unsup/24h/classify/test_2019_eq.csv',
    },
    'small': {
        'train_file': '../../data/unsup/24h/classify/cro_train.csv',
        'val_file': '../../data/unsup/24h/classify/cro_val.csv',
        'test_file': '../../data/unsup/24h/classify/cro_test.csv',
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
    # Reading CSV File

    df = pd.read_csv(file_name, lineterminator='\n')
    # df = df.head(100)
    print('Processing', file_name, df.shape)
    texts = df.content.tolist()
    labels = df.label.tolist()

    return texts, labels


class HRDataset(torch.utils.data.Dataset):
    # 24Sata Dataset Processing
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
    parser.add_argument("--output_dir", type=str, default="../results/classify/xxx", help='Output Directory')
    parser.add_argument("--logging_dir", type=str, default="../logs/classify/xxx", help='Logging Directory')
    parser.add_argument("--num_train_epochs", type=int, default=10, help='Number of training Epochs')
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help='Training Batch Size')
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help='Evaluation Batch Size')
    parser.add_argument("--warmup_steps", type=int, default=500, help='Warmup Steps')
    parser.add_argument("--weight_decay", type=int, default=0.01, help='Weight Decay Rate')
    parser.add_argument("--logging_steps", type=int, default=500, help='Logging Steps')
    parser.add_argument("--save_steps", type=int, default=500,
                        help='Number of updates steps before two checkpoint saves')
    parser.add_argument("--save_total_limit", type=int, default=500,
                        help='If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints')
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help='The checkpoint save strategy to adopt during training')

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )


    # Dataset
    parser.add_argument("-dataset", type=str, default='small', help='Training validation set large/small')

    # Model
    parser.add_argument("-model_card", type=str, default='bert-base-multilingual-cased', help='The model directory checkpoint for weights initialization.')
    #     parser.add_argument("-all_steps", action='store_true',
    #                         help='To Train on all steps check point')

    # TODO: Currently expects tokenizer to be present in the model directory only. Better Change this in future

    # parser.add_argument("-tokenizer_name", type=str, required=True,
    #                     help='Pretrained tokenizer name or path if not the same as model_name')

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

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
    max_train_steps = args.max_train_steps
    gradient_accumulation_steps =args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    lr_scheduler_type = args.lr_scheduler_type
    num_warmup_steps = args.num_warmup_steps
    max_length = args.max_length



    dataset = args.dataset
    train_file = datasets[dataset]['train_file']
    val_file = datasets[dataset]['val_file']
    test_file = datasets[dataset]['test_file']
    # encode_data = args.encode_data
    model_card= args.model_card
    # all_steps = args.all_steps

    #     model_dir = model_cards[model_card]
    # model_dirs = []
    #     if all_steps:
    #         list_dir = os.listdir(model_dir)
    #         for item in list_dir:
    #             if 'checkpoint' in item:
    #                 logging_steps = 50000 #TODO: maybe change letter. For now, a large Logging step
    #                 tmp_dir = os.path.join(model_dir, item + '/')
    #                 model_dirs.append(tmp_dir)
    # model_dirs.append(model_card)

    model = AutoModelForSequenceClassification.from_pretrained(model_card, num_labels=2)

    # print(model_dirs)

    tokenizer = AutoTokenizer.from_pretrained(model_card)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO: Maybe want to save the dataset, so that processing is less
    train_texts, train_labels = read_data(train_file)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    train_dataset = HRDataset(train_encodings, train_labels)
    val_texts, val_labels = read_data(val_file)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    val_dataset = HRDataset(val_encodings, val_labels)

    test_texts, test_labels = read_data(test_file)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    test_dataset = HRDataset(test_encodings, test_labels)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )
    val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    all_train_loss = []
    all_val_loss = []
    prev_val_loss = 9999999999999999 # A very large model
    for epoch in range(num_train_epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss += loss.item()
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        model.eval()
        with torch.no_grad():
            for split, eval_dataloader in zip(['val', 'test'], [val_dataloader, test_dataloader]):

                metric_acc = load_metric("accuracy")
                metric_f1 = load_metric("f1")

                all_predictions = []
                all_references = []

                for step, batch in enumerate(eval_dataloader):
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)

                    all_predictions.extend(predictions.tolist())
                    all_references.extend(batch["labels"].tolist())

                    if split == 'val':
                        if step == 0:
                            val_loss = 0
                        loss = outputs.loss
                        loss = loss / gradient_accumulation_steps
                        val_loss +=loss.item()

                #Save All results for Future
                result = pd.DataFrame([all_references,all_predictions])
                result = result.transpose()
                result.columns = ['ref', 'pred']
                result.head()
                file_name = output_dir + '/'+split +'_'+str(epoch)+'.csv'
                result.to_csv(file_name, index=False)
                print('output saved to ', file_name)

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        # Save Model when validation loss decreased
        if output_dir is not None and prev_val_loss > val_loss:
            print('Saving Model to ', output_dir, epoch)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            prev_val_loss = val_loss

    # Save All Loss for the Best Model
    result = pd.DataFrame([all_train_loss, all_val_loss])
    result = result.transpose()
    result.columns = ['train', 'val']
    result.head()
    loss_file_name = output_dir + '/loss.csv'
    result.to_csv(loss_file_name, index=False)
    print('Loss saved to ', loss_file_name)

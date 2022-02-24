import argparse
import random
import math
import sys

import subprocess

data = {"small": {"gen": "/scratch-local/ravi/datasets/cro_corpus/use/small/LM_general_small.txt",
                  "24h":"/scratch-local/ravi/datasets/cro_corpus/use/small/LM_24h_small.txt"
                  },
        "large": {

        },
        }

model_cards ={"mbert":"bert-base-multilingual-cased",
       "csebert":"EMBEDDIA/crosloengual-bert",
       }
per_device_train_batch_sizes= {"mbert":'24', "csebert":'40'}
# vocab_init_types=["no","random","avg","sum","max"]
vocab_init_types=["no"]


import wandb
wandb.init(project="train-LLM", entity="hahackathon")

if __name__ == "__main__":

    berts = ["mbert","csebert"]
    dataset_sizes =["small","large"]
    dataset_types = ["gen","24h"]
    dataset_sizes = ["small"]

    python_path = "/homes/ravi/anaconda3/bin/python"
    cache_dir = "./cache"
    gradient_accumulation_steps = '64'
    per_device_eval_batch_size = '18'
    max_seq_length = '256'
    num_train_epochs = '5'
    validation_split_percentage = '10'

    for bert in berts:
        model_card = model_cards[bert]
        per_device_train_batch_size = per_device_train_batch_sizes[bert]
        for dataset_size in dataset_sizes:
            for dataset_type in dataset_types:

                train_file = data[dataset_size][dataset_type]


                for vocab_init_type in vocab_init_types:
                    output_dir = '/import/cogsci/ravi/codes/NLE-abusive/model/output/' + bert + '_finetune_' + dataset_size + '_' + dataset_type +'_vocab'+'_'+vocab_init_type

                    list_arg = [
                                python_path, "run_mlm_domain_adaptation.py",
                                "--model_name_or_path", model_card,
                                "--cache_dir", cache_dir,
                                "--train_file", train_file,
                                "--output_dir", output_dir,
                                "--do_train",
                                "--gradient_accumulation_steps",gradient_accumulation_steps,
                                "--per_device_train_batch_size", per_device_train_batch_size,
                                "--per_device_eval_batch_size", per_device_eval_batch_size,
                                "--fp16",
                                "--overwrite_output_dir",
                                "--line_by_line",
                                "--max_seq_length",max_seq_length,
                                "--num_train_epochs",num_train_epochs,
                                "--validation_split_percentage",validation_split_percentage,
                                "--vocab_init_type",vocab_init_type,
                                "--report_to","wandb",
                                "--evaluation_strategy","steps"
                                ]

                    print(list_arg)
                    subprocess.call(list_arg)

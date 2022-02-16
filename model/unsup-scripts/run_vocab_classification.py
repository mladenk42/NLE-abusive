import subprocess
from pathlib import Path
import argparse

per_device_train_batch_sizes = {"mbert": '24', "csebert": '40'}
vocab_init_types = ["random", "avg", "sum", "max"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--berts", nargs="+", default=["mbert", "csebert"])
    parser.add_argument("-all_steps", action='store_true',
                        help='To Train on all steps check point')
    args = parser.parse_args()
    berts = args.berts
    all_steps = agrs.all_steps

    # berts = ["mbert", "csebert"]
    # berts = ["csebert"]
    # dataset_sizes =["small","large"]

    # These two are with respect to LM Model training
    dataset_types_LM = ["gen", "24h"]
    dataset_sizes_LM = ["small"]

    datasets = ['small','small25','small50','small75']

    python_path = "/homes/ravi/anaconda3/bin/python"
    cache_dir = "./cache"

    for bert in berts:
        # model_card = model_cards[bert]
        per_device_train_batch_size = per_device_train_batch_sizes[bert]
        for dataset_size in dataset_sizes_LM:
            for dataset_type in dataset_types_LM:
                for dataset in datasets:
                    for vocab_init_type in vocab_init_types:

                        if all_steps:
                            checkpoints = ['/checkpoint-500','/checkpoint-1000','/checkpoint-1500']
                        
                        for checkpoint in checkpoints:
                            model_dir_str = bert + '_finetune_' + dataset_size + '_' + dataset_type + '_vocab' + '_' + vocab_init_type+checkpoint
                            model_card = '../output/' + model_dir_str

                            output_dir = '../results/classify/' + model_dir_str +'_'+dataset
                            logging_dir = '../logs/classify/' + model_dir_str+'_'+dataset

                            # Recursively create Directory, even if it exits
                            Path(output_dir).mkdir(parents=True, exist_ok=True)
                            Path(logging_dir).mkdir(parents=True, exist_ok=True)
                            list_arg = [
                                python_path, "train_classification_vocab.py",
                                "--output_dir", output_dir,
                                "--logging_dir", logging_dir,
                                "--model_card", model_card,
                                "--dataset", dataset

                            ]

                            print(list_arg)
                            subprocess.call(list_arg)
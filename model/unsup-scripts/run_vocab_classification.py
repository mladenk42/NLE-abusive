import subprocess
from pathlib import Path

per_device_train_batch_sizes = {"mbert": '24', "csebert": '40'}
vocab_init_types = ["random", "avg", "sum", "max"]

if __name__ == "__main__":

    # berts = ["mbert", "csebert"]
    berts = ["csebert"]
    # dataset_sizes =["small","large"]

    # These two are with respect to LM Model training
    dataset_types_LM = ["gen", "24h"]
    dataset_sizes_LM = ["small"]

    python_path = "/homes/ravi/anaconda3/bin/python"
    cache_dir = "./cache"

    for bert in berts:
        # model_card = model_cards[bert]
        per_device_train_batch_size = per_device_train_batch_sizes[bert]
        for dataset_size in dataset_sizes_LM:
            for dataset_type in dataset_types_LM:

                for vocab_init_type in vocab_init_types:
                    model_card = '../output/' + bert + '_finetune_' + dataset_size + '_' + dataset_type + '_vocab' + '_' + vocab_init_type

                    output_dir = '../results/classify/' + bert + '_finetune_' + dataset_size + '_' + dataset_type + '_vocab' + '_' + vocab_init_type
                    logging_dir = '../logs/classify/' + bert + '_finetune_' + dataset_size + '_' + dataset_type + '_vocab' + '_' + vocab_init_type

                    # Recursively create Directory, even if it exits
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    Path(logging_dir).mkdir(parents=True, exist_ok=True)
                    list_arg = [
                        python_path, "train_classification_vocab.py",
                        "--output_dir", output_dir,
                        "--logging_dir", logging_dir,
                        "--model_card", model_card

                    ]

                    print(list_arg)
                    subprocess.call(list_arg)
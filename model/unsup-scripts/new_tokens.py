from pathlib import Path

from transformers_domain_adaptation import DataSelector

from transformers import AutoModelForMaskedLM, AutoTokenizer

from transformers_domain_adaptation import VocabAugmentor

# import itertools as it
from pathlib import Path
# from typing import Sequence, Union, Generator
#
# from datasets import load_dataset
# from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

dpt_corpus_train = '../data/unsup/24h/24sata_LM.txt'

# model_cards = ['EMBEDDIA/crosloengual-bert', 'bert-base-multilingual-cased']
# new_tokens_files = ['cse.txt', 'mbert.txt']


model_cards = ['bert-base-multilingual-cased']
new_tokens_files = [ 'mbert.txt']

for model_card, new_tokens_file in zip(model_cards,new_tokens_files):

    # dpt_corpus_train_data_selected = dpt_corpus_train[:-2]+'_'+model_card+'_selected.txt'

    model = AutoModelForMaskedLM.from_pretrained(model_card)
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    # selector = DataSelector(
    #     keep=0.5,  # TODO Replace with `keep`
    #     tokenizer=tokenizer,
    #     similarity_metrics=['euclidean'],
    #     diversity_metrics=[
    #         "type_token_ratio",
    #         "entropy",
    #     ],
    # )

    training_texts = Path(dpt_corpus_train).read_text().splitlines()

    # Select relevant documents from in-domain training corpus
    # selected_corpus = selector.transform(training_texts)
    # Save selected corpus to disk under `dpt_corpus_train_data_selected`
    # Path(dpt_corpus_train_data_selected).write_text('\n'.join(selected_corpus))

    print('Old', len(tokenizer))
    target_vocab_size = len(tokenizer) + 1000
    augmentor = VocabAugmentor(
        tokenizer=tokenizer,
        cased=False,
        target_vocab_size=target_vocab_size
    )
    print('New', len(tokenizer))
    # Obtain new domain-specific terminology based on the fine-tuning corpus
    new_tokens = augmentor.get_new_tokens(training_texts)

    # new_tokens_file = model_card +'.txt'
    with open(new_tokens_file, 'w') as f:
        for item in new_tokens:
            f.write("%s\n" % item)

    print(new_tokens_file, len(new_tokens), 'done.')
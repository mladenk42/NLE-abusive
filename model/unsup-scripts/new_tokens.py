from pathlib import Path

from transformers_domain_adaptation import DataSelector

from transformers import AutoModelForMaskedLM, AutoTokenizer

from transformers_domain_adaptation import VocabAugmentor

import os
# import itertools as it
from pathlib import Path
# from typing import Sequence, Union, Generator
#
# from datasets import load_dataset
# from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# dpt_corpus_train = '../../data/unsup/24h/2007.txt'

# model_cards = ['EMBEDDIA/crosloengual-bert', 'bert-base-multilingual-cased']
# new_tokens_files = ['cse.txt', 'mbert.txt']


# model_cards = ['bert-base-multilingual-cased']
# new_tokens_files = [ 'mbert_.txt']
#
# for model_card, new_tokens_file in zip(model_cards,new_tokens_files):
#
#     # dpt_corpus_train_data_selected = dpt_corpus_train[:-2]+'_'+model_card+'_selected.txt'
#
#     model = AutoModelForMaskedLM.from_pretrained(model_card)
#     tokenizer = AutoTokenizer.from_pretrained(model_card)
#
#     # selector = DataSelector(
#     #     keep=0.5,  # TODO Replace with `keep`
#     #     tokenizer=tokenizer,
#     #     similarity_metrics=['euclidean'],
#     #     diversity_metrics=[
#     #         "type_token_ratio",
#     #         "entropy",
#     #     ],
#     # )
#
#     # training_texts = Path(dpt_corpus_train).read_text().splitlines()
#
#     # Select relevant documents from in-domain training corpus
#     # selected_corpus = selector.transform(training_texts)
#     # Save selected corpus to disk under `dpt_corpus_train_data_selected`
#     # Path(dpt_corpus_train_data_selected).write_text('\n'.join(selected_corpus))
#
#     print('Old', len(tokenizer))
#     target_vocab_size = len(tokenizer) + 1000
#     augmentor = VocabAugmentor(
#         tokenizer=tokenizer,
#         cased=False,
#         target_vocab_size=target_vocab_size
#     )
#
#     # Obtain new domain-specific terminology based on the fine-tuning corpus
#     new_tokens = augmentor.get_new_tokens(dpt_corpus_train)
#     print('New', len(tokenizer), len(new_tokens))
#     # new_tokens_file = model_card +'.txt'
#     with open(new_tokens_file, 'w') as f:
#         for item in new_tokens:
#             f.write("%s\n" % item)
#
#     print(new_tokens_file, len(new_tokens), 'done.')

def extend_token(tokenizer,input_file_name,vocab_size,vocab_ext_by=500,use_existing=True):

    dir_name = os.path.dirname(input_file_name)+'/'
    file_name = os.path.basename(input_file_name)
    new_vocab_size = vocab_size + vocab_ext_by

    # TODO: Is it better to include the BERT name instead of size
    vocab_str = file_name[:-4] + '_' + str(vocab_size) + '_' + str(vocab_ext_by)

    new_token_file = dir_name + vocab_str + '-vocab.txt'

    augmentor = VocabAugmentor(
        tokenizer=tokenizer,
        cased=False,
        target_vocab_size=new_vocab_size
    )

    if not Path(new_token_file).is_file() or not use_existing:
        # Obtain new domain-specific terminology based on the fine-tuning corpus
        new_tokens = augmentor.get_new_tokens(input_file_name)
        print('New', len(tokenizer), len(new_tokens))

        with open(new_token_file, 'w') as f:
            for item in new_tokens:
                f.write("%s\n" % item)

    print("Load Vocab file", new_token_file)
    new_tokens = []
    with open(new_token_file) as fid:
        lines = fid.readlines()
        for line in lines:
            # toks = tokenizer.encode(line, add_special_tokens=False)
            # if len(toks) > 1:
            new_tokens.append(line.strip())

    print(len(new_tokens))

    return new_tokens


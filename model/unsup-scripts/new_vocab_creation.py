from pathlib import Path
import os

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def extend_vocab(tokenizer,input_file_name,vocab_size,vocab_ext_by=500,use_existing=True):
    paths = [input_file_name]
    dir_name = os.path.dirname(input_file_name)+'/'
    file_name = os.path.basename(input_file_name)
    new_vocab_size = vocab_size + vocab_ext_by

    #TODO: Is it better to include the BERT name instead of size
    vocab_str = file_name[:-4]+'_'+str(vocab_size)+'_'+str(vocab_ext_by)

    new_token_file = dir_name+vocab_str+'-vocab.txt'

    if not Path(new_token_file).is_file() or not use_existing:

        print("Creating Vocab ")
        # Initialize a tokenizer
        new_tokenizer = BertWordPieceTokenizer(lowercase=False)

        # Customize training
        new_tokenizer.train(files=paths, vocab_size=new_vocab_size, min_frequency=2)

        # Save files to disk
        new_tokenizer.save_model(dir_name, vocab_str)

    print("Load Vocab file", new_token_file)
    new_tokens=[]
    with open(new_token_file) as fid:
        lines = fid.readlines()
        for line in lines:
            toks = tokenizer.encode(line, add_special_tokens=False)
            if len(toks)>1:
                new_tokens.append(line.strip())

    print(len(new_tokens))

    return new_tokens


# # dir_name = '../../data/unsup/24h/'
# file_name = '../../data/unsup/24h/2007.txt'
# vocab_size = 1000
# tokenizer = BertTokenizer.from_pretrained('EMBEDDIA/crosloengual-bert')
#
# new_tokens = extend_vocab(tokenizer,file_name,vocab_size,vocab_ext_by=5000,use_existing=False)
# # print(new_tokens)

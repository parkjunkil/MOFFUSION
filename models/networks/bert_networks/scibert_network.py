from transformers import AutoModel, AutoTokenizer
#from transformers import *
import os
def create_scibert():

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    path = "./data/models/scibert_scivocab_uncased"
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    model = AutoModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
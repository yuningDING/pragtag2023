import json
import pandas as pd
from utils import get_data
import sys
import os
from sentence_transformers import SentenceTransformer
from train_sbert import train_sbert
from write_eval_stats_compare_within import evaluate

train_file = '/data/our_split/train.json'
val_file = '/data/our_split/validation.json'

for prompt in ['case', 'iscb', 'rpkg', 'diso', 'scip']:

    # Grab respective data from train, val
    train_data = get_data(train_file, prompt)
    val_data = get_data(val_file, prompt)

    train_sbert(run_path='results/' + prompt, df_train=train_data, df_test=val_data, df_val=val_data, answer_column="sentence", target_column="label", id_column="sent_id", base_model="all-MiniLM-L6-v2", num_pairs_per_example=None, save_model=True, num_epochs=5, batch_size=8, do_warmup=True)
    evaluate(model_path=os.path.join('results', prompt), df_train=train_data, df_test=val_data)
import json
import sys
import os
from utils import get_data
import pandas as pd
import random
from train_sbert import train_sbert, eval_sbert_within_domain
import copy
from write_eval_stats_compare_within import evaluate, evaluate_from_df

num_folds = 4

data_path = 'data/train_inputs_low.json'
target_folder = 'results/low_data_similarity_folds/'

df_data = get_data(data_path, 'ALL')
# print(df_data)

def get_pids(review_ids, df_data):

    target_df = pd.DataFrame()

    for review_id in review_ids:

        target_df = pd.concat([target_df, df_data[df_data['pid'] == review_id]])
    
    return target_df

print(len(df_data['pid'].unique()))
print(df_data['domain'].value_counts())

domain_map = {}

for review, df in df_data.groupby('pid'):

    review_list = domain_map.get(df['domain'][0], [])
    review_list.append(review)
    domain_map[df['domain'][0]] = review_list


#for each domain, take one as testing
for i in range(num_folds):

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for domain in domain_map.keys():

        domain_reviews = copy.deepcopy(domain_map[domain])
        # print(domain, len(domain_reviews))
        test_review = [domain_reviews[i]]
        domain_reviews.remove(test_review[0])

        df_train = pd.concat([df_train, get_pids(domain_reviews, df_data)])
        df_test = pd.concat([df_test, get_pids(test_review, df_data)])

    print(len(df_data))
    print(len(df_train))
    print(len(df_test))
    prompt = 'fold_' + str(i)

    df_train.to_csv(os.path.join(target_folder, prompt, 'train.csv'))
    df_test.to_csv(os.path.join(target_folder, prompt, 'test.csv'))

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    train_sbert(run_path=os.path.join(target_folder, prompt), df_train=df_train, df_test=df_test, df_val=df_test, answer_column="sentence", target_column="label", id_column="sent_id", base_model="all-MiniLM-L12-v2", num_pairs_per_example=None, save_model=True, num_epochs=10, batch_size=8, do_warmup=True, respect_domains=True)

    # Evaluate this split
    evaluate(model_path=os.path.join(target_folder, prompt), df_train=df_train, df_test=df_test)


df_preds = pd.DataFrame()

# Evaluate union of all splits
for i in range(num_folds):

    # Concat all folds together
    df_preds_fold = pd.read_csv('results/low_data_similarity_folds/fold_' + str(i) + '/predictions_sim.csv')

    df_preds = pd.concat([df_preds, df_preds_fold])

evaluate_from_df(target_folder=os.path.join(target_folder), df_pred=df_preds, suffix='overall')
evaluate_from_df(target_folder=os.path.join(target_folder), df_pred=df_preds[df_preds['domain'] == 'case'], suffix='case')
evaluate_from_df(target_folder=os.path.join(target_folder), df_pred=df_preds[df_preds['domain'] == 'diso'], suffix='diso')
evaluate_from_df(target_folder=os.path.join(target_folder), df_pred=df_preds[df_preds['domain'] == 'iscb'], suffix='iscb')
evaluate_from_df(target_folder=os.path.join(target_folder), df_pred=df_preds[df_preds['domain'] == 'rpkg'], suffix='rpkg')
evaluate_from_df(target_folder=os.path.join(target_folder), df_pred=df_preds[df_preds['domain'] == 'scip'], suffix='scip')
    
    





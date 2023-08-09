import pandas as pd
import copy
from statistics import mode
from utils import get_data
from write_eval_stats_compare_within import evaluate_from_df
import os

## All results are on full data

# Map of submissions: Approach and path to result file
methods = {'similarity-based': 'results/full_data_similarity_overall_within_ours/predicted_our_split.json', 'instance-based': 'results/roberta_large_validate_predicted.json',
'longformer': 'results/longformer_validate_predicted.json', 'chatgpt': 'results/chatgpt_validation_predicted.json'}


def get_df(file_path):

    ## Build df with gold labels
    map_gold = {}
    df_gold = pd.read_json(file_path)
    for _, row in df_gold.iterrows():

        row_id = row['id']
        row_labels = row['labels']

        for i in range(len(row_labels)):

            map_gold[str(row_id) + '_' + str(i)] = row_labels[i]


    df_results = pd.DataFrame.from_dict(map_gold, orient='index')
    return df_results


df_results = get_df('/data/our_split/validation.json')
df_results.columns = ['gold']


for method, method_path in methods.items():

    df_method = get_df(method_path)
    df_method.columns = [method + '_pred']

    df_results = pd.merge(left=df_results, right=df_method, left_index=True, right_index=True)
    df_results[method + '_is_correct'] = df_results['gold'] == df_results[method + '_pred']


# Find which ones are found by none of the methods
def find_none(row):

    noone = True
    for method in methods.keys():

        if row[method+ '_is_correct']:
            noone = False
    
    return noone

# Find which ones are found by all methods
def find_all(row, method_list):

    remaining_methods = copy.deepcopy(list(methods.keys()))
    for method in method_list:
        remaining_methods.remove(method)

    all = True
    for method in method_list:

        if not row[method+ '_is_correct']:
            all = False
    
    # None of the other methods should get it right
    for method in remaining_methods:

        if row[method+'_is_correct']:
            all=False
    
    return all

# Find which ones would be found in oracle condition
def find_oracle(row):

    oracle = False
    for method in methods.keys():

        if row[method+ '_is_correct']:
            oracle = True
    
    return oracle

# Find which ones would be classified by majority voting
def find_majority(row):

    votes = [row[method + '_pred'] for method in methods.keys()]
    vote = mode(votes)
    
    return vote


df_results['none'] = df_results.apply(find_none, axis=1)
df_results['all'] = df_results.apply(find_all, axis=1, method_list=list(methods.keys()))
df_results['oracle'] = df_results.apply(find_oracle, axis=1)

df_results['majority_pred'] = df_results.apply(find_majority, axis=1)
df_results['majority_is_correct'] = df_results['majority_pred'] == df_results['gold']

print(df_results)

print('Total number of sentences: ', len(df_results))
print('How many none of the methods get: ', sum(df_results['none']))
print('How many all of the methods get: ', sum(df_results['all']))
print('How many are predicted correctly in oracle condition: ', sum(df_results['oracle']))
print('How many are predicted correctly in majority condition: ', sum(df_results['majority_is_correct']))

overlap_map = {
## One:
'inst': ['instance-based'],
'simi': ['similarity-based'],
'longformer': ['longformer'],
'chat': ['chatgpt'],

## Two:
'inst-simi': ['instance-based', 'similarity-based'],
'inst-long': ['instance-based', 'longformer'],
'inst-chat': ['instance-based', 'chatgpt'],

'simi-long': ['similarity-based', 'longformer'],
'simi-chat': ['similarity-based', 'chatgpt'],

'long-chat': ['longformer', 'chatgpt'],

## Three:
'inst-simi-long': ['instance-based', 'similarity-based', 'longformer'],
'inst-simi-chat': ['instance-based', 'similarity-based', 'chatgpt'],
'inst-long-chat': ['instance-based', 'longformer', 'chatgpt'],
'simi-long-chat': ['similarity-based', 'longformer', 'chatgpt']
}

for setting, method_list in overlap_map.items():

    df_results[setting] = df_results.apply(find_all, axis=1, method_list=method_list)
    print(setting, sum(df_results[setting]))


# print(df_results)

for method, results in methods.items():

    target_folder = os.path.join('results', 'label_performance', method)
    
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # df_results = get_data(results, 'ALL')
    # df_results = df_results.rename(columns={'label': 'pred_max'})

    df_method_results = df_results[[method + '_pred', 'gold']]
    df_method_results = df_method_results.rename(columns={method + '_pred': 'pred_max', 'gold': 'label'})
    df_method_results['pred_avg'] = df_method_results['pred_max']
    evaluate_from_df(target_folder=target_folder, df_pred=df_method_results, suffix='')
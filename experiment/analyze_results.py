import pandas as pd

## All results are on full data

# Map of submissions: Approach and path to result file
methods = {'similarity-based': 'results/full_data_similarity_overall_within_ours/predicted_our_split.json', 'instance-based': 'results/full_data_similarity_overall_cross_ours/predicted_our_split.json'}


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
def find_all(row):

    all = True
    for method in methods.keys():

        if not row[method+ '_is_correct']:
            all = False
    
    return all

# Find which ones would be found in oracle condition
def find_oracle(row):

    oracle = False
    for method in methods.keys():

        if row[method+ '_is_correct']:
            oracle = True
    
    return oracle


df_results['none'] = df_results.apply(find_none, axis=1)
df_results['all'] = df_results.apply(find_all, axis=1)
df_results['oracle'] = df_results.apply(find_oracle, axis=1)

print(df_results)

print('Total number of sentences: ', len(df_results))
print('How many none of the methods get: ', sum(df_results['none']))
print('How many all of the methods get: ', sum(df_results['all']))
print('How are predicted correctly in oracle condition: ', sum(df_results['oracle']))



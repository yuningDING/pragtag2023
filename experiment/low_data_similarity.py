import json
import sys
from utils import get_data
import pandas as pd
import random
from train_sbert import train_sbert

data_path = 'data/train_inputs_low.json'

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


df_train = pd.DataFrame()
df_test = pd.DataFrame()

#for each domain, take one as testing
for domain in domain_map.keys():

    random.seed(1334)

    domain_reviews = domain_map[domain]
    # print(domain, len(domain_reviews))
    test_review = [random.choice(domain_reviews)]
    domain_reviews.remove(test_review[0])

    df_train = pd.concat([df_train, get_pids(domain_reviews, df_data)])
    df_test = pd.concat([df_test, get_pids(test_review, df_data)])


df_train.to_csv('data/our_split/train_low.csv')
df_test.to_csv('data/our_split/val_low.csv')

print(len(df_data))
print(len(df_train))
print(len(df_test))
prompt = 'low_data_similarity_our_low_split'

train_sbert(run_path='/results/' + prompt, df_train=df_train, df_test=df_test, df_val=df_test, answer_column="sentence", target_column="label", id_column="sent_id", base_model="all-MiniLM-L12-v2", num_pairs_per_example=None, save_model=True, num_epochs=10, batch_size=8, do_warmup=True, respect_domains=True)







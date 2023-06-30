import pandas as pd
import os
from numpy import mean, std


def process(json_df):
    essay_id = []
    label = []
    text = []
    prediction_string = []
    domain = []
    for index, row in json_df.iterrows():
        sentences = row['sentences']
        essay_id.extend([row['id']]*len(sentences))
        domain.extend([row['domain']]*len(sentences))
        count_t = 0
        for i in range(len(sentences)):
            text.append(sentences[i])
            label.append(row['labels'][i])
            token_indexes = []
            for t in sentences[i].split():
                token_indexes.append(count_t)
                count_t+=1
            prediction_string.append(' '.join( [str(x) for x in token_indexes]))
    csv_df = pd.DataFrame(list(zip(essay_id, domain, label, text, prediction_string)),columns =['essay_id', 'domain', 'label', 'text', 'prediction_string'])
    return csv_df


def get_txt(json_df, txt_dir):
    os.makedirs(txt_dir, exist_ok=True)
    for index, row in json_df.iterrows():
        text = ''
        essay_id = row['id']
        sentences = row['sentences']
        for s in sentences:
            text = text + ' ' + s
        with open(txt_dir+'/'+essay_id+'.txt', 'w', encoding='utf-8') as text_file:
            text_file.write(text)


def get_tokens_statistic_review(json_df):
    number_of_tokens = []
    for index, row in json_df.iterrows():
        text = ''
        sentences = row['sentences']
        for s in sentences:
            text = text + ' ' + s
        number_of_tokens.append(len(text.strip().split()))
    print('Max number of tokens:', max(number_of_tokens))
    print('Min number of tokens:', min(number_of_tokens))
    print('Average number of tokens:', mean(number_of_tokens))
    print('Standard deviation of tokens:', std(number_of_tokens))


def get_tokens_statistic_sentence(json_df):
    number_of_tokens = []
    for index, row in json_df.iterrows():
        sentences = row['sentences']
        for s in sentences:
            number_of_tokens.append(len(s.strip().split()))
    print('Max number of tokens:', max(number_of_tokens))
    print('Min number of tokens:', min(number_of_tokens))
    print('Average number of tokens:', mean(number_of_tokens))
    print('Standard deviation of tokens:', std(number_of_tokens))


def split_train_validate(json_df):
    domain_dict = {'case':32, 'diso':24, 'iscb':22, 'rpkg':22, 'scip':18}
    validation = pd.DataFrame()
    for d in domain_dict.keys():
        sub_df = json_df.loc[json_df['domain'] == d]
        sample = sub_df.sample(n=int(domain_dict.get(d)/10))
        validation = pd.concat([validation,sample], ignore_index=True, sort=False)
    print(validation)
    df_all = json_df.merge(validation, on=['id', 'pid','domain'],how='outer', indicator=True)
    train = df_all[df_all['_merge'] == 'left_only']
    train = train.drop(columns=['sentences_y','labels_y','_merge'])
    train.rename(columns={'sentences_x': 'sentences', 'labels_x': 'labels'}, inplace=True)
    print(train)
    return train, validation


all = pd.read_json('train.json')

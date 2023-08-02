import json
import pandas as pd


def write_submission(pred_path, gold_path, target_filepath):

    # from id to list
    pred_dict = {}

    ## For each element: Build list of predicted labels and add to gold file
    df_pred = pd.read_csv(pred_path)
    for element_id, sub_df in df_pred.groupby('id'):

        pred_list = pred_dict.get(element_id, [])

        for _, row in sub_df.iterrows():
            sent_id = row['sent_id']
            sent_idx = int(sent_id[sent_id.rindex('_')+1:])
            pred_list.append(row['pred_max'])
        
        pred_dict[element_id] = pred_list


    with open(gold_path) as in_file:
        json_data = json.load(in_file)
        
        with open(target_filepath, 'w') as target_file:

            target_file.write('[')

            # for element in json_data:

            #     element_id = element['id']
            #     element['labels'] = pred_dict[element_id]

            #     json.dump(element, target_file)
            #     target_file.write(',')

            for i in range(len(json_data)):
                
                element = json_data[i]

                element_id = element['id']
                element['labels'] = pred_dict[element_id]

                json.dump(element, target_file)

                if not i == len(json_data) - 1:
                    target_file.write(',')

            target_file.write(']')

def get_data(json_file_name, prompt):

    data = {}

    with open(json_file_name) as in_file:
        json_data = json.load(in_file)
        
        for element in json_data:

            if element['domain'] == prompt or prompt.startswith('ALL'): 
                
                paper_id = element['pid']
                report_id = element['id']
                domain = element['domain']
                sentences = element['sentences']
                labels = element['labels']

                for i in range(len(sentences)):
                    sent_id = paper_id + '_' + report_id + '_' + str(i)

                    data[sent_id] = {
                        'sent_id': sent_id,
                        'id': report_id,
                        'pid': paper_id,
                        'domain': domain,
                        'sentence': sentences[i],
                        'label': labels[i] 
                    }

    
    data = pd.DataFrame.from_dict(data, orient='index')
    return data

def get_test_data(json_file_name, prompt):

    data = {}

    with open(json_file_name) as in_file:
        json_data = json.load(in_file)
        
        for element in json_data:

            if element['domain'] == prompt or prompt.startswith('ALL'): 
                
                paper_id = element['pid']
                report_id = element['id']
                domain = element['domain']
                sentences = element['sentences']
                # labels = element['labels']

                for i in range(len(sentences)):
                    sent_id = paper_id + '_' + report_id + '_' + str(i)

                    data[sent_id] = {
                        'sent_id': sent_id,
                        'id': report_id,
                        'pid': paper_id,
                        'domain': domain,
                        'sentence': sentences[i],
                        # 'label': labels[i] 
                    }

    
    data = pd.DataFrame.from_dict(data, orient='index')
    return data


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse


# given gold and prediction, compute the f1 macro across reviews
def evaluate(gold, pred):
    rids = list(gold.keys())

    return f1_score([l for r in rids for l in gold[r]["labels"]], [l for r in rids for l in pred[r]["labels"]],
                    average="macro")


# compute the confusion matrix across reviews
def confusion(gold, pred):
    rids = list(gold.keys())

    return confusion_matrix([l for r in rids for l in gold[r]], [l for r in rids for l in pred[r]])


# computes f1 macro for each domains and computes the mean across domains
def eval_across_domains(gold, pred):
    domains = set(g["domain"] for g in gold.values())
    rid_to_domain = {rid : g["domain"] for rid, g in gold.items()}

    per_domain = {d: evaluate(dict(filter(lambda x: x[1]["domain"] == d, gold.items())),
                              dict(filter(lambda x: rid_to_domain[x[1]["id"]] == d, pred.items()))) for d in domains}

    return per_domain, np.mean(list(per_domain.values()))
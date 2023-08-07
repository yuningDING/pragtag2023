import os
import sys
import pandas as pd
import json
import random
from tqdm import tqdm
from scipy import spatial
from glob import glob
from utils import get_data
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from train_sbert import eval_sbert, get_train_examples_limited
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SimilarityFunction
from sklearn.metrics import accuracy_score, classification_report

predictions_file = 'predictions_sim.csv'
answer_column = 'sentence'
id_column = 'sent_id'
target_column = 'label'

source_folder = 'results/low_data_similarity_folds'
aux_path = '/Users/mariebexte/Coding/Datasets/PragTag/requestitem_export_09ebbebb-8b76-40c3-b2a5-03533ea6d7c0/auxiliary_data/F1000-22/data'
gw_lookup_path = 'data/gw_lookup.json'

domains = ['case', 'diso', 'iscb', 'rpkg', 'scip']
respect_domains = True

overall_pred = []
overall_gold = []


def pred_auxiliary(df_test, df_ref, id_column, answer_column, target_column):

    print('Evaluating: ', len(df_test))

    max_predictions = []
    avg_predictions = []

    # Later used to create dataframe with classification results
    predictions = {}
    predictions_index = 0

    for domain in df_ref['domain'].unique():

        # df_test_domain = df_test[df_test[domain]==domain]
        df_ref_domain = df_ref[df_ref['domain']==domain]

        # Cross every test embedding with every train embedding
        for idx, test_answer in tqdm(df_test.iterrows(), total=len(df_test)):

            # Copy reference answer dataframe
            copy_eval = df_ref_domain[[id_column, answer_column, target_column, "embedding"]].copy()
            # Put reference answers as 'answers 2'
            copy_eval.columns = ["id2", "text2", "score2", "embedding2"]
            # Put current testing answer as 'answer 1': Copy it num_ref_answers times into the reference dataframe to compare it to all of the reference answers
            copy_eval["id1"] = [test_answer[id_column]]*len(copy_eval)
            copy_eval["text1"] = [test_answer[answer_column]]*len(copy_eval)
            test_emb = test_answer['embedding']
            copy_eval['cos_sim'] = copy_eval["embedding2"].apply(lambda _emb: 1 - spatial.distance.cosine(test_emb, _emb))

            # Determine prediction: MAX
            max_row = copy_eval.iloc[[copy_eval["cos_sim"].argmax()]]
            max_sim = max_row.iloc[0]["cos_sim"]
            max_pred = max_row.iloc[0]["score2"]
            max_sim_id = max_row.iloc[0]["id1"]
            max_sim_answer = max_row.iloc[0]["text1"]

            # Determine prediction: AVG
            label_avgs = {}
            for label in set(copy_eval["score2"]):
                label_subset = copy_eval[copy_eval["score2"] == label]
                label_avgs[label] = label_subset["cos_sim"].mean()
            avg_pred = max(label_avgs, key=label_avgs.get)
            avg_sim = max(label_avgs.values())

            max_predictions.append(max_pred)
            avg_predictions.append(avg_pred)

            predictions[predictions_index] = {"id": test_answer[id_column], "pred_avg": avg_pred, "sim_score_avg": avg_sim,"pred_max": max_pred, "sim_score_max": max_sim, "most_similar_answer_id": max_sim_id, "most_similar_answer_text": max_sim_answer}
            predictions_index += 1

    copy_test = df_test.copy()
    df_predictions = pd.DataFrame.from_dict(predictions, orient='index')
    df_predictions = pd.merge(copy_test, df_predictions, left_on=id_column, right_on="id")
    # df_predictions.to_csv(os.path.join(run_path, "predictions_sim.csv"), index=None)

    return df_predictions



def get_aux_data(domain):

    data = {}

    with open(gw_lookup_path) as gw_file:

        gw_json = json.load(gw_file)
        desired_ids = gw_json[domain]

        for desired_id in desired_ids:

            review_path = os.path.join(aux_path, desired_id, 'v1/reviews.json')
            with open(review_path, 'r') as review_file:

                review_json = json.load(review_file)


                for review in review_json:

                    rid = review['rid']
                    report = review['report']['main']
                    sentence_indexes = review['meta']['sentences']['main']

                    sent_num = 0
                    for sentence_index in sentence_indexes:

                        element_index = rid + '_' + str(sent_num)
                        data[element_index] = {
                            'rid': rid,
                            'sent_id': element_index,
                            'domain': domain,
                            'sentence': report[sentence_index[0]:sentence_index[1]].strip()
                        }

                        sent_num += 1
                    
    df_data = pd.DataFrame.from_dict(data, orient='index')
    return df_data

########################

for fold in range (4):

    train_file = os.path.join(source_folder, 'fold_' + str(fold), 'train.csv')
    val_file = os.path.join(source_folder, 'fold_' + str(fold), 'predictions_sim.csv')
    model = SentenceTransformer(os.path.join(source_folder, 'fold_' + str(fold), 'finetuned_model'))

    # Take n with highest similiarity (for each label)
    train = False
    num_per_label = 10
    num_rounds = 3
    num_train = 1000
    condition_name = str(num_per_label) + '_per_label_' + str(num_rounds) + 'rounds_' + str (num_train)
    if train:
        condition_name = condition_name + '_train'
    

    df_train = pd.read_csv(train_file)
    df_train['embedding'] = df_train[answer_column].apply(model.encode)

    aux_data_dict = {}
    for domain in domains:
        print('Getting aux data for', domain)
        df_aux = get_aux_data(domain)
        # Enrich with embeddings
        df_aux['embedding'] = df_aux[answer_column].apply(model.encode)
        aux_data_dict[domain] = df_aux


    for round in range(num_rounds):

        for domain in domains:

            df_largest = pd.DataFrame()

            print('STARTING', fold, round, domain)

            ## Get auxiliary data for this domain
            df_aux = aux_data_dict[domain]
            df_train_domain = df_train[df_train['domain']==domain]
            df_aux_with_preds = pred_auxiliary(df_test=df_aux, df_ref=df_train_domain, id_column='sent_id', answer_column=answer_column, target_column='label')

            for label in ['Structure', 'Recap', 'Other', 'Todo', 'Strength', 'Weakness']:
                df_label = df_aux_with_preds[df_aux_with_preds['pred_avg'] == label]
                df_largest = pd.concat([df_largest, df_label.nlargest(num_per_label, "sim_score_avg")])

            # df_largest = df_aux_with_preds[df_aux_with_preds['sim_score_avg'] > .95]
            print('LABEL DISTRIBUTION', df_largest['pred_avg'].value_counts())
            # Rename to take prediction as labeled column
            df_largest = df_largest.rename(columns={'pred_avg': 'label', 'rid': 'pid'})


            # Add to training
            print(len(df_largest))
            print(len(df_train))
            df_train = pd.concat([df_train, df_largest])
            print(len(df_train))
            # Remove from auxiliary, 
            print(len(df_aux))
            df_aux = df_aux[~df_aux['sent_id'].isin(df_largest['sent_id'])]
            print(len(df_aux))

            aux_data_dict[domain] = df_aux


        if train:
        
            batch_size=8
            # Define list of training pairs: Create as many as possible
            train_examples = get_train_examples_limited(df_train=df_train, respect_domains=respect_domains, id_column=id_column, target_column=target_column, answer_column=answer_column)
            if len(train_examples) > num_train:
                random.seed(1334)
                train_examples = random.sample(train_examples, num_train)
            # Define train dataset, dataloader, train loss
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            train_loss = losses.OnlineContrastiveLoss(model)
            model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=0, show_progress_bar=True)

            ## Update embeddings
            df_aux['embedding'] = df_aux[answer_column].apply(model.encode)
            df_train['embedding'] = df_train[answer_column].apply(model.encode)
            

    df_test = pd.read_csv(val_file) 
    df_test['embedding'] = df_test[answer_column].apply(model.encode)
    # df_test = df_test[['sent_id', 'pid', 'domain', 'sentence', 'label', 'embedding']]
    df_test = df_test[['sent_id', 'pid', 'domain', 'sentence', 'embedding']]

    run_path = os.path.join(source_folder, condition_name, 'fold_' + str(fold))
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    if train:
        model.save(os.path.join(run_path, 'finetuned_model'))
    
    df_train.to_csv(os.path.join(run_path, 'ref.csv'))
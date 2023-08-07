import os
import pandas as pd
from utils import get_data
from train_sbert import eval_sbert_within_domain
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SimilarityFunction
from sklearn.metrics import accuracy_score, classification_report

train_file = '/data/our_split/train.json'
# train_file = '/data/train_inputs_low.json'
# val_file = '/data/our_split/validation.json'
val_file = '/data/our_split/validation.json'

def evaluate(model_path, df_train, df_test, answer_column='sentence'):

    if len(df_test) > 0:

        model = SentenceTransformer(os.path.join(model_path, 'finetuned_model'))

        # Eval testing data: Get sentence embeddings for all testing and reference answers
        df_test['embedding'] = df_test[answer_column].apply(model.encode)
        df_test = df_test[['sent_id', 'pid', 'domain', 'sentence', 'label', 'embedding']]

        df_ref = df_train
        df_ref['embedding'] = df_ref[answer_column].apply(model.encode)

        target_dir = os.path.join(model_path)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        pred_max, pred_avg = eval_sbert_within_domain(run_path=target_dir, df_test=df_test, df_ref=df_ref, id_column='sent_id', answer_column=answer_column, target_column='label')
        gold = df_test['label'] 

        acc_avg = accuracy_score(gold, pred_avg)
        acc_max = accuracy_score(gold, pred_max)

        with open(os.path.join(target_dir, 'results.txt'), 'w') as res_file:

            res_file.write('acc avg:\t' + str(acc_avg) + '\n')
            res_file.write('acc max:\t' + str(acc_max) + '\n\n')
            res_file.write('classification report (avg):\n')
            res_file.write(classification_report(gold, pred_avg) + '\n\n')
            res_file.write('classification report (max):\n')
            res_file.write(classification_report(gold, pred_max) + '\n')

            y_gold = pd.Series(gold, name='Gold')
            y_pred_avg = pd.Series(pred_max, name='Pred (avg)')
            y_pred_max = pd.Series(pred_max, name='Pred (max)')
            df_confusion_avg = pd.crosstab(y_gold, y_pred_avg)
            df_confusion_max = pd.crosstab(y_gold, y_pred_max)

            res_file.write(str(df_confusion_avg) + '\n\n')
            res_file.write(str(df_confusion_max) + '\n\n')


def evaluate_from_df(target_folder, df_pred, suffix=''):

        pred_max = df_pred['pred_max']
        pred_avg = df_pred['pred_avg']
        gold = df_pred['label'] 

        acc_avg = accuracy_score(gold, pred_avg)
        acc_max = accuracy_score(gold, pred_max)

        with open(os.path.join(target_folder, 'results_' + suffix + '.txt'), 'w') as res_file:

            res_file.write('acc avg:\t' + str(acc_avg) + '\n')
            res_file.write('acc max:\t' + str(acc_max) + '\n\n')
            res_file.write('classification report (avg):\n')
            res_file.write(classification_report(gold, pred_avg) + '\n\n')
            res_file.write('classification report (max):\n')
            res_file.write(classification_report(gold, pred_max) + '\n')

            y_gold = pd.Series(gold, name='Gold')
            y_pred_avg = pd.Series(pred_max, name='Pred (avg)')
            y_pred_max = pd.Series(pred_max, name='Pred (max)')
            df_confusion_avg = pd.crosstab(y_gold, y_pred_avg)
            df_confusion_max = pd.crosstab(y_gold, y_pred_max)

            res_file.write(str(df_confusion_avg) + '\n\n')
            res_file.write(str(df_confusion_max) + '\n\n')




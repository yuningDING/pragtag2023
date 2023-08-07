import os
import sys
import pandas as pd
from statistics import mode
from utils import get_data, get_test_data, write_submission
from load import load_prediction_and_gold
from eval import eval_across_domains
from train_sbert import eval_sbert_within_domain, eval_sbert
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report

result_folder = 'results_cross'
predictions_file = 'predictions_sim.csv'

prediction_column = 'max'

def vote(voters, setting, train_file, val_file, eval=False):

    # MAP: element_id to MAP: domain to label
    overall_preds = {}

    for domain, soi in voters.items():

        val_subset = get_test_data(val_file, soi)
        df_train = get_data(train_file, soi)

        # Predict on the respective split of interest for this model
        model = SentenceTransformer(os.path.join(domain, 'finetuned_model'))
        df_train['embedding'] = df_train['sentence'].apply(model.encode)
        val_subset['embedding'] = val_subset['sentence'].apply(model.encode)
        # df_test = eval_sbert_within_domain(run_path='', df_test=val_subset, df_ref=df_train, id_column='sent_id', answer_column='sentence', target_column='label', save=False)
        df_test = eval_sbert(run_path='', df_test=val_subset, df_ref=df_train, id_column='sent_id', answer_column='sentence', target_column='label', save=False)

        for _, row in df_test.iterrows():

            id_map = overall_preds.get(row['sent_id'], {})
            id_map[domain] = (row['pred_' + prediction_column], row['sim_score_' + prediction_column])
            overall_preds[row['sent_id']] = id_map

    final_preds = {}
    for (element, preds) in overall_preds.items():

        labels = [pred_tuple[0] for pred_tuple in preds.values()]
        print(labels)
        pred = mode(labels)
        final_preds[element] = pred

    df_pred = pd.DataFrame.from_dict(final_preds, orient='index')
    print(df_pred)
    df_pred.columns = ['pred_max']
    df_pred['sent_id'] = df_pred.index
    print(df_pred)

    df_test = get_test_data(val_file, 'ALL')
    df_overall = pd.merge(left=df_test, right=df_pred)

    target_dir = os.path.join(result_folder, setting)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    # df_test = get_data(val_file, 'ALL')
    df_test = get_test_data(val_file, 'ALL')
    df_test = pd.merge(df_test, df_pred)
    df_test.to_csv(os.path.join(target_dir, 'predictions_sim.csv'))

    # Prepare submission
    write_submission(os.path.join(target_dir, 'predictions_sim.csv'), val_file, os.path.join(target_dir, 'predicted.json'))

    ## If it is the internal split, calculate metrics:
    if eval:

        pred, gold = load_prediction_and_gold(os.path.join(target_dir, 'predicted.json'), val_file)
        per_domain, mean = eval_across_domains(gold, pred)

        with open(os.path.join(target_dir, 'scores.txt'), "w+") as f:
            for k, v in per_domain.items():
                f.write(f"f1_{k}:{v}\n")
            f.write(f"f1_mean:{mean}")


### Voting: Define maps: Path to model and which subset of the data to use
## Full data

# Domain-experts (no real voting)
voters = {
    'results/case': 'case',
    'results/diso': 'diso',
    'results/iscb': 'iscb',
    'results/rpkg': 'rpkg',
    'results/scip': 'scip'
}
setting = 'full_data_similarity_domain_experts'
vote(voters=voters, setting=setting + '_our_split', train_file='data/our_split/train.json', val_file='data/our_split/validation.json', eval=True)
vote(voters=voters, setting=setting + '_test_data', train_file='data/our_split/train.json', val_file='data/test_inputs.json')
# Domain experts + cross-domain models
voters = {
    'results/full_data_similarity_overall_cross_ours': 'ALL',
    'results/full_data_similarity_overall_within_ours': 'ALL',
    'results/full_data_similarity_overall_within_large_ours': 'ALL',
    'results/case': 'case',
    'results/diso': 'diso',
    'results/iscb': 'iscb',
    'results/rpkg': 'rpkg',
    'results/scip': 'scip'
}
setting = 'full_data_similarity_voting'
vote(voters=voters, setting=setting + '_our_split', train_file='data/our_split/train.json', val_file='data/our_split/validation.json', eval=True)
vote(voters=voters, setting=setting + '_test_data', train_file='data/our_split/train.json', val_file='data/test_inputs.json')

## Low data
voters = {
    'results/low_data_similarity_folds/fold_0': 'ALL',
    'results/low_data_similarity_folds/fold_1': 'ALL',
    'results/low_data_similarity_folds/fold_2': 'ALL',
    'results/low_data_similarity_folds/fold_3': 'ALL',
    'results/low_data_similarity_ours': 'ALL',
}
setting = 'low_data_similarity_voting'
vote(voters=voters, setting=setting + '_our_split', train_file='data/our_split/train_low.json', val_file='data/our_split/validation.json', eval=True)
vote(voters=voters, setting=setting + '_test_data', train_file='data/our_split/train_low.json', val_file='data/test_inputs.json')
import pandas as pd
from utils import get_test_data, write_submission
from eval import eval_across_domains
from load import load_prediction_and_gold
from scipy import spatial
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import os


def no_data(val_path, target_path, eval=False):
    df_val = get_test_data(val_path, 'ALL')

    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(os.path.join(target_path))

    labels_short={'Todo': ['should', 'could', 'need'], 'Strength': ['good', 'strength', 'clear'], 'Weakness': ['weakness', 'shortcoming', 'flaw'], 'Structure': ['reviewer'], 'Recap': ['authors', 'describe', 'article'], 'Other': ['other']}
    labels={'Todo': ['suggests the ways a manuscript can be improved', "Could the authors devise a standard procedure to obtain the data?"],
        'Strength': ['points out the merits of the work', "It is very well written and the contribution is significant."],
        'Weakness': ['points out a limitation', "However, the data is not publicly available, making the work hard to reproduce"],
        'Structure': ['used to organize the reviewing report', "Typos:"],
        'Recap': ['summarizes the manuscript', "The paper proposes a new method for"],
        'Other': ['contains additional information, e.g. reviewers thoughts, background knowledge and performative statements', "Few examples from prior work: [1], [2], [3]", "Once this is clarified, the paper can be accepted."]}

    for key, value in labels_short.items():
        labels[key] = labels[key] + value

    label_encodings = {}

    model = SentenceTransformer('all-MiniLM-L12-v2')
    for label, label_list in labels.items():
        label_encodings[label] = [model.encode(label_element) for label_element in label_list]

    df_val['embedding'] = df_val['sentence'].apply(model.encode)
    
    for _, row in df_val.iterrows():

        max_sim_label = None
        highest_similarity_label = -1

        for label, label_embeddings in label_encodings.items():
            
            similarities =  [1 - spatial.distance.cosine(row['embedding'], label_encoding) for label_encoding in label_embeddings]
                
            avg_sim = np.average(np.array(similarities))
            if avg_sim > highest_similarity_label:
                highest_similarity_label = avg_sim
                max_sim_label = label

        df_val.loc[df_val['sent_id'] == row['sent_id'],'pred_max'] = max_sim_label

    df_val.to_csv(os.path.join(target_path, 'predictions_sim.csv'))

    write_submission(os.path.join(target_path, 'predictions_sim.csv'), val_path, os.path.join(target_path, 'predicted.json'))

    ## If it is the internal split, calculate metrics:
    if eval:

        pred, gold = load_prediction_and_gold(os.path.join(target_path, 'predicted.json'), val_path)
        per_domain, mean = eval_across_domains(gold, pred)

        with open(os.path.join(target_path, 'scores.txt'), "w+") as f:
            for k, v in per_domain.items():
                f.write(f"f1_{k}:{v}\n")
            f.write(f"f1_mean:{mean}")


val_path = 'data/our_split/validation.json'
target_path = 'results/no_data_similarity_our_split'
no_data(val_path=val_path, target_path=target_path, eval=True)

val_path = 'data/test_inputs.json'
target_path = 'results/no_data_similarity_test_data'
no_data(val_path=val_path, target_path=target_path)
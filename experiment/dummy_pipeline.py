import pandas as pd
from eval import eval_across_domains
from load import load_prediction_and_gold
import os


def evaluate(pred_path, gold_path, out_path):
    pred, gold = load_prediction_and_gold(pred_path, gold_path)

    per_domain, mean = eval_across_domains(gold, pred)

    with open(out_path, "w+") as f:
        for k, v in per_domain.items():
            f.write(f"f1_{k}:{v}\n")
        f.write(f"f1_mean:{mean}")


if __name__ == "__main__":
    validation = pd.read_json("../data/validation.json")
    validation_dummy = validation[['id']].copy()
    final_pred = {}
    for index, row in validation.iterrows():
        sentences = row['sentences']
        essay_id = row['id']
        prediction = []
        for i in range(len(sentences)):
            if i==0:
                prediction.append('Structure')
            else:
                prediction.append('Todo')
            i += 1
        final_pred[essay_id]=prediction
    validation_dummy['labels'] = validation_dummy['id'].map(final_pred)

    output_dir = 'dummy_output'
    os.makedirs(output_dir, exist_ok=True)
    validation_dummy.to_json(output_dir+ '/' + 'validate_predicted.json', orient="records")
    evaluate(output_dir + '/' + 'validate_predicted.json', "../data/validation.json", output_dir + '/' + 'scores.txt')

import json
import pandas as pd
import os
from eval import eval_across_domains
from load import load_prediction_and_gold

def evaluate(pred_path, gold_path, out_path):
    pred, gold = load_prediction_and_gold(pred_path, gold_path)

    per_domain, mean = eval_across_domains(gold, pred)

    with open(out_path, "w+") as f:
        for k, v in per_domain.items():
            f.write(f"f1_{k}:{v}\n")
        f.write(f"f1_mean:{mean}")


#validation = pd.read_json("../data/public_dat/train_inputs_low.json")
validation = pd.read_json("../data/public_dat/validation.json")
#print(validation)
validation_dummy = validation[['id']].copy()
output_dir = 'gpt_output_validation'
final_pred = {}

for index, row in validation.iterrows():
    print(index)
    essay_id = row['id']
    num_sentences = len(row['sentences'])
    with open(output_dir + '/' + str(index) + "_out.txt", "r") as text_file:
        content_string = text_file.read()
        try:
            content = json.loads(content_string)
            sentences = content['sentences']
            labels = []
            for sentence in sentences:
                label = sentence['label']
                if label == '':
                    label = 'Other'
                if not (label == 'Other' or label == 'Recap' or label == 'Strength' or label == 'Weakness' or label == 'Todo' or label == 'Structure'):
                    label = 'Other'
                labels.append(label)
            if(len(labels) > num_sentences):
                labels =labels[0:num_sentences]
            while (len(labels) != num_sentences):
                labels.append("Other")
            # print(labels)
        except Exception as e:
            print(e)
            #print("Problem with: "+str(content_string))
            print(str(content_string)[1280:])
        final_pred[essay_id] = labels
        validation_dummy['labels'] = validation_dummy['id'].map(final_pred)


validation_dummy.to_json(output_dir+ '/' + 'predicted.json', orient="records")

evaluate(output_dir + '/' + 'predicted.json', "../data/public_dat/validation.json", output_dir + '/' + 'scores.txt')
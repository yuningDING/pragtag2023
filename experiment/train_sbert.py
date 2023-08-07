import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import SimilarityFunction
import pandas as pd
from torch.utils.data import DataLoader
import torch
import shutil
import sys
from scipy import spatial
import shutil
import logging
from load import load_prediction_and_gold
from eval import eval_across_domains
from utils import get_data, get_test_data, write_submission
from tqdm import tqdm

def get_train_examples_limited(df_train, respect_domains=True, domain_col='domain', id_column='id', target_column='label', answer_column='text', max_num_samples=10000):
    
    num_samples_per_train = int(max_num_samples/len(df_train))
    
    train_examples = []
    for _, example_1 in tqdm(df_train.iterrows(), total=len(df_train)):

        random_pairs = df_train.sample(n=num_samples_per_train)
        for _, example_2 in random_pairs.iterrows():

            if not example_1[id_column] == example_2[id_column]:

                if respect_domains and example_1[domain_col] == example_2[domain_col]:

                    label = 0
                    if example_1[target_column] == example_2[target_column]:
                        label = 1

                    train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))
                
                elif not respect_domains:

                    label = 0
                    if example_1[target_column] == example_2[target_column]:
                        label = 1

                    train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))

    return train_examples
    

def get_train_examples(df_train, respect_domains=True, domain_col='domain', id_column='id', target_column='label', answer_column='text'):
    train_examples = []
    for _, example_1 in tqdm(df_train.iterrows(), total=len(df_train)):
        for _, example_2 in df_train.iterrows():

            if not example_1[id_column] == example_2[id_column]:

                if respect_domains and example_1[domain_col] == example_2[domain_col]:



                    label = 0
                    if example_1[target_column] == example_2[target_column]:
                        label = 1

                    train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))
                
                elif not respect_domains:

                    label = 0
                    if example_1[target_column] == example_2[target_column]:
                        label = 1

                    train_examples.append(InputExample(texts=[example_1[answer_column], example_2[answer_column]], label=label*1.0))

    return train_examples


def get_val_examples(df_train, df_val, respect_domains=True, domain_col='domain', id_column='id', target_column='label', answer_column='text'):

    # Define validation pairs: Create as many as possible
    val_example_dict = {}
    val_example_index = 0
    for _, example_1 in tqdm(df_val.iterrows(), total=len(df_val)):
        for _, example_2 in df_train.iterrows():

            if not example_1[id_column] == example_2[id_column]:
                
                # If pairs should only be built within domains:
                if respect_domains and example_1[domain_col] == example_2[domain_col]:

                    label = 0
                    if example_1[target_column] == example_2[target_column]:
                        label = 1

                    val_example_dict[val_example_index] = {"text_1": example_1[answer_column], "text_2": example_2[answer_column], "sim_label": label}
                    val_example_index += 1
                
                elif not respect_domains:

                    label = 0
                    if example_1[target_column] == example_2[target_column]:
                        label = 1

                    val_example_dict[val_example_index] = {"text_1": example_1[answer_column], "text_2": example_2[answer_column], "sim_label": label}
                    val_example_index += 1

    val_examples = pd.DataFrame.from_dict(val_example_dict, "index")
    return val_examples


def eval_sbert_within_domain(run_path, df_test, df_ref, id_column, answer_column, target_column, save=True):

    max_predictions = []
    avg_predictions = []

    # Later used to create dataframe with classification results
    predictions = {}
    predictions_index = 0

    for domain in df_ref['domain'].unique():

        df_test_domain = df_test[df_test['domain']==domain]
        df_ref_domain = df_ref[df_ref['domain']==domain]

        # Cross every test embedding with every train embedding
        for idx, test_answer in df_test_domain.iterrows():

            # Copy reference answer dataframe
            copy_eval = df_ref_domain[[id_column, answer_column, target_column, "embedding"]].copy()
            # Put reference answers as 'answers 2'
            copy_eval.columns = ["id2", "text2", "score2", "embedding2"]
            # Put current testing answer as 'answer 1': Copy it num_ref_answers times into the reference dataframe to compare it to all of the reference answers
            copy_eval["id1"] = [test_answer[id_column]]*len(copy_eval)
            copy_eval["text1"] = [test_answer[answer_column]]*len(copy_eval)
            # copy_eval["score1"] = [test_answer[target_column]]*len(copy_eval)
            copy_eval["embedding1"] = [test_answer["embedding"]]*len(copy_eval)
            emb1 = list(copy_eval["embedding1"])
            emb2 = list(copy_eval["embedding2"])
            copy_eval["cos_sim"] = [1 - spatial.distance.cosine(emb1[i], emb2[i]) for i in range(len(copy_eval))]

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

            predictions[predictions_index] = {id_column: test_answer[id_column], "pred_avg": avg_pred, "sim_score_avg": avg_sim,"pred_max": max_pred, "sim_score_max": max_sim, "most_similar_answer_id": max_sim_id, "most_similar_answer_text": max_sim_answer}
            predictions_index += 1

    copy_test = df_test.copy()
    df_predictions = pd.DataFrame.from_dict(predictions, orient='index')
    df_predictions = pd.merge(copy_test, df_predictions, left_on=id_column, right_on=id_column)

    if save:
        df_predictions.to_csv(os.path.join(run_path, "predictions_sim.csv"), index=None)
        return max_predictions, avg_predictions
    
    else:
        return df_predictions

    
def eval_sbert(run_path, df_test, df_ref, id_column, answer_column, target_column, save=True):

    max_predictions = []
    avg_predictions = []

    # Later used to create dataframe with classification results
    predictions = {}
    predictions_index = 0

    # Cross every test embedding with every train embedding
    for idx, test_answer in df_test.iterrows():

        # Copy reference answer dataframe
        copy_eval = df_ref[[id_column, answer_column, target_column, "embedding"]].copy()
        # Put reference answers as 'answers 2'
        copy_eval.columns = ["id2", "text2", "score2", "embedding2"]
        # Put current testing answer as 'answer 1': Copy it num_ref_answers times into the reference dataframe to compare it to all of the reference answers
        copy_eval["id1"] = [test_answer[id_column]]*len(copy_eval)
        copy_eval["text1"] = [test_answer[answer_column]]*len(copy_eval)
        # copy_eval["score1"] = [test_answer[target_column]]*len(copy_eval)
        copy_eval["embedding1"] = [test_answer["embedding"]]*len(copy_eval)
        emb1 = list(copy_eval["embedding1"])
        emb2 = list(copy_eval["embedding2"])
        copy_eval["cos_sim"] = [1 - spatial.distance.cosine(emb1[i], emb2[i]) for i in range(len(copy_eval))]

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

        predictions[predictions_index] = {id_column: test_answer[id_column], "pred_avg": avg_pred, "sim_score_avg": avg_sim,"pred_max": max_pred, "sim_score_max": max_sim, "most_similar_answer_id": max_sim_id, "most_similar_answer_text": max_sim_answer}
        predictions_index += 1

    copy_test = df_test.copy()
    df_predictions = pd.DataFrame.from_dict(predictions, orient='index')
    df_predictions = pd.merge(copy_test, df_predictions, left_on=id_column, right_on=id_column)

    if save:
        df_predictions.to_csv(os.path.join(run_path, "predictions_sim.csv"), index=None)
        return max_predictions, avg_predictions
    
    else:
        return df_predictions


# For larger amounts of training data: Do not create all possible pairs, but limit to a fixed number per epoch (if possible, have different pairs across different epochs)
def train_sbert(run_path, df_train, df_test, df_val, answer_column="text", target_column="label", id_column="id", base_model="all-MiniLM-L6-v2", num_pairs_per_example=None, save_model=False, num_epochs=20, batch_size=8, do_warmup=True, evaluation_steps=3000, respect_domains=True):

    if num_pairs_per_example is not None:
        num_samples = len(df_train) * num_pairs_per_example
        num_batches_per_round = int(num_samples/batch_size)
        logging.info("LIMITING SBERT TRAINING PAIRS: "+str(num_pairs_per_example)+" pairs per sample!")

    device = "cpu"
    #device = "mps"
    if torch.cuda.is_available():
        device = "cuda"

    model = SentenceTransformer(base_model, device=device)

    # Where to store finetuned model: In LC this is just temporary, will be deleted at the end of the run
    model_path = os.path.join(run_path, "finetuned_model")

    # Define list of training pairs: Create as many as possible
    train_examples = get_train_examples(df_train=df_train, respect_domains=respect_domains, id_column=id_column, target_column=target_column, answer_column=answer_column)
    val_examples = get_val_examples(df_train=df_train, df_val=df_val, respect_domains=respect_domains, id_column=id_column, target_column=target_column, answer_column=answer_column)

    # Define train dataset, dataloader, train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.OnlineContrastiveLoss(model)

    # Define evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_examples["text_1"].tolist(), val_examples["text_2"].tolist(), val_examples["sim_label"].tolist())

    num_warm_steps = 0
    if do_warmup == True:
        steps_per_epoch = len(train_examples)/batch_size
        total_num_steps = steps_per_epoch * num_epochs
        num_warm_steps = round(0.1*total_num_steps)

    # Tune the model
    if num_pairs_per_example is not None:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=num_warm_steps, evaluator=evaluator, output_path=model_path, save_best_model=True, show_progress_bar=True, steps_per_epoch=num_batches_per_round)
    else:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=num_warm_steps, evaluator=evaluator, output_path=model_path, save_best_model=True, show_progress_bar=True, evaluation_steps=evaluation_steps)

    logging.info("SBERT number of epochs: "+str(num_epochs))
    logging.info("SBERT batch size: "+str(batch_size))
    logging.info("SBERT warmup steps: "+str(num_warm_steps))
    logging.info("SBERT evaluator: "+str(evaluator.__class__)+" Batch size: "+str(evaluator.batch_size)+" Main similarity:"+str(evaluator.main_similarity))
    logging.info("SBERT loss: "+str(train_loss.__class__))

    # Evaluate best model: Can only do this if training was sucessful, otherwise keep pretrained
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer(base_model)

    # Eval testing data: Get sentence embeddings for all testing and reference answers
    df_test['embedding'] = df_test[answer_column].apply(model.encode)

    df_ref = pd.concat([df_val, df_train])
    df_ref['embedding'] = df_ref[answer_column].apply(model.encode)

    # Copy training statistic into run folder
    if os.path.exists(model_path):
        shutil.copyfile(os.path.join(model_path, "eval", "similarity_evaluation_results.csv"), os.path.join(run_path, "eval_training.csv"))

    # Delete model to save space
    if os.path.exists(model_path) and save_model==False:
        shutil.rmtree(model_path)

    if respect_domains:
        return eval_sbert_within_domain(run_path=run_path, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
    else:
        return eval_sbert(run_path=run_path, df_test=df_test, df_ref=df_ref, id_column=id_column, answer_column=answer_column, target_column=target_column)
    

def train_and_eval_sbert(train_path, val_path, test_path, prompt, target_folder='results', bs=8, epochs=10, model="all-MiniLM-L12-v2", respect_domains=True):

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    df_train = get_data(train_path, 'ALL')
    df_val = get_data(val_path, 'ALL')
    df_test = get_test_data(test_path, 'ALL')

    prompt_ours = prompt + '_ours'
    prompt_test = prompt + '_test_data'

    # train_sbert(run_path=target_folder + '/' + prompt_ours, df_train=df_train, df_test=df_val, df_val=df_val, answer_column="sentence", target_column="label", id_column="sent_id", base_model=model, num_pairs_per_example=None, save_model=True, num_epochs=epochs, batch_size=bs, do_warmup=True, respect_domains=respect_domains)
    model = SentenceTransformer(os.path.join(target_folder, prompt_ours, 'finetuned_model'))

    df_train['embedding'] = df_train['sentence'].apply(model.encode)
    df_val['embedding'] = df_val['sentence'].apply(model.encode)
    df_test['embedding'] = df_test['sentence'].apply(model.encode)

    target_folder = 'results_cross'

    if not os.path.exists(os.path.join(target_folder, prompt_ours)):
        os.mkdir(os.path.join(target_folder, prompt_ours))
    # Predict on our split
    # eval_sbert_within_domain(run_path=os.path.join(target_folder, prompt_ours), df_test=df_val, df_ref=df_train, id_column='sent_id', answer_column='sentence', target_column='label')
    eval_sbert(run_path=os.path.join(target_folder, prompt_ours), df_test=df_val, df_ref=df_train, id_column='sent_id', answer_column='sentence', target_column='label')
    write_submission(os.path.join(target_folder, prompt_ours, 'predictions_sim.csv'), val_path, os.path.join(target_folder, prompt_ours, 'predicted_our_split.json'))
    # Predict on challenge test data
    if not os.path.exists(os.path.join(target_folder, prompt_test)):
        os.mkdir(os.path.join(target_folder, prompt_test))
    # eval_sbert_within_domain(run_path=os.path.join(target_folder, prompt_test), df_test=df_test, df_ref=df_train, id_column='sent_id', answer_column='sentence', target_column='label')
    eval_sbert(run_path=os.path.join(target_folder, prompt_test), df_test=df_test, df_ref=df_train, id_column='sent_id', answer_column='sentence', target_column='label')
    write_submission(os.path.join(target_folder, prompt_test, 'predictions_sim.csv'), test_path, os.path.join(target_folder, prompt_test, 'predicted.json'))

    ## Calculate metrics for the internal split
    pred, gold = load_prediction_and_gold(os.path.join(target_folder, prompt_ours, 'predicted_our_split.json'), val_path)
    per_domain, mean = eval_across_domains(gold, pred)

    with open(os.path.join(target_folder, prompt_ours, 'scores.txt'), "w+") as f:
        for k, v in per_domain.items():
            f.write(f"f1_{k}:{v}\n")
        f.write(f"f1_mean:{mean}")
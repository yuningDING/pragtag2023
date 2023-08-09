import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm
import os
from transformers import LongformerTokenizerFast, LongformerForTokenClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score


class Constants:
    OUTPUT_LABELS = ['O', 'B-Other', 'I-Other', 'B-Recap', 'I-Recap', 'B-Strength', 'I-Strength', 'B-Structure','I-Structure', 'B-Todo', 'I-Todo', 'B-Weakness', 'I-Weakness']
    LABELS_TO_IDS = {v: k for k, v in enumerate(OUTPUT_LABELS)}
    IDS_TO_LABELS = {k: v for k, v in enumerate(OUTPUT_LABELS)}


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def json_to_sequence_tagging_dataframe(json_df, test=False):
    essay_id = []
    label = []
    text = []
    prediction_string = []
    domain = []
    sentence_id = []
    for index, row in json_df.iterrows():
        sentences = row['sentences']
        essay_id.extend([row['id']] * len(sentences))
        domain.extend([row['domain']] * len(sentences))
        count_t = 0
        for i in range(len(sentences)):
            sentence_id.append(row['id']+'_'+str(i))
            text.append(sentences[i])
            if test is False:
                label.append(row['labels'][i])
                token_indexes = []
                for t in sentences[i].split():
                    token_indexes.append(count_t)
                    count_t += 1
                prediction_string.append(' '.join([str(x) for x in token_indexes]))

    if test is False:
        csv_df = pd.DataFrame(list(zip(sentence_id, essay_id, domain, label, text, prediction_string)), columns=['sentence_id','essay_id', 'domain', 'label', 'text', 'prediction_string'])
    else:
        csv_df = pd.DataFrame(list(zip(sentence_id, essay_id, domain, text)), columns=['sentence_id', 'essay_id', 'domain', 'text'])

    return csv_df


def agg_essays(folder):
    names, texts = [], []
    for f in tqdm(list(os.listdir(folder))):
        names.append(f.replace('.txt', ''))
        texts.append(open(folder + '/' + f, 'r', encoding='utf-8').read().strip())
    df_texts = pd.DataFrame({'essay_id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts


def ner(df_texts, df_train):
    all_entities = []
    for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        total = len(row.text_split)
        entities = ['O'] * total

        for _, row2 in df_train[df_train['essay_id'] == row['essay_id']].iterrows():
            discourse = row2['label']
            list_ix = [int(x) for x in row2['prediction_string'].split(' ')]
            entities[list_ix[0]] = f'B-{discourse}'
            for k in list_ix[1:]:
                try:
                    entities[k] = f'I-{discourse}'
                except IndexError:
                    print(row['essay_id'])
                    print(row2['discourse_text'])
                    print('predictionstring index:', k)
                    print('max length of text:', total)
        all_entities.append(entities)

    df_texts['BIO_Tags'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts


def preprocess(df_gold, txt_folder, test=False):
    df_texts = agg_essays(txt_folder)
    if test:
        #df_gold = json_to_sequence_tagging_dataframe(json_df, True)
        df_texts = df_texts[df_texts['essay_id'].isin(set(df_gold['essay_id'].unique()))].reset_index()
    else:
        #df_gold = json_to_sequence_tagging_dataframe(json_df)
        df_texts = df_texts[df_texts['essay_id'].isin(set(df_gold['essay_id'].unique()))].reset_index()
        df_texts = ner(df_texts, df_gold)
    return df_texts, df_gold


def build_model_tokenizer(model, model_path=None):
    tokenizer = LongformerTokenizerFast.from_pretrained(model, add_prefix_space=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = LongformerForTokenClassification.from_pretrained(model, num_labels=len(Constants.OUTPUT_LABELS))
    model.resize_token_embeddings(len(tokenizer))
    model.longformer.embeddings.word_embeddings.padding_idx = 1
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    return model, tokenizer


class ReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.text_split[index]
        word_labels = self.data.BIO_Tags[index]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  padding='max_length',
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [Constants.LABELS_TO_IDS[label] for label in word_labels]

        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                try:
                    encoded_labels[idx] = labels[i]
                    i += 1
                except IndexError:
                    #print("Length of labels:" + str(len(labels)))
                    #print("Length of encoded_labels:" + str(len(encoded_labels)))
                    #print("IndexError with idx [" + str(idx) + "] and label number [" + str(i) + "]")
                    continue

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['label'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len


def load_data(dataframe, batch_size):
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }
    return DataLoader(dataframe, **train_params)


def model_train(training_loader, model, optimizer, device, max_norm):

    loss_log = []

    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.to(device)
    model.train()

    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['label'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        tr_logits = outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print("Training loss per 100 training steps: " + str(loss_step))
            loss_log.append(loss_step)

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_norm
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps

    loss_log.append(epoch_loss)

    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

    return loss_log, tr_accuracy


def get_sentence_predictions(device, model, max_len, tokenizer, sentence):
    inputs = tokenizer(sentence,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=max_len,
                       return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits,
                                         axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [Constants.OUTPUT_LABELS[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue

    # If the last wordpiece is the end of the input sequence, there is no padding, i.e. the entire text was processed
    # But not everything fit into longformer: Fill remaining token positions with O prediction
    if (len(sentence) != len(prediction)) and wp_preds[-1][0] == "</s>":
        while (len(prediction) < len(sentence)):
            prediction.append("O-Argument")
    return prediction


def most_frequent(List):
    unique, counts = np.unique(List, return_counts=True)
    index = np.argmax(counts)
    return unique[index]


def get_final_prediction(json_df, pred):
    final_pred = {}
    for essay_id in pred.keys():
        review_pred = pred.get(essay_id)
        sentences = json_df[json_df['id'] == essay_id]['sentences']
        #print(essay_id)
        for s_list in sentences:
            final_pred_per_review = []
            sentence_start = 0
            for s in s_list:
                sentence_end = sentence_start + len(s.split())
                #print('start', sentence_start)
                #print('end', sentence_end)
                tokens = review_pred[sentence_start: sentence_end]
                if sentence_end == len(s.split()):
                    tokens = review_pred[sentence_start:]
                #print(tokens)
                prediction = most_frequent(tokens).replace('B-', '').replace('I-','')
                if sentence_start == 0:
                    prediction = "Structure"
                if prediction.startswith('O'):
                    prediction = "Todo"      
                final_pred_per_review.append(prediction)
                sentence_start = sentence_end
            final_pred[essay_id] = final_pred_per_review
    #print(final_pred)
    json_df['labels'] = json_df['id'].map(final_pred)
    return json_df


def model_predict(device, model, max_len, tokenizer, dataframe):
    pred = {}
    for index, row in dataframe.iterrows():
        o = get_sentence_predictions(device, model, max_len, tokenizer, row['text_split'])
        pred[row['essay_id']] = o

    return pred

'''
def model_evaluate(data_pred, data_gold, label):
    col_list_gold = ["essay_id", label, "prediction_string"]
    data_gold = data_gold[col_list_gold]

    # print(data_pred.columns)
    col_list_pred = ["essay_id", label, "prediction_string"]
    data_pred.reset_index(drop=True)
    data_pred = data_pred[col_list_pred]

    overall_f1, scores = score_feedback_comp(data_gold, data_pred,label)
    print("Overall F1 evaluation:", overall_f1)

    gold = pd.Series(list(data_gold[label]), name='Gold')
    pred = pd.Series(list(data_pred[label]), name='Predicted')
    try:
        cm = pd.crosstab(gold, pred, margins=True)
    except:
        cm = pd.DataFrame()

    return overall_f1, scores, cm
'''

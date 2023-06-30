import gc
import os
import random

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, recall_score, confusion_matrix, cohen_kappa_score)
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, AutoConfig, AutoModel, BertTokenizer, BertModel, logging

np.set_printoptions(threshold=10_000)
logging.set_verbosity_error()

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Constants:
    OUTPUT_LABELS = ['Other', 'Recap', 'Strength', 'Structure', 'Todo', 'Weakness']
    IDS_TO_LABELS = {k: v for k, v in enumerate(OUTPUT_LABELS)}


class PragTagModel(torch.nn.Module):
    def __init__(self, model_name):
        super(PragTagModel, self).__init__()
        if 'roberta' in model_name:
            self.model = RobertaModel.from_pretrained(model_name)
        else:
            self.model = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        if model_name == 'roberta-base':
            self.linear = torch.nn.Linear(768, len(Constants.OUTPUT_LABELS))
        else:
            self.linear = torch.nn.Linear(1024, len(Constants.OUTPUT_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        output = self.dropout(output.pooler_output)
        output = self.linear(output)

        return output


class PragTagModelWithExtraFeatures(torch.nn.Module):
    def __init__(self, model_name, num_extra_dims, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.dropout = torch.nn.Dropout(0.3)
        if model_name == 'roberta-base':
            self.linear = torch.nn.Linear(768, len(Constants.OUTPUT_LABELS))
        else:
            self.linear = torch.nn.Linear(1024, len(Constants.OUTPUT_LABELS))

        num_hidden_size = self.transformer.config.hidden_size
        self.classifier = torch.nn.Linear(num_hidden_size + num_extra_dims, num_labels)


    def forward(self, input_ids, extra_data, attention_mask=None):
        hidden_states = self.transformer(input_ids=input_ids,attention_mask=attention_mask)  # [batch size, sequence length, hidden size]
        cls_embeds = hidden_states.last_hidden_state[:, 0, :]  # [batch size, hidden size]
        concat = torch.cat((cls_embeds, extra_data), dim=-1)  # [batch size, hidden size+num extra dims]
        output = self.classifier(concat)  # [batch size, num labels]
        return output


def build_model_tokenizer(model_name, with_extra_features, num_extra_dims, model_path=None):
    # Tokenizer
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    # Modell
    if with_extra_features:
        model = PragTagModelWithExtraFeatures(model_name, num_extra_dims, len(Constants.OUTPUT_LABELS))
    else:
        model = PragTagModel(model_name)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    return tokenizer, model


class PragTagDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, data_path, plus_text, extra_feature):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.text = data['text'].values
        self.type = data['label'].values
        self.domain = data['domain'].values
        self.targets = data[Constants.OUTPUT_LABELS].values
        self.essay_id = data['essay_id'].values
        self.plus_text = plus_text
        self.extra_feature = extra_feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(text.lower(),
                                            truncation=True,
                                            padding='max_length',
                                            add_special_tokens=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            max_length=self.max_len,
                                            return_tensors='pt')
        if self.plus_text:
            essay_path = os.path.join(self.data_path, f"{self.essay_id[index]}.txt")
            # --- discourse [SEP] essay ---
            essay = open(essay_path, 'r').read()
            inputs = self.tokenizer(text.lower(), essay.lower(), truncation=True, padding='max_length',
                                    add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True,
                                    max_length=self.max_len, return_tensors='pt')

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs['token_type_ids'].flatten()
        targets = torch.FloatTensor(self.targets[index])

        if len(self.extra_feature) > 0:
            return {'input_ids': input_ids, 'attention_mask': attention_mask,
                    'extra_data': torch.FloatTensor(self.extra_feature[index]), 'token_type_ids': token_type_ids,
                    'targets': targets}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'targets': targets}
    

class PragTagDatasetWithoutTargets(Dataset):
    def __init__(self, data, max_len, tokenizer, data_path, plus_text, extra_feature):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.text = data['text'].values
        self.domain = data['domain'].values
        self.essay_id = data['essay_id'].values
        self.plus_text = plus_text
        self.extra_feature = extra_feature
        self.sentence_id = data['sentence_id'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence_id = self.sentence_id
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(text.lower(),
                                            truncation=True,
                                            padding='max_length',
                                            add_special_tokens=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            max_length=self.max_len,
                                            return_tensors='pt')
        if self.plus_text:
            essay_path = os.path.join(self.data_path, f"{self.essay_id[index]}.txt")
            # --- discourse [SEP] essay ---
            essay = open(essay_path, 'r').read()
            inputs = self.tokenizer(text.lower(), essay.lower(), truncation=True, padding='max_length',
                                    add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True,
                                    max_length=self.max_len, return_tensors='pt')

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs['token_type_ids'].flatten()

        if len(self.extra_feature) > 0:
            return {'input_ids': input_ids, 'attention_mask': attention_mask,
                    'extra_data': torch.FloatTensor(self.extra_feature[index]), 'token_type_ids': token_type_ids}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def get_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    return optimizer


def train_model(n_epochs,
                train_loader,
                val_loader,
                test_loader,
                model, lr,
                device, extra_data=None):
    test_preds = {}
    val_preds = {}
    optimizer = get_optimizer(model, lr)
    model.to(device)
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        print(f' Epoch: {epoch + 1} - Train Set '.center(50, '='))
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.float)
            if extra_data is None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                extra_data = batch['extra_data'].to(device, dtype=torch.long)
                outputs = model(input_ids, extra_data, attention_mask)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            del input_ids, attention_mask, token_type_ids, targets, outputs
            gc.collect()

        print(f' Epoch: {epoch + 1} - Validation Set '.center(50, '='))
        val_targets = []
        val_outputs = []
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader)):
                input_ids = data['input_ids'].to(device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                if extra_data is None:
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    extra_data = data['extra_data'].to(device, dtype=torch.long)
                    outputs = model(input_ids, extra_data, attention_mask)
                loss = loss_fn(outputs, targets)
                val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.item() - val_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                del input_ids, attention_mask, token_type_ids, targets, outputs
                gc.collect()
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            print('Epoch: {} \tAvgerage Training Loss: {:.6f}'.format(
                epoch + 1,
                train_loss,
            ))
            #print(val_targets)
            val_targets = np.argmax(val_targets, axis=1)
            #print(val_targets)
            #print("-------")
            #print(val_outputs)
            val_outputs = np.argmax(val_outputs, axis=1)
            #print(val_outputs)
            accuracy = accuracy_score(val_targets, val_outputs)
            f1_score_macro = f1_score(val_targets, val_outputs, average='macro')
            print(f"Accuracy Score: {round(accuracy, 4)}")
            print(f"F1 Score (Macro): {round(f1_score_macro, 4)} \n")
            cm = confusion_matrix(val_targets, val_outputs)
            print("Confusion Matrix:")
            print(cm)
        val_preds[epoch] = val_outputs
        if extra_data is None:
            test_preds[epoch] = model_predict(device, model, test_loader)
        else:
            test_preds[epoch] = model_predict(device, model, test_loader, extra_data)

    return test_preds, val_preds, model


def model_predict(device, model, test_loader, extra_data=None):
    print('Test')
    model.eval()
    test_outputs = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            if extra_data is None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                extra_data = data['extra_data'].to(device, dtype=torch.long)
                outputs = model(input_ids, extra_data, attention_mask)
            test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        test_outputs = np.argmax(test_outputs, axis=1)     

    return test_outputs


def get_final_prediction(json_df, pred):
    final_pred = {}
    i = 0
    for index, row in json_df.iterrows():
        sentences = row['sentences']
        essay_id = row['id']
        prediction = []
        for s in sentences:
            #print(s)
            if i<len(pred):
                #print(pred[i])
                prediction.append(Constants.OUTPUT_LABELS[pred[i]])
                i+=1
            else:
                break
        final_pred[essay_id]=prediction
    json_df['labels'] = json_df['id'].map(final_pred)
    return json_df


def get_position(dataframe):
    dataframe['sentence_index'] = dataframe.groupby('essay_id').cumcount()
    a = dataframe.groupby('essay_id').size().reset_index(name='size')
    a.index = a['essay_id']
    dataframe['sentence_number'] = dataframe['essay_id'].map(a['size'] - 1)
    dataframe['sentence_position'] = dataframe['sentence_index'] / dataframe['sentence_number']
    return dataframe[['sentence_index', 'sentence_position']].to_numpy()
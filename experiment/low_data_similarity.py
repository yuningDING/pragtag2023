from train_sbert import train_and_eval_sbert

# To prepare submission
test_path = 'data/test_inputs.json'

# To adjust model on
train_path = 'data/our_split/train_low.json'
val_path = 'data/our_split/validation_low.json'

# Result subfolder
prompt = 'low_data_similarity'

train_and_eval_sbert(train_path=train_path, val_path=val_path, test_path=test_path, prompt=prompt)
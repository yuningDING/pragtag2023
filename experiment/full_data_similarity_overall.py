from train_sbert import train_and_eval_sbert

train_and_eval_sbert(train_path='data/our_split/train.json', val_path='data/our_split/validation.json', test_path='data/test_inputs.json', prompt='full_data_similarity_overall_within', bs=32, epochs=5)
train_and_eval_sbert(train_path='data/our_split/train.json', val_path='data/our_split/validation.json', test_path='data/test_inputs.json', prompt='full_data_similarity_overall_within_large_run2', bs=32, epochs=5, model="all-MiniLM-L12-v2")
train_and_eval_sbert(train_path='data/our_split/train.json', val_path='data/our_split/validation.json', test_path='data/test_inputs.json', prompt='full_data_similarity_overall_cross', bs=64, epochs=5, respect_domains=False)
    


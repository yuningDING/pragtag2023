import pandas as pd

validation_ids = set()

with open('data/our_split/val_ids.txt', 'r') as val_ids:
    for val_id in val_ids:
        validation_ids.add(val_id.strip())

print("Validation IDs:", validation_ids)

df_full = pd.read_json('data/train_inputs_full.json')

df_val = df_full[df_full['id'].isin(validation_ids)]
df_train = df_full.drop(df_val.index)

# # print(len(df_full))
# # print(len(df_train))
# # print(len(df_val))

df_train.to_json('data/our_split/train.json', orient='records')
df_val.to_json('data/our_split/validation.json', orient='records') 
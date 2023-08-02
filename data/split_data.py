import pandas as pd


def split(val_id_file, full_data_json, condition):

    validation_ids = set()

    with open(val_id_file, 'r') as val_ids:
        for val_id in val_ids:
            validation_ids.add(val_id.strip())

    print("Validation IDs:", validation_ids)

    df_full = pd.read_json(full_data_json)

    df_val = df_full[df_full['id'].isin(validation_ids)]
    df_train = df_full.drop(df_val.index)

    # # print(len(df_full))
    # # print(len(df_train))
    # # print(len(df_val))

    df_train.to_json('data/our_split/train' + condition + '.json', orient='records')
    df_val.to_json('data/our_split/validation' + condition + '.json', orient='records') 


## LIMITED DATA
split('data/our_split/val_ids_low.txt', 'data/train_inputs_low.json', '_low')

## FULL DATA
split('data/our_split/val_ids.txt', 'data/train_inputs_full.json', '')